"""F002c-only TinyViT sequence encoder and train/validation image loader.

The frozen F002 sequence inventory supplies deterministic nested frame indices.
This module changes only F002c's preprocessing, model, and optimizer recipe.
"""
from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src import f002_pipeline as pipe
from src.f002_protocol import key


MODEL_NAME = "tiny_vit_5m_224.dist_in22k_ft_in1k"
PRETRAINED_REVISION = "b46989c2a2c95e7612b919a6f5e2cd7dcf4f1271"


class TinyViTSequence(nn.Module):
    """Mean pretrained per-frame embeddings, followed by one binary logit.

    ``num_classes=0`` keeps timm's pretrained spatial pooling and feature
    normalization. Only the original ImageNet classifier is discarded.
    Inputs are B x K x 3 x 224 x 224; outputs are B sequence logits.
    """

    def __init__(self, pretrained: bool = True, drop_path_rate: float = 0.1,
                 head_dropout: float = 0.1):
        super().__init__()
        self.encoder = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,
            pretrained_cfg_overlay={"hf_hub_id": f"timm/{MODEL_NAME}@{PRETRAINED_REVISION}"},
        )
        if self.encoder.num_features != 320:
            raise ValueError("F002c requires the TinyViT-5M 320-feature encoder")
        for module in self.encoder.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(320, 1))
        self.train()

    def train(self, mode: bool = True):
        """Freeze BN statistics while retaining trainable affine parameters.

        Consecutive frames are correlated: a 32-frame forward contains only
        two independent sequence rows. Reapply this after every mode switch.
        The remaining encoder modules retain their requested training mode.
        """
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 224, 224):
            raise ValueError("Expected B x K x 3 x 224 x 224 frames")
        batch, k = frames.shape[:2]
        if batch < 1 or k not in (4, 8, 16):
            raise ValueError("F002c requires nonempty batches and K in {4,8,16}")
        features = self.encoder(frames.reshape(batch * k, 3, 224, 224))
        features = features.reshape(batch, k, 320).mean(dim=1)
        return self.head(features).squeeze(-1)


def build_optimizer(model: TinyViTSequence, backbone_lr: float = 5e-5,
                    head_lr: float = 5e-4, weight_decay: float = 0.01):
    """AdamW with separate encoder/head rates and no bias/norm/position decay."""
    if backbone_lr <= 0 or head_lr <= 0 or weight_decay < 0:
        raise ValueError("Learning rates must be positive and decay nonnegative")
    encoder_exclusions = set(model.encoder.no_weight_decay()) if hasattr(
        model.encoder, "no_weight_decay") else set()
    excluded_keywords = set(model.encoder.no_weight_decay_keywords()) if hasattr(
        model.encoder, "no_weight_decay_keywords") else set()
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        local_name = name[len("encoder."):] if is_encoder else name
        no_decay = (parameter.ndim <= 1 or name.endswith(".bias")
                    or "attention_biases" in name
                    or (is_encoder and local_name in encoder_exclusions)
                    or (is_encoder and any(keyword in local_name
                                           for keyword in excluded_keywords)))
        group_key = (is_encoder, no_decay)
        if group_key not in groups:
            groups[group_key] = {
                "params": [],
                "lr": backbone_lr if is_encoder else head_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": ("encoder" if is_encoder else "head")
                        + ("_no_decay" if no_decay else "_decay"),
            }
        groups[group_key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8)


class F002cSequences(pipe.Sequences):
    """Frozen frame selection with F002c's bicubic, uncropped preprocessing.

    ``augment=True`` samples one horizontal flip for the whole train sequence.
    Construction rejects test rows before the base class can inventory files.
    """

    def __init__(self, rows: list[dict], k: int = 16, raw: Path = pipe.RAW,
                 augment: bool = False):
        partitions = {row.get("partition") for row in rows}
        if not rows or partitions not in ({"train"}, {"validation"}):
            raise ValueError("F002c image loading requires only train or validation rows")
        if augment and partitions != {"train"}:
            raise ValueError("Augmentation is restricted to train rows")
        # The frozen verifier reads only tracked cohort metadata, including test
        # manifest hashes. It neither inventories nor opens test images. Check
        # complete row equality so relabeling a test row as train cannot bypass
        # this loader's restriction, or silently change retained image paths.
        partition = next(iter(partitions))
        allowed = {key(row): row for row in pipe.load_inputs()[partition]}
        if len({key(row) for row in rows}) != len(rows) or any(
                allowed.get(key(row)) != row for row in rows):
            raise ValueError("Rows must match the verified T006 train/validation manifest")
        self.augment = augment
        super().__init__(rows, k, raw)

    def frames(self, index: int) -> torch.Tensor:
        tensors = []
        for path in self.selected[index][1]:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                tensors.append(TF.normalize(TF.to_tensor(image), pipe.MEAN, pipe.STD))
        frames = torch.stack(tensors)
        if self.augment and torch.rand(()).item() < 0.5:
            frames = frames.flip(-1)
        return frames


class SequenceDataset(Dataset):
    """DataLoader adapter returning frames, target, row weight, and row index.

    ``weights`` maps frozen (exam_id, zone_id) keys to duplicate-aware weights.
    Omit it for validation. Rows and sampled-file provenance stay accessible
    through ``sequences.rows`` and ``sequences.identity(index)``.
    """

    def __init__(self, sequences: F002cSequences, weights: dict | None = None):
        self.sequences = sequences
        self.weights = [float(weights[key(row)]) if weights is not None else 1.0
                        for row in sequences.rows]
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("Sequence weights must be finite and positive")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):
        row = self.sequences.rows[index]
        return (self.sequences.frames(index),
                torch.tensor(float(row["binary_target"]), dtype=torch.float32),
                torch.tensor(self.weights[index], dtype=torch.float32), index)
