"""Preprocessed K16 frame cache for F005b training and evaluation.

F002c preprocessing is RGB → full-square bicubic 224 → to_tensor → ImageNet
normalization. The cache stores the uint8 image after the resize, so
``normalize(cache / 255)`` reproduces ``F002cSequences.frames`` bit for bit
while skipping JPEG decoding and resizing on every epoch.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src import f002_pipeline as pipe
from src.f002_protocol import key
from experiments.F002c.model import F002cSequences

MEAN = torch.tensor(pipe.MEAN).view(1, 3, 1, 1)
STD = torch.tensor(pipe.STD).view(1, 3, 1, 1)


def build(rows: list[dict]) -> dict:
    """uint8 K16 frames (N×16×3×224×224) for verified train or validation rows."""
    sequences = F002cSequences(rows, 16)  # rejects test rows and unverified manifests
    frames = torch.empty(len(rows), 16, 3, 224, 224, dtype=torch.uint8)
    for i in range(len(rows)):
        for j, path in enumerate(sequences.selected[i][1]):
            with Image.open(path) as image:
                image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                frames[i, j] = TF.pil_to_tensor(image)
    return {"frames": frames, "keys": [key(r) for r in rows],
            "frame_indices": [sequences.identity(i)["frame_indices"] for i in range(len(rows))]}


def normalize(frames: torch.Tensor) -> torch.Tensor:
    """uint8 …×3×224×224 → F002c float input."""
    return (frames.float() / 255 - MEAN.to(frames.device)) / STD.to(frames.device)


def load(path: Path, rows: list[dict]) -> torch.Tensor:
    """Cached uint8 frames, checked to belong to exactly these rows in this order."""
    record = torch.load(path, weights_only=False)
    if record["keys"] != [key(r) for r in rows]:
        raise ValueError(f"{path} does not match the requested rows")
    return record["frames"]


class CachedSequences(Dataset):
    """(frames uint8 16×3×224×224, target, weight, index); optional sequence flip."""

    def __init__(self, frames: torch.Tensor, rows: list[dict], weights: dict | None = None,
                 augment: bool = False):
        if augment and {r["partition"] for r in rows} != {"train"}:
            raise ValueError("Augmentation is restricted to train rows")
        self.frames, self.rows, self.augment = frames, rows, augment
        self.weights = [float(weights[key(r)]) if weights else 1.0 for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        frames = self.frames[index]
        if self.augment and torch.rand(()).item() < 0.5:
            frames = frames.flip(-1)
        return (frames, torch.tensor(float(self.rows[index]["binary_target"])),
                torch.tensor(self.weights[index]), index)
