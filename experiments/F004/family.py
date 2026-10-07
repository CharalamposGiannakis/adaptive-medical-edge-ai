"""F004 (T010): the TinyViT frame-count operating points as one deployable model.

One set of TinyViT weights serves every operating point. An operating point is
the number of frames K taken from a clip with the nested F003 sampler, so
K1 ⊂ K2 ⊂ K4 ⊂ K8 ⊂ K12 ⊂ K16 and a larger K reuses every frame a smaller K already
encoded. The computation per point is F002c's: encode each frame, average the
per-frame features over K, apply the binary head.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from src import f002_pipeline as pipe
from experiments.F002c.model import TinyViTSequence
from experiments.F003.adaptive import CHECKPOINT, CHECKPOINT_SHA256, frame_indices

OPERATING_POINTS = (1, 2, 4, 8, 12, 16)
FEATURES = 320


class AdaptiveTinyViT(nn.Module):
    """Frames B×K×3×224×224 → B logits, for any K in ``OPERATING_POINTS``.

    Holds the encoder and head of one ``TinyViTSequence``; no other parameters.
    ``exit_name`` lets F003's strict fvcore counter trace it unchanged.
    """

    exit_name = "final"

    def __init__(self, backbone: TinyViTSequence):
        super().__init__()
        self.encoder, self.head = backbone.encoder, backbone.head

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """n×3×224×224 frames → n×320 per-frame features."""
        return self.encoder(frames)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """B×K×320 per-frame features → B sequence logits."""
        return self.head(features.mean(dim=1)).squeeze(-1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 224, 224):
            raise ValueError("Expected B × K × 3 × 224 × 224 frames")
        batch, k = frames.shape[:2]
        if batch < 1 or k not in OPERATING_POINTS:
            raise ValueError(f"K must be one of {OPERATING_POINTS}")
        features = self.encode(frames.reshape(batch * k, 3, 224, 224))
        return self.classify(features.reshape(batch, k, FEATURES))


def load_family(checkpoint: Path = CHECKPOINT, sha256: str = CHECKPOINT_SHA256,
                device="cpu") -> AdaptiveTinyViT:
    """A hash-checked TinyViTSequence state dict as the family, eval mode, unfused attention."""
    if pipe.sha256_file(checkpoint) != sha256:
        raise ValueError(f"{checkpoint} hash differs from the expected {sha256}")
    backbone = TinyViTSequence(pretrained=False)
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return AdaptiveTinyViT(backbone).to(device).eval()


class AnytimeClip:
    """One clip served under a changing budget.

    Keeps the feature of every frame it has encoded. ``predict(k)`` encodes only
    the frames K adds to what is cached, so K4 → K16 encodes 12 frames and any
    move down to a smaller K encodes none. ``load(frame_numbers)`` returns those
    frames of the clip as n×3×224×224, on the model's device. ``encode`` maps
    such frames to n×320 features; it defaults to one batched ``model.encode``
    call (F005's INT8 model passes a per-frame encoder instead).
    """

    def __init__(self, model: AdaptiveTinyViT, frame_count: int,
                 load: Callable[[list[int]], torch.Tensor],
                 encode: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.model, self.frame_count, self.load = model, frame_count, load
        self.encode = model.encode if encode is None else encode
        self.features: dict[int, torch.Tensor] = {}

    def predict(self, k: int) -> tuple[torch.Tensor, int]:
        """Sequence logit at K and the number of frames encoded for it."""
        if k not in OPERATING_POINTS:
            raise ValueError(f"K must be one of {OPERATING_POINTS}")
        chosen = frame_indices(self.frame_count, k)
        missing = [i for i in chosen if i not in self.features]
        if missing:
            self.features.update(zip(missing, self.encode(self.load(missing))))
        stacked = torch.stack([self.features[i] for i in chosen]).unsqueeze(0)
        return self.model.classify(stacked)[0], len(missing)


class Upgrade(AdaptiveTinyViT):
    """FLOPs view of moving one clip from K = ``cached`` to a larger K.

    Input: the 1×(K − cached)×3×224×224 new frames. The cached features are a
    buffer, so the count is the new frames' encoding plus pooling and head.
    """

    def __init__(self, model: AdaptiveTinyViT, cached: int):
        nn.Module.__init__(self)
        self.encoder, self.head = model.encoder, model.head
        self.register_buffer("cached", torch.zeros(1, cached, FEATURES))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        new = self.encode(frames.reshape(-1, 3, 224, 224)).reshape(1, -1, FEATURES)
        return self.classify(torch.cat([self.cached, new], dim=1))


def read_parameters(model: AdaptiveTinyViT, k: int) -> dict[str, torch.Tensor]:
    """Every parameter a forward at K reads, by name, captured with module hooks."""
    names = {module: name for name, module in model.named_modules()}
    used: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        for local, parameter in module.named_parameters(recurse=False):
            used[f"{names[module]}.{local}" if names[module] else local] = parameter
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if any(True for _ in m.parameters(recurse=False))]
    try:
        with torch.no_grad():
            model(torch.zeros(1, k, 3, 224, 224, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return used


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
