"""F006b inference on the frozen F005b width × K family.

The controller chooses (width, K); this module serves that choice for one clip.
Every clip gets a fresh F005 ``NestedAnytimeClip`` cache. Its entries are keyed
by both width and original frame index, so a K upgrade can reuse features at
the same width without reusing features from another width or clip.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch
from torch import nn

from experiments.F003.adaptive import frame_indices
from experiments.F005 import prune
from experiments.F005.run_f005b import load_deployable


FrameLoader = Callable[[list[int]], torch.Tensor]


@dataclass(frozen=True)
class InferenceResult:
    width: float
    k: int
    frame_indices: tuple[int, ...]
    logit: float
    probability_abnormal: float
    predicted_class: int
    new_frames: int
    cached_frames: int
    cache_entries: int
    elapsed_ms: float


def load_runtime(device: str | torch.device = "cpu") -> nn.Module:
    """Load only the D029 canonical seed-2026 bundle, with its hash checks."""
    return load_deployable(device=device)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _validated_loader(load: FrameLoader, device: torch.device) -> FrameLoader:
    def checked(numbers: list[int]) -> torch.Tensor:
        frames = load(numbers)
        if (not isinstance(frames, torch.Tensor) or frames.ndim != 4 or
                tuple(frames.shape[1:]) != (3, 224, 224) or
                frames.shape[0] != len(numbers) or not frames.is_floating_point()):
            raise ValueError("Frame loader must return N × 3 × 224 × 224 floating-point frames")
        if frames.device != device:
            raise ValueError(f"Frame loader returned {frames.device}; model is on {device}")
        return frames
    return checked


def _validate_point(width: float, k: int, frame_count: int) -> tuple[int, ...]:
    if width not in prune.WIDTHS:
        raise ValueError(f"Width must be one of {prune.WIDTHS}")
    return frame_indices(frame_count, k)


def _check_chunk_size(chunk_size: int | None) -> None:
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer or None")


class _ChunkedEncoder(nn.Module):
    """Batch size limit around the same nested encoder; no copied weights."""

    def __init__(self, encoder: nn.Module, chunk_size: int):
        super().__init__()
        self.base = encoder
        self.chunk_size = chunk_size

    @property
    def _width_bns(self):
        # set_width scans nested groups and also updates this optional list.
        # Keep the base encoder's width-aware BN keys visible through the
        # chunking adapter rather than leaving them at a previous width.
        return getattr(self.base, "_width_bns", ())

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.base(part) for part in frames.split(self.chunk_size)], dim=0)


class _ServingModel(nn.Module):
    def __init__(self, model: nn.Module, chunk_size: int | None):
        super().__init__()
        self.encoder = model.encoder if chunk_size is None else _ChunkedEncoder(model.encoder, chunk_size)
        self.head = model.head
        self.eval()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def direct_predict(model: nn.Module, frame_count: int, load: FrameLoader,
                   width: float, k: int, *, chunk_size: int | None = None) -> float:
    """Uncached sequence logit at any F005 width × F004 K.

    ``load`` receives original clip frame indices, not K16 positions. The
    callback supplies already preprocessed float tensors on the model device.
    This computes each selected frame once and is the parity reference for the
    width-aware anytime cache. ``chunk_size`` changes batching only.
    """
    chosen = _validate_point(width, k, frame_count)
    _check_chunk_size(chunk_size)
    if model.training:
        raise ValueError("The deployable model must be in eval mode")
    device = _model_device(model)
    frames = _validated_loader(load, device)(list(chosen))
    encoder = model.encoder if chunk_size is None else _ChunkedEncoder(model.encoder, chunk_size)
    with torch.inference_mode():
        prune.set_width(model.encoder, width)
        features = encoder(frames)
        logit = model.head(features.mean(dim=0, keepdim=True)).squeeze(-1)[0]
        value = float(logit)
    if not math.isfinite(value):
        raise ValueError("Direct inference produced a non-finite logit")
    return value


class NestedClipRuntime:
    """One clip served across changing width × K requests.

    Construct a new instance (or call ``reset_clip``) for each clip. ``load``
    receives original frame indices and must return preprocessed float frames
    on the model device. The optional chunk size limits encoder batch size;
    it leaves the nested weights, frame selection and cache keys unchanged.
    ``elapsed_ms`` includes the supplied in-memory loader callback and is a
    local per-call observation, not the comparable CPU benchmark result.
    """

    def __init__(self, model: nn.Module, frame_count: int, load: FrameLoader,
                 *, chunk_size: int | None = None):
        if model.training:
            raise ValueError("The deployable model must be in eval mode")
        _check_chunk_size(chunk_size)
        self.model = model
        self.device = _model_device(model)
        self.chunk_size = chunk_size
        self._serving_model = _ServingModel(model, chunk_size)
        self.reset_clip(frame_count, load)

    def reset_clip(self, frame_count: int, load: FrameLoader) -> None:
        """Start another clip with an empty width-and-frame cache."""
        frame_indices(frame_count, 16)  # the F003 sampler requires at least 16 frames
        self.frame_count = frame_count
        self._cache = prune.NestedAnytimeClip(
            self._serving_model, frame_count, _validated_loader(load, self.device))

    def predict(self, width: float, k: int) -> InferenceResult:
        chosen = _validate_point(width, k, self.frame_count)
        _synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            logit, new_frames = self._cache.predict(width, k)
            value = float(logit)
            probability = float(torch.sigmoid(logit))
        _synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not math.isfinite(value) or not math.isfinite(probability):
            raise ValueError("Cached inference produced a non-finite prediction")
        return InferenceResult(
            width=width, k=k, frame_indices=chosen, logit=value,
            probability_abnormal=probability, predicted_class=int(probability >= 0.5),
            new_frames=new_frames, cached_frames=k - new_frames,
            cache_entries=len(self._cache.features), elapsed_ms=elapsed_ms)
