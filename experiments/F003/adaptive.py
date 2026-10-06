"""F003 (T009): adaptive configurations around the frozen D026 TinyViT checkpoint.

Every configuration reuses the same checkpoint weights. The configuration axes
are frame count K, exit depth, and input resolution. Exit heads are the only
new parameters; they are fitted on train-split features of the frozen trunk.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from src import f002_pipeline as pipe
from src.f002_flops import BOOKKEEPING_ALLOWLIST, CUSTOM_HANDLERS
from src.f002_protocol import nested_frame_indices
from experiments.F002c.model import F002cSequences, TinyViTSequence

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "f002c_tinyvit5m_epoch8.pt"
CHECKPOINT_SHA256 = "30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2"
K_VALUES = (1, 2, 4, 8, 12, 16)
EXITS = ("stage1", "stage2_mid", "stage2", "final")
EXIT_DIMS = {"stage1": 128, "stage2_mid": 160, "stage2": 160, "final": 320}
RESOLUTIONS = (160, 192, 224)
WINDOW_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
TEMPORAL_K = (4, 8, 16)


# ---------------------------------------------------------------- frames

def frame_indices(frame_count: int, k: int) -> tuple[int, ...]:
    """Nested frame selection extending the frozen F002 sampler to more K.

    K4/K8/K16 equal ``nested_frame_indices``. K12 is the same widest-gap
    bisection stopped at 12 frames. K2 keeps the two interior thirds anchors
    and K1 the first of them, so K1 ⊂ K2 ⊂ K4 ⊂ K8 ⊂ K12 ⊂ K16.
    """
    if k not in K_VALUES or frame_count < 16:
        raise ValueError(f"F003 requires K in {K_VALUES} and at least 16 frames")
    anchors = [j * (frame_count - 1) // 3 for j in range(4)]
    if k == 1:
        return (anchors[1],)
    if k == 2:
        return (anchors[1], anchors[2])
    selected = set(anchors)
    while len(selected) < k:
        ordered = sorted(selected)
        left, right = max(zip(ordered, ordered[1:]),
                          key=lambda pair: (pair[1] - pair[0], -pair[0]))
        selected.add((left + right) // 2)
    result = tuple(sorted(selected))
    if k in (4, 8, 16) and result != nested_frame_indices(frame_count, k):
        raise AssertionError("F003 sampler diverged from the frozen F002 sampler")
    return result


def positions_in_k16(frame_count: int, k: int) -> list[int]:
    """Positions of the K frames inside the frozen K16 selection."""
    k16 = nested_frame_indices(frame_count, 16)
    return [k16.index(index) for index in frame_indices(frame_count, k)]


def window_indices(frame_count: int, fraction: float, k: int) -> tuple[int, ...]:
    """K frames spread over the first ``fraction`` of the clip.

    Scan-duration axis: the window is the first max(16, ceil(fraction × N))
    frames (never more than N), and the usual nested rule spreads K frames
    over that window. Fraction 1.0 equals ``frame_indices``. Frame rate is
    unknown, so the window is a share of the clip, not a number of seconds.
    """
    if fraction not in WINDOW_FRACTIONS or k not in TEMPORAL_K:
        raise ValueError("Unknown F003 temporal window")
    length = min(frame_count, max(16, math.ceil(fraction * frame_count)))
    return frame_indices(length, k)


def dense_indices(frame_count: int, k: int) -> tuple[int, ...]:
    """The first K consecutive frames: native sampling rate, shortest scan."""
    if k not in TEMPORAL_K or frame_count < k:
        raise ValueError("Unknown F003 dense configuration")
    return tuple(range(k))


def load_frame(path: Path, resolution: int = 224) -> torch.Tensor:
    """F002c preprocessing: RGB, full-square bicubic resize, ImageNet norm."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
        return TF.normalize(TF.to_tensor(image), pipe.MEAN, pipe.STD)


class F003Sequences(F002cSequences):
    """F002c train/validation loader at K16, with resolution and K subsets.

    Construction always inventories the frozen K16 selection; ``frames``
    returns any nested K subset at any square resolution, using F002c's
    bicubic full-square preprocessing.
    """

    def __init__(self, rows: list[dict], resolution: int = 224):
        super().__init__(rows, 16)
        if resolution not in RESOLUTIONS:
            raise ValueError(f"F003 resolution must be one of {RESOLUTIONS}")
        self.resolution = resolution

    def positions(self, index: int, k: int) -> list[int]:
        return positions_in_k16(int(self.rows[index]["frame_count"]), k)

    def frames(self, index: int, k: int = 16) -> torch.Tensor:
        paths = [self.selected[index][1][p] for p in self.positions(index, k)]
        return torch.stack([load_frame(path, self.resolution) for path in paths])

    def frames_at(self, index: int, frame_numbers) -> torch.Tensor:
        """Arbitrary retained frames of a row, by zero-based index (hash-checked)."""
        paths = pipe.retained_frames(self.rows[index], self.raw)
        return torch.stack([load_frame(paths[i], self.resolution) for i in frame_numbers])


# ---------------------------------------------------------------- model

def load_backbone(device="cpu") -> TinyViTSequence:
    """The D026 checkpoint, hash-checked, in eval mode with unfused attention."""
    if pipe.sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("D026 TinyViT checkpoint hash differs from the frozen record")
    model = TinyViTSequence(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    return model.to(device).eval()


class Trunk(nn.Module):
    """The TinyViT encoder split at the F003 exits; shares the backbone modules.

    Returns pooled per-frame features for every exit up to ``until``. The
    ``final`` feature is exactly the encoder output used by F002c.
    """

    def __init__(self, encoder: nn.Module, until: str = "final"):
        super().__init__()
        if until not in EXITS:
            raise ValueError(f"Unknown exit {until}")
        self.encoder, self.until = encoder, until

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        e, out = self.encoder, {}
        x = e.stages[1](e.stages[0](e.patch_embed(frames)))
        out["stage1"] = x.mean(dim=(2, 3))
        if self.until == "stage1":
            return out
        stage = e.stages[2]
        y = stage.downsample(x).permute(0, 2, 3, 1)
        y = stage.blocks[:3](y)
        out["stage2_mid"] = y.mean(dim=(1, 2))
        if self.until == "stage2_mid":
            return out
        y = stage.blocks[3:](y)
        out["stage2"] = y.mean(dim=(1, 2))
        if self.until == "stage2":
            return out
        out["final"] = e.forward_head(e.stages[3](y.permute(0, 3, 1, 2)))
        return out


class ExitSequenceModel(nn.Module):
    """One F003 configuration: frames B×K×3×R×R → one logit per sequence.

    Mean-pools the chosen exit's per-frame features over K, then applies the
    exit head. ``final`` uses the checkpoint's own binary head.
    """

    def __init__(self, backbone: TinyViTSequence, exit_name: str = "final",
                 head: nn.Linear | None = None):
        super().__init__()
        if (exit_name == "final") != (head is None):
            raise ValueError("Only non-final exits take a fitted head")
        self.trunk = Trunk(backbone.encoder, exit_name)
        self.exit_name = exit_name
        self.head = backbone.head if head is None else head

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3 or frames.shape[3] != frames.shape[4]:
            raise ValueError("Expected B × K × 3 × R × R frames")
        batch, k = frames.shape[:2]
        features = self.trunk(frames.reshape(batch * k, *frames.shape[2:]))[self.exit_name]
        return self.head(features.reshape(batch, k, -1).mean(dim=1)).squeeze(-1)


def set_fused_attention(model: nn.Module, enabled: bool) -> None:
    modules = [m for m in model.modules() if hasattr(m, "fused_attn")]
    if not modules:
        raise ValueError("Model has no attention modules")
    for module in modules:
        module.fused_attn = enabled


# ---------------------------------------------------------------- exit heads

def fit_exit_head(features, targets, weights) -> tuple[nn.Linear, dict]:
    """Standardized L2 logistic regression folded into one ``nn.Linear``.

    Fixed C=1, no class weights, threshold 0.5 at use, duplicate-aware sample
    weights, train split only. Folding keeps the head a single linear layer so
    FLOPs and runtime include it like the F002 heads.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", class_weight=None,
                               random_state=2026)
    model.fit(scaler.transform(x), targets, sample_weight=weights)
    coef = model.coef_[0] / scaler.scale_
    bias = model.intercept_[0] - float((coef * scaler.mean_).sum())
    head = nn.Linear(x.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(coef, dtype=torch.float32).unsqueeze(0))
        head.bias.fill_(bias)
    return head, {"n_iter": int(model.n_iter_[0]), "C": 1.0,
                  "train_rows": int(x.shape[0]), "dim": int(x.shape[1])}


# ---------------------------------------------------------------- FLOPs

def _scalar_only(name):
    """Shape arithmetic traced as 0-d tensors (window padding sizes): no FLOPs.

    Fails closed if the operator ever acts on a real tensor.
    """
    def handler(inputs, outputs):
        sizes = outputs[0].type().sizes()
        if sizes is None or math.prod(sizes) != 1:
            raise RuntimeError(f"{name} on a non-scalar tensor must be counted")
        return Counter()
    return handler


# TinyViT-specific bookkeeping reviewed for F003: views/splits/padding/copies
# move data; ``aten::index`` gathers the attention-bias table.
TINYVIT_BOOKKEEPING = frozenset({"aten::view", "aten::split_with_sizes", "aten::index",
                                 "aten::pad", "aten::constant_pad_nd", "aten::contiguous"})
HANDLERS = CUSTOM_HANDLERS | {name: _scalar_only(name) for name in
                              ("aten::rsub", "aten::remainder", "aten::floor_divide")}


def count_flops(model: ExitSequenceModel, k: int, resolution: int = 224) -> dict:
    """Strict fvcore count of one batch-one request, F002 convention."""
    from fvcore.nn import FlopCountAnalysis

    allowed = BOOKKEEPING_ALLOWLIST | TINYVIT_BOOKKEEPING

    class Strict(FlopCountAnalysis):
        def _should_ignore_node(self, node):
            kind = node.kind()
            if kind in allowed:
                return True
            if super()._should_ignore_node(node):
                raise RuntimeError(f"Unapproved fvcore-ignored F003 operator: {kind}")
            return False

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, k, 3, resolution, resolution, device=device)
    analysis = Strict(model, dummy).set_op_handle(**HANDLERS)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = int(analysis.total())
    unsupported = dict(analysis.unsupported_ops())
    if unsupported:
        raise RuntimeError(f"Uncounted F003 operators: {unsupported}")
    by_operator = dict(analysis.by_operator())
    linear = model.head if isinstance(model.head, nn.Linear) else model.head[-1]
    head = int(analysis.by_module_and_operator().get(
        "head" if isinstance(model.head, nn.Linear) else "head.1", {}).get("linear", 0))
    if head != linear.in_features or by_operator.get("mean_pool", 0) <= 0 or total <= 0:
        raise RuntimeError("Incomplete F003 exit/pool/head FLOPs trace")
    if by_operator.get("matmul", 0) <= 0 or by_operator.get("softmax", 0) <= 0:
        raise RuntimeError("Attention FLOPs omitted")
    return {"flops": total, "by_operator": by_operator, "head_flops": head, "k": k,
            "resolution": resolution, "exit": model.exit_name,
            "convention": "fvcore_one_fma_one_flop; F002 handlers; TinyViT bookkeeping reviewed in F003"}
