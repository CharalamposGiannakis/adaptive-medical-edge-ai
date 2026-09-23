"""Run F001: shared-weight depth operating points for DeiT-Tiny.

This is intentionally a script rather than a notebook so the experiment can be
re-run with an explicit command and its configuration is saved beside its
outputs. The test split is not referenced anywhere in this file.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms


MODEL_NAME = "deit_tiny_patch16_224"
NUM_CLASSES = 7
IMAGE_SIZE = 224
DEPTHS = (4, 8, 12)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow global determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    """Convert common tensor/library values to JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")


def make_dataset(split: str, data_root: Path):
    """Create only an allowed train/validation DermaMNIST-224 split."""

    if split not in {"train", "val"}:
        raise ValueError("F001 only permits the train and val splits; the test split is held out.")
    data_root.mkdir(parents=True, exist_ok=True)

    import medmnist
    from medmnist import DermaMNIST

    dataset = DermaMNIST(
        split=split,
        root=str(data_root),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        download=True,
        size=IMAGE_SIZE,
        as_rgb=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"MedMNIST returned an empty {split} split.")
    if getattr(medmnist, "__version__", None):
        # Kept as a side effect-free validation hook for the saved run metadata.
        pass
    return dataset


def make_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
):
    train_dataset = make_dataset("train", data_root)
    val_dataset = make_dataset("val", data_root)
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def create_backbone(pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)


def forward_features_to_depth(backbone: nn.Module, images: Tensor, depth: int) -> Tensor:
    """Run the timm VisionTransformer through exactly ``depth`` blocks."""

    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}, got {depth}")

    # These calls mirror timm's VisionTransformer.forward_features while making
    # the block loop truncatable. They are kept in one place so training,
    # validation, FLOP counting, and latency use exactly the same computation.
    x = backbone.patch_embed(images)
    if hasattr(backbone, "_pos_embed"):
        x = backbone._pos_embed(x)
    else:  # Compatibility with older timm releases.
        if getattr(backbone, "cls_token", None) is not None:
            cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if getattr(backbone, "pos_embed", None) is not None:
            x = x + backbone.pos_embed
        x = backbone.pos_drop(x)
    if hasattr(backbone, "patch_drop"):
        x = backbone.patch_drop(x)
    if hasattr(backbone, "norm_pre"):
        x = backbone.norm_pre(x)
    x = backbone.blocks[:depth](x)
    x = backbone.norm(x)

    if getattr(backbone, "global_pool", "token") == "avg":
        x = x[:, getattr(backbone, "num_prefix_tokens", 1) :].mean(dim=1)
    else:
        x = x[:, 0]
    if hasattr(backbone, "fc_norm"):
        x = backbone.fc_norm(x)
    return x


class DepthClassifier(nn.Module):
    """A trained DeiT backbone plus heads for the early operating points."""

    def __init__(self, backbone: nn.Module, depths: Iterable[int] = (4, 8)) -> None:
        super().__init__()
        self.backbone = backbone
        feature_dim = int(backbone.num_features)
        self.early_heads = nn.ModuleDict(
            {str(depth): nn.Linear(feature_dim, NUM_CLASSES) for depth in depths}
        )

    def logits_at_depth(self, images: Tensor, depth: int) -> Tensor:
        features = forward_features_to_depth(self.backbone, images, depth)
        if depth == 12:
            features = self.backbone.head_drop(features)
            return self.backbone.head(features)
        return self.early_heads[str(depth)](features)

    def forward(self, images: Tensor, depth: int = 12) -> Tensor:
        return self.logits_at_depth(images, depth)


class FixedDepthModel(nn.Module):
    """Adapter that gives fvcore and latency code a fixed-depth forward method."""

    def __init__(self, model: DepthClassifier, depth: int) -> None:
        super().__init__()
        self.model = model
        self.depth = depth

    def forward(self, images: Tensor) -> Tensor:
        return self.model.logits_at_depth(images, self.depth)


def target_tensor(targets: Tensor) -> Tensor:
    return targets.reshape(-1).long()


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create the current device-aware AMP scaler without changing enablement."""

    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(scaler_device, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: DepthClassifier,
    loader: DataLoader,
    depth: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.backbone.eval()
    model.early_heads.eval()
    all_predictions: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = target_tensor(targets).to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model.logits_at_depth(images, depth)
            loss = criterion(logits, targets)
        total_loss += float(loss.item()) * targets.numel()
        total_items += targets.numel()
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    predictions = np.concatenate(all_predictions)
    target_values = np.concatenate(all_targets)
    return {
        "loss": total_loss / total_items,
        "accuracy": float(np.mean(predictions == target_values)),
        "macro_f1": float(
            f1_score(target_values, predictions, average="macro", zero_division=0)
        ),
    }


def train_full_model(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
) -> tuple[List[Dict[str, float]], Dict[str, Tensor]]:
    backbone.to(device)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(device, amp)
    history: List[Dict[str, float]] = []
    probe = DepthClassifier(backbone).to(device)
    best_state: Dict[str, Tensor] | None = None
    best_accuracy = float("-inf")

    for epoch in range(1, epochs + 1):
        backbone.train()
        running_loss = 0.0
        items = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = target_tensor(targets).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits = backbone(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * targets.numel()
            items += targets.numel()
        validation = evaluate(probe, val_loader, 12, device, amp)
        epoch_metrics = {
            "phase": "full_backbone",
            "epoch": epoch,
            "train_loss": running_loss / items,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_metrics)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy(backbone.state_dict())
        print(
            f"[full] epoch {epoch}/{epochs} train_loss={running_loss / items:.4f} "
            f"val_acc={validation['accuracy']:.4f}"
        )
    if best_state is None:
        raise RuntimeError("full-model training did not produce a checkpoint")
    return history, best_state


def train_early_head(
    model: DepthClassifier,
    depth: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
) -> tuple[List[Dict[str, float]], Dict[str, Tensor]]:
    """Train exactly one early head while keeping every backbone parameter frozen."""

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.early_heads.parameters():
        parameter.requires_grad_(False)
    head = model.early_heads[str(depth)]
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    model.backbone.eval()  # Frozen dropout/normalization behavior is deterministic.
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(device, amp)
    history: List[Dict[str, float]] = []
    best_state: Dict[str, Tensor] | None = None
    best_accuracy = float("-inf")

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0
        items = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = target_tensor(targets).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits = model.logits_at_depth(images, depth)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * targets.numel()
            items += targets.numel()
        val_metrics = evaluate(model, val_loader, depth, device, amp)
        epoch_metrics = {
            "phase": f"head_{depth}",
            "epoch": epoch,
            "train_loss": running_loss / items,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(head.state_dict())
        print(
            f"[head {depth}] epoch {epoch}/{epochs} "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}"
        )
    if best_state is None:
        raise RuntimeError(f"head training did not produce a checkpoint for depth {depth}")
    return history, best_state


def count_flops(model: DepthClassifier, depth: int, device: torch.device) -> Dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    # fvcore traces this fixed-depth wrapper internally. Keep its default
    # operator methodology and record, rather than replace, unsupported ops.
    fixed_model = FixedDepthModel(model, depth).to(device).eval()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    analysis = FlopCountAnalysis(fixed_model, (dummy,))
    analysis.tracer_warnings("none")
    total = analysis.total()
    return {
        "flops": int(total),
        "counting_library": "fvcore.nn.FlopCountAnalysis",
        "reported_unit": "FLOPs",
        "macs_relabelled_as_flops": False,
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "unsupported_operators": dict(analysis.unsupported_ops()),
        "uncalled_modules": sorted(analysis.uncalled_modules()),
    }


@torch.no_grad()
def measure_latency(
    model: DepthClassifier,
    depth: int,
    device: torch.device,
    warmup: int,
    runs: int,
    amp: bool,
) -> Dict[str, Any]:
    if runs < 1:
        raise ValueError("latency runs must be at least 1")
    fixed_model = FixedDepthModel(model, depth).to(device).eval()
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    for _ in range(warmup):
        with autocast_context(device, amp):
            fixed_model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast_context(device, amp):
            fixed_model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "batch_size": 1,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "device": str(device),
        "amp": amp,
    }


def save_operating_points(path: Path, points: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "depth",
        "role",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_loss",
        "flops",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for point in points:
            writer.writerow({column: point.get(column, "") for column in columns})


def save_plot(path: Path, points: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib is unavailable; metrics were still written.")
        return

    x = [float(point["flops"]) / 1e9 for point in points]
    y = [float(point["validation_accuracy"]) * 100.0 for point in points]
    labels = [f"depth {point['depth']}" for point in points]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(x, y, marker="o", linewidth=1.5)
    for x_value, y_value, label in zip(x, y, labels):
        axis.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("fvcore FLOPs (G)")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("F001 validation accuracy vs. compute")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "medmnist")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=repo_root / "checkpoints" / "feasibility_001",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "artifacts" / "feasibility_001",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA automatic mixed precision")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.head_epochs < 1:
        raise ValueError("epochs and head-epochs must be at least 1")
    seed_everything(args.seed)
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    print(f"F001 device={device} model={MODEL_NAME} depths={DEPTHS}")
    print("F001 holds out the official test split; only train and val are loaded.")

    train_loader, val_loader = make_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
    )
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    full_checkpoint = args.checkpoint_root / "full_backbone.pt"
    history: List[Dict[str, float]] = []

    backbone = create_backbone(pretrained=True)
    if full_checkpoint.exists() and not args.force_retrain:
        backbone.load_state_dict(torch.load(full_checkpoint, map_location="cpu"))
        print(f"[full] reused {full_checkpoint}")
    else:
        phase_history, best_state = train_full_model(
                backbone,
                train_loader,
                val_loader,
                device,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                amp,
            )
        history.extend(phase_history)
        backbone.load_state_dict(best_state)
        torch.save(backbone.state_dict(), full_checkpoint)
        print(f"[full] saved {full_checkpoint}")

    model = DepthClassifier(backbone).to(device)
    model.backbone.eval()
    for depth in (4, 8):
        head_checkpoint = args.checkpoint_root / f"head_depth_{depth}.pt"
        if head_checkpoint.exists() and not args.force_retrain:
            model.early_heads[str(depth)].load_state_dict(
                torch.load(head_checkpoint, map_location="cpu")
            )
            print(f"[head {depth}] reused {head_checkpoint}")
        else:
            phase_history, best_state = train_early_head(
                    model,
                    depth,
                    train_loader,
                    val_loader,
                    device,
                    args.head_epochs,
                    args.head_learning_rate,
                    args.weight_decay,
                    amp,
                )
            history.extend(phase_history)
            model.early_heads[str(depth)].load_state_dict(best_state)
            torch.save(model.early_heads[str(depth)].state_dict(), head_checkpoint)
            print(f"[head {depth}] saved {head_checkpoint}")

    model.eval()
    adaptive_checkpoint = args.checkpoint_root / "depth_adaptive_model.pt"
    torch.save(model.state_dict(), adaptive_checkpoint)
    print(f"[adaptive] saved shared-backbone artifact {adaptive_checkpoint}")
    points: List[Dict[str, Any]] = []
    for depth in DEPTHS:
        validation = evaluate(model, val_loader, depth, device, amp)
        flops = count_flops(model, depth, device)
        latency = measure_latency(
            model,
            depth,
            device,
            args.latency_warmup,
            args.latency_runs,
            amp,
        )
        point = {
            "depth": depth,
            "role": "static_full_depth_baseline" if depth == 12 else "shared_weight_operating_point",
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_loss": validation["loss"],
            **flops,
            "latency": latency,
            "latency_mean_ms": latency["mean_ms"],
            "latency_p50_ms": latency["p50_ms"],
            "latency_p95_ms": latency["p95_ms"],
        }
        points.append(point)
        print(
            f"[depth {depth}] val_acc={validation['accuracy']:.4f} "
            f"macro_f1={validation['macro_f1']:.4f} "
            f"flops={flops['flops'] / 1e9:.4f}G "
            f"latency_mean={latency['mean_ms']:.2f}ms"
        )

    run_metadata = {
        "experiment": "F001",
        "question": "Can a single shared ViT backbone provide useful accuracy-compute operating points?",
        "model_name": MODEL_NAME,
        "dataset": "DermaMNIST",
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "splits_loaded": ["train", "val"],
        "test_split_loaded": False,
        "depths": list(DEPTHS),
        "device": str(device),
        "amp": amp,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "python_version": os.sys.version,
        "history": history,
        "operating_points": points,
    }
    write_json(args.artifact_root / "results.json", run_metadata)
    save_operating_points(args.artifact_root / "operating_points.csv", points)
    save_plot(args.artifact_root / "accuracy_vs_flops.png", points)
    write_json(
        args.artifact_root / "run_config.json",
        {key: value for key, value in vars(args).items()} | {"device_resolved": str(device), "amp": amp},
    )
    print(f"[done] results written to {args.artifact_root}")


if __name__ == "__main__":
    main()
