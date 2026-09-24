"""Characterize every transformer depth from 4 through 12 for F001b.

The script imports F001's model, data, head-training, metric, latency, and
fvcore utilities directly. It loads the canonical F001 checkpoint, reuses its
trained depth-4 and depth-8 heads, and trains only the missing frozen-backbone
heads. It never constructs the official test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


DEPTHS = tuple(range(4, 13))
REUSED_HEAD_DEPTHS = (4, 8)


def load_f001_module(repo_root: Path):
    source = repo_root / "experiments" / "feasibility_001" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("f001_runner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load F001 runner from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # F001's helpers validate requested depth against the original sparse set.
    module.DEPTHS = DEPTHS
    return module


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_f001_inputs(repo_root: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    checkpoint_root = repo_root / "checkpoints" / "feasibility_001"
    artifact_root = repo_root / "artifacts" / "feasibility_001"
    checkpoint_paths = {
        "full_backbone": checkpoint_root / "full_backbone.pt",
        "head_depth_4": checkpoint_root / "head_depth_4.pt",
        "head_depth_8": checkpoint_root / "head_depth_8.pt",
        "depth_adaptive_model": checkpoint_root / "depth_adaptive_model.pt",
    }
    required = [*checkpoint_paths.values(), artifact_root / "results.json", artifact_root / "run_config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required completed F001 files are missing:\n" + "\n".join(missing))

    results = read_json(artifact_root / "results.json")
    config = read_json(artifact_root / "run_config.json")
    if results.get("experiment") != "F001":
        raise ValueError("Canonical result file does not identify experiment F001")
    if results.get("test_split_loaded") is not False or set(results.get("splits_loaded", [])) != {"train", "val"}:
        raise ValueError("F001 metadata does not confirm train/validation-only use")
    if config.get("epochs") != 5 or config.get("head_epochs") != 5:
        raise ValueError("Existing F001 checkpoints are not from the documented 5+5 epoch run")
    if config.get("seed") != 41 or config.get("batch_size") != 64:
        raise ValueError("Existing F001 seed/batch configuration differs from the expected run")
    if results.get("depths") != [4, 8, 12]:
        raise ValueError("Canonical F001 results do not contain depths 4, 8, and 12")
    saved_points = {int(point["depth"]): point for point in results.get("operating_points", [])}
    if not set((4, 8, 12)).issubset(saved_points):
        raise ValueError("F001 result metadata is missing a required sparse operating point")

    hashes = {name: file_sha256(path) for name, path in checkpoint_paths.items()}
    for folder_name, folder in (("checkpoint_file", checkpoint_root), ("artifact_file", artifact_root)):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                hashes[f"{folder_name}:{path.relative_to(folder).as_posix()}"] = file_sha256(path)
    return results, config, hashes


def load_and_verify_checkpoint_bundle(
    module,
    repo_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    checkpoint_root = repo_root / "checkpoints" / "feasibility_001"
    backbone = module.create_backbone(pretrained=False)
    backbone.load_state_dict(
        torch.load(checkpoint_root / "full_backbone.pt", map_location="cpu", weights_only=True)
    )

    model = module.DepthClassifier(backbone, depths=range(4, 12)).to(device)
    model.early_heads["4"].load_state_dict(
        torch.load(checkpoint_root / "head_depth_4.pt", map_location="cpu", weights_only=True)
    )
    model.early_heads["8"].load_state_dict(
        torch.load(checkpoint_root / "head_depth_8.pt", map_location="cpu", weights_only=True)
    )

    bundled = torch.load(
        checkpoint_root / "depth_adaptive_model.pt", map_location="cpu", weights_only=True
    )
    for key, value in backbone.state_dict().items():
        bundled_key = f"backbone.{key}"
        if bundled_key not in bundled or not torch.equal(value.cpu(), bundled[bundled_key].cpu()):
            raise ValueError(f"F001 combined checkpoint disagrees with full_backbone.pt at {key}")
    for depth in REUSED_HEAD_DEPTHS:
        phase_head = model.early_heads[str(depth)].state_dict()
        for key, value in phase_head.items():
            bundled_key = f"early_heads.{depth}.{key}"
            if bundled_key not in bundled or not torch.equal(value.cpu(), bundled[bundled_key].cpu()):
                raise ValueError(f"F001 combined checkpoint disagrees with head depth {depth} at {key}")
    return model, bundled


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def add_consecutive_deltas(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index, point in enumerate(points):
        if index == 0:
            point.update(
                {
                    "transition_from_previous": "",
                    "delta_validation_accuracy": "",
                    "delta_accuracy_percentage_points": "",
                    "delta_validation_macro_f1": "",
                    "delta_flops": "",
                    "delta_gflops": "",
                    "marginal_accuracy_per_gflop": "",
                    "marginal_macro_f1_per_gflop": "",
                }
            )
            continue
        previous = points[index - 1]
        delta_flops = int(point["fvcore_flops"]) - int(previous["fvcore_flops"])
        delta_gflops = delta_flops / 1e9
        delta_accuracy = float(point["validation_accuracy"]) - float(previous["validation_accuracy"])
        delta_macro_f1 = float(point["validation_macro_f1"]) - float(previous["validation_macro_f1"])
        if delta_flops <= 0:
            raise ValueError(
                f"FLOPs did not increase from depth {previous['depth']} to {point['depth']}"
            )
        point.update(
            {
                "transition_from_previous": f"{previous['depth']}->{point['depth']}",
                "delta_validation_accuracy": delta_accuracy,
                "delta_accuracy_percentage_points": delta_accuracy * 100.0,
                "delta_validation_macro_f1": delta_macro_f1,
                "delta_flops": delta_flops,
                "delta_gflops": delta_gflops,
                "marginal_accuracy_per_gflop": delta_accuracy / delta_gflops,
                "marginal_macro_f1_per_gflop": delta_macro_f1 / delta_gflops,
            }
        )
    return points


def save_csv(path: Path, points: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "depth",
        "validation_accuracy",
        "validation_macro_f1",
        "fvcore_flops",
        "gflops",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_device",
        "trainable_exit_head_parameters",
        "selected_exit_head_parameters",
        "shared_backbone_parameters",
        "combined_deployable_parameters",
        "transition_from_previous",
        "delta_validation_accuracy",
        "delta_accuracy_percentage_points",
        "delta_validation_macro_f1",
        "delta_flops",
        "delta_gflops",
        "marginal_accuracy_per_gflop",
        "marginal_macro_f1_per_gflop",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: point.get(key, "") for key in columns} for point in points)


def read_saved_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_tradeoff(path: Path, points: Sequence[Mapping[str, Any]], metric: str, ylabel: str, title: str) -> None:
    xs = [float(point["gflops"]) for point in points]
    ys = [float(point[metric]) for point in points]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(xs, ys, color="#627b9a", linewidth=1.2, alpha=0.75, zorder=1)
    ax.scatter(xs, ys, color="#176b87", s=42, zorder=2)
    for x, y, point in zip(xs, ys, points):
        ax.annotate(str(point["depth"]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("fvcore FLOPs (GFLOPs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.01,
        -0.22,
        "Markers are measured depths; connecting segments are visual guides only (no fitted curve).",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_marginal(path: Path, points: Sequence[Mapping[str, Any]]) -> None:
    transitions = points[1:]
    labels = [point["transition_from_previous"] for point in transitions]
    values = [float(point["marginal_accuracy_per_gflop"]) for point in transitions]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#176b87" if value >= 0 else "#bb5a52" for value in values]
    ax.bar(labels, values, color=colors, width=0.72)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Consecutive depth transition")
    ax.set_ylabel("Δ validation accuracy / Δ GFLOPs")
    ax.set_title("Marginal validation accuracy gain per added GFLOP")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_observations(points: Sequence[Mapping[str, Any]]) -> List[str]:
    transitions = points[1:]
    max_accuracy_gain = max(transitions, key=lambda point: point["delta_validation_accuracy"])
    max_f1_gain = max(transitions, key=lambda point: point["delta_validation_macro_f1"])
    best_marginal = max(transitions, key=lambda point: point["marginal_accuracy_per_gflop"])
    accuracy_drops = [point for point in transitions if point["delta_validation_accuracy"] < 0]
    f1_drops = [point for point in transitions if point["delta_validation_macro_f1"] < 0]
    accuracy_flat = [point for point in transitions if abs(point["delta_accuracy_percentage_points"]) < 0.5]
    observations = [
        f"The largest consecutive validation-accuracy increase was {max_accuracy_gain['delta_accuracy_percentage_points']:.2f} percentage points at transition {max_accuracy_gain['transition_from_previous']}; the largest macro-F1 increase was {max_f1_gain['delta_validation_macro_f1']:.4f} at transition {max_f1_gain['transition_from_previous']}.",
        f"The highest measured marginal accuracy gain was {best_marginal['marginal_accuracy_per_gflop']:.4f} accuracy fraction per GFLOP for transition {best_marginal['transition_from_previous']}.",
    ]
    if accuracy_drops:
        observations.append(
            "Validation accuracy decreased at "
            + ", ".join(point["transition_from_previous"] for point in accuracy_drops)
            + "."
        )
    else:
        observations.append("Validation accuracy increased or held steady at every measured consecutive depth transition.")
    if f1_drops:
        observations.append(
            "Validation macro-F1 decreased at "
            + ", ".join(point["transition_from_previous"] for point in f1_drops)
            + "."
        )
    else:
        observations.append("Validation macro-F1 increased or held steady at every measured consecutive depth transition.")
    if accuracy_flat:
        observations.append(
            "Accuracy changes under 0.5 percentage points occurred at "
            + ", ".join(point["transition_from_previous"] for point in accuracy_flat)
            + "."
        )
    dominated = []
    for candidate in points:
        for comparator in points:
            if int(comparator["depth"]) >= int(candidate["depth"]):
                continue
            if (
                comparator["validation_accuracy"] >= candidate["validation_accuracy"]
                and comparator["validation_macro_f1"] >= candidate["validation_macro_f1"]
                and (
                    comparator["validation_accuracy"] > candidate["validation_accuracy"]
                    or comparator["validation_macro_f1"] > candidate["validation_macro_f1"]
                )
            ):
                dominated.append(f"depth {candidate['depth']} by depth {comparator['depth']}")
                break
    if dominated:
        observations.append("Among these measured points, predictive-metric dominated pairs include " + "; ".join(dominated) + ".")
    else:
        observations.append("No measured point is dominated on both accuracy and macro-F1 by a cheaper measured point.")
    observations.append(
        "Any knee or high-efficiency region is a candidate for later scenario-specific evaluation; this descriptive sweep does not select an optimal depth."
    )
    return observations


def write_report(path: Path, points: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    rows = []
    for point in points:
        latency = f"{point['latency_mean_ms']:.3f} ms"
        marginal = point["marginal_accuracy_per_gflop"]
        marginal_text = "" if marginal == "" else f"{marginal:.4f}"
        rows.append(
            f"| {point['depth']} | {point['validation_accuracy']:.4f} | {point['validation_macro_f1']:.4f} | {point['fvcore_flops']:,} | {point['gflops']:.4f} | {latency} | {point['selected_exit_head_parameters']:,} | {marginal_text} |"
        )
    table = "\n".join(
        [
            "| Depth | Val accuracy | Val macro-F1 | fvcore FLOPs | GFLOPs | Mean latency | Selected head params | Δaccuracy/ΔGFLOPs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
        ]
    )
    observations = "\n".join(f"- {item}" for item in summarize_observations(points))
    report = f"""# F001b — Full Depth Characterization

## Research question
How does predictive performance evolve as transformer depth increases from 4 to 12 in the same shared-weight DeiT backbone?

## Method
- Reused the completed F001 DeiT-Tiny checkpoint and its trained depth-4 and depth-8 heads. Their tensors were checked against the combined F001 checkpoint before use.
- Dataset: DermaMNIST-224. Only the official training and validation splits were loaded; the official test split remains untouched. Validation metrics are used for this feasibility-stage characterization and best-head checkpoint selection.
- The F001 backbone was frozen while training heads at depths 5, 6, 7, 9, 10, and 11. Each new head is a linear 7-class head trained using F001's existing head-training procedure. The F001 depth-4 and depth-8 heads and depth-12 classifier were reused.
- Seed: {metadata['seed']}; head epochs: {metadata['head_epochs']}; batch size: {metadata['batch_size']}; head learning rate: {metadata['head_learning_rate']}; weight decay: {metadata['weight_decay']}.
- FLOPs use F001's `fvcore.nn.FlopCountAnalysis.total()` with a single 1x3x224x224 input. Values are explicitly fvcore FLOPs. Unsupported operators are retained in `results.json`; no custom handlers were added.
- Latency uses F001's batch-1 warmup and timed CUDA measurements ({metadata['latency_warmup']} warmup, {metadata['latency_runs']} measured runs) on one fixed all-zero 1x3x224x224 input, on {metadata['device_name']} with AMP={str(metadata['amp']).lower()}. This measures model inference only, excluding data loading and preprocessing; latency is hardware-specific.

## Results
{table}

Per-depth raw measurements, unsupported operators, uncalled modules, parameter counts, and consecutive-depth deltas are in `results.json` and `results.csv`.

## Observations
{observations}

## Relationship to F001
F001 sampled depths 4, 8, and 12. This denser set measures the intervening depths directly on the same F001 fine-tuned backbone, making the shape of observed accuracy, macro-F1, and compute changes clearer. No values were interpolated or fitted. Connecting lines in the trade-off plots are visual guides between measurements.

## Implications
- Future operating-point selection can use measured intermediate depths rather than assuming a smooth curve between 4, 8, and 12.
- The per-input marginal gains and latency measurements can inform later controller constraints, but do not alone define a policy.
- Future Pareto analysis should retain the measured points and compare them under the eventual scenario's latency, energy, and predictive-quality requirements.

No depth is declared optimal by this experiment.
"""
    path.write_text(report, encoding="utf-8")


def parse_args(repo_root: Path, f001_config: Mapping[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(f001_config["data_root"]))
    parser.add_argument("--checkpoint-root", type=Path, default=repo_root / "checkpoints" / "feasibility_001")
    parser.add_argument("--artifact-root", type=Path, default=repo_root / "artifacts" / "F001b_depth_sweep")
    parser.add_argument("--head-epochs", type=int, default=int(f001_config["head_epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(f001_config["batch_size"]))
    parser.add_argument("--num-workers", type=int, default=int(f001_config["num_workers"]))
    parser.add_argument("--head-learning-rate", type=float, default=float(f001_config["head_learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=float(f001_config["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(f001_config["seed"]))
    parser.add_argument("--latency-warmup", type=int, default=int(f001_config["latency_warmup"]))
    parser.add_argument("--latency-runs", type=int, default=int(f001_config["latency_runs"]))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=f001_config["device_resolved"])
    parser.add_argument(
        "--reuse-trained-heads",
        action="store_true",
        help="reuse F001b head checkpoints already present under the artifact directory",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    f001_results, f001_config, input_hashes = verify_f001_inputs(repo_root)
    args = parse_args(repo_root, f001_config)
    if args.head_epochs < 1 or args.latency_runs < 1:
        raise ValueError("head epochs and latency runs must be positive")
    module = load_f001_module(repo_root)
    module.seed_everything(args.seed)
    device = module.choose_device(args.device)
    if args.device == "cpu":
        amp = False
    else:
        amp = device.type == "cuda" and not bool(f001_config.get("no_amp", False))

    model, _ = load_and_verify_checkpoint_bundle(module, repo_root, device)
    train_loader, val_loader = module.make_loaders(
        args.data_root, args.batch_size, args.num_workers, device, args.seed
    )
    history: List[Dict[str, Any]] = []
    head_checkpoint_dir = args.artifact_root / "checkpoints"
    head_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_results_path = args.artifact_root / "results.json"
    if args.reuse_trained_heads and previous_results_path.is_file():
        history = read_json(previous_results_path).get("head_training_history", [])

    for depth in DEPTHS:
        if depth in REUSED_HEAD_DEPTHS or depth == 12:
            continue
        head_path = head_checkpoint_dir / f"head_depth_{depth}.pt"
        if args.reuse_trained_heads:
            if not head_path.is_file():
                raise FileNotFoundError(f"Cannot reuse missing F001b head checkpoint {head_path}")
            model.early_heads[str(depth)].load_state_dict(
                torch.load(head_path, map_location="cpu", weights_only=True)
            )
            continue
        phase_history, best_state = module.train_early_head(
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
        torch.save(model.early_heads[str(depth)].state_dict(), head_path)
        print(f"[head {depth}] saved best validation checkpoint")

    model.eval()
    shared_backbone_parameters = count_parameters(model.backbone)
    combined_parameters = count_parameters(model)
    points: List[Dict[str, Any]] = []
    f001_points = {int(point["depth"]): point for point in f001_results["operating_points"]}
    for depth in DEPTHS:
        validation = module.evaluate(model, val_loader, depth, device, amp)
        flops_info = module.count_flops(model, depth, device)
        latency = module.measure_latency(
            model, depth, device, args.latency_warmup, args.latency_runs, amp
        )
        selected_head = model.backbone.head if depth == 12 else model.early_heads[str(depth)]
        point = {
            "depth": depth,
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_loss": validation["loss"],
            "fvcore_flops": int(flops_info["flops"]),
            "gflops": float(flops_info["flops"]) / 1e9,
            "counting_library": flops_info["counting_library"],
            "reported_unit": flops_info["reported_unit"],
            "unsupported_operators": flops_info["unsupported_operators"],
            "uncalled_modules": flops_info["uncalled_modules"],
            "input_shape": flops_info["input_shape"],
            "batch1_latency": latency,
            "latency_mean_ms": latency["mean_ms"],
            "latency_p50_ms": latency["p50_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "latency_device": latency["device"],
            "trainable_exit_head_parameters": count_parameters(selected_head),
            "selected_exit_head_parameters": count_parameters(selected_head),
            "shared_backbone_parameters": shared_backbone_parameters,
            "combined_deployable_parameters": combined_parameters,
            "head_training": "reused_from_F001" if depth in REUSED_HEAD_DEPTHS else ("full_finetune_from_F001" if depth == 12 else "frozen_backbone_linear_head"),
        }
        if depth in f001_points:
            baseline = f001_points[depth]
            for metric in ("validation_accuracy", "validation_macro_f1"):
                if not math.isclose(point[metric], float(baseline[metric]), rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(
                        f"Reused depth {depth} {metric} differs from F001 output: "
                        f"{point[metric]} vs {baseline[metric]}"
                    )
        points.append(point)
        print(
            f"[depth {depth}] val_acc={point['validation_accuracy']:.4f} "
            f"macro_f1={point['validation_macro_f1']:.4f} "
            f"fvcore_flops={point['fvcore_flops'] / 1e9:.4f}G "
            f"latency_mean={point['latency_mean_ms']:.3f}ms"
        )

    add_consecutive_deltas(points)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.artifact_root / "results.csv"
    save_csv(csv_path, points)
    saved_points = read_saved_csv(csv_path)
    saved_depths = [int(point["depth"]) for point in saved_points]
    if saved_depths != list(DEPTHS) or len(set(saved_depths)) != len(DEPTHS):
        raise RuntimeError(f"Saved CSV depth rows are missing, duplicated, or unordered: {saved_depths}")
    saved_flops = [int(point["fvcore_flops"]) for point in saved_points]
    if any(later <= earlier for earlier, later in zip(saved_flops, saved_flops[1:])):
        raise RuntimeError(f"Saved fvcore FLOPs do not increase with depth: {saved_flops}")
    (args.artifact_root / "results.json").write_text(
        json.dumps(
            {
                "experiment": "F001b",
                "dataset": "DermaMNIST",
                "image_size": module.IMAGE_SIZE,
                "model_name": module.MODEL_NAME,
                "splits_loaded": ["train", "val"],
                "test_split_loaded": False,
                "depths": list(DEPTHS),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "amp": amp,
                "seed": args.seed,
                "head_epochs": args.head_epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
                "latency_warmup": args.latency_warmup,
                "latency_runs": args.latency_runs,
                "source_f001_checkpoint_sha256": input_hashes,
                "head_training_history": history,
                "operating_points": points,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_tradeoff(
        args.artifact_root / "validation_accuracy_vs_flops.png",
        saved_points,
        "validation_accuracy",
        "Validation accuracy",
        "F001b validation accuracy vs. compute",
    )
    plot_tradeoff(
        args.artifact_root / "validation_macro_f1_vs_flops.png",
        saved_points,
        "validation_macro_f1",
        "Validation macro-F1",
        "F001b validation macro-F1 vs. compute",
    )
    plot_marginal(args.artifact_root / "marginal_accuracy_gain_vs_depth.png", saved_points)
    report_metadata = {
        "seed": args.seed,
        "head_epochs": args.head_epochs,
        "batch_size": args.batch_size,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "latency_warmup": args.latency_warmup,
        "latency_runs": args.latency_runs,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": amp,
    }
    write_report(args.artifact_root / "summary.md", points, report_metadata)
    _, _, hashes_after_run = verify_f001_inputs(repo_root)
    if hashes_after_run != input_hashes:
        raise RuntimeError("An F001 checkpoint or artifact changed during the F001b run")
    print(f"[done] F001b results written to {args.artifact_root}")


if __name__ == "__main__":
    main()
