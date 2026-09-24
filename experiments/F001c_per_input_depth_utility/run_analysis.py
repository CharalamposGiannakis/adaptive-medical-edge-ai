"""Analyze per-validation-sample utility across the existing F001b depth heads.

This is inference and analysis only: it never creates a train or test dataset,
trains a model, or writes to F001/F001b inputs. Run from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


DEPTHS = tuple(range(4, 13))
PAIRS = ((6, 7), (7, 8), (8, 9), (9, 12), (6, 9), (6, 12))
ORACLE_SETS = {
    "all_4_12": DEPTHS,
    "depths_6_7_8_9": (6, 7, 8, 9),
    "depths_6_7_8_9_12": (6, 7, 8, 9, 12),
    "depths_6_9": (6, 9),
    "depths_7_9": (7, 9),
    "depths_8_9": (8, 9),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_inputs(root: Path) -> dict[str, str]:
    folders = (
        "experiments/feasibility_001",
        "experiments/F001b_depth_sweep",
        "checkpoints/feasibility_001",
        "artifacts/feasibility_001",
        "artifacts/F001b_depth_sweep",
    )
    files: dict[str, str] = {}
    for folder in folders:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = sha256(path)
    return files


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_f001(root: Path):
    path = root / "experiments/feasibility_001/run_experiment.py"
    spec = importlib.util.spec_from_file_location("f001_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import F001 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DEPTHS = DEPTHS
    return module


def load_inputs(root: Path, module, device: torch.device):
    f001_root = root / "checkpoints/feasibility_001"
    f001b_root = root / "artifacts/F001b_depth_sweep"
    f001_results = read_json(root / "artifacts/feasibility_001/results.json")
    f001_config = read_json(root / "artifacts/feasibility_001/run_config.json")
    f001b_results = read_json(f001b_root / "results.json")
    if f001_results.get("test_split_loaded") is not False:
        raise ValueError("F001 metadata does not confirm that the test split was untouched")
    if f001b_results.get("test_split_loaded") is not False:
        raise ValueError("F001b metadata does not confirm that the test split was untouched")
    if f001b_results.get("depths") != list(DEPTHS):
        raise ValueError("F001b results do not contain exactly depths 4 through 12")
    for depth in DEPTHS:
        checkpoint = (f001_root / f"head_depth_{depth}.pt") if depth in (4, 8) else (
            f001_root / "full_backbone.pt" if depth == 12 else
            f001b_root / "checkpoints" / f"head_depth_{depth}.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required trained checkpoint missing: {checkpoint}")

    backbone = module.create_backbone(pretrained=False)
    backbone.load_state_dict(torch.load(f001_root / "full_backbone.pt", map_location="cpu", weights_only=True))
    model = module.DepthClassifier(backbone, depths=range(4, 12)).to(device)
    for depth in DEPTHS:
        if depth == 12:
            continue
        checkpoint = (f001_root / f"head_depth_{depth}.pt") if depth in (4, 8) else (
            f001b_root / "checkpoints" / f"head_depth_{depth}.pt"
        )
        model.early_heads[str(depth)].load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
    model.eval()

    f001b_points = {int(point["depth"]): point for point in f001b_results["operating_points"]}
    if set(f001b_points) != set(DEPTHS):
        raise ValueError("F001b measured results are missing depth FLOPs")
    flops = {depth: int(f001b_points[depth]["fvcore_flops"]) for depth in DEPTHS}
    return model, f001_config, f001b_results, flops


def infer_validation(module, model, device: torch.device, amp: bool, batch_size: int,
                     num_workers: int, data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Intentionally instantiate only the validation dataset; no test split API call exists here.
    dataset = module.make_dataset("val", data_root)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    labels_meta = __import__("medmnist").INFO["dermamnist"]["label"]
    names = {int(key): value for key, value in labels_meta.items()}
    true_labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    if len(dataset) != len(true_labels):
        raise ValueError("Validation label count does not match dataset length")
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for images, _ in loader:
            count = images.shape[0]
            images = images.to(device, non_blocking=True)
            for depth in DEPTHS:
                with module.autocast_context(device, amp):
                    logits = model.logits_at_depth(images, depth)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                batch_true = true_labels[offset:offset + count]
                if len(batch_true) != count:
                    raise ValueError("Validation loader yielded more rows than its labels")
                pred = probs.argmax(axis=1)
                conf = probs.max(axis=1)
                entropy = -(probs * np.log(np.clip(probs, 1e-30, 1.0))).sum(axis=1)
                for i in range(count):
                    row: dict[str, Any] = {
                        "sample_index": offset + i,
                        "sample_id": f"dermamnist224_val_{offset + i:04d}",
                        "true_label": int(batch_true[i]),
                        "true_class_name": names[int(batch_true[i])],
                        "depth": depth,
                        "predicted_label": int(pred[i]),
                        "predicted_class_name": names[int(pred[i])],
                        "correct": int(pred[i] == batch_true[i]),
                        "predicted_class_confidence": float(conf[i]),
                        "max_softmax_probability": float(conf[i]),
                        "predictive_entropy_nats": float(entropy[i]),
                    }
                    row.update({f"p_class_{j}": float(prob) for j, prob in enumerate(probs[i])})
                    rows.append(row)
            offset += count
    if offset != len(dataset) or len(rows) != len(dataset) * len(DEPTHS):
        raise ValueError(f"Expected {len(dataset)} samples at each depth; observed {offset}")
    for depth in DEPTHS:
        depth_rows = [r for r in rows if r["depth"] == depth]
        if [r["sample_index"] for r in depth_rows] != list(range(len(dataset))):
            raise ValueError(f"Validation index sequence differs at depth {depth}")
    return rows, {"sample_count": len(dataset), "class_names": names}


def compare(rows: Sequence[Mapping[str, Any]], shallow: int, deep: int) -> dict[str, Any]:
    left = [r for r in rows if int(r["depth"]) == shallow]
    right = [r for r in rows if int(r["depth"]) == deep]
    if len(left) != len(right) or not left:
        raise ValueError(f"Inconsistent counts for {shallow}->{deep}")
    if [r["sample_index"] for r in left] != [r["sample_index"] for r in right]:
        raise ValueError(f"Mismatched sample order for {shallow}->{deep}")
    a = np.asarray([int(r["correct"]) for r in left], dtype=bool)
    b = np.asarray([int(r["correct"]) for r in right], dtype=bool)
    rescue = int(np.sum(~a & b))
    regression = int(np.sum(a & ~b))
    shallow_wrong, shallow_right = int((~a).sum()), int(a.sum())
    return {
        "shallow_depth": shallow, "deeper_depth": deep, "sample_count": len(left),
        "shallow_accuracy": float(a.mean()), "deeper_accuracy": float(b.mean()),
        "rescue_count": rescue,
        "rescue_rate_given_shallow_wrong": rescue / shallow_wrong if shallow_wrong else "",
        "shallow_wrong_denominator": shallow_wrong,
        "regression_count": regression,
        "regression_rate_given_shallow_correct": regression / shallow_right if shallow_right else "",
        "shallow_correct_denominator": shallow_right,
        "stable_correct_count": int(np.sum(a & b)),
        "stable_wrong_count": int(np.sum(~a & ~b)),
        "prediction_change_count": sum(int(x["predicted_label"] != y["predicted_label"]) for x, y in zip(left, right)),
        "prediction_change_rate": float(np.mean([x["predicted_label"] != y["predicted_label"] for x, y in zip(left, right)])),
        "net_correction": rescue - regression,
        "mean_confidence_delta_deeper_minus_shallow": float(np.mean([float(y["predicted_class_confidence"]) - float(x["predicted_class_confidence"]) for x, y in zip(left, right)])),
        "mean_entropy_delta_nats_deeper_minus_shallow": float(np.mean([float(y["predictive_entropy_nats"]) - float(x["predictive_entropy_nats"]) for x, y in zip(left, right)])),
    }


def difficulty_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for shallow in (6, 7, 8, 9):
        for metric in ("predicted_class_confidence", "predictive_entropy_nats"):
            base = sorted((r for r in rows if int(r["depth"]) == shallow), key=lambda r: int(r["sample_index"]))
            order = np.argsort([float(r[metric]) for r in base], kind="stable")
            bins = np.array_split(order, 5)
            deeper_values = tuple(dict.fromkeys((shallow + 1, 12)))
            for comparison_depth in deeper_values:
                by_index = {int(r["sample_index"]): r for r in rows if int(r["depth"]) == comparison_depth}
                for quantile, indexes in enumerate(bins, 1):
                    sub = [base[int(i)] for i in indexes]
                    wrong = [r for r in sub if not int(r["correct"])]
                    rescues = sum(int(by_index[int(r["sample_index"])] ["correct"]) for r in wrong)
                    result.append({
                        "shallow_depth": shallow, "comparison_depth": comparison_depth,
                        "uncertainty_measure": metric, "quantile_bin": quantile,
                        "bin_label": f"Q{quantile}", "samples_in_bin": len(sub),
                        "mean_measure_value": float(np.mean([float(r[metric]) for r in sub])),
                        "shallow_errors_in_bin": len(wrong), "rescued_shallow_errors": rescues,
                        "rescue_rate_given_shallow_wrong": rescues / len(wrong) if wrong else "",
                        "rescues_per_all_bin_samples": rescues / len(sub) if sub else "",
                    })
    return result


def oracle_analysis(rows: Sequence[Mapping[str, Any]], flops: Mapping[int, int]):
    grouped = {d: sorted((r for r in rows if int(r["depth"]) == d), key=lambda r: int(r["sample_index"])) for d in DEPTHS}
    n = len(grouped[4])
    oracle_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    selected_all: dict[str, list[int | None]] = {}
    for policy, depths in ORACLE_SETS.items():
        chosen: list[int | None] = []
        for i in range(n):
            hits = [d for d in depths if int(grouped[d][i]["correct"])]
            chosen.append(min(hits) if hits else None)
        selected_all[policy] = chosen
        covered = sum(d is not None for d in chosen)
        costs = [flops[d] if d is not None else flops[max(depths)] for d in chosen]
        mean_depth = float(np.mean([d for d in chosen if d is not None])) if covered else ""
        oracle_rows.append({
            "policy": policy, "allowed_depths": ";".join(map(str, depths)), "sample_count": n,
            "oracle_covered_count": covered, "oracle_coverage": covered / n,
            "oracle_accuracy_all_samples": covered / n,
            "mean_selected_depth_among_covered": mean_depth,
            "mean_fvcore_gflops_all_samples_failure_at_max_allowed": float(np.mean(costs)) / 1e9,
            "mean_fvcore_gflops_among_covered": float(np.mean([flops[d] for d in chosen if d is not None])) / 1e9 if covered else "",
            "failure_cost_convention": "charge maximum allowed depth for samples correct at no allowed depth",
            "interpretation": "unreachable oracle upper bound; not a deployable controller",
        })
        for depth in depths:
            count = sum(d == depth for d in chosen)
            dist_rows.append({"policy": policy, "selected_depth": depth, "count": count, "fraction_all_samples": count / n})
        failures = sum(d is None for d in chosen)
        dist_rows.append({"policy": policy, "selected_depth": "oracle_failure", "count": failures, "fraction_all_samples": failures / n})
    fixed = []
    for d in DEPTHS:
        accuracy = float(np.mean([int(r["correct"]) for r in grouped[d]]))
        fixed.append({"depth": d, "fixed_validation_accuracy": accuracy, "fvcore_flops": flops[d], "fvcore_gflops": flops[d] / 1e9})
    return oracle_rows, dist_rows, fixed, selected_all


def oracle_fixed_comparison(oracle_rows: Sequence[Mapping[str, Any]], fixed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    full = next(row for row in oracle_rows if row["policy"] == "all_4_12")
    return [{
        "oracle_policy": "all_4_12",
        "fixed_depth": int(row["depth"]),
        "oracle_accuracy_or_coverage": float(full["oracle_accuracy_all_samples"]),
        "oracle_mean_fvcore_gflops_failure_at_max_allowed": float(full["mean_fvcore_gflops_all_samples_failure_at_max_allowed"]),
        "fixed_validation_accuracy": float(row["fixed_validation_accuracy"]),
        "fixed_fvcore_gflops": float(row["fvcore_gflops"]),
        "oracle_minus_fixed_accuracy_percentage_points": 100 * (float(full["oracle_accuracy_all_samples"]) - float(row["fixed_validation_accuracy"])),
        "oracle_minus_fixed_mean_fvcore_gflops": float(full["mean_fvcore_gflops_all_samples_failure_at_max_allowed"]) - float(row["fvcore_gflops"]),
        "oracle_is_unreachable": True,
    } for row in fixed_rows]


def later_depth_value(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    correct = {d: np.asarray([int(r["correct"]) for r in sorted((x for x in rows if int(x["depth"]) == d), key=lambda x: int(x["sample_index"]))], dtype=bool) for d in DEPTHS}
    out = []
    for late in (9, 12):
        earlier = tuple(range(4, late))
        first_correct = ~np.logical_or.reduce([correct[d] for d in earlier]) & correct[late]
        out.append({
            "later_depth": late,
            "wrong_at_9_correct_at_12": int(np.sum(~correct[9] & correct[12])) if late == 12 else "",
            "correct_at_9_wrong_at_12": int(np.sum(correct[9] & ~correct[12])) if late == 12 else "",
            "wrong_at_all_depths_4_to_8_correct_at_9": int(np.sum(first_correct)) if late == 9 else "",
            "wrong_at_all_depths_4_to_11_correct_at_12": int(np.sum(first_correct)) if late == 12 else "",
            "first_correct_at_this_depth_count": int(np.sum(first_correct)),
        })
    return out


def plot_transition(artifact: Path, filename: str, transitions: Sequence[Mapping[str, Any]], key: str,
                    ylabel: str, zero: bool = False) -> None:
    xs = [f"{r['shallow_depth']}→{r['deeper_depth']}" for r in transitions]
    ys = [float(r[key]) for r in transitions]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(xs, ys, color=["#277da1" if y >= 0 else "#c8553d" for y in ys])
    if zero:
        ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Consecutive depth transition")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(artifact / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_plots(artifact: Path) -> None:
    transitions = read_csv(artifact / "transition_analysis.csv")
    plot_transition(artifact, "rescue_rate_by_transition.png", transitions, "rescue_rate_given_shallow_wrong", "Rescue rate among shallower errors")
    plot_transition(artifact, "regression_rate_by_transition.png", transitions, "regression_rate_given_shallow_correct", "Regression rate among shallower correct predictions")
    plot_transition(artifact, "net_correction_by_transition.png", transitions, "net_correction", "Rescues − regressions", zero=True)
    plot_transition(artifact, "prediction_change_rate_by_transition.png", transitions, "prediction_change_rate", "Prediction-change rate")

    dist = read_csv(artifact / "oracle_selected_depth_distribution.csv")
    policies = list(ORACLE_SETS)
    categories = [*map(str, DEPTHS), "oracle_failure"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bottom = np.zeros(len(policies))
    for category in categories:
        vals = [float(next((r["fraction_all_samples"] for r in dist if r["policy"] == p and str(r["selected_depth"]) == category), 0.0)) for p in policies]
        ax.bar(policies, vals, bottom=bottom, label=category)
        bottom += np.asarray(vals)
    ax.set_ylabel("Fraction of validation samples")
    ax.set_title("Cheapest correct depth selected by unreachable oracle")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Oracle selection", ncol=5, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(artifact / "oracle_selected_depth_distribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    bins = read_csv(artifact / "difficulty_analysis.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, measure in zip(axes, ("predicted_class_confidence", "predictive_entropy_nats")):
        for shallow in (6, 7, 8, 9):
            selected = [r for r in bins if r["uncertainty_measure"] == measure and int(r["shallow_depth"]) == shallow and int(r["comparison_depth"]) == shallow + 1]
            selected.sort(key=lambda r: int(r["quantile_bin"]))
            valid = [r for r in selected if r["rescue_rate_given_shallow_wrong"] != ""]
            ax.plot([int(r["quantile_bin"]) for r in valid], [float(r["rescue_rate_given_shallow_wrong"]) for r in valid], marker="o", label=f"{shallow}→{shallow+1}")
        ax.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)])
        ax.set_xlabel("Shallow prediction quantile (low → high)")
        ax.set_title("Confidence" if measure == "predicted_class_confidence" else "Predictive entropy")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Rescue rate among shallow errors")
    axes[1].legend(title="Depth transition")
    fig.suptitle("Uncertainty vs. next-depth rescue (descriptive validation analysis)")
    fig.tight_layout()
    fig.savefig(artifact / "confidence_or_entropy_vs_rescue.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(artifact: Path, transitions, pairwise, oracle, fixed, later, bins) -> str:
    t_by = {(int(r["shallow_depth"]), int(r["deeper_depth"])): r for r in transitions}
    t9_12 = next(r for r in pairwise if int(r["shallow_depth"]) == 9 and int(r["deeper_depth"]) == 12)
    all_oracle = next(r for r in oracle if r["policy"] == "all_4_12")
    subset_69 = next(r for r in oracle if r["policy"] == "depths_6_9")
    subset_6912 = next(r for r in oracle if r["policy"] == "depths_6_7_8_9_12")
    late12 = next(r for r in later if int(r["later_depth"]) == 12)
    confidence_q1 = [r for r in bins if r["uncertainty_measure"] == "predicted_class_confidence" and int(r["comparison_depth"]) == int(r["shallow_depth"]) + 1 and int(r["quantile_bin"]) == 1]
    confidence_q5 = [r for r in bins if r["uncertainty_measure"] == "predicted_class_confidence" and int(r["comparison_depth"]) == int(r["shallow_depth"]) + 1 and int(r["quantile_bin"]) == 5]
    entropy_q5 = [r for r in bins if r["uncertainty_measure"] == "predictive_entropy_nats" and int(r["comparison_depth"]) == int(r["shallow_depth"]) + 1 and int(r["quantile_bin"]) == 5]
    low_rates = [float(r["rescue_rate_given_shallow_wrong"]) for r in confidence_q1 if r["rescue_rate_given_shallow_wrong"] != ""]
    high_entropy_rates = [float(r["rescue_rate_given_shallow_wrong"]) for r in entropy_q5 if r["rescue_rate_given_shallow_wrong"] != ""]
    conf_line = (
        f"At depths 6–9, the lowest-confidence quintile had {min(int(r['shallow_errors_in_bin']) for r in confidence_q1)}–{max(int(r['shallow_errors_in_bin']) for r in confidence_q1)} shallow errors per bin and next-depth rescue rates of {min(low_rates):.1%}–{max(low_rates):.1%}; the highest-confidence quintile had no shallow errors at any of these exits, so its conditional rescue rate is undefined."
        if low_rates else "Confidence bins did not contain enough shallow errors for a conditional rescue-rate summary."
    )
    entropy_line = (
        f"The highest-entropy quintile had next-depth rescue rates of {min(high_entropy_rates):.1%}–{max(high_entropy_rates):.1%}; the lowest-entropy quintile had no shallow errors, so its conditional rescue rate is undefined."
        if high_entropy_rates else "Entropy bins did not contain enough shallow errors for a conditional rescue-rate summary."
    )
    transitions_table = "\n".join(
        f"| {r['shallow_depth']}→{r['deeper_depth']} | {r['rescue_count']} | {float(r['rescue_rate_given_shallow_wrong']):.3f} | {r['regression_count']} | {float(r['regression_rate_given_shallow_correct']):.3f} | {r['net_correction']} | {float(r['prediction_change_rate']):.3f} |"
        for r in transitions
    )
    candidate = sorted(fixed, key=lambda r: float(r["fixed_validation_accuracy"]), reverse=True)[:4]
    candidates = ", ".join(f"{r['depth']} ({float(r['fixed_validation_accuracy']):.1%})" for r in candidate)
    summary = f"""# F001c — Per-input depth utility and rescue analysis

## Purpose and method

F001c asks whether deeper computation corrects examples misclassified by shallower operating points, including where aggregate validation accuracy has flattened or declined. This is analysis-only: it loads the existing F001 shared DeiT-Tiny backbone and existing F001/F001b heads, performs inference on the 1,003-example DermaMNIST-224 validation split, and does not train or tune anything. Only the validation split is instantiated; the official test split remains untouched. Predictions are stored by stable validation index at every depth 4–12, with class probabilities, confidence, and entropy.

Oracle calculations use only these measured per-sample predictions and the existing F001b measured fvcore FLOP counts. They are unreachable upper bounds, not deployable controllers. When no allowed depth is correct, mean oracle compute charges the maximum allowed depth as a conservative fallback; coverage separately marks such samples as failures. All plotted values are re-read from the saved CSV analyses.

## Consecutive transitions

| Transition | Rescues | Rescue rate* | Regressions | Regression rate** | Net correction | Prediction change rate |
|---|---:|---:|---:|---:|---:|---:|
{transitions_table}

* Conditional on the shallow prediction being wrong. **Conditional on it being correct. Counts and denominators are in `transition_analysis.csv`.

## Interpretation

1. **Do deeper computations rescue shallow errors?** Across 9→12, depth 12 rescues {t9_12['rescue_count']} of {t9_12['shallow_wrong_denominator']} depth-9 errors (conditional rate {float(t9_12['rescue_rate_given_shallow_wrong']):.1%}); the complete transition table gives all adjacent changes.
2. **Are rescues offset by regressions?** At 9→12 there are {t9_12['regression_count']} regressions against {t9_12['rescue_count']} rescues (net {t9_12['net_correction']}). Other transitions are listed above; per-input changes can coexist with a flat aggregate score.
3. **Does depth 12 add distinct value beyond depth 9?** {late12['wrong_at_9_correct_at_12']} samples are wrong at 9 and correct at 12; {late12['correct_at_9_wrong_at_12']} go the other way. Depth 12 is the first correct depth for {late12['first_correct_at_this_depth_count']} samples across depths 4–12.
4. **Is there oracle headroom?** The all-depth oracle covers {float(all_oracle['oracle_coverage']):.1%} of validation samples ({all_oracle['oracle_covered_count']}/{all_oracle['sample_count']}) at mean {float(all_oracle['mean_fvcore_gflops_all_samples_failure_at_max_allowed']):.4f} GFLOPs using the stated failure-cost convention. The {subset_69['allowed_depths']} oracle covers {float(subset_69['oracle_coverage']):.1%}. The {subset_6912['allowed_depths']} oracle covers {float(subset_6912['oracle_coverage']):.1%}, versus {float(next(r['oracle_coverage'] for r in oracle if r['policy'] == 'depths_6_7_8_9')):.1%} without depth 12. This idealized hindsight result is not achievable routing performance.
5. **Could uncertainty be a routing signal?** {conf_line} {entropy_line} These are descriptive validation-set comparisons, not evidence of calibration or a validated policy; inspect `difficulty_analysis.csv` including bin counts.
6. **Candidate depths to retain:** The strongest aggregate validation accuracies are at {candidates}. These are candidates for scenario-specific trade-off analysis, not “optimal” choices; lower-cost depths may still be useful under tighter constraints.
7. **Still unknown:** whether the rescue patterns persist on held-out test data under a pre-specified protocol; whether uncertainty yields a stable deployable policy; latency/energy budgets, target device and scenario utility; and whether later-depth corrections justify their incremental cost. Resolve these with Dolly/TA before choosing the final scenario or controller.

## Artifacts

- `per_sample_predictions.csv`: one row per validation sample and depth; full 7-class probability vector is included.
- `transition_analysis.csv`, `pairwise_analysis.csv`, `difficulty_analysis.csv`, `oracle_results.csv`, `oracle_selected_depth_distribution.csv`, `fixed_depth_comparison.csv`, `oracle_vs_fixed_depth.csv`, `later_depth_value.csv`.
- Six plots: rescue, regression, net correction, prediction change, oracle selected-depth distribution, and uncertainty-vs-rescue.
- `run_metadata.json`: exact checkpoint locations/hashes, F001b FLOPs, runtime configuration, split guard and input hashes.

This characterization uses DermaMNIST validation data and makes no clinical interpretation of errors. It does not select an optimal depth or implement a controller.
"""
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data/medmnist")
    parser.add_argument("--artifact-root", type=Path, default=root / "artifacts/F001c_per_input_depth_utility")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    before = snapshot_inputs(root)
    module = load_f001(root)
    device = module.choose_device(args.device)
    f001_config = read_json(root / "artifacts/feasibility_001/run_config.json")
    amp = device.type == "cuda" and not args.no_amp and bool(f001_config.get("amp", True))
    model, f001_config, f001b_results, flops = load_inputs(root, module, device)
    rows, data_info = infer_validation(module, model, device, amp, args.batch_size, args.num_workers, args.data_root)
    artifact = args.artifact_root.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    write_csv(artifact / "per_sample_predictions.csv", rows)

    transition = [compare(rows, d, d + 1) for d in range(4, 12)]
    pairwise = [compare(rows, a, b) for a, b in PAIRS]
    write_csv(artifact / "transition_analysis.csv", transition)
    write_csv(artifact / "pairwise_analysis.csv", pairwise)
    bins = difficulty_bins(rows)
    write_csv(artifact / "difficulty_analysis.csv", bins)
    oracle, distribution, fixed, _ = oracle_analysis(rows, flops)
    write_csv(artifact / "oracle_results.csv", oracle)
    write_csv(artifact / "oracle_selected_depth_distribution.csv", distribution)
    write_csv(artifact / "fixed_depth_comparison.csv", fixed)
    write_csv(artifact / "oracle_vs_fixed_depth.csv", oracle_fixed_comparison(oracle, fixed))
    later = later_depth_value(rows)
    write_csv(artifact / "later_depth_value.csv", later)

    make_plots(artifact)
    # Report is derived from the saved tabular analyses, so plots and prose share sources.
    report = summarize(artifact, read_csv(artifact / "transition_analysis.csv"),
                       read_csv(artifact / "pairwise_analysis.csv"),
                       read_csv(artifact / "oracle_results.csv"),
                       read_csv(artifact / "fixed_depth_comparison.csv"),
                       read_csv(artifact / "later_depth_value.csv"),
                       read_csv(artifact / "difficulty_analysis.csv"))
    (artifact / "summary.md").write_text(report, encoding="utf-8")
    if args.data_root.resolve() != (root / "data/medmnist").resolve():
        data_root = str(args.data_root.resolve())
    else:
        data_root = "data/medmnist"
    metadata = {
        "experiment": "F001c", "analysis_only": True, "trained_any_weights": False,
        "dataset": "DermaMNIST-224", "split_loaded": "val", "test_split_loaded": False,
        "validation_sample_count": data_info["sample_count"], "evaluated_depths": list(DEPTHS),
        "probabilities_saved": True, "entropy_unit": "nats", "data_root": data_root,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": amp, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "f001_config": f001_config, "f001b_source_metadata": {"experiment": f001b_results.get("experiment"), "depths": f001b_results.get("depths"), "splits_loaded": f001b_results.get("splits_loaded"), "test_split_loaded": f001b_results.get("test_split_loaded")},
        "checkpoint_paths": {"backbone": "checkpoints/feasibility_001/full_backbone.pt", "depth_4": "checkpoints/feasibility_001/head_depth_4.pt", "depth_8": "checkpoints/feasibility_001/head_depth_8.pt", **{f"depth_{d}": f"artifacts/F001b_depth_sweep/checkpoints/head_depth_{d}.pt" for d in DEPTHS if d not in (4, 8, 12)}, "depth_12_classifier": "checkpoints/feasibility_001/full_backbone.pt (classifier included)"},
        "f001b_measured_fvcore_flops": {str(d): flops[d] for d in DEPTHS},
        "oracle_cost_failure_convention": "maximum allowed measured fvcore FLOPs charged when no allowed depth is correct",
        "input_sha256_before_analysis": before,
    }
    write_json(artifact / "run_metadata.json", metadata)
    after = snapshot_inputs(root)
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if changed:
        raise RuntimeError("F001/F001b inputs changed during F001c: " + ", ".join(changed))
    metadata["input_sha256_after_analysis"] = after
    metadata["input_hashes_unchanged"] = True
    write_json(artifact / "run_metadata.json", metadata)
    print(f"F001c complete: {data_info['sample_count']} validation samples × {len(DEPTHS)} depths; test split untouched.")
    print(f"Artifacts: {artifact}")
    print(f"All {len(before)} F001/F001b input-file hashes match after analysis.")


if __name__ == "__main__":
    main()
