"""Deterministic T012b resource traces, alternatives, and optional validation inference.

The tracked simulation is policy/cost evidence. ``--inference`` additionally
serves three real validation clips through the canonical F005b model, recording
predictions and incidental local per-call latency in a separate trace. It never
opens the T006 test partition or computes mixed-policy quality from cell means.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.F003.adaptive import frame_indices
from experiments.F006a import run_analysis as f006a
from experiments.F006a_2 import run_analysis as f006a_2
from experiments.F006b import controller

HERE = Path(__file__).resolve().parent
VALIDATION = ROOT / "data" / "openpocus" / "t006" / "validation.csv"
SCENARIOS = {
    "decreasing": ("1.00", "0.83", "0.67", "0.25", "0"),
    "increasing": ("0", "0.25", "0.67", "0.83", "1.00"),
    "fluctuating": ("0.83", "0.25", "0.67", "1.00", "0.50"),
}
SWEEP = ("0", "0.05", "0.25", "0.50", "0.5001", "0.67", "0.75",
         "0.7501", "0.83", "0.99", "1.00")


def validation_cases(path: Path = VALIDATION) -> tuple[dict, ...]:
    """Use the first row of three distinct validation dependency groups."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 265 or any(r["partition"] != "validation" for r in rows):
        raise ValueError("Expected the tracked 265-row T006 validation manifest")
    chosen, seen = [], set()
    for row in rows:
        if row["split_group"] in seen:
            continue
        seen.add(row["split_group"])
        if int(row["frame_count"]) < 16:
            raise ValueError("F005b serving requires at least 16 frames")
        chosen.append(row)
        if len(chosen) == len(SCENARIOS):
            break
    if len(chosen) != len(SCENARIOS):
        raise ValueError("Three distinct validation groups are required")
    return tuple(chosen)


def clip_id(row: dict) -> str:
    return f'{row["exam_id"]}:{row["zone_id"]}'


def exact_decimal(value: Fraction) -> str:
    return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def trace_rows(points: tuple[controller.OperatingPoint, ...],
               baseline: f006a_2.Quality, cases: tuple[dict, ...]) -> list[dict]:
    """Simulate the F005 width/frame cache for each policy on each one-clip trace."""
    if len(cases) != len(SCENARIOS):
        raise ValueError("One independent validation clip is needed per scenario")
    rows = []
    for (scenario, resources), case in zip(SCENARIOS.items(), cases):
        count = int(case["frame_count"])
        for policy in controller.POLICIES:
            cached: set[tuple[float, int]] = set()
            previous = None
            for index, resource in enumerate(resources, start=1):
                selection = controller.select(points, baseline, resource, policy)
                point = selection.point
                name = point.name if point else "DEFER"
                chosen_frames = frame_indices(count, point.k) if point else ()
                new = sum((point.width, frame) not in cached for frame in chosen_frames) if point else 0
                reused = len(chosen_frames) - new
                if point:
                    cached.update((point.width, frame) for frame in chosen_frames)
                incremental = (controller.incremental_flops(points, point.width, new)
                               if point else 0)
                if point and incremental > point.flops:
                    raise AssertionError("Incremental model work exceeds full request work")
                rows.append({
                    "scenario": scenario, "request": index, "policy": policy,
                    "clip_id": clip_id(case), "validation_group": case["split_group"],
                    "frame_count": count, "resource_b": resource,
                    "permitted_flops": exact_decimal(selection.permitted_flops),
                    "configuration": name,
                    "width": f"{point.width:g}" if point else "",
                    "k": point.k if point else "",
                    "configuration_flops": point.flops if point else 0,
                    "incremental_modeled_flops": incremental,
                    "new_encoded_frames": new, "reused_frames": reused,
                    "cached_width_frame_pairs": len(cached),
                    "selection_change": (str(name != previous).lower() if previous is not None else "false"),
                    "model_switch": (str(point is not None and previous not in (None, "DEFER")
                                         and name != previous).lower()),
                    "reference_accuracy": repr(float(point.quality.accuracy)) if point else "",
                    "reference_macro_f1": repr(float(point.quality.macro_f1)) if point else "",
                    "reference_abnormal_recall": repr(float(point.quality.abnormal_recall)) if point else "",
                    "predicted_class": "", "probability_abnormal": "",
                    "observed_local_ms": "", "direct_logit_abs_diff": "",
                    "deferred": str(point is None).lower(),
                    "reason": selection.reason,
                })
                previous = name
    return rows


def comparison_rows(trace: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in trace:
        grouped[(row["scenario"], row["policy"])].append(row)
    result = []
    for scenario in SCENARIOS:
        for policy in controller.POLICIES:
            rows = grouped[(scenario, policy)]
            selected = [r for r in rows if r["deferred"] == "false"]
            full = sum(int(r["configuration_flops"]) for r in selected)
            incremental = sum(int(r["incremental_modeled_flops"]) for r in selected)
            violations = sum(int(r["configuration_flops"]) > Fraction(r["resource_b"]) * controller.C_MAX
                             for r in selected)
            if violations:
                raise AssertionError("Policy attempted an over-budget inference")
            names = list(dict.fromkeys(r["configuration"] for r in selected))
            result.append({
                "scenario": scenario, "policy": policy,
                "requests": len(rows), "inferences": len(selected),
                "deferrals": len(rows) - len(selected),
                "selection_changes_including_defer": sum(r["selection_change"] == "true" for r in rows),
                "active_model_switches": sum(r["model_switch"] == "true" for r in rows),
                "budget_violations": violations,
                "budget_blocked_requests": sum(r["reason"].endswith("over_budget") for r in rows),
                "total_full_configuration_flops": full,
                "mean_full_flops_per_request": exact_decimal(Fraction(full, len(rows))),
                "total_incremental_modeled_flops": incremental,
                "mean_incremental_flops_per_request": exact_decimal(Fraction(incremental, len(rows))),
                "encoded_frames": sum(int(r["new_encoded_frames"]) for r in rows),
                "reused_frames": sum(int(r["reused_frames"]) for r in rows),
                "selected_configurations": ";".join(names),
                "measured_mixed_policy_macro_f1": "",
            })
    return result


def budget_sweep_rows(points: tuple[controller.OperatingPoint, ...],
                      baseline: f006a_2.Quality) -> list[dict]:
    rows = []
    for resource in SWEEP:
        for policy in controller.POLICIES:
            selected = controller.select(points, baseline, resource, policy)
            p = selected.point
            rows.append({"resource_b": resource, "policy": policy,
                         "permitted_flops": exact_decimal(selected.permitted_flops),
                         "configuration": p.name if p else "DEFER",
                         "configuration_flops": p.flops if p else 0,
                         "abnormal_recall": repr(float(p.quality.abnormal_recall)) if p else "",
                         "macro_f1": repr(float(p.quality.macro_f1)) if p else "",
                         "reason": selected.reason})
    return rows


def boundary_rows(points: tuple[controller.OperatingPoint, ...],
                  baseline: f006a_2.Quality) -> list[dict]:
    selected = controller.selected_configurations(points, baseline)
    rows = []
    for point in selected:
        threshold = Fraction(point.flops, controller.C_MAX)
        for relation, budget_flops in (("one_flop_below", point.flops - 1),
                                       ("exact", point.flops),
                                       ("one_flop_above", point.flops + 1)):
            b = Fraction(budget_flops, controller.C_MAX)
            if b > 1:
                rows.append({"threshold_configuration": point.name, "relation": relation,
                             "threshold_flops": point.flops, "cmax_flops": controller.C_MAX,
                             "threshold_fraction": str(threshold),
                             "threshold_b_decimal": repr(point.flops / controller.C_MAX),
                             "resource_fraction": str(b), "permitted_flops": budget_flops,
                             "selected_configuration": "OUTSIDE_DOMAIN", "reason": "b_above_one"})
                continue
            choice = controller.select(points, baseline, b)
            rows.append({"threshold_configuration": point.name, "relation": relation,
                         "threshold_flops": point.flops, "cmax_flops": controller.C_MAX,
                         "threshold_fraction": str(threshold), "resource_fraction": str(b),
                         "threshold_b_decimal": repr(point.flops / controller.C_MAX),
                         "permitted_flops": budget_flops,
                         "selected_configuration": choice.point.name if choice.point else "DEFER",
                         "reason": choice.reason})
    return rows


def validate_envelope(points: tuple[controller.OperatingPoint, ...],
                      baseline: f006a_2.Quality) -> tuple[controller.OperatingPoint, ...]:
    selected = controller.selected_configurations(points, baseline)
    if not selected or any(controller.three_dimensional_dominators(p, points) for p in selected):
        raise ValueError("A selected point is dominated in FLOPs/F1/abnormal recall")
    choices = [controller.select(points, baseline, Fraction(p.flops, controller.C_MAX)).point
               for p in sorted(points, key=lambda p: (p.flops, p.name))]
    recalls = [p.quality.abnormal_recall for p in choices if p is not None]
    if recalls != sorted(recalls):
        raise ValueError("Recall fell as the available FLOP budget increased")
    return selected


def products(points: tuple[controller.OperatingPoint, ...], baseline: f006a_2.Quality,
             cases: tuple[dict, ...]) -> tuple[dict[str, bytes], list[dict]]:
    selected = validate_envelope(points, baseline)
    trace = trace_rows(points, baseline, cases)
    tables = {"resource_trace.csv": trace,
              "comparison.csv": comparison_rows(trace),
              "budget_sweep.csv": budget_sweep_rows(points, baseline),
              "budget_boundaries.csv": boundary_rows(points, baseline)}
    rendered = {name: f006a.render_csv(list(rows[0]), rows) for name, rows in tables.items()}
    source = {"f005b_run_sha256": f006a.RUN_SHA256,
              "f005b_checkpoint_sha256": "18d601e3ea04a1ba572ef99484c8681c5a3011dd7d3eb28309cb32906ff9e702",
              "f005b_importance_sha256": _canonical_importance_sha256(),
              "f002b_checkpoint_sha256": f006a_2.F002_CHECKPOINT_SHA256,
              "validation_manifest_sha256": hashlib.sha256(VALIDATION.read_bytes()).hexdigest(),
              "cmax_flops": controller.C_MAX,
              "policy": "quality-qualified; maximize abnormal recall; lower FLOPs; name",
              "selected": [{"configuration": p.name, "flops": p.flops,
                            "threshold_b_fraction": str(Fraction(p.flops, controller.C_MAX)),
                            "threshold_b_decimal": repr(p.flops / controller.C_MAX),
                            "macro_f1": repr(float(p.quality.macro_f1)),
                            "abnormal_recall": repr(float(p.quality.abnormal_recall)),
                            "three_dimensional_dominators": []} for p in selected],
              "scenarios": SCENARIOS,
              "validation_clips": [{"scenario": name, "clip_id": clip_id(case),
                                    "split_group": case["split_group"],
                                    "frame_count": int(case["frame_count"])}
                                   for name, case in zip(SCENARIOS, cases)],
              "mixed_policy_quality": "not computed: the deterministic trace has no unit predictions"}
    rendered["simulation_summary.json"] = (json.dumps(source, indent=2, sort_keys=True) + "\n").encode()
    return rendered, trace


def _canonical_importance_sha256() -> str:
    from experiments.F005.run_f005b import CANONICAL_NESTED_IMPORTANCE_SHA256

    return CANONICAL_NESTED_IMPORTANCE_SHA256


def render_figure(trace: list[dict]) -> bytes:
    """One figure: chosen model work follows the budget under three traces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"recall_priority": "#007E71", "minimum_quality": "#BE663B",
              "f006a_rule_a": "#5172AB", "fixed_full_k16": "#8D6397"}
    labels = {"recall_priority": "Recall-priority", "minimum_quality": "Minimum quality cost",
              "f006a_rule_a": "F006a Rule A", "fixed_full_k16": "Fixed full K16"}
    with plt.rc_context({"font.size": 9, "axes.labelsize": 10, "savefig.dpi": 200}):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
        for ax, scenario in zip(axes, SCENARIOS):
            per_scenario = [r for r in trace if r["scenario"] == scenario]
            main = [r for r in per_scenario if r["policy"] == "recall_priority"]
            xs = [int(r["request"]) for r in main]
            ax.plot(xs, [float(r["permitted_flops"]) / 1e9 for r in main],
                    color="#333333", linestyle="--", marker="|", label="Budget ceiling")
            for policy in controller.POLICIES:
                policy_rows = [r for r in per_scenario if r["policy"] == policy]
                ys = [int(r["configuration_flops"]) / 1e9 if r["deferred"] == "false"
                      else float("nan") for r in policy_rows]
                ax.plot(xs, ys, color=colors[policy], marker="o", markersize=4,
                        linewidth=1.6, label=labels[policy])
            for row in main:
                x = int(row["request"])
                if row["deferred"] == "true":
                    ax.annotate("DEFER", (x, 0.4), ha="center", fontsize=7, color=colors["recall_priority"])
                else:
                    ax.annotate(row["configuration"],
                                (x, int(row["configuration_flops"]) / 1e9),
                                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
                                color=colors["recall_priority"])
            ax.set_title(scenario.capitalize())
            ax.set_xticks(xs)
            ax.set_xlabel("Request")
            ax.set_ylim(0, 22.8)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("GFLOPs per request (full configuration)")
        handles, legends = axes[0].get_legend_handles_labels()
        fig.legend(handles, legends, loc="lower center", ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("T012b provisional controller under changing FLOP budgets")
        fig.tight_layout(rect=(0, 0.1, 1, 0.93))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", metadata={"Software": "F006b simulate.py"})
        plt.close(fig)
    return stream.getvalue()


def validation_inference_rows(trace: list[dict], cases: tuple[dict, ...]) -> list[dict]:
    """Run the same trace on real validation clips, checking cache/direct parity."""
    import torch

    from src import f002_pipeline as pipe
    from experiments.F006b.runtime import NestedClipRuntime, direct_predict, load_runtime

    model = load_runtime(device="cpu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    observed = []
    try:
        for (scenario, _), case in zip(SCENARIOS.items(), cases):
            retained = pipe.retained_frames(case, pipe.RAW)
            k16 = frame_indices(len(retained), 16)
            preloaded = {i: pipe.preprocess_frame(retained[i]) for i in k16}

            def load(numbers: list[int]):
                return torch.stack([preloaded[i] for i in numbers])

            for policy in controller.POLICIES:
                runtime = NestedClipRuntime(model, len(retained), load, chunk_size=4)
                for row in (r for r in trace if r["scenario"] == scenario and r["policy"] == policy):
                    record = row.copy()
                    if row["deferred"] == "false":
                        result = runtime.predict(float(row["width"]), int(row["k"]))
                        direct = direct_predict(model, len(retained), load, result.width,
                                                result.k, chunk_size=4)
                        gap = abs(result.logit - direct)
                        if gap > 5e-5 or result.new_frames != row["new_encoded_frames"]:
                            raise ValueError("Canonical cached/direct parity or cache accounting failed")
                        record.update({"predicted_class": result.predicted_class,
                                       "probability_abnormal": f"{result.probability_abnormal:.9f}",
                                       "observed_local_ms": f"{result.elapsed_ms:.3f}",
                                       "direct_logit_abs_diff": f"{gap:.9g}"})
                    observed.append(record)
    finally:
        torch.set_num_threads(previous_threads)
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare deterministic tables with saved bytes")
    parser.add_argument("--inference", action="store_true", help="also serve three validation clips on CPU")
    args = parser.parse_args()
    if args.check and args.inference:
        parser.error("--check compares deterministic policy outputs only")
    baseline = f006a_2.read_baseline()
    points = controller.load_policy_points()
    cases = validation_cases()
    rendered, trace = products(points, baseline, cases)
    for name, data in rendered.items():
        path = HERE / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Generated output differs: {path}")
        else:
            path.write_bytes(data)
    if not args.check:
        figure = HERE / "figures" / "controller_response.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(render_figure(trace))
    if args.inference:
        rows = validation_inference_rows(trace, cases)
        raw_output = ROOT / "artifacts" / "F006b" / "validation_inference_trace.csv"
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_bytes(f006a.render_csv(list(rows[0]), rows))
    print(json.dumps({"selected": [p.name for p in validate_envelope(points, baseline)],
                      "scenario_requests": len(trace), "tables_checked" if args.check else "tables_written": len(rendered),
                      "validation_inference_requests": len(rows) if args.inference else 0}, sort_keys=True))


if __name__ == "__main__":
    main()
