"""F006a_2: validation-only quality screening of all 24 F005b configurations.

The F002b reference is reconstructed from tracked duplicate-aware confusion
counts. F005b cell counts and selected metrics come from its canonical run;
F006a supplies the hash-checked checkpoint, exact strict FLOPs and rounded
seed-2026 AUROC. No model training or dataset inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.F006a import run_analysis as f006a

HERE = Path(__file__).resolve().parent
F002_README = ROOT / "experiments" / "F002" / "README.md"
F002_CHECKPOINT = ROOT / "checkpoints" / "f002b_mobilenetv2_seed2026.pt"
F002_CHECKPOINT_SHA256 = "5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823"
EXPECTED_CMAX = 20_596_068_563
NORMAL_UNITS, ABNORMAL_UNITS = 178, 76


@dataclass(frozen=True)
class Quality:
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: Fraction
    macro_f1: Fraction
    normal_recall: Fraction
    abnormal_recall: Fraction


@dataclass(frozen=True)
class Profile:
    point: f006a.Point
    quality: Quality


@dataclass(frozen=True)
class Thresholds:
    accuracy: Fraction
    macro_f1: Fraction
    abnormal_recall: Fraction


def quality_from_counts(tn: int, fp: int, fn: int, tp: int) -> Quality:
    """Class 0 is normal; class 1 is abnormal (F002 binary_metrics)."""
    if any(not isinstance(n, int) or n < 0 for n in (tn, fp, fn, tp)):
        raise ValueError("Confusion counts must be nonnegative integers")
    if tn + fp != NORMAL_UNITS or fn + tp != ABNORMAL_UNITS:
        raise ValueError("Confusion counts must cover 178 normal and 76 abnormal units")
    normal_f1 = Fraction(2 * tn, 2 * tn + fp + fn)
    abnormal_f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    return Quality(tn, fp, fn, tp,
                   Fraction(tn + tp, NORMAL_UNITS + ABNORMAL_UNITS),
                   (normal_f1 + abnormal_f1) / 2,
                   Fraction(tn, NORMAL_UNITS), Fraction(tp, ABNORMAL_UNITS))


def read_baseline(readme_path: Path = F002_README, checkpoint_path: Path = F002_CHECKPOINT) -> Quality:
    """Derive exact F002b values from documented recall numerators/denominators."""
    if (readme_path == F002_README and checkpoint_path == F002_CHECKPOINT and
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != F002_CHECKPOINT_SHA256):
        raise ValueError("Frozen F002b checkpoint hash changed")
    readme = readme_path.read_text(encoding="utf-8")
    section = readme.split("## F002b frozen static reference", 1)[1].split("## Interpretation", 1)[0]

    def recalled(label: str) -> tuple[int, int]:
        match = re.search(rf"^\| {label} \| [0-9.]+ \((\d+)/(\d+)\) \|$", section, re.MULTILINE)
        if not match:
            raise ValueError(f"F002b {label} confusion count is unavailable")
        return int(match.group(1)), int(match.group(2))

    tn, normal_total = recalled("Normal recall")
    tp, abnormal_total = recalled("Abnormal recall")
    if (normal_total, abnormal_total) != (NORMAL_UNITS, ABNORMAL_UNITS):
        raise ValueError("F002b evaluation-unit totals changed")
    baseline = quality_from_counts(tn, normal_total - tn, abnormal_total - tp, tp)
    # Guard against accidentally parsing an unrelated table or a stale record.
    for metric, rounded in (("accuracy", "0.886"), ("macro_f1", "0.859")):
        label = "Accuracy" if metric == "accuracy" else "Macro-F1"
        if f"| {label} | {rounded} |" not in section or abs(float(getattr(baseline, metric)) - float(rounded)) >= 0.0005:
            raise ValueError(f"F002b rounded {label} disagrees with reconstructed counts")
    return baseline


def reconstruct_profiles() -> list[Profile]:
    """Recount strict FLOPs through F006a and validate all selected-run counts."""
    points = f006a.reconstruct_points()
    cells = f006a.read_selected_cells()
    if len(points) != 24 or {p.name for p in points} != set(cells):
        raise ValueError("Expected all 24 F005b width × K configurations")
    profiles = []
    for point in points:
        cell = cells[point.name]
        quality = quality_from_counts(*(cell[field] for field in ("tn", "fp", "fn", "tp")))
        for field in ("accuracy", "macro_f1", "normal_recall", "abnormal_recall"):
            recorded = point.__dict__[field] if field in ("accuracy", "macro_f1") else cell[field]
            if not math.isclose(float(getattr(quality, field)), recorded, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{point.name}: {field} disagrees with selected-run confusion counts")
        profiles.append(Profile(point, quality))
    return sorted(profiles, key=lambda p: (p.point.flops, p.point.name))


def baseline_thresholds(baseline: Quality) -> Thresholds:
    return Thresholds(baseline.accuracy, baseline.macro_f1, baseline.abnormal_recall)


def screen(profile: Profile, requirements: Thresholds) -> dict[str, bool]:
    q = profile.quality
    flags = {"accuracy": q.accuracy >= requirements.accuracy,
             "macro_f1": q.macro_f1 >= requirements.macro_f1,
             "abnormal_recall": q.abnormal_recall >= requirements.abnormal_recall}
    return flags | {"all": all(flags.values())}


def screened_rows(profiles: list[Profile], baseline: Quality) -> list[dict]:
    thresholds = baseline_thresholds(baseline)
    rows = []
    for profile in sorted(profiles, key=lambda p: (p.point.flops, p.point.name)):
        p, q = profile.point, profile.quality
        flags = screen(profile, thresholds)
        rows.append({
            "configuration": p.name, "width": f"{p.width:g}", "k": p.k,
            "flops": p.flops, "gflops": f"{p.flops / 1e9:.9f}",
            "tn": q.tn, "fp": q.fp, "fn": q.fn, "tp": q.tp,
            "accuracy": repr(float(q.accuracy)), "macro_f1": repr(float(q.macro_f1)),
            "normal_recall": repr(float(q.normal_recall)),
            "abnormal_recall": repr(float(q.abnormal_recall)),
            "auroc_rounded_3dp": f"{p.auroc:.3f}",
            "delta_accuracy_vs_f002b": repr(float(q.accuracy - baseline.accuracy)),
            "delta_macro_f1_vs_f002b": repr(float(q.macro_f1 - baseline.macro_f1)),
            "delta_normal_recall_vs_f002b": repr(float(q.normal_recall - baseline.normal_recall)),
            "delta_abnormal_recall_vs_f002b": repr(float(q.abnormal_recall - baseline.abnormal_recall)),
            "pass_accuracy": "PASS" if flags["accuracy"] else "FAIL",
            "pass_macro_f1": "PASS" if flags["macro_f1"] else "FAIL",
            "pass_abnormal_recall": "PASS" if flags["abnormal_recall"] else "FAIL",
            "pass_all": "PASS" if flags["all"] else "FAIL",
            "failed_requirements": ";".join(key for key in ("accuracy", "macro_f1", "abnormal_recall")
                                            if not flags[key]),
        })
    return rows


def screening_summary(profiles: list[Profile], requirements: Thresholds) -> dict[str, int]:
    decisions = [screen(p, requirements) for p in profiles]
    return {key: sum(d[key] for d in decisions) for key in ("accuracy", "macro_f1", "abnormal_recall", "all")}


def summary_rows(profiles: list[Profile], baseline: Quality) -> list[dict]:
    counts = screening_summary(profiles, baseline_thresholds(baseline))
    thresholds = baseline_thresholds(baseline)
    return [{"requirement": key, "minimum": repr(float(getattr(thresholds, key))) if key != "all" else "",
             "passing_of_24": counts[key], "failing_of_24": 24 - counts[key]}
            for key in ("accuracy", "macro_f1", "abnormal_recall", "all")]


def cheapest(profiles: list[Profile]) -> Profile | None:
    return min(profiles, key=lambda p: (p.point.flops, p.point.name)) if profiles else None


def budget_rows(profiles: list[Profile], baseline: Quality) -> list[dict]:
    by_name = {p.point.name: p for p in profiles}
    if len(by_name) != 24 or by_name["w1.0_k16"].point.flops != EXPECTED_CMAX:
        raise ValueError("F005b Cmax differs from the proposed 20.596068563 GFLOPs")
    thresholds = baseline_thresholds(baseline)
    qualified = [p for p in profiles if screen(p, thresholds)["all"]]
    points = [p.point for p in profiles]
    rows = []
    for b in f006a.BUDGETS:
        limit = f006a.budget_flops(b, EXPECTED_CMAX)
        feasible = [p for p in profiles if b > 0 and p.point.flops <= limit]
        joint = [p for p in feasible if screen(p, thresholds)["all"]]
        chosen = cheapest(joint)
        rule_a = f006a.select(points, limit, "max_f1") if b > 0 else None
        exact_limit = Decimal(limit.numerator) / Decimal(limit.denominator)
        rows.append({
            "budget_state": f"{float(b):.2f}", "budget_flops": format(exact_limit, "f"),
            "budget_gflops": f"{exact_limit / Decimal(10**9):.11f}",
            "budget_feasible_count": len(feasible),
            "budget_feasible_configurations": ";".join(p.point.name for p in feasible),
            "quality_qualified_total": len(qualified),
            "quality_and_budget_count": len(joint),
            "quality_and_budget_configurations": ";".join(p.point.name for p in joint),
            "cheapest_qualified": chosen.point.name if chosen else "DEFER",
            "cheapest_qualified_gflops": f"{chosen.point.flops / 1e9:.9f}" if chosen else "",
            "cheapest_qualified_abnormal_recall": repr(float(chosen.quality.abnormal_recall)) if chosen else "",
            "f006a_rule_a": rule_a.name if rule_a else "DEFER",
            "f006a_rule_a_passes_quality": ("PASS" if screen(by_name[rule_a.name], thresholds)["all"]
                                           else "FAIL") if rule_a else "",
            "status": "FEASIBLE" if chosen else "DEFER",
        })
    return rows


def sensitivity_scenarios(baseline: Quality) -> list[tuple[str, Thresholds]]:
    """One-factor perturbations set by one validation unit or 0.005 F1."""
    base = baseline_thresholds(baseline)
    cases = [("baseline", base)]
    for units in (-1, 1, 2, 3):
        cases.append((f"accuracy_{units:+d}_unit", Thresholds(base.accuracy + Fraction(units, 254),
                                                                 base.macro_f1, base.abnormal_recall)))
    for delta in (Fraction(-5, 1000), Fraction(5, 1000), Fraction(10, 1000)):
        cases.append((f"macro_f1_{float(delta):+0.3f}", Thresholds(base.accuracy,
                                                                    base.macro_f1 + delta,
                                                                    base.abnormal_recall)))
    for units in (-1, 1, 2, 3):
        cases.append((f"abnormal_recall_{units:+d}_unit",
                      Thresholds(base.accuracy, base.macro_f1,
                                 base.abnormal_recall + Fraction(units, 76))))
    return cases


def sensitivity_rows(profiles: list[Profile], baseline: Quality) -> list[dict]:
    rows = []
    for name, requirements in sensitivity_scenarios(baseline):
        survivors = [p for p in profiles if screen(p, requirements)["all"]]
        feasible_by_b = {}
        for b in f006a.BUDGETS:
            limit = f006a.budget_flops(b, EXPECTED_CMAX)
            feasible = [p for p in survivors if b > 0 and p.point.flops <= limit]
            feasible_by_b[f"b{float(b):.2f}_qualified_count"] = len(feasible)
            feasible_by_b[f"b{float(b):.2f}_cheapest"] = cheapest(feasible).point.name if feasible else "DEFER"
        rows.append({
            "scenario": name,
            "minimum_accuracy": repr(float(requirements.accuracy)),
            "minimum_macro_f1": repr(float(requirements.macro_f1)),
            "minimum_abnormal_recall": repr(float(requirements.abnormal_recall)),
            "qualified_count": len(survivors),
            "qualified_configurations": ";".join(p.point.name for p in survivors),
            **feasible_by_b,
        })
    return rows


def render_products(profiles: list[Profile], baseline: Quality) -> dict[str, bytes]:
    profiles = sorted(profiles, key=lambda p: (p.point.flops, p.point.name))
    screened = screened_rows(profiles, baseline)
    summary = summary_rows(profiles, baseline)
    budgets = budget_rows(profiles, baseline)
    sensitivity = sensitivity_rows(profiles, baseline)
    return {"quality_screen.csv": f006a.render_csv(list(screened[0]), screened),
            "screening_summary.csv": f006a.render_csv(list(summary[0]), summary),
            "budget_feasibility.csv": f006a.render_csv(list(budgets[0]), budgets),
            "sensitivity.csv": f006a.render_csv(list(sensitivity[0]), sensitivity)}


def render_figure(profiles: list[Profile], baseline: Quality) -> bytes:
    """Show the recall dimension absent from the existing F006a Pareto plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = baseline_thresholds(baseline)
    profiles = sorted(profiles, key=lambda p: (p.point.flops, p.point.name))
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11, "savefig.dpi": 200}):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        for qualifies, color, label, marker in (
                (False, "#8B9299", "Fails at least one quality requirement", "x"),
                (True, "#007E71", "Passes all three provisional requirements", "o")):
            subset = [p for p in profiles if screen(p, thresholds)["all"] == qualifies]
            ax.scatter([p.point.flops / 1e9 for p in subset],
                       [float(p.quality.abnormal_recall) for p in subset],
                       color=color, marker=marker, s=58, alpha=0.9, label=label, zorder=3)
        ax.axhline(float(baseline.abnormal_recall), color="#B35E00", linestyle="--",
                   linewidth=1.4, label="F002b abnormal recall: 57/76", zorder=2)
        offsets = {"w0.5_k2": (7, 10), "w0.75_k8": (7, -17),
                   "w1.0_k12": (-7, 9), "w1.0_k16": (-7, -18)}
        for p in profiles:
            if p.point.name in offsets:
                dx, dy = offsets[p.point.name]
                ax.annotate(p.point.name, (p.point.flops / 1e9, float(p.quality.abnormal_recall)),
                            xytext=(dx, dy), textcoords="offset points", fontsize=8,
                            ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.set_xlabel("GFLOPs per sequence (strict fvcore)")
        ax.set_ylabel("Abnormal recall on 76 validation units")
        ax.set_title("F005b seed 2026: abnormal recall and provisional quality screen")
        ax.set_ylim(0.58, 0.9)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", metadata={"Software": "F006a_2 run_analysis.py"})
        plt.close(fig)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare regenerated table bytes")
    args = parser.parse_args()
    baseline = read_baseline()
    profiles = reconstruct_profiles()
    products = render_products(profiles, baseline)
    for filename, data in products.items():
        path = HERE / filename
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Generated output differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if not args.check:
        figure_path = HERE / "figures" / "quality_vs_work.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure_path.write_bytes(render_figure(profiles, baseline))
    print(json.dumps({"source_f005b_run_sha256": f006a.RUN_SHA256,
                      "baseline_accuracy": float(baseline.accuracy),
                      "baseline_macro_f1": float(baseline.macro_f1),
                      "baseline_abnormal_recall": float(baseline.abnormal_recall),
                      "points": len(profiles),
                      "passing": screening_summary(profiles, baseline_thresholds(baseline)),
                      "tables_checked" if args.check else "tables_written": len(products)}, sort_keys=True))


if __name__ == "__main__":
    main()
