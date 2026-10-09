"""T012a: validation-only design analysis of the deployable F005b family.

The canonical F005b run supplies selected-epoch macro-F1 and accuracy. Strict
fvcore FLOPs are recomputed from its hash-checked checkpoint. The original raw
evaluation artifact is unavailable here, so AUROC is read at the three-decimal
precision printed for seed 2026 in the tracked F005 README.
"""
from __future__ import annotations

import argparse
import csv
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
RUN = ROOT / "checkpoints" / "f005b_nested_seed2026" / "run.json"
RUN_SHA256 = "47710bc9f918576b5698d2f0eeaa3f6f1c7de127b483cd7c3e387f735e131bec"
F005_README = ROOT / "experiments" / "F005" / "README.md"
HERE = Path(__file__).resolve().parent
WIDTHS = (0.25, 0.5, 0.75, 1.0)
KS = (1, 2, 4, 8, 12, 16)
BUDGETS = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
# Illustrative sensitivity grid, not a selected policy parameter.
EPSILONS = (0.0, 0.005, 0.01, 0.015, 0.02)
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.03)


@dataclass(frozen=True)
class Point:
    name: str
    width: float
    k: int
    flops: int
    macro_f1: float
    accuracy: float = 0.0
    auroc: float = 0.0


def order_key(point: Point) -> tuple:
    return (point.flops, -point.macro_f1, point.width, point.k, point.name)


def dominates(a: Point, b: Point) -> bool:
    return (a.flops <= b.flops and a.macro_f1 >= b.macro_f1
            and (a.flops < b.flops or a.macro_f1 > b.macro_f1))


def dominance_witnesses(points: list[Point]) -> dict[str, str | None]:
    """One reproducible witness: highest F1, then cheapest, then name."""
    if len({p.name for p in points}) != len(points):
        raise ValueError("Duplicate configuration name")
    return {p.name: (min((q for q in points if dominates(q, p)),
                         key=lambda q: (-q.macro_f1, q.flops, q.name)).name
                     if any(dominates(q, p) for q in points) else None)
            for p in points}


def pareto_front(points: list[Point]) -> list[Point]:
    witnesses = dominance_witnesses(points)
    return sorted((p for p in points if witnesses[p.name] is None), key=order_key)


def budget_flops(b: Fraction, cmax: int) -> Fraction:
    if not 0 <= b <= 1 or cmax <= 0:
        raise ValueError("Budget state and Cmax must be valid")
    return b * cmax


def select(points: list[Point], limit: Fraction, rule: str,
           parameter: float = 0.0, cmax: int | None = None) -> Point | None:
    """Choose only feasible Pareto points; exact ties prefer lower work."""
    if parameter < 0 or not math.isfinite(parameter):
        raise ValueError("Selection parameter must be finite and nonnegative")
    feasible = [p for p in pareto_front(points) if p.flops <= limit]
    if not feasible:
        return None
    if rule == "max_f1":
        return min(feasible, key=lambda p: (-p.macro_f1, p.flops, p.name))
    if rule == "near_best":
        best = max(p.macro_f1 for p in feasible)
        return min((p for p in feasible if p.macro_f1 >= best - parameter), key=order_key)
    if rule == "cost_aware":
        if cmax is None or cmax <= 0:
            raise ValueError("cost_aware needs positive Cmax")
        return min(feasible, key=lambda p: (-(p.macro_f1 - parameter * p.flops / cmax),
                                             p.flops, p.name))
    raise ValueError(f"Unknown rule: {rule}")


def read_rounded_auroc(path: Path = F005_README) -> dict[tuple[float, int], float]:
    """Parse only Table 3b: seed-2026 AUROC, printed to three decimals."""
    readme = path.read_text(encoding="utf-8")
    section = readme.split("**Table 3b. Seed 2026", 1)[1].split("Paired against D026", 1)[0]
    values = {}
    for line in section.splitlines():
        match = re.fullmatch(r"\| (100|75|50|25)% \|[^|]*\| ([0-9. /]+) \|", line.strip())
        if not match:
            continue
        width = int(match.group(1)) / 100
        numbers = [float(x.strip()) for x in match.group(2).split("/")]
        if len(numbers) != len(KS) or any(not 0 <= x <= 1 for x in numbers):
            raise ValueError("Incomplete seed-2026 AUROC row in F005 Table 3b")
        values.update({(width, k): x for k, x in zip(KS, numbers)})
    if set(values) != {(w, k) for w in WIDTHS for k in KS}:
        raise ValueError("Missing seed-2026 AUROC cells in F005 Table 3b")
    return values


def read_selected_cells(path: Path = RUN) -> dict:
    raw = path.read_bytes()
    if path == RUN and hashlib.sha256(raw).hexdigest() != RUN_SHA256:
        raise ValueError("Canonical F005b run.json hash changed")
    run = json.loads(raw)
    if (run.get("status"), run.get("mode"), run.get("seed"), run.get("test_image_inference")) != (
            "completed", "nested", 2026, False):
        raise ValueError("F005b canonical run metadata is not the deployed validation run")
    matches = [entry for entry in run["history"] if entry["epoch"] == run["selected"]["epoch"]]
    if len(matches) != 1:
        raise ValueError("Selected F005b epoch is ambiguous")
    cells = matches[0]["cells"]
    expected = {f"w{w}_k{k}" for w in WIDTHS for k in KS}
    if set(cells) != expected:
        raise ValueError("Selected F005b epoch does not have exactly 24 cells")
    for cell in cells.values():
        if any(not math.isfinite(cell[m]) or not 0 <= cell[m] <= 1 for m in ("macro_f1", "accuracy")):
            raise ValueError("Invalid F005b validation metric")
        if sum(cell[m] for m in ("tn", "fp", "fn", "tp")) != 254:
            raise ValueError("F005b metrics are not for the 254 evaluation units")
    return cells


def recompute_flops() -> dict[tuple[float, int], int]:
    """Use exactly the strict counter and physical-width path used by F005b."""
    sys.path.insert(0, str(ROOT))
    import torch
    from experiments.F003.adaptive import count_flops
    from experiments.F004.family import AdaptiveTinyViT, OPERATING_POINTS
    from experiments.F005.prune import WIDTHS as TRAINED_WIDTHS, physical
    from experiments.F005.run_f005b import _Holder, load_deployable

    if set(TRAINED_WIDTHS) != set(WIDTHS) or tuple(OPERATING_POINTS) != KS:
        raise ValueError("F005b operating space changed")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_deployable(device="cpu")  # validates recorded checkpoint and importance hashes
    counts = {}
    for width in WIDTHS:
        family = AdaptiveTinyViT(_Holder(physical(model.encoder, width), model.head)).eval()
        for k in KS:
            counts[(width, k)] = count_flops(family, k)["flops"]
            if counts[(width, k)] <= 0:
                raise ValueError("Nonpositive strict FLOP count")
    return counts


def reconstruct_points() -> list[Point]:
    cells, auroc, flops = read_selected_cells(), read_rounded_auroc(), recompute_flops()
    return [Point(name=f"w{w}_k{k}", width=w, k=k, flops=flops[(w, k)],
                  macro_f1=cells[f"w{w}_k{k}"]["macro_f1"],
                  accuracy=cells[f"w{w}_k{k}"]["accuracy"], auroc=auroc[(w, k)])
            for w in WIDTHS for k in KS]


def render_csv(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def configuration_csv(points: list[Point]) -> bytes:
    witnesses = dominance_witnesses(points)
    front = pareto_front(points)
    previous = {p.name: front[i - 1] if i else None for i, p in enumerate(front)}
    rows = []
    for p in sorted(points, key=order_key):
        prior = previous.get(p.name)
        rows.append({
            "configuration": p.name, "width": f"{p.width:g}", "k": p.k,
            "flops": p.flops, "gflops": f"{p.flops / 1e9:.9f}",
            "macro_f1": repr(p.macro_f1), "accuracy": repr(p.accuracy),
            "auroc_rounded_3dp": f"{p.auroc:.3f}",
            "pareto_macro_f1": str(witnesses[p.name] is None).lower(),
            "dominance_witness": witnesses[p.name] or "",
            "prior_frontier_point": prior.name if prior else "",
            "additional_gflops": f"{(p.flops - prior.flops) / 1e9:.9f}" if prior else "",
            "additional_macro_f1": repr(p.macro_f1 - prior.macro_f1) if prior else "",
            "relative_cost_increase": repr(p.flops / prior.flops - 1) if prior else "",
            "macro_f1_gain_per_gflop": repr((p.macro_f1 - prior.macro_f1) /
                                              ((p.flops - prior.flops) / 1e9)) if prior else "",
        })
    return render_csv(list(rows[0]), rows)


def budget_rows(points: list[Point]) -> list[dict]:
    cmax = next(p.flops for p in points if (p.width, p.k) == (1.0, 16))
    rules = [("max_f1", 0.0)] + [("near_best", value) for value in EPSILONS] + [
        ("cost_aware", value) for value in LAMBDAS]
    rows = []
    for b in BUDGETS:
        limit = budget_flops(b, cmax)
        for rule, parameter in (("defer", 0.0),) if b == 0 else rules:
            chosen = None if b == 0 else select(points, limit, rule, parameter, cmax)
            if b > 0 and chosen is None:
                raise ValueError(f"No feasible point at b={float(b):.2f}")
            exact_limit = Decimal(limit.numerator) / Decimal(limit.denominator)
            rows.append({
                "budget_state": f"{float(b):.2f}", "rule": rule,
                "parameter": f"{parameter:.3f}" if rule not in ("max_f1", "defer") else "",
                "budget_flops": format(exact_limit, "f"),
                "budget_gflops": f"{exact_limit / Decimal(10**9):.11f}",
                "configuration": chosen.name if chosen else "defer",
                "width": f"{chosen.width:g}" if chosen else "",
                "k": chosen.k if chosen else "",
                "actual_flops": chosen.flops if chosen else "",
                "actual_gflops": f"{chosen.flops / 1e9:.9f}" if chosen else "",
                "macro_f1": repr(chosen.macro_f1) if chosen else "",
                "budget_utilization": repr(float(Fraction(chosen.flops, 1) / limit)) if chosen else "",
            })
    return rows


def budget_csv(points: list[Point]) -> bytes:
    rows = budget_rows(points)
    return render_csv(list(rows[0]), rows)


def save_figure(points: list[Point], directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {0.25: "#CC79A7", 0.5: "#D55E00", 0.75: "#009E73", 1.0: "#0072B2"}
    markers = {0.25: "D", 0.5: "^", 0.75: "s", 1.0: "o"}
    front = pareto_front(points)
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11, "savefig.dpi": 220}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
        for ax, metric, title in zip(axes, ("accuracy", "macro_f1"),
                                     ("Accuracy (F1 front highlighted)", "Macro-F1 (defines front)")):
            for width in WIDTHS:
                group = [p for p in points if p.width == width]
                ax.scatter([p.flops / 1e9 for p in group], [getattr(p, metric) for p in group],
                           c=colours[width], marker=markers[width], s=45, alpha=0.78,
                           label=f"{width:.0%} width")
            if metric == "macro_f1":
                ax.plot([p.flops / 1e9 for p in front], [p.macro_f1 for p in front],
                        color="#222222", linewidth=1.7, linestyle="--", marker="o", markersize=3,
                        label="Macro-F1 Pareto set")
            else:
                ax.scatter([p.flops / 1e9 for p in front], [p.accuracy for p in front],
                           facecolors="none", edgecolors="#222222", linewidths=1.1, s=85,
                           label="Macro-F1 Pareto points")
            ax.set_xlabel("GFLOPs per sequence (strict fvcore)")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.set_xscale("log")
        label_positions = {(0.5, 2): (4, -17, "left"),
                           (0.5, 8): (0, 11, "right"),
                           (0.75, 8): (4, -17, "left"),
                           (1.0, 16): (-4, 9, "right")}
        for p in (p for p in front if (p.width, p.k) in label_positions):
            dx, dy, align = label_positions[(p.width, p.k)]
            axes[1].annotate(f"{p.width:g}×K{p.k}", (p.flops / 1e9, p.macro_f1),
                             textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=align)
        left_handles, left_labels = axes[0].get_legend_handles_labels()
        right_handles, right_labels = axes[1].get_legend_handles_labels()
        handles, labels = left_handles[:4] + right_handles[-1:], left_labels[:4] + right_labels[-1:]
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), fontsize=8)
        fig.suptitle("F005b seed 2026: 24 measured validation configurations", fontsize=12)
        fig.tight_layout(rect=(0, 0.11, 1, 0.96))
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "pareto.png", metadata={"Software": "F006a run_analysis.py"})
        fig.savefig(directory / "pareto.pdf", metadata={"Creator": "F006a run_analysis.py",
                                                      "CreationDate": None, "ModDate": None})
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare regenerated tables with tracked bytes")
    args = parser.parse_args()
    points = reconstruct_points()
    products = {HERE / "configurations.csv": configuration_csv(points),
                HERE / "budget_sensitivity.csv": budget_csv(points)}
    for path, data in products.items():
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Generated table differs: {path}")
        else:
            path.write_bytes(data)
    if not args.check:
        save_figure(points, HERE / "figures")
    print(json.dumps({"source_run_sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
                      "points": len(points), "pareto": len(pareto_front(points)),
                      "tables_checked" if args.check else "tables_written": len(products)}, sort_keys=True))


if __name__ == "__main__":
    main()
