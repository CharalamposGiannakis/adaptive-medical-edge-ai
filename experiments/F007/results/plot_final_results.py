"""Render the two F007 presentation figures from the copied official report.

No predictions, metrics, intervals, or operating points are recomputed here.
Run from the repository root with the existing bde-project Python environment:

    C:\\venvs\\bde-project\\Scripts\\python.exe -m experiments.F007.results.plot_final_results
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPORT = HERE / "final_test_report.json"
REPORT_SHA256 = "1e5c3846bd01b03f9c407702f5a3a116b6622f1a32e363fac10f1079364824a5"
STATES = ("0", "0.25", "0.50", "0.75", "1.00")
SYSTEMS = ("A", "B", "C", "D", "E", "F")
LABELS = {
    "A": "A · MobileNetV2 K16 (external static)",
    "B": "B · original TinyViT K16 (architecture baseline)",
    "C": "C · nested TinyViT w0.5 K8",
    "D": "D · minimum quality cost",
    "E": "E · validation macro-F1 policy",
    "F": "F · recall-priority policy",
}
COLORS = {"A": "#747b83", "B": "#193a63", "C": "#3193a1",
          "D": "#7651a8", "E": "#c89320", "F": "#bd4b35"}


def load_report() -> dict:
    if hashlib.sha256(REPORT.read_bytes()).hexdigest() != REPORT_SHA256:
        raise ValueError("F007 published report differs from the official artifact")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (report.get("partition") != "test" or report.get("diagnostic_only") is not False
            or len(report.get("cells", {})) != 38
            or len(report.get("contrasts", {})) != 24):
        raise ValueError("F007 official report structure is incomplete")
    return report


def _save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=220, facecolor="white")
    svg = HERE / f"{stem}.svg"
    fig.savefig(svg, metadata={"Date": None}, facecolor="white")
    # Matplotlib indents multiline SVG paths with trailing spaces. Strip those
    # layout-only spaces so the generated vector remains clean in Git.
    svg.write_bytes(("\n".join(line.rstrip() for line in
                               svg.read_text(encoding="utf-8").splitlines()) + "\n")
                    .encode("utf-8"))
    plt.close(fig)


def quality_vs_flops(report: dict) -> None:
    """Use reported factorial macro-F1 and full independent-request FLOPs."""
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.20)
    fig.patch.set_facecolor("white")
    for system in SYSTEMS:
        seen: set[tuple[float, float]] = set()
        points = []
        for state in STATES[1:]:
            point = report["cells"][f"factorial_b{state}:{system}"]["point"]
            if point["macro_f1"] is None:
                continue
            xy = (point["mean_full_flops_per_requested_row"] / 1e9,
                  point["macro_f1"])
            if xy not in seen:
                points.append((state, *xy))
                seen.add(xy)
        if not points:
            continue
        marker = {"A": "D", "B": "*", "C": "s", "D": "o", "E": "^", "F": "o"}[system]
        ax.plot([p[1] for p in points], [p[2] for p in points],
                color=COLORS[system], lw=2.3 if system == "F" else 1.5,
                linestyle="-" if system in ("D", "E", "F") else "None",
                marker=marker, markersize=12 if system == "B" else 7,
                label=LABELS[system], zorder=5 if system == "B" else 3)
        if system == "F":
            for state, x, y in points:
                ax.annotate(f"b={state}", (x, y), xytext=(5, -15),
                            textcoords="offset points", fontsize=8,
                            color=COLORS[system])
    shared = report["cells"]["factorial_b0.25:F"]["point"]
    ax.annotate("D=E=F", (shared["mean_full_flops_per_requested_row"] / 1e9,
                            shared["macro_f1"]), xytext=(-7, 12),
                textcoords="offset points", fontsize=8, color="#3f4852")
    ax.set(xlim=(0, 22), ylim=(0.835, 0.892),
           xlabel="Full independent-request GFLOPs per assigned alias row",
           ylabel="Completed-unit macro-F1",
           title="F007 held-out factorial evaluation: quality and model work")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True, fontsize=8.2)
    fig.text(0.10, 0.035,
             "Reported values only; each plotted served factorial cell has 265/265 unit coverage. "
             "At b=0 every system defers (no predictive metric).",
             fontsize=8, color="#3f4852")
    _save(fig, "heldout_quality_vs_flops")


def heterogeneous_coverage(report: dict) -> None:
    """Use the report's realized per-state completed unit counts."""
    counts = report["heterogeneous_state_counts"]
    state_colors = {"0": "#d9dde2", "0.25": "#d7a443", "0.50": "#64a9ad",
                    "0.75": "#627cb3", "1.00": "#374978"}
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.30)
    fig.patch.set_facecolor("white")
    for index, system in enumerate(SYSTEMS):
        requested = sum(counts[system][state]["requested_units"] for state in STATES)
        if requested != 265:
            raise ValueError("Heterogeneous schedule does not cover 265 units")
        bottom = 0.0
        for state in STATES:
            cell = counts[system][state]
            completed = cell["completed_units"]
            height = 100 * completed / requested
            if completed:
                ax.bar(index, height, bottom=bottom, color=state_colors[state],
                       width=0.65, edgecolor="white", linewidth=0.6,
                       label=f"b={state}" if index == 0 else None)
                ax.text(index, bottom + height / 2, str(completed),
                        ha="center", va="center", fontsize=9,
                        color="white" if state in ("0.75", "1.00") else "#24303b")
            bottom += height
        total = report["cells"][f"heterogeneous:{system}"]["point"]
        if (total["completed_units"] != sum(counts[system][s]["completed_units"]
                                             for s in STATES)
                or abs(total["coverage"] * 100 - bottom) > 1e-10):
            raise ValueError("Heterogeneous coverage differs from official cell")
        ax.text(index, bottom + 2, f'{total["completed_units"]}/265',
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=COLORS[system])
    ax.set(ylim=(0, 95), yticks=(0, 20, 40, 60, 80),
           ylabel="Completed evaluation units / all assigned units (%)",
           title="F007 held-out heterogeneous schedule: coverage by resource state")
    ax.set_xticks(range(len(SYSTEMS)),
                  ["A\nMobileNetV2\nexternal", "B\noriginal TinyViT\nmain baseline",
                   "C\nnested static", "D\nmin-cost policy",
                   "E\nmacro-F1 policy", "F\nrecall policy"])
    ax.get_xticklabels()[1].set_fontweight("bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(title="Served assigned state", ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=8.5)
    fig.text(0.11, 0.025,
             "Each state contains 53 of 265 frozen units. All systems defer at b=0; "
             "unserved states contribute no predictive metric.",
             fontsize=8, color="#3f4852")
    _save(fig, "heterogeneous_coverage")


def main() -> None:
    plt.rcParams["svg.hashsalt"] = "F007-official-heldout-results"
    report = load_report()
    quality_vs_flops(report)
    heterogeneous_coverage(report)


if __name__ == "__main__":
    main()
