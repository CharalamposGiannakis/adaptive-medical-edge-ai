"""Figures for F005 (T011), made only from saved results in artifacts/F005/.

No model is run and no result is changed. Inputs: evaluation.json and
benchmark/*.json written by run_f005.py, plus F004's flops.json. Output: PDF and
PNG per figure in artifacts/F005/figures/, copied to the tracked
experiments/F005/figures/.

    python experiments/F005/create_figures.py            # needs matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.F004.family import OPERATING_POINTS  # noqa: E402

WIDTHS_STR = ("1.0", "0.75", "0.5", "0.25")

ART = ROOT / "artifacts" / "F005"
F004_ART = ROOT / "artifacts" / "F004"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (successes + z * z / 2) / (n + z * z)
    half = z * math.sqrt(successes * (n - successes) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def create_figures(copy_to: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "lines.linewidth": 1.4, "lines.markersize": 5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
        "grid.linewidth": 0.5, "grid.alpha": 0.4})
    # Okabe–Ito palette; each series also has its own line style and marker.
    S = {"fp32_batched": ("#0072B2", "o-", "FP32"),
         "int8_dynamic_per_frame": ("#D55E00", "s--", "INT8 dynamic, frames encoded one at a time"),
         "int8_dynamic_batched": ("#009E73", "^:", "INT8 dynamic, K frames encoded together")}

    out = ART / "figures"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def panel(axis, letter, title):
        axis.set_title(f"({letter}) {title}", loc="left", fontsize=9, pad=6)

    def figure_title(fig, text):
        fig.suptitle(text, fontsize=10, fontweight="bold", y=1.03)

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
        print("wrote", out / name, flush=True)

    evaluation = read(ART / "evaluation.json")
    results = evaluation["results"]
    flops = read(F004_ART / "flops.json")
    gflops = [flops["points"][str(k)]["flops"] / 1e9 for k in OPERATING_POINTS]

    def metric(result, name):
        da = result["duplicate_aware"]
        if name == "accuracy":
            n = da["tp"] + da["tn"] + da["fp"] + da["fn"]
            return (da["accuracy"], *wilson(da["tp"] + da["tn"], n))
        if name == "macro_f1":
            return (da["macro_f1"], *result["macro_f1_ci95"])
        return (result["auroc"], *result["auroc_ci95"])

    # Figure 1: quality against FLOPs, FP32 and INT8.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for axis, letter, (name, label) in zip(axes, "abc", (("accuracy", "Accuracy at threshold 0.5"),
                                                         ("macro_f1", "Macro-F1 at threshold 0.5"),
                                                         ("auroc", "AUROC"))):
        for variant, (colour, style, legend) in S.items():
            values = [metric(results[variant][str(k)], name) for k in OPERATING_POINTS]
            if variant == "fp32_batched":
                axis.fill_between(gflops, [v[1] for v in values], [v[2] for v in values], color=colour,
                                  alpha=0.12, linewidth=0)
            axis.plot(gflops, [v[0] for v in values], style, color=colour, markersize=4, label=legend)
        for k, x, r in zip(OPERATING_POINTS, gflops, [results["fp32_batched"][str(k)] for k in OPERATING_POINTS]):
            axis.annotate(f"K{k}", (x, metric(r, name)[0]), textcoords="offset points", xytext=(4, -11),
                          fontsize=7, color=S["fp32_batched"][0])
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("GFLOPs per sequence (fvcore multiply-adds, log scale)")
        axis.set_ylabel(label)
        axis.grid(True)
        panel(axis, letter, label)
    figure_title(fig, "Validation quality of the frame-count family in FP32 and with INT8 dynamic quantization")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig1_quality_vs_flops")

    # Figure 2: CPU latency per K and weight-file size.
    benches = [read(p) for p in sorted((ART / "benchmark").glob("*.json"))]
    if benches:
        bench = benches[0]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
        axis = axes[0]
        for variant, (colour, style, legend) in S.items():
            medians = [bench["configs"][f"{variant}_k{k}"]["median_latency_ms"] for k in OPERATING_POINTS]
            reps = [bench["configs"][f"{variant}_k{k}"]["repeat_median_latency_ms"] for k in OPERATING_POINTS]
            axis.errorbar(OPERATING_POINTS, medians, yerr=[[m - min(r) for m, r in zip(medians, reps)],
                                                           [max(r) - m for m, r in zip(medians, reps)]],
                          fmt=style, color=colour, capsize=2, markersize=4, label=legend)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(OPERATING_POINTS, [str(k) for k in OPERATING_POINTS])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Frames per sequence, K")
        axis.set_ylabel("Median model-only latency (ms, log scale)")
        axis.legend(loc="upper left", fontsize=7)
        axis.grid(True)
        panel(axis, "a", f"CPU (Xeon E5-2630 v3, {bench['threads']} threads)")
        axis = axes[1]
        sizes = evaluation["size_bytes"]
        names = [("fp32", "FP32", "#0072B2"), ("int8_dynamic", "INT8\ndynamic", "#D55E00")]
        axis.bar(range(2), [sizes[n] / 1e6 for n, _, _ in names], color=[c for _, _, c in names], width=0.6)
        for i, (n, _, _) in enumerate(names):
            axis.text(i, sizes[n] / 1e6, f"{sizes[n] / 1e6:.1f} MB", ha="center", va="bottom", fontsize=7.5)
        axis.set_xticks(range(2), [label for _, label, _ in names])
        axis.set_ylabel("Weight file (MB)")
        axis.grid(True, axis="y")
        panel(axis, "b", "One file for all six K")
        figure_title(fig, "Cost of the frame-count family in FP32 and INT8 on CPU")
        fig.tight_layout()
        save(fig, "fig2_cpu_latency_and_size")

    # Figure 3: F005b nested widths against the unpruned family, three seeds.
    if (ART / "f005b_evaluation.json").exists():
        f5b = read(ART / "f005b_evaluation.json")
        mobilenet = read(ROOT / "artifacts/F003/mobilenet_k/mobilenet_k.json")["results"]["16"]
        width_style = {"1.0": ("#0072B2", "o-"), "0.75": ("#009E73", "s--"), "0.5": ("#D55E00", "^-."),
                       "0.25": ("#CC79A7", "D:")}
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))
        metric_names = (("accuracy", "Accuracy at threshold 0.5"), ("macro_f1", "Macro-F1 at threshold 0.5"),
                        ("auroc", "AUROC"))
        for axis, letter, (name, label) in zip(axes, "abc", metric_names):
            d026 = [results["fp32_batched"][str(k)] for k in OPERATING_POINTS]
            d026_y = [r["auroc"] if name == "auroc" else r["duplicate_aware"][name] for r in d026]
            axis.plot(gflops, d026_y, color="0.55", linestyle="-", marker="o", markerfacecolor="white",
                      markersize=4, linewidth=1.0, label="TinyViT, unpruned", zorder=2)
            for width, (colour, style) in width_style.items():
                xs = [f5b["flops"][width][str(k)] / 1e9 for k in OPERATING_POINTS]
                per_seed = [[seed["cells"][f"w{width}_k{k}"][name] for k in OPERATING_POINTS]
                            for seed in f5b["results"]["nested"].values()]
                mean = [statistics.fmean(col) for col in zip(*per_seed)]
                lo = [max(0.0, m - min(col)) for m, col in zip(mean, zip(*per_seed))]  # rounding
                hi = [max(0.0, max(col) - m) for m, col in zip(mean, zip(*per_seed))]
                axis.errorbar(xs, mean, yerr=[lo, hi], fmt=style, color=colour, capsize=2, markersize=4,
                              linewidth=1.2, label=f"nested width {float(width):.0%}", zorder=3)
            static = mobilenet["auroc"] if name == "auroc" else mobilenet["duplicate_aware"][name]
            axis.scatter([mobilenet["flops"]["flops"] / 1e9], [static], marker="*", s=130, facecolors="none",
                         edgecolors="black", linewidths=1.1, zorder=6, label="MobileNetV2 static baseline (16 frames)")
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel("GFLOPs per sequence (fvcore multiply-adds, log scale)")
            axis.set_ylabel(label)
            axis.grid(True)
            panel(axis, letter, label)
        figure_title(fig, "Nested pruned widths × frame count against the unpruned network")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        save(fig, "fig3_pruned_pareto")

    # Figure 4: latency of the pruned widths (physical copies and the sliced nested file).
    timing = {p.stem: read(p) for p in sorted((ART / "benchmark").glob("f005b_*.json"), reverse=True)}  # GPU first
    if timing:
        fig, axes = plt.subplots(1, len(timing), figsize=(4.4 * len(timing), 3.3), squeeze=False)
        for axis, letter, (stem, bench) in zip(axes[0], "abcd", timing.items()):
            ks = sorted({int(n.rsplit("_k", 1)[1]) for n in bench["configs"]})
            for width, (colour, style) in (("1.0", ("#0072B2", "o-")), ("0.75", ("#009E73", "s--")),
                                           ("0.5", ("#D55E00", "^-.")), ("0.25", ("#CC79A7", "D:"))):
                for mode, face, alpha in (("physical", None, 1.0), ("sliced", "white", 0.7)):
                    ys = [bench["configs"][f"{mode}_w{width}_k{k}"]["median_latency_ms"] for k in ks]
                    axis.plot(ks, ys, style, color=colour, markerfacecolor=face, alpha=alpha, markersize=4,
                              label=f"{float(width):.0%}, {mode}")
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_xticks(ks, [str(k) for k in ks])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel("Frames per sequence, K")
            axis.set_ylabel("Median model-only latency (ms, log scale)")
            axis.grid(True)
            panel(axis, letter, "GPU (GTX 1080 Ti)" if bench["device"] == "cuda"
                  else f"CPU (Xeon E5-2630 v3, {bench['threads']} threads)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize=7)
        figure_title(fig, "Latency of the nested pruned widths")
        fig.tight_layout()
        save(fig, "fig4_pruned_latency")

    # Figure 5: the deployable file's accuracy-FLOPs frontier against the baselines.
    if (ART / "f005b_evaluation.json").exists():
        f5b = read(ART / "f005b_evaluation.json")
        cells = f5b["results"]["nested"]["2026"]["cells"]
        single = f5b["results"]["single"]["2026"]["cells"]
        mob = read(ROOT / "artifacts/F003/mobilenet_k/mobilenet_k.json")["results"]
        mob_ks = sorted(int(k) for k in mob)
        markers = {"1.0": ("#0072B2", "o"), "0.75": ("#009E73", "s"), "0.5": ("#D55E00", "^"), "0.25": ("#CC79A7", "D")}
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
        for axis, letter, (name, label) in zip(axes, "ab", (("macro_f1", "Macro-F1 at threshold 0.5"),
                                                           ("auroc", "AUROC"))):
            points = []
            for width, (colour, marker) in markers.items():
                for k in OPERATING_POINTS:
                    cell = cells[f"w{width}_k{k}"]
                    points.append((f5b["flops"][width][str(k)] / 1e9, cell[name], cell[f"{name}_ci95"], width, k))
                xs = [p[0] for p in points if p[3] == width]
                ys = [p[1] for p in points if p[3] == width]
                axis.scatter(xs, ys, marker=marker, s=22, color=colour, alpha=0.45, linewidths=0,
                             label=f"pruned width {float(width):.0%}, K = 1–16")
            front = sorted((p for p in points if not any(q[1] >= p[1] and q[0] <= p[0] and (q[1] > p[1] or q[0] < p[0])
                                                         for q in points)), key=lambda p: p[0])
            axis.errorbar([p[0] for p in front], [p[1] for p in front],
                          yerr=[[p[1] - p[2][0] for p in front], [p[2][1] - p[1] for p in front]],
                          fmt="-", color="black", ecolor="0.35", elinewidth=0.8, capsize=2, linewidth=1.6,
                          drawstyle="steps-post", label="frontier of the one pruned file (95% CI); colour = width",
                          zorder=5)
            for j, p in enumerate(front):
                colour, marker = markers[p[3]]
                axis.scatter([p[0]], [p[1]], marker=marker, s=40, color=colour, edgecolors="black",
                             linewidths=0.6, zorder=7)
                if name == "macro_f1":  # K labels on the primary-metric panel only
                    axis.annotate(f"K{p[4]}", (p[0], p[1]), textcoords="offset points",
                                  xytext=(-4, 7 if j % 2 == 0 else -13), fontsize=6.5, ha="right")
            unpruned = [results["fp32_batched"][str(k)] for k in OPERATING_POINTS]
            axis.plot(gflops, [u["auroc"] if name == "auroc" else u["duplicate_aware"][name] for u in unpruned],
                      color="0.55", linestyle="--", marker="o", markerfacecolor="white", markersize=4, linewidth=1.0,
                      label="TinyViT, unpruned, K = 1–16")
            axis.plot([f5b["flops"]["0.5"][str(k)] / 1e9 for k in OPERATING_POINTS],  # same FLOPs as nested 50%
                      [single[f"w0.5_k{k}"][name] for k in OPERATING_POINTS], color="#E69F00", linestyle=":",
                      marker="v", markersize=4, linewidth=1.0, label="statically pruned 50% model, K = 1–16")
            axis.plot([mob[str(k)]["flops"]["flops"] / 1e9 for k in mob_ks],
                      [mob[str(k)]["auroc"] if name == "auroc" else mob[str(k)]["duplicate_aware"][name] for k in mob_ks],
                      color="black", linestyle=(0, (1, 1.5)), marker="x", markersize=4, linewidth=0.9,
                      label="MobileNetV2 baseline fed fewer frames")
            static = mob["16"]
            axis.scatter([static["flops"]["flops"] / 1e9], [static["auroc"] if name == "auroc" else static["duplicate_aware"][name]],
                         marker="*", s=150, facecolors="none", edgecolors="black", linewidths=1.1, zorder=6,
                         label="MobileNetV2 static baseline (16 frames)")
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel("GFLOPs per sequence (fvcore multiply-adds, log scale)")
            axis.set_ylabel(label)
            axis.grid(True)
            panel(axis, letter, label)
        axes[0].set_ylim(0.69, 0.93)
        axes[1].set_ylim(0.76, 0.99)
        figure_title(fig, "Accuracy–compute frontier of the single pruned weight file and the baselines")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), fontsize=7.5)
        fig.tight_layout()
        save(fig, "fig5_frontier")

    # Figure 6: slide version: accuracy (brief) and macro-F1 (primary) against compute.
    if (ART / "f005b_evaluation.json").exists():
        f5b = read(ART / "f005b_evaluation.json")
        cells = f5b["results"]["nested"]["2026"]["cells"]
        single = f5b["results"]["single"]["2026"]["cells"]
        mob = read(ROOT / "artifacts/F003/mobilenet_k/mobilenet_k.json")["results"]
        with plt.rc_context({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10,
                             "xtick.labelsize": 11, "ytick.labelsize": 11}):
            fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), sharey=False)
            for axis, letter, (name, label) in zip(axes, "ab", (("accuracy", "Validation accuracy"),
                                                               ("macro_f1", "Validation macro-F1"))):
                def interval(cell):
                    if name == "macro_f1":
                        return cell["macro_f1_ci95"]
                    n = 254
                    return wilson(round(cell["accuracy"] * n), n)
                points = [(f5b["flops"][w][str(k)] / 1e9, cells[f"w{w}_k{k}"][name], interval(cells[f"w{w}_k{k}"]))
                          for w in WIDTHS_STR for k in OPERATING_POINTS]
                front = sorted((p for p in points if not any(q[1] >= p[1] and q[0] <= p[0] and (q[1] > p[1] or q[0] < p[0])
                                                             for q in points)), key=lambda p: p[0])
                axis.fill_between([p[0] for p in front], [p[2][0] for p in front], [p[2][1] for p in front],
                                  step="post", color="black", alpha=0.08, linewidth=0)
                axis.plot([p[0] for p in front], [p[1] for p in front], "-", drawstyle="steps-post", color="black",
                          linewidth=2.2, label="adaptive TinyViT, one pruned file (frontier, 95% CI)")
                axis.plot(gflops, [results["fp32_batched"][str(k)]["duplicate_aware"][name] for k in OPERATING_POINTS],
                          "o--", color="#0072B2", markerfacecolor="white", linewidth=1.6,
                          label="TinyViT, unpruned, K = 1–16 frames")
                axis.plot([f5b["flops"]["0.5"][str(k)] / 1e9 for k in OPERATING_POINTS],
                          [single[f"w0.5_k{k}"][name] for k in OPERATING_POINTS], "v:", color="#E69F00",
                          linewidth=1.6, label="statically pruned TinyViT (50% width), K = 1–16")
                static = mob["16"]
                axis.scatter([static["flops"]["flops"] / 1e9], [static["duplicate_aware"][name]], marker="*", s=260,
                             facecolors="none", edgecolors="black", linewidths=1.4, zorder=6,
                             label="MobileNetV2 static baseline")
                axis.set_xscale("log")
                axis.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.xaxis.set_minor_formatter(NullFormatter())
                axis.set_xlabel("GFLOPs per lung-zone clip (multiply-adds, log scale)")
                axis.set_ylabel(label)
                axis.grid(True)
                axis.set_title(f"({letter}) {label}", loc="left")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13), frameon=False)
            fig.suptitle("Accuracy against compute: one adaptive file and the static baselines",
                         fontweight="bold", y=1.02)
            fig.tight_layout()
            save(fig, "fig6_frontier_slide")

    if copy_to:
        target = ROOT / copy_to
        if target.exists():
            for old in list(target.glob("*.png")) + list(target.glob("*.pdf")):
                old.unlink()
        target.mkdir(parents=True, exist_ok=True)
        for figure in sorted(list(out.glob("*.png")) + list(out.glob("*.pdf"))):
            shutil.copy2(figure, target / figure.name)
        print("copied figures to", target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copy-to", default="experiments/F005/figures",
                        help="tracked folder for the figures; empty string to skip")
    create_figures(parser.parse_args().copy_to or None)


if __name__ == "__main__":
    main()
