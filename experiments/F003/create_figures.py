"""Figures for F003 (T009), made only from saved results in artifacts/F003/.

No model is run and no result is changed. Inputs: summary.json,
analysis_extra.json, analysis_robustness.json and benchmark/*.json, written by
run_f003.py. Output: PDF and PNG per figure in artifacts/F003/figures/, copied
to the tracked experiments/F003/figures/ for the README.

    python experiments/F003/create_figures.py            # needs matplotlib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.F003.adaptive import K_VALUES, TEMPORAL_K  # noqa: E402

ART = ROOT / "artifacts" / "F003"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def create_figures(copy_to: str | None) -> None:
    """Five publication-style figures from saved results; no model is run.

    Titles are neutral panel labels; descriptions and interpretation belong in
    the captions and text. Each figure is written as PDF (vector) and PNG.
    """
    import math as _math
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "lines.linewidth": 1.4, "lines.markersize": 5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
        "grid.linewidth": 0.5, "grid.alpha": 0.4})
    # Okabe–Ito colour-blind-safe palette.
    C = {"k": "#0072B2", "stage2": "#E69F00", "stage2_mid": "#009E73", "stage1": "#D55E00",
         "res": "#7F7F7F", "dense": "#CC79A7", "window": "#56B4E9", "ref": "#000000"}

    summary = read(ART / "summary.json")
    extra = read(ART / "analysis_extra.json")
    robust = read(ART / "analysis_robustness.json")
    rows = {r["config"]: r for r in summary["configs"]}
    auc, auc_ci = robust["auroc"], robust["auroc_ci95"]
    mobilenet_k = read(ART / "mobilenet_k" / "mobilenet_k.json")["results"]
    ref = summary["reference"]
    mobilenet = {"flops": ref["f002b_recorded"]["flops"] / 1e9, "f1": ref["f002b_recorded"]["macro_f1"],
                 "f1_ci": robust["f002b_macro_f1_ci95"], "auc": auc["f002b_mobilenet_v2_k16"],
                 "auc_ci": auc_ci["f002b_mobilenet_v2_k16"]}
    out = ART / "figures"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def log_x(axis, ticks):
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())

    def panel(axis, letter, title):
        """Panel label and a descriptive (not interpretive) panel title."""
        axis.set_title(f"({letter}) {title}", loc="left", fontsize=9, pad=6)

    def figure_title(fig, text):
        fig.suptitle(text, fontsize=10, fontweight="bold", y=1.02)

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
        print("wrote", out / name, flush=True)

    reference_marker = dict(marker="*", s=130, facecolors="none", edgecolors=C["ref"],
                            linewidths=1.1, zorder=6)
    k_names = [f"final_k{k}_r224" for k in K_VALUES]

    def k_label(axis, xs, ys, offset=(-3, 6)):
        for k, x, y in zip(K_VALUES, xs, ys):
            if k == 12:  # unlabelled: it is the point between K8 and K16
                continue
            axis.annotate(f"K{k}", (x, y), textcoords="offset points", xytext=offset, fontsize=7,
                          color=C["k"], ha="center")

    # Figure 1: predictive quality against arithmetic cost.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for axis, metric in zip(axes, ("f1", "auc")):
        value = (lambda n: rows[n]["macro_f1"]) if metric == "f1" else (lambda n: auc[n])
        interval = (lambda n: rows[n]["macro_f1_ci95"]) if metric == "f1" else (lambda n: auc_ci[n])
        xs = [rows[n]["flops"] / 1e9 for n in k_names]
        ys = [value(n) for n in k_names]
        axis.fill_between(xs, [interval(n)[0] for n in k_names], [interval(n)[1] for n in k_names],
                          color=C["k"], alpha=0.12, linewidth=0)
        axis.plot(xs, ys, "o-", color=C["k"], zorder=4)
        k_label(axis, xs, ys)
        for exit_name, marker in (("stage2", "s"), ("stage2_mid", "^"), ("stage1", "v")):
            names = [f"{exit_name}_k{k}_r224" for k in K_VALUES]
            axis.plot([rows[n]["flops"] / 1e9 for n in names], [value(n) for n in names],
                      marker=marker, linestyle=":", color=C[exit_name], markersize=4)
        for resolution, marker in ((192, ">"), (160, "<")):
            names = [f"final_k{k}_r{resolution}" for k in K_VALUES]
            axis.plot([rows[n]["flops"] / 1e9 for n in names], [value(n) for n in names],
                      marker=marker, linestyle="none", color=C["res"], markersize=4,
                      alpha=0.9 if resolution == 192 else 0.6)
        temporal_x = [rows[f"final_k{k}_r224"]["flops"] / 1e9 for k in TEMPORAL_K]
        dense_y = [extra["temporal"][f"dense_k{k}"]["duplicate_aware"]["macro_f1"] if metric == "f1"
                   else auc[f"temporal_dense_k{k}"] for k in TEMPORAL_K]
        axis.plot(temporal_x, dense_y, "D--", color=C["dense"], markersize=4)
        mk = [mobilenet_k[str(k)] for k in K_VALUES]
        axis.plot([m["flops"]["flops"] / 1e9 for m in mk],
                  [m["duplicate_aware"]["macro_f1"] if metric == "f1" else m["auroc"] for m in mk],
                  "o--", color=C["ref"], markerfacecolor="white", markersize=4, linewidth=1.0)
        axis.scatter([mobilenet["flops"]], [mobilenet["f1"] if metric == "f1" else mobilenet["auc"]],
                     **reference_marker)
        log_x(axis, [0.5, 1, 2, 5, 10, 20])
        axis.set_xlabel("Cost (GFLOPs per sequence, log scale)")
        axis.set_ylabel("Macro-F1 (threshold 0.5)" if metric == "f1" else "AUROC")
        axis.grid(True)
    panel(axes[0], "a", "Macro-F1")
    panel(axes[1], "b", "AUROC")
    figure_title(fig, "Predictive quality versus computational cost")
    handles = [Line2D([], [], color=C["k"], marker="o", label="TinyViT, K spread frames (224 px)"),
               Line2D([], [], color=C["stage2"], marker="s", linestyle=":", label="exit after stage 2"),
               Line2D([], [], color=C["stage2_mid"], marker="^", linestyle=":", label="exit within stage 2"),
               Line2D([], [], color=C["stage1"], marker="v", linestyle=":", label="exit after stage 1"),
               Line2D([], [], color=C["res"], marker=">", linestyle="none", label="192 px input"),
               Line2D([], [], color=C["res"], marker="<", linestyle="none", alpha=0.6, label="160 px input"),
               Line2D([], [], color=C["dense"], marker="D", linestyle="--", label="K consecutive frames"),
               Line2D([], [], color=C["ref"], marker="o", markerfacecolor="white", linestyle="--",
                      label="MobileNetV2 (F002b), K spread frames"),
               Line2D([], [], color=C["ref"], marker="*", markerfacecolor="none", linestyle="none",
                      markersize=9, label="MobileNetV2 static reference (F002b, K16)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    save(fig, "fig1_quality_vs_cost")

    # Figure 2: runtime on the GTX 1080 Ti.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    k_rows = [rows[n] for n in k_names]
    axes[0].plot([r["pipeline_seq_s"] for r in k_rows], [r["macro_f1"] for r in k_rows], "o-",
                 color=C["k"])
    k_label(axes[0], [r["pipeline_seq_s"] for r in k_rows], [r["macro_f1"] for r in k_rows], offset=(0, 6))
    for name, marker, color in (("stage2_k16_r224", "s", C["stage2"]), ("stage2_mid_k16_r224", "^", C["stage2_mid"]),
                                ("stage1_k16_r224", "v", C["stage1"]), ("final_k16_r192", ">", C["res"]),
                                ("final_k16_r160", "<", C["res"])):
        axes[0].scatter([rows[name]["pipeline_seq_s"]], [rows[name]["macro_f1"]], marker=marker,
                        color=color, s=22, zorder=5)
    axes[0].scatter([ref["f002b_same_job"]["pipeline_seq_s"]], [mobilenet["f1"]], **reference_marker)
    log_x(axes[0], [8, 10, 15, 20, 30, 45, 60])
    axes[0].set_xlabel(r"End-to-end throughput (sequences s$^{-1}$, log scale)")
    axes[0].set_ylabel("Macro-F1 (threshold 0.5)")
    axes[0].grid(True)
    bench = {n: read(ART / "benchmark" / f"{n}.json") for n in k_names}
    reference_bench = read(ART / "benchmark" / "f002b_mobilenet_v2_k16_r224.json")
    labels = [f"K{k}" for k in K_VALUES] + ["MobileNetV2\n(F002b, K16)"]
    model_paths = [bench[n]["paths"]["model_only_unfused"] for n in k_names] + \
        [reference_bench["paths"]["model_only"]]
    pipe_paths = [bench[n]["paths"][bench[n]["pipeline_path"]] for n in k_names] + \
        [reference_bench["paths"]["pipeline"]]
    pos = list(range(len(K_VALUES))) + [len(K_VALUES) + 0.6]
    for offset, paths, color, label in ((-0.2, model_paths, C["k"], "model only (input on GPU)"),
                                        (0.2, pipe_paths, C["stage2"], "end to end (JPEG decode + model)")):
        medians = [p["median_latency_ms"] for p in paths]
        upper = [p["p95_latency_ms"] - p["median_latency_ms"] for p in paths]
        axes[1].bar([x + offset for x in pos], medians, width=0.4, color=color, label=label,
                    yerr=[[0] * len(paths), upper], capsize=2, error_kw={"linewidth": 0.8})
    axes[1].set_xticks(pos, labels)
    axes[1].get_xticklabels()[-1].set_fontsize(7)
    axes[1].set_ylabel("Latency per sequence (ms)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, axis="y")
    panel(axes[0], "a", "Macro-F1 versus end-to-end throughput")
    panel(axes[1], "b", "Latency per sequence")
    figure_title(fig, "Runtime on a GTX 1080 Ti")
    handles = [Line2D([], [], color=C["k"], marker="o", label="TinyViT, K spread frames"),
               Line2D([], [], color=C["stage2"], marker="s", linestyle="none", label="exit after stage 2, K16"),
               Line2D([], [], color=C["stage2_mid"], marker="^", linestyle="none", label="exit within stage 2, K16"),
               Line2D([], [], color=C["stage1"], marker="v", linestyle="none", label="exit after stage 1, K16"),
               Line2D([], [], color=C["res"], marker=">", linestyle="none", label="192 px, K16"),
               Line2D([], [], color=C["res"], marker="<", linestyle="none", label="160 px, K16"),
               Line2D([], [], color=C["ref"], marker="*", markerfacecolor="none", linestyle="none",
                      markersize=9, label="MobileNetV2 (F002b)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig2_runtime")

    # Figure 3: which frames are used, at fixed K.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    categories = ["consecutive", "first 25%", "first 50%", "first 75%", "whole clip"]
    k_colors = {4: C["stage1"], 8: C["stage2"], 16: C["k"]}
    for axis, metric in zip(axes, ("f1", "auc")):
        for offset, k in zip((-0.15, 0.0, 0.15), TEMPORAL_K):
            names = [f"dense_k{k}"] + [f"window{w}_k{k}" for w in (25, 50, 75, 100)]
            if metric == "f1":
                values = [extra["temporal"][n]["duplicate_aware"]["macro_f1"] for n in names]
                intervals = [extra["temporal"][n]["macro_f1_ci95"] for n in names]
            else:
                values = [auc[f"temporal_{n}"] for n in names]
                intervals = [auc_ci[f"temporal_{n}"] for n in names]
            xs = [i + offset for i in range(len(names))]
            axis.errorbar(xs, values, yerr=[[v - lo for v, (lo, _) in zip(values, intervals)],
                                            [hi - v for v, (_, hi) in zip(values, intervals)]],
                          fmt="o", color=k_colors[k], capsize=2, markersize=4, linewidth=0.9,
                          label=f"K = {k}")
        axis.axvline(0.5, color="0.6", linewidth=0.7, linestyle="--")
        axis.set_xticks(range(len(categories)), categories, rotation=20)
        axis.set_ylabel("Macro-F1 (threshold 0.5)" if metric == "f1" else "AUROC")
        axis.set_xlabel("Frame selection")
        axis.grid(True, axis="y")
    axes[0].legend(loc="lower right")
    panel(axes[0], "a", "Macro-F1")
    panel(axes[1], "b", "AUROC")
    figure_title(fig, "Frame selection at fixed frame count K")
    fig.tight_layout()
    save(fig, "fig3_frame_selection")

    # Figure 4: escalation from a cheap K to K16.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    style = {"confidence": dict(color="#D55E00", line="--", dot="o", main="X"),
             "disagreement": dict(color="#009E73", line=":", dot="^", main="P")}
    for axis, cheap_k, letter in zip(axes, (2, 4), ("a", "b")):
        static = [rows[f"final_k{k}_r224"] for k in (cheap_k, 8, 16)]
        axis.plot([r["flops"] / 1e9 for r in static], [r["macro_f1"] for r in static], "o-",
                  color=C["k"], zorder=3)
        for r in static:
            axis.annotate(f"K{r['k']}", (r["flops"] / 1e9, r["macro_f1"]), textcoords="offset points",
                          xytext=(4, -11), fontsize=7, color=C["k"])
        for signal, st in style.items():
            sweep = extra["cascade_curves_by_escalation_rate"][f"final_k{cheap_k}_to_k16_{signal}"]
            axis.plot([c["expected_flops"] / 1e9 for c in sweep], [c["macro_f1"] for c in sweep],
                      linestyle=st["line"], color=st["color"], linewidth=1.1)
            reruns = [r for name, runs in robust["cascade_cv_sensitivity"].items()
                      if name.startswith(f"k{cheap_k}_{signal}") for r in runs]
            axis.scatter([r["expected_flops"] / 1e9 for r in reruns], [r["macro_f1"] for r in reruns],
                         s=16, marker=st["dot"], facecolors="none", edgecolors=st["color"],
                         linewidths=0.8, alpha=0.9)
            primary = extra["cascade_cross_validated"][f"final_k{cheap_k}_to_k16_{signal}"]
            axis.scatter([primary["expected_flops"] / 1e9], [primary["held_out"]["macro_f1"]],
                         marker=st["main"], s=70, color=st["color"], edgecolors="black",
                         linewidths=0.6, zorder=6)
        axis.scatter([mobilenet["flops"]], [mobilenet["f1"]], **reference_marker)
        log_x(axis, [3, 5, 10, 20] if cheap_k == 2 else [5, 7, 10, 15, 20])
        axis.set_xlabel("Expected cost (GFLOPs per sequence, log scale)")
        axis.grid(True)
        panel(axis, letter, f"Escalation from K{cheap_k} to K16")
    axes[0].set_ylabel("Macro-F1 (threshold 0.5)")
    figure_title(fig, "Confidence-based and disagreement-based escalation to K16")
    handles = [Line2D([], [], color=C["k"], marker="o", label="static K, no escalation"),
               Line2D([], [], color=C["ref"], marker="*", markerfacecolor="none", linestyle="none",
                      markersize=9, label="MobileNetV2 (F002b)"),
               Line2D([], [], linestyle="none", label=" ")]  # keeps one signal per legend column
    for signal, st in style.items():
        name = "confidence" if signal == "confidence" else "frame disagreement"
        handles += [Line2D([], [], color=st["color"], linestyle=st["line"],
                           label=f"{name}: in-sample threshold sweep"),
                    Line2D([], [], color=st["color"], marker=st["main"], markeredgecolor="black",
                           linestyle="none", markersize=8,
                           label=f"{name}: cross-validated, pre-specified rule"),
                    Line2D([], [], color=st["color"], marker=st["dot"], markerfacecolor="none",
                           linestyle="none", markersize=5,
                           label=f"{name}: cross-validated reruns")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig4_escalation")

    # Figure 5: recall per acquisition device, with Wilson 95% intervals.
    def wilson(successes, n, z=1.96):
        if n == 0:
            return (float("nan"),) * 3
        p = successes / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return p, centre - half, centre + half

    devices_table = extra["per_device"]
    models = [("final_k1_r224", "TinyViT K1", "#56B4E9"), ("final_k4_r224", "TinyViT K4", "#0072B2"),
              ("final_k16_r224", "TinyViT K16", "#003F63"),
              ("f002b_mobilenet_v2_k16", "MobileNetV2 (F002b)", "#E69F00")]
    shown = ["0", "1", "4"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axis, (correct, wrong), ylabel, letter in zip(
            axes, (("tn", "fp"), ("tp", "fn")), ("Normal recall (specificity)",
                                                 "Abnormal recall (sensitivity)"), ("a", "b")):
        for i, (name, label, color) in enumerate(models):
            stats = [wilson(devices_table[name][d][correct],
                            devices_table[name][d][correct] + devices_table[name][d][wrong]) for d in shown]
            xs = [j + (i - 1.5) * 0.2 for j in range(len(shown))]
            values = [v for v, _, _ in stats]
            axis.bar(xs, values, width=0.2, color=color, label=label)
            axis.errorbar(xs, values, yerr=[[v - lo for v, lo, _ in stats], [hi - v for v, _, hi in stats]],
                          fmt="none", ecolor="0.25", capsize=1.5, linewidth=0.7)
        axis.set_xticks(range(len(shown)), [
            f"device {d}\nn = {devices_table['final_k4_r224'][d]['units'] - devices_table['final_k4_r224'][d]['abnormal_units'] if correct == 'tn' else devices_table['final_k4_r224'][d]['abnormal_units']}"
            for d in shown])
        if correct == "tp":
            axis.text(1, 0.05, "no abnormal\nunits", ha="center", fontsize=7, color="0.4")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1.05)
        axis.grid(True, axis="y")
        panel(axis, letter, ylabel)
    figure_title(fig, "Recall per acquisition device")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig5_per_device")

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
    parser.add_argument("--copy-to", default="experiments/F003/figures",
                        help="tracked folder for the figures; empty string to skip")
    create_figures(parser.parse_args().copy_to or None)


if __name__ == "__main__":
    main()
