"""Figures for F004 (T010), made only from saved results in artifacts/F004/.

No model is run and no result is changed. Inputs: evaluation.json, flops.json,
benchmark/*.json and comparison.json (with train/*/run.json), written by
run_f004.py, plus F003's saved MobileNetV2 F002b results at each K. Output: PDF
and PNG per figure in artifacts/F004/figures/, copied to the tracked
experiments/F004/figures/.

    python experiments/F004/create_figures.py            # needs matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.F004.family import OPERATING_POINTS  # noqa: E402
from experiments.F004.run_f004 import TRAIN_KS, UPGRADES  # noqa: E402

ART = ROOT / "artifacts" / "F004"
F003_ART = ROOT / "artifacts" / "F003"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (successes + z * z / 2) / (n + z * z)
    half = z * math.sqrt(successes * (n - successes) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def create_figures(copy_to: str | None) -> None:
    """Publication-style figures from saved results; descriptive titles only."""
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
    # Okabe–Ito colour-blind-safe palette. Every compared series also differs in
    # line style and marker; no colour means two different things in one figure.
    C = {"tinyvit": "#0072B2", "mobilenet": "#000000", "multik": "#D55E00", "k16": "#009E73",
         "gpu": "#0072B2", "cpu": "#E69F00"}

    out = ART / "figures"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def panel(axis, letter, title):
        axis.set_title(f"({letter}) {title}", loc="left", fontsize=9, pad=6)

    def figure_title(fig, text):
        fig.suptitle(text, fontsize=10, fontweight="bold", y=1.03)

    def log_x(axis, ticks):
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())

    def k_axis(axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(OPERATING_POINTS, [str(k) for k in OPERATING_POINTS])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Frames per sequence, K")

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
        print("wrote", out / name, flush=True)

    evaluation = read(ART / "evaluation.json")["results"]
    flops = read(ART / "flops.json")
    gflops = {k: flops["points"][str(k)]["flops"] / 1e9 for k in OPERATING_POINTS}
    mobilenet = read(F003_ART / "mobilenet_k" / "mobilenet_k.json")["results"]
    mobilenet_ks = sorted(int(k) for k in mobilenet)

    def metric(result, name):
        """(value, low, high) for accuracy (Wilson), macro-F1 or AUROC (unit bootstrap)."""
        da = result["duplicate_aware"]
        if name == "accuracy":
            n = da["tp"] + da["tn"] + da["fp"] + da["fn"]
            return (da["accuracy"], *wilson(da["tp"] + da["tn"], n))
        if name == "macro_f1":
            return (da["macro_f1"], *result["macro_f1_ci95"])
        return (result["auroc"], *result["auroc_ci95"])

    # Figure 1: the family against the static baseline, three quality measures.
    metrics = (("accuracy", "Accuracy at threshold 0.5"), ("macro_f1", "Macro-F1 at threshold 0.5"),
               ("auroc", "AUROC"))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for axis, letter, (name, label) in zip(axes, "abc", metrics):
        xs = [gflops[k] for k in OPERATING_POINTS]
        values = [metric(evaluation[str(k)], name) for k in OPERATING_POINTS]
        axis.fill_between(xs, [v[1] for v in values], [v[2] for v in values], color=C["tinyvit"],
                          alpha=0.12, linewidth=0)
        axis.plot(xs, [v[0] for v in values], "o-", color=C["tinyvit"], zorder=4,
                  label="TinyViT, one network at K = 1–16 frames")
        for k, x, v in zip(OPERATING_POINTS, xs, values):
            axis.annotate(f"K{k}", (x, v[0]), textcoords="offset points", fontsize=7, color=C["tinyvit"],
                          xytext=(-4, 7) if k == 12 else (4, -11), ha="right" if k == 12 else "left")
        mx = [mobilenet[str(k)]["flops"]["flops"] / 1e9 for k in mobilenet_ks]
        mv = [metric(mobilenet[str(k)], name) for k in mobilenet_ks]
        axis.plot(mx, [v[0] for v in mv], "D--", color=C["mobilenet"], markerfacecolor="white",
                  markersize=4, linewidth=1.0, label="MobileNetV2 baseline fed the same K (not retrained)")
        for k, x, v in zip(mobilenet_ks[:-1], mx, mv):  # K16 is the starred baseline
            axis.annotate(f"K{k}", (x, v[0]), textcoords="offset points", xytext=(-4, 5), fontsize=6.5,
                          color=C["mobilenet"], ha="right")
        static = mv[-1]
        axis.errorbar([mx[-1]], [static[0]], yerr=[[static[0] - static[1]], [static[2] - static[0]]],
                      fmt="*", markersize=13, markerfacecolor="white", markeredgecolor=C["mobilenet"],
                      ecolor=C["mobilenet"], capsize=2, elinewidth=0.8, zorder=6,
                      label="MobileNetV2 static baseline (16 frames)")
        log_x(axis, [0.5, 1, 2, 5, 10, 20])
        axis.set_xlabel("GFLOPs per sequence (fvcore multiply-adds, log scale)")
        axis.set_ylabel(label)
        axis.grid(True)
        panel(axis, letter, label)
    figure_title(fig, "Validation quality against compute: TinyViT frame-count family and MobileNetV2 baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig1_quality_vs_flops")

    # Figure 2: runtime on GPU (three GTX 1080 Ti hosts) and CPU.
    benches = {p.stem: read(p) for p in sorted((ART / "benchmark").glob("*.json"))}
    gpu = {n: b for n, b in benches.items() if b["device"] == "cuda"}
    cpu = {n: b for n, b in benches.items() if b["device"] == "cpu"}
    if gpu or cpu:
        def across(group, config):
            """Median over hosts of each host's median latency, with the host range."""
            values = [b["configs"][config]["median_latency_ms"] for b in group.values()]
            return statistics.median(values), min(values), max(values)

        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
        for axis, letter, group, device in ((axes[0, 0], "a", gpu, "GPU"), (axes[0, 1], "b", cpu, "CPU")):
            if not group:
                axis.set_visible(False)
                continue
            for prefix, style, name in (("point", "o-", "model only"), ("pipeline", "s--", "end to end")):
                stats = [across(group, f"{prefix}_k{k}") for k in OPERATING_POINTS]
                axis.errorbar(OPERATING_POINTS, [s[0] for s in stats],
                              yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                              fmt=style, color=C["gpu" if device == "GPU" else "cpu"], capsize=2,
                              markerfacecolor="white" if prefix == "pipeline" else None, label=f"TinyViT, {name}")
            for config, marker, name in (("f002b_model_k16", "*", "MobileNetV2 baseline, model only"),
                                         ("f002b_pipeline_k16", "P", "MobileNetV2 baseline, end to end")):
                s = across(group, config)
                axis.errorbar([16], [s[0]], yerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt=marker, markersize=9,
                              color=C["mobilenet"], markerfacecolor="white", capsize=2, label=name)
            k_axis(axis)
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_minor_formatter(FuncFormatter(
                lambda value, _: f"{value:g}" if str(int(value))[0] in "1235" else ""))
            axis.set_ylabel("Median latency per sequence (ms, log scale)")
            axis.grid(True)
            axis.legend(loc="upper left", fontsize=6.5)
            hosts = (f"{len(group)} × GTX 1080 Ti" if device == "GPU"
                     else "Xeon E5-2630 v3, 4 threads")
            panel(axis, letter, f"{device} ({hosts})")

        axis = axes[1, 0]
        moves = [f"upgrade_k{s}_to_k{l}" for s, l in UPGRADES]
        width = 0.38
        for offset, group, device in ((-width / 2, gpu, "GPU"), (width / 2, cpu, "CPU")):
            if not group:
                continue
            ratios = []
            for (small, large), move in zip(UPGRADES, moves):
                per_host = [b["configs"][move]["median_latency_ms"] / b["configs"][f"point_k{large}"]["median_latency_ms"]
                            for b in group.values()]
                ratios.append(statistics.median(per_host))
            axis.bar([i + offset for i in range(len(moves))], ratios, width=width,
                     color=C["gpu" if device == "GPU" else "cpu"], hatch=None if device == "GPU" else "//",
                     edgecolor="white" if device == "GPU" else C["cpu"],
                     fill=device == "GPU", label=device)
        flops_ratio = [flops["upgrades"][f"k{s}_to_k{l}"]["flops"] / flops["points"][str(l)]["flops"]
                       for s, l in UPGRADES]
        axis.plot(range(len(moves)), flops_ratio, "k_", markersize=14, markeredgewidth=1.6, linestyle="none",
                  label="FLOPs ratio")
        axis.axvline(4.5, color="0.6", linewidth=0.8, linestyle=":")
        axis.set_xticks(range(len(moves)), [f"K{s}→K{l}" for s, l in UPGRADES], rotation=35)
        axis.set_ylabel("Upgrade latency / target K from scratch")
        axis.set_ylim(0, 1.25)
        axis.legend(loc="upper left", ncol=3, fontsize=6.5)
        axis.grid(True, axis="y")
        panel(axis, "c", "Upgrade with cached frames, relative cost")

        axis = axes[1, 1]
        host_styles = iter([("o-", "#0072B2"), ("v--", "#56B4E9"), ("^-.", "#009E73"), ("s:", "#E69F00")])
        for group, device in ((gpu, "GPU"), (cpu, "CPU")):
            for i, b in enumerate(group.values()):
                style, colour = next(host_styles)
                by_k = b["switching"]["by_k"]
                base = {k: b["configs"][f"point_k{k}"]["median_latency_ms"] for k in OPERATING_POINTS}
                rel = [100 * by_k[str(k)]["median_difference_ms"] / base[k] for k in OPERATING_POINTS]
                lo = [100 * by_k[str(k)]["min_difference_ms"] / base[k] for k in OPERATING_POINTS]
                hi = [100 * by_k[str(k)]["max_difference_ms"] / base[k] for k in OPERATING_POINTS]
                shift = 1 + 0.04 * (i - (len(group) - 1) / 2) + (0.12 if device == "CPU" else 0)
                axis.errorbar([k * shift for k in OPERATING_POINTS], rel,
                              yerr=[[r - l for r, l in zip(rel, lo)], [h - r for r, h in zip(rel, hi)]],
                              fmt=style, color=colour, capsize=2, markersize=4,
                              markerfacecolor="white" if device == "CPU" else None, alpha=0.9,
                              label=f"{device} {i + 1}" if device == "GPU" else "CPU")
        axis.axhline(0, color="0.3", linewidth=0.8)
        k_axis(axis)
        axis.set_ylabel("Interleaved − grouped median (% of latency)")
        axis.legend(loc="upper right", fontsize=6.5)
        axis.grid(True)
        panel(axis, "d", "K drawn per request vs grouped by K")
        figure_title(fig, "Runtime of the TinyViT frame-count operating points")
        fig.tight_layout()
        save(fig, "fig2_runtime")

    # Figure 3: F004b fine-tunes against the D026 weights, per K, over seeds.
    comparison = read(ART / "comparison.json") if (ART / "comparison.json").exists() else None
    if comparison:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
        for axis, letter, (key, mkey, label) in zip(axes, "abc", (
                ("accuracy", "per_k_accuracy", "Accuracy at threshold 0.5"),
                ("macro_f1", "per_k_macro_f1", "Macro-F1 at threshold 0.5"),
                ("auroc", "per_k_auroc", "AUROC"))):
            d026 = [metric(evaluation[str(k)], key)[0] for k in OPERATING_POINTS]
            axis.plot(OPERATING_POINTS, d026, "o-", color=C["tinyvit"], label="TinyViT, original weights")
            for loss, style, name, dx in (("multik", "s--", "fine-tune, multi-K loss", 0.97),
                                          ("k16", "^:", "fine-tune, K16-only control", 1.03)):
                seeds = comparison["seeds"].get(loss, {})
                if not seeds:
                    continue
                per_seed = [[s[mkey][str(k)] for k in OPERATING_POINTS] for s in seeds.values()]
                mean = [statistics.fmean(col) for col in zip(*per_seed)]
                axis.errorbar([k * dx for k in OPERATING_POINTS], mean,
                              yerr=[[m - min(col) for m, col in zip(mean, zip(*per_seed))],
                                    [max(col) - m for m, col in zip(mean, zip(*per_seed))]],
                              fmt=style, color=C[loss], capsize=2, markersize=4,
                              label=f"{name}, mean of {len(per_seed)} seeds (bars: range)")
            k_axis(axis)
            axis.set_ylabel(label)
            axis.grid(True)
            panel(axis, letter, label)
        figure_title(fig, "Fine-tuning across frame counts against the original TinyViT weights")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        save(fig, "fig3_finetune_per_k")

        # Figure 4: validation curves per epoch for every F004b run.
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), sharey=True)
        train_ks = "/".join(f"K{k}" for k in TRAIN_KS)
        for axis, letter, (which, title) in zip(axes, "abc", (("mean", f"Mean over {train_ks}"),
                                                              ("1", "K1"), ("16", "K16"))):
            for loss, style, name in (("multik", "s--", "multi-K loss"), ("k16", "^:", "K16-only control")):
                for j, seed in enumerate(sorted(comparison["seeds"].get(loss, {}))):
                    run = read(ART / "train" / f"{loss}_seed{seed}" / "run.json")
                    entries = [{"epoch": 0, **run["epoch0_d026"]}] + run["history"]
                    ys = [e["mean_macro_f1"] if which == "mean" else e["per_k"][which]["macro_f1"] for e in entries]
                    axis.plot([e["epoch"] for e in entries], ys, style, color=C[loss], markersize=3.5,
                              alpha=1.0 - 0.3 * j, label=f"{name}, seed {seed}")
                    selected = run["selected"]["epoch"]
                    axis.scatter([selected], [ys[selected]], s=55, facecolors="none", edgecolors=C[loss])
            axis.set_xlabel("Fine-tuning epoch (0 = original weights)")
            axis.grid(True)
            panel(axis, letter, title)
        axes[0].set_ylabel("Validation macro-F1 at threshold 0.5")
        figure_title(fig, "Validation macro-F1 per fine-tuning epoch (circles: selected epoch)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        save(fig, "fig4_finetune_epochs")

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
    parser.add_argument("--copy-to", default="experiments/F004/figures",
                        help="tracked folder for the figures; empty string to skip")
    create_figures(parser.parse_args().copy_to or None)


if __name__ == "__main__":
    main()
