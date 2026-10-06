"""F003 (T009) commands: cost, cached features, validation analysis, runtime.

Run order: flops → features (train/validation at 224; validation at 160/192)
→ analyze → benchmark → report. Everything writes to ignored artifacts/F003/.
No command builds a loader for the T006 test partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, benchmark_request_ids, key
from experiments.F003.adaptive import (CHECKPOINT_SHA256, EXITS, EXIT_DIMS, K_VALUES,
                                       RESOLUTIONS, ExitSequenceModel, F003Sequences,
                                       Trunk, count_flops, fit_exit_head, load_backbone,
                                       set_fused_attention)

ART = ROOT / "artifacts" / "F003"
SOURCES = ("experiments/F003/adaptive.py", "experiments/F003/run_f003.py",
           "experiments/F002c/model.py")
F002B_UNITS = ROOT / "artifacts/F002/f002b/mobilenet_v2_seed2026/lr1e-4/validation_selected_units.json"
F002B = {"macro_f1": 0.8588753712041384, "flops": 5107788033,
         "model_only_median_latency_ms": 11.40, "pipeline_median_latency_ms": 89.17}
BENCHMARK_CONFIGS = ([("final", k, 224) for k in K_VALUES] +
                     [(e, 16, 224) for e in EXITS[:-1]] +
                     [("final", 16, r) for r in RESOLUTIONS[:-1]])
TAUS = [round(0.05 * i, 2) for i in range(11)]


def environment() -> dict:
    record = pipe.environment_record()
    record["f003_source_sha256"] = {s: pipe.sha256_file(ROOT / s) for s in SOURCES}
    record["checkpoint_sha256"] = CHECKPOINT_SHA256
    return record


def config_name(exit_name: str, k: int, resolution: int) -> str:
    return f"{exit_name}_k{k}_r{resolution}"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- flops

def cmd_flops(args) -> None:
    backbone = load_backbone("cpu")
    heads = {e: torch.nn.Linear(EXIT_DIMS[e], 1) for e in EXITS[:-1]}  # shape only
    result = {}
    for exit_name in EXITS:
        model = ExitSequenceModel(backbone, exit_name, heads.get(exit_name))
        for resolution in RESOLUTIONS:
            for k in K_VALUES:
                if exit_name != "final" and resolution != 224:
                    continue
                record = count_flops(model, k, resolution)
                result[config_name(exit_name, k, resolution)] = record
        print(exit_name, "done", flush=True)
    pipe.write_json(ART / "flops.json", {"configs": result, "environment": environment()})
    print(json.dumps({n: round(r["flops"] / 1e9, 3) for n, r in result.items()}, indent=1))


# ---------------------------------------------------------------- features

class FrameSet(Dataset):
    def __init__(self, sequences: F003Sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences.frames(index, 16), index


def cmd_features(args) -> None:
    """One K16 pass per row; caches pooled per-frame features at every exit."""
    if args.partition == "train" and args.resolution != 224:
        raise SystemExit("Exit heads are fitted at 224 only; train features at other sizes are unused")
    out = ART / "features" / f"{args.partition}_r{args.resolution}.pt"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    sequences = F003Sequences(inputs[args.partition], args.resolution)
    backbone = load_backbone(device)
    trunk = Trunk(backbone.encoder).eval()
    loader = DataLoader(FrameSet(sequences), batch_size=1, num_workers=args.workers)
    features = {e: torch.zeros(len(sequences), 16, EXIT_DIMS[e]) for e in EXITS}
    logits = torch.full((len(sequences),), float("nan"))
    verify = args.partition == "validation" and args.resolution == 224
    started = time.time()
    with torch.inference_mode():
        for frames, index in loader:
            frames = frames[0].to(device)
            out_features = trunk(frames)
            for e in EXITS:
                features[e][index.item()] = out_features[e].float().cpu()
            if verify:  # direct sequence forward on the same frames checks the cache
                logits[index.item()] = backbone(frames.unsqueeze(0)).float().cpu()[0]
            if index.item() % 200 == 0:
                print(f"{args.partition} r{args.resolution}: {index.item()}/{len(sequences)}", flush=True)
    rows = sequences.rows
    record = {"features": features, "direct_k16_logits": logits,
              "keys": [key(r) for r in rows], "targets": [int(r["binary_target"]) for r in rows],
              "positions": {k: [sequences.positions(i, k) for i in range(len(rows))] for k in K_VALUES},
              "frame_indices": [sequences.identity(i)["frame_indices"] for i in range(len(rows))],
              "partition": args.partition, "resolution": args.resolution,
              "seconds": time.time() - started, "environment": environment()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out)
    print(f"wrote {out} in {record['seconds']:.0f} s", flush=True)


def load_features(partition: str, resolution: int) -> dict:
    record = torch.load(ART / "features" / f"{partition}_r{resolution}.pt", weights_only=False)
    if record["environment"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Cached features belong to another checkpoint")
    return record


def pooled(record: dict, exit_name: str, k: int) -> torch.Tensor:
    """Mean over each row's nested K frames, exactly as the sequence model does."""
    features = record["features"][exit_name]
    return torch.stack([features[i, positions].mean(dim=0)
                        for i, positions in enumerate(record["positions"][k])])


# ---------------------------------------------------------------- analysis

def evaluate(record: dict, rows: list[dict], unit_of: dict, head, exit_name, k) -> dict:
    with torch.no_grad():
        logits = head(pooled(record, exit_name, k)).squeeze(-1).double()
    preds = [{"exam_id": r["exam_id"], "zone_id": r["zone_id"],
              "binary_target": int(r["binary_target"]), "device": r["device"],
              "unit_id": unit_of[key(r)], "k": k, "exit": exit_name,
              "resolution": record["resolution"], "logit": float(l),
              "probability": float(torch.sigmoid(l))} for r, l in zip(rows, logits)]
    return aggregate_validation(rows, preds, unit_of)


def bootstrap(units: list[dict], reps: int = 1000, seed: int = 2026) -> list[float]:
    """Unit-level bootstrap 95% interval of duplicate-aware macro-F1."""
    from src.f002_protocol import binary_metrics
    rng = random.Random(seed)
    scores = []
    for _ in range(reps):
        sample = [units[rng.randrange(len(units))] for _ in units]
        scores.append(binary_metrics([u["binary_target"] for u in sample],
                                     [u["probability"] for u in sample])["macro_f1"])
    scores.sort()
    return [scores[int(0.025 * reps)], scores[int(0.975 * reps) - 1]]


def paired_delta(a: list[dict], b: list[dict], reps: int = 1000, seed: int = 2026) -> dict:
    """Paired unit bootstrap of macro-F1(a) − macro-F1(b) on the same units."""
    from src.f002_protocol import binary_metrics
    pa = {u["unit_id"]: u for u in a}
    pb = {u["unit_id"]: u for u in b}
    if pa.keys() != pb.keys():
        raise ValueError("Paired comparison needs identical evaluation units")
    ids = sorted(pa)

    def f1(units, chosen):
        return binary_metrics([units[i]["binary_target"] for i in chosen],
                              [units[i]["probability"] for i in chosen])["macro_f1"]
    rng = random.Random(seed)
    deltas = sorted(f1(pa, s) - f1(pb, s) for s in
                    ([ids[rng.randrange(len(ids))] for _ in ids] for _ in range(reps)))
    return {"delta": f1(pa, ids) - f1(pb, ids),
            "ci95": [deltas[int(0.025 * reps)], deltas[int(0.975 * reps) - 1]],
            "a_better_fraction": sum(d > 0 for d in deltas) / reps}


def cascade(cheap: list[dict], full: list[dict], cheap_flops: int, full_flops: int,
            reused_flops: int) -> list[dict]:
    """Escalate a row from cheap to full when |p − 0.5| < τ (descriptive).

    Expected cost per row = cheap + escalation × (full − reused), where
    ``reused`` is the cheap work the full configuration need not repeat.
    """
    from src.f002_protocol import binary_metrics
    by_unit_full = {u["unit_id"]: u for u in full}
    points = []
    for tau in TAUS:
        escalated = [abs(u["probability"] - 0.5) < tau for u in cheap]
        chosen = [by_unit_full[u["unit_id"]] if esc else u for u, esc in zip(cheap, escalated)]
        rate = sum(escalated) / len(cheap)
        metrics = binary_metrics([u["binary_target"] for u in chosen],
                                 [u["probability"] for u in chosen])
        points.append({"tau": tau, "escalation_rate": rate,
                       "expected_flops": cheap_flops + rate * (full_flops - reused_flops),
                       **metrics})
    return points


def cmd_analyze(args) -> None:
    inputs = pipe.load_inputs()
    unit_of = inputs["unit_of"]
    train, val = load_features("train", 224), load_features("validation", 224)
    if train["keys"] != [key(r) for r in inputs["train"]] or \
            val["keys"] != [key(r) for r in inputs["validation"]]:
        raise ValueError("Cached feature rows differ from the frozen manifests")
    backbone = load_backbone("cpu")
    flops = read(ART / "flops.json")["configs"]

    # Cache check: final head on cached K16 features versus a direct forward.
    with torch.no_grad():
        cached = backbone.head(pooled(val, "final", 16)).squeeze(-1)
    cache_error = float((cached - val["direct_k16_logits"]).abs().max())
    if cache_error > 1e-4:
        raise ValueError(f"Cached features do not reproduce direct logits ({cache_error})")

    weights = [inputs["train_weights"][k] for k in train["keys"]]
    heads, head_info = {}, {}
    for exit_name in EXITS[:-1]:
        heads[exit_name], head_info[exit_name] = fit_exit_head(
            pooled(train, exit_name, 16).numpy(), train["targets"], weights)
    heads["final"] = backbone.head
    torch.save({e: h.state_dict() for e, h in heads.items() if e != "final"},
               ART / "exit_heads.pt")

    results, evaluations = {}, {}
    records = {224: val} | {r: load_features("validation", r) for r in RESOLUTIONS[:-1]
                            if (ART / "features" / f"validation_r{r}.pt").exists()}
    for resolution, record in records.items():
        for exit_name in EXITS:
            if resolution != 224 and exit_name != "final":
                continue
            for k in K_VALUES:
                name = f"{exit_name}_k{k}_r{resolution}"
                evaluation = evaluate(record, inputs["validation"], unit_of,
                                      heads[exit_name], exit_name, k)
                evaluations[name] = evaluation
                pipe.write_predictions(ART / "predictions" / f"{name}.csv", evaluation)
                results[name] = {"exit": exit_name, "k": k, "resolution": resolution,
                                 "duplicate_aware": evaluation["duplicate_aware"],
                                 "row_level": evaluation["row_level"],
                                 "macro_f1_ci95": bootstrap(evaluation["units"]),
                                 "flops": flops[name]["flops"]}
                print(name, round(evaluation["duplicate_aware"]["macro_f1"], 4),
                      round(flops[name]["flops"] / 1e9, 2), "GF", flush=True)

    comparisons = {}
    if F002B_UNITS.exists():
        mobilenet = read(F002B_UNITS)
        for name in ("final_k4_r224", "final_k8_r224", "final_k16_r224"):
            comparisons[f"{name}_vs_F002b"] = paired_delta(evaluations[name]["units"], mobilenet)
    for k in (1, 2, 4, 8, 12):
        comparisons[f"final_k{k}_vs_k16"] = paired_delta(
            evaluations[f"final_k{k}_r224"]["units"], evaluations["final_k16_r224"]["units"])

    full = evaluations["final_k16_r224"]["units"]
    per_frame = (flops["final_k16_r224"]["flops"] - flops["final_k8_r224"]["flops"]) / 8
    cascades = {}
    for k in (1, 2, 4, 8):
        # Nested frames: the K16 pass reuses the k cheap frames' features.
        cascades[f"final_k{k}_to_k16"] = cascade(
            evaluations[f"final_k{k}_r224"]["units"], full,
            flops[f"final_k{k}_r224"]["flops"], flops["final_k16_r224"]["flops"], k * per_frame)
    for exit_name in EXITS[:-1]:
        # Continuing the trunk reuses everything computed up to the exit.
        trunk_share = flops[f"{exit_name}_k16_r224"]["flops"] - flops[f"{exit_name}_k16_r224"]["head_flops"]
        cascades[f"{exit_name}_k16_to_final"] = cascade(
            evaluations[f"{exit_name}_k16_r224"]["units"], full,
            flops[f"{exit_name}_k16_r224"]["flops"], flops["final_k16_r224"]["flops"], trunk_share)

    pipe.write_json(ART / "analysis.json", {
        "configs": results, "comparisons": comparisons, "cascades": cascades,
        "exit_heads": head_info, "cache_max_abs_logit_error": cache_error,
        "f002b_reference": F002B, "threshold": 0.5,
        "notes": ["validation is also the tuning split for cascades; values are optimistic",
                  "exit heads fitted on train-split features of the frozen trunk"],
        "environment": environment()})
    print("cache error", cache_error)
    print(json.dumps(comparisons, indent=1))


# ---------------------------------------------------------------- runtime

def cmd_benchmark(args) -> None:
    """F002 §9 timing protocol for F003 configurations on one CUDA node."""
    device = torch.device("cuda")
    pipe.set_benchmark_flags()
    env = environment()
    if not env.get("slurm_job_id") or not env.get("gpu_model"):
        raise SystemExit("F003 timing must run as a SLURM GPU job")
    inputs = pipe.load_inputs()
    requests = benchmark_request_ids(inputs["validation"])
    by_key = {key(r): r for r in inputs["validation"]}
    rows = [by_key[node] for node in requests["timed"]]
    backbone = load_backbone(device)
    saved = torch.load(ART / "exit_heads.pt", weights_only=True)
    configs = [c for c in BENCHMARK_CONFIGS if config_name(*c) in args.configs] \
        if args.configs else BENCHMARK_CONFIGS
    sync = lambda: torch.cuda.synchronize(device)  # noqa: E731
    for exit_name, k, resolution in configs:
        name = config_name(exit_name, k, resolution)
        out = ART / "benchmark" / f"{name}.json"
        if out.exists():
            print("skip existing", name)
            continue
        head = None
        if exit_name != "final":
            head = torch.nn.Linear(EXIT_DIMS[exit_name], 1)
            head.load_state_dict(saved[exit_name])
        model = ExitSequenceModel(backbone, exit_name, head).to(device).eval()
        sequences = F003Sequences(rows, resolution)
        tensors = [sequences.frames(i, k).unsqueeze(0).to(device) for i in range(len(rows))]
        set_fused_attention(model, False)
        with torch.inference_mode():
            unfused = [model(t) for t in tensors]
            set_fused_attention(model, True)
            fused = [model(t) for t in tensors]
        difference = max(float((a - b).abs().max()) for a, b in zip(unfused, fused))
        equivalent = math.isfinite(difference) and difference <= 1e-5
        paths = {}
        torch.cuda.reset_peak_memory_stats(device)
        for attention in ("fused", "unfused"):
            set_fused_attention(model, attention == "fused")

            def model_only(index):
                with torch.inference_mode():
                    model(tensors[index])
            repeats = pipe._timed_requests(model_only, list(range(200)), list(range(50)), sync)
            paths[f"model_only_{attention}"] = pipe.summarize_timing(repeats, k) | {"raw": repeats}
        primary = "fused" if equivalent else "unfused"
        set_fused_attention(model, primary == "fused")
        for index in range(len(rows)):  # untimed warm-cache decode pass
            sequences.frames(index, k)

        def pipeline(index):
            with torch.inference_mode():
                model(sequences.frames(index, k).unsqueeze(0).to(device))
        repeats = pipe._timed_requests(pipeline, list(range(200)), list(range(50)), sync)
        paths[f"pipeline_{primary}"] = pipe.summarize_timing(repeats, k) | {"raw": repeats}
        set_fused_attention(model, False)
        result = {"config": name, "exit": exit_name, "k": k, "resolution": resolution,
                  "primary_path": f"model_only_{primary}", "pipeline_path": f"pipeline_{primary}",
                  "fused_equivalence": {"max_abs_logit_difference": difference,
                                        "tolerance": 1e-5, "equivalent": equivalent,
                                        "requests": len(tensors)},
                  "paths": paths, "request_ids": [f"{e}/{z}" for e, z in requests["timed"]],
                  "warmup_per_repeat": 50, "timed_requests": 200, "repeats": 5,
                  "peak_memory_mb_including_preloaded_requests":
                      torch.cuda.max_memory_allocated(device) / 2**20,
                  "precision": "float32", "benchmark_flags": dict(pipe.PRIMARY_BENCHMARK_FLAGS),
                  "environment": env}
        pipe.write_json(out, result)
        summary = paths[f"model_only_{primary}"]
        print(name, primary, f"{summary['median_latency_ms']:.2f} ms",
              f"{summary['sequence_throughput_s']:.1f} seq/s",
              f"pipeline {paths[f'pipeline_{primary}']['median_latency_ms']:.1f} ms", flush=True)
        del tensors, unfused, fused
        torch.cuda.empty_cache()


def cmd_benchmark_reference(args) -> None:
    """Re-time frozen F002b MobileNetV2 K16 in the same job as the F003 timings.

    Reads the frozen checkpoint (hash-checked by ``pipe.load_checkpoint``) and
    writes only under artifacts/F003, so F002b evidence is never touched.
    """
    out = ART / "benchmark" / "f002b_mobilenet_v2_k16_r224.json"
    if out.exists():
        print("skip existing reference")
        return
    device = torch.device("cuda")
    pipe.set_benchmark_flags()
    env = environment()
    if not env.get("slurm_job_id") or not env.get("gpu_model"):
        raise SystemExit("F003 timing must run as a SLURM GPU job")
    inputs = pipe.load_inputs()
    requests = benchmark_request_ids(inputs["validation"])
    by_key = {key(r): r for r in inputs["validation"]}
    sequences = pipe.Sequences([by_key[node] for node in requests["timed"]], 16)
    run, model = pipe.load_checkpoint(ROOT / "artifacts/F002/f002b/mobilenet_v2_seed2026", device)
    tensors = [sequences.frames(i).unsqueeze(0).to(device) for i in range(len(sequences))]
    sync = lambda: torch.cuda.synchronize(device)  # noqa: E731

    def model_only(index):
        with torch.inference_mode():
            model(tensors[index])

    def pipeline(index):
        with torch.inference_mode():
            model(sequences.frames(index).unsqueeze(0).to(device))
    paths = {}
    repeats = pipe._timed_requests(model_only, list(range(200)), list(range(50)), sync)
    paths["model_only"] = pipe.summarize_timing(repeats, 16) | {"raw": repeats}
    for index in range(len(sequences)):
        sequences.frames(index)
    repeats = pipe._timed_requests(pipeline, list(range(200)), list(range(50)), sync)
    paths["pipeline"] = pipe.summarize_timing(repeats, 16) | {"raw": repeats}
    pipe.write_json(out, {"config": "f002b_mobilenet_v2_k16_r224", "exit": "f002b", "k": 16,
                          "resolution": 224, "checkpoint_sha256": run["checkpoint_sha256"],
                          "primary_path": "model_only", "pipeline_path": "pipeline",
                          "paths": paths, "precision": "float32",
                          "benchmark_flags": dict(pipe.PRIMARY_BENCHMARK_FLAGS),
                          "environment": env})
    print("F002b reference", f"{paths['model_only']['median_latency_ms']:.2f} ms",
          f"pipeline {paths['pipeline']['median_latency_ms']:.1f} ms", flush=True)


# ---------------------------------------------------------------- report

def pareto_front(points: list[dict]) -> list[str]:
    """Configurations no other point beats on both macro-F1 and FLOPs."""
    front = []
    for p in points:
        if not any(q["macro_f1"] >= p["macro_f1"] and q["flops"] <= p["flops"] and
                   (q["macro_f1"] > p["macro_f1"] or q["flops"] < p["flops"]) for q in points):
            front.append(p["config"])
    return front


def cmd_report(args) -> None:
    analysis = read(ART / "analysis.json")
    rows = []
    for name, result in analysis["configs"].items():
        bench_path = ART / "benchmark" / f"{name}.json"
        row = {"config": name, "exit": result["exit"], "k": result["k"],
               "resolution": result["resolution"], "flops": result["flops"],
               **{m: result["duplicate_aware"][m] for m in
                  ("macro_f1", "accuracy", "normal_recall", "abnormal_recall")},
               "macro_f1_ci95": result["macro_f1_ci95"]}
        if bench_path.exists():
            # One path for every configuration: canonical unfused attention, as
            # traced for FLOPs. Fused timing stays a secondary column, because the
            # fused/unfused logit gap straddles 1e-5 across configurations.
            bench = read(bench_path)
            unfused, fused = bench["paths"]["model_only_unfused"], bench["paths"]["model_only_fused"]
            pipeline = bench["paths"][bench["pipeline_path"]]
            row.update(model_only_median_ms=unfused["median_latency_ms"],
                       model_only_p95_ms=unfused["p95_latency_ms"],
                       model_only_seq_s=unfused["sequence_throughput_s"],
                       model_only_frames_s=unfused["frame_throughput_s"],
                       fused_model_only_median_ms=fused["median_latency_ms"],
                       fused_max_abs_logit_difference=bench["fused_equivalence"]["max_abs_logit_difference"],
                       pipeline_attention=bench["pipeline_path"].split("_")[-1],
                       pipeline_median_ms=pipeline["median_latency_ms"],
                       pipeline_seq_s=pipeline["sequence_throughput_s"],
                       pipeline_frames_s=pipeline["frame_throughput_s"])
        rows.append(row)
    front = pareto_front(rows)
    for row in rows:
        row["on_pareto_front"] = row["config"] in front
    reference = {"f002b_recorded": analysis["f002b_reference"]}
    if (ART / "benchmark" / "f002b_mobilenet_v2_k16_r224.json").exists():
        bench = read(ART / "benchmark" / "f002b_mobilenet_v2_k16_r224.json")
        reference["f002b_same_job"] = {
            "model_only_median_ms": bench["paths"]["model_only"]["median_latency_ms"],
            "model_only_seq_s": bench["paths"]["model_only"]["sequence_throughput_s"],
            "pipeline_median_ms": bench["paths"]["pipeline"]["median_latency_ms"],
            "pipeline_seq_s": bench["paths"]["pipeline"]["sequence_throughput_s"]}
    pipe.write_json(ART / "summary.json", {"configs": rows, "pareto_front": front,
                                           "reference": reference})
    print("pareto front:", front)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; summary.json written without a plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"final": ("o", "TinyViT, frame count K (224 px)"),
              "stage2": ("s", "exit after stage 2"), "stage2_mid": ("^", "exit mid stage 2"),
              "stage1": ("v", "exit after stage 1")}
    for exit_name, (marker, label) in styles.items():
        pts = sorted((r for r in rows if r["exit"] == exit_name and r["resolution"] == 224),
                     key=lambda r: r["flops"])
        ax.plot([p["flops"] / 1e9 for p in pts], [p["macro_f1"] for p in pts],
                marker=marker, label=label, linestyle="-" if exit_name == "final" else ":")
        if exit_name == "final":
            for p in pts:
                ax.annotate(f"K{p['k']}", (p["flops"] / 1e9, p["macro_f1"]),
                            textcoords="offset points", xytext=(4, -10), fontsize=8)
    steps = analysis["cascades"]["final_k4_to_k16"]
    ax.plot([c["expected_flops"] / 1e9 for c in steps], [c["macro_f1"] for c in steps],
            linestyle="--", color="gray", marker=".",
            label="K4→K16 escalation if |p−0.5|<τ (τ tuned on validation)")
    res = [r for r in rows if r["resolution"] != 224 and r["k"] == 16]
    ax.scatter([p["flops"] / 1e9 for p in res], [p["macro_f1"] for p in res], marker="D",
               label="K16 at 160/192 px")
    ref = analysis["f002b_reference"]
    ax.scatter([ref["flops"] / 1e9], [ref["macro_f1"]], marker="*", s=200, color="black",
               label="static MobileNetV2 (F002b, K16)", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("sequence GFLOPs (fvcore, log scale)")
    ax.set_ylabel("duplicate-aware validation macro-F1")
    ax.set_title("F003: TinyViT configurations on validation (threshold 0.5)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "pareto_f003.png", dpi=160)
    print("wrote", ART / "pareto_f003.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("flops").set_defaults(func=cmd_flops)
    features = sub.add_parser("features")
    features.add_argument("--partition", choices=("train", "validation"), required=True)
    features.add_argument("--resolution", type=int, choices=RESOLUTIONS, default=224)
    features.add_argument("--device", default="cuda")
    features.add_argument("--workers", type=int, default=4)
    features.set_defaults(func=cmd_features)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--configs", nargs="*", help="subset of config names; default all")
    bench.set_defaults(func=cmd_benchmark)
    sub.add_parser("benchmark-reference").set_defaults(func=cmd_benchmark_reference)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
