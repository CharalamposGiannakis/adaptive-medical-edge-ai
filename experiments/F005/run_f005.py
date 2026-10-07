"""F005 (T011) commands: INT8 dynamic quantization of the shared F004 family.

evaluate (CPU, validation) → benchmark (CPU, exclusive node). Writes to ignored
artifacts/F005/. Figures: create_figures.py. No command builds a loader for the
T006 test partition.

Two INT8 encoding modes are compared because dynamic quantization picks each
Linear's activation scale from the whole input tensor:
- ``per_frame``: every frame is encoded alone, so a frame's features never
  depend on which other frames share the request, and the F004 anytime cache
  stays exact. This is the deployable INT8 mode.
- ``batched``: the K frames of a request are encoded together, as F004 does in
  FP32. Its results depend on the frame set and are reported for comparison.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, benchmark_request_ids, key
from experiments.F003.adaptive import F003Sequences, positions_in_k16
from experiments.F003.run_f003 import auroc, auroc_ci, bootstrap, mcnemar, paired_delta
from experiments.F004.family import OPERATING_POINTS
from experiments.F004.run_f004 import prediction, timed_pass
from experiments.F005.compress import (SCHEMES, build, quantized_linear_count, serialized_bytes,
                                       set_quantized_engine)

ART = ROOT / "artifacts" / "F005"
F004_ART = ROOT / "artifacts" / "F004"
SOURCES = ("experiments/F005/compress.py", "experiments/F005/run_f005.py",
           "experiments/F004/family.py", "experiments/F004/run_f004.py")
PROTOCOL = {"warmup": 5, "timed": 50, "repeats": 5}


def environment() -> dict:
    record = pipe.environment_record()
    record["f005_source_sha256"] = {s: pipe.sha256_file(ROOT / s) for s in SOURCES}
    record["quantized_engine"] = torch.backends.quantized.engine
    return record


def read(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def encode_per_frame(model, frames: torch.Tensor) -> torch.Tensor:
    """n×3×224×224 → n×320, one frame per encoder call."""
    return torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])


def per_k_from_features(model, features: torch.Tensor, frame_count: int) -> dict[int, float]:
    """Logit at every K from one row's 16 per-frame features (nested positions)."""
    return {k: float(model.classify(features[positions_in_k16(frame_count, k)].unsqueeze(0))[0])
            for k in OPERATING_POINTS}


def summarize(rows, predictions, unit_of) -> tuple[dict, dict]:
    evaluation = aggregate_validation(rows, predictions, unit_of)
    units = evaluation["units"]
    return evaluation, {"duplicate_aware": evaluation["duplicate_aware"], "row_level": evaluation["row_level"],
                        "macro_f1_ci95": bootstrap(units), "auroc": auroc(units), "auroc_ci95": auroc_ci(units)}


def cmd_evaluate(args) -> None:
    """Validation at every K for FP32 and INT8 (per-frame and batched), on CPU."""
    torch.set_num_threads(args.threads)
    engine = set_quantized_engine()
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    sequences = F003Sequences(rows, 224)
    models = {scheme: build(scheme) for scheme in SCHEMES}
    variants = ("fp32_per_frame", "fp32_batched", "int8_dynamic_per_frame", "int8_dynamic_batched")
    logits = {v: {k: [] for k in OPERATING_POINTS} for v in variants}
    started = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            frames16 = sequences.frames(i, 16)
            n = int(row["frame_count"])
            for scheme, model in models.items():
                per_frame = per_k_from_features(model, encode_per_frame(model, frames16), n)
                for k in OPERATING_POINTS:
                    logits[f"{scheme}_per_frame"][k].append(per_frame[k])
                    batched = model(frames16[positions_in_k16(n, k)].unsqueeze(0))[0]
                    logits[f"{scheme}_batched"][k].append(float(batched))
            if i % 25 == 0:
                print(f"validation {i}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)
    results, evaluations = {}, {}
    for variant in variants:
        results[variant] = {}
        for k in OPERATING_POINTS:
            preds = [prediction(row, unit_of, k, logit) for row, logit in zip(rows, logits[variant][k])]
            evaluation, metrics = summarize(rows, preds, unit_of)
            pipe.write_predictions(ART / "predictions" / f"{variant}_k{k}.csv", evaluation)
            evaluations[(variant, k)] = evaluation["units"]
            results[variant][str(k)] = metrics
    comparisons = {}
    for variant in ("int8_dynamic_per_frame", "int8_dynamic_batched"):
        comparisons[variant] = {}
        for k in OPERATING_POINTS:
            a, b = logits[variant][k], logits["fp32_batched"][k]
            comparisons[variant][str(k)] = {
                "minus_fp32": paired_delta(evaluations[(variant, k)], evaluations[("fp32_batched", k)]),
                "auroc_minus_fp32": results[variant][str(k)]["auroc"] - results["fp32_batched"][str(k)]["auroc"],
                "mcnemar": mcnemar(evaluations[(variant, k)], evaluations[("fp32_batched", k)]),
                "max_abs_logit_difference": max(abs(x - y) for x, y in zip(a, b)),
                "median_abs_logit_difference": statistics.median(abs(x - y) for x, y in zip(a, b)),
                "row_decision_flips": sum((x >= 0) != (y >= 0) for x, y in zip(a, b))}
    # FP32 on CPU against the F004 GPU evaluation of the same family.
    reference = {}
    for k in OPERATING_POINTS:
        path = F004_ART / "predictions" / f"family_k{k}.csv"
        if path.exists():
            import csv
            with path.open(newline="", encoding="utf-8") as handle:
                f004 = {(r["exam_id"], r["zone_id"]): float(r["logit"]) for r in csv.DictReader(handle)}
            ours = logits["fp32_batched"][k]
            reference[str(k)] = max(abs(f004[(r["exam_id"], r["zone_id"])] - x) for r, x in zip(rows, ours))
    anytime_gap = {variant: max(abs(x - y) for k in OPERATING_POINTS
                                for x, y in zip(logits[f"{variant}_per_frame"][k], logits[f"{variant}_batched"][k]))
                   for variant in SCHEMES}
    quantized, plain = quantized_linear_count(models["int8_dynamic"])
    record = {"results": results, "int8_vs_fp32": comparisons,
              "fp32_cpu_vs_f004_gpu_max_abs_logit_difference": reference,
              "per_frame_vs_batched_max_abs_logit_difference": anytime_gap,
              "size_bytes": {s: serialized_bytes(m) for s, m in models.items()},
              "int8_linear_modules": quantized, "fp32_linear_modules_left": plain,
              "flops_split": flops_split(), "quantized_engine": engine, "threads": args.threads,
              "rows": len(rows), "units": len(evaluations[("fp32_batched", 16)]), "threshold": 0.5,
              "partition": "validation", "test_image_inference": False, "calibration_data": "none",
              "seconds": time.time() - started, "environment": environment()}
    pipe.write_json(ART / "evaluation.json", record)
    for variant in variants:
        print(variant, {k: round(results[variant][str(k)]["duplicate_aware"]["macro_f1"], 4)
                        for k in OPERATING_POINTS}, flush=True)
    print("sizes", record["size_bytes"], "per-frame vs batched", anytime_gap)


def flops_split() -> dict:
    """Per K, the share of F004's fvcore FLOPs that the INT8 Linear layers execute."""
    flops = read(F004_ART / "flops.json")["points"]
    return {k: {"total": v["flops"], "linear_int8": v["by_operator"].get("linear", 0),
                "linear_share": v["by_operator"].get("linear", 0) / v["flops"]} for k, v in flops.items()}


def cmd_benchmark(args) -> None:
    """CPU model-only timing of FP32 and INT8 at every K, round-robin in one job."""
    env = environment()
    if not env.get("slurm_job_id"):
        raise SystemExit("F005 timing must run as a SLURM job")
    out = ART / "benchmark" / f"cpu_{env['hostname']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to mix timing sessions")
    torch.set_num_threads(args.threads)
    engine = set_quantized_engine()
    inputs = pipe.load_inputs()
    requests = benchmark_request_ids(inputs["validation"])
    by_key = {key(r): r for r in inputs["validation"]}
    rows = [by_key[node] for node in requests["timed"][:PROTOCOL["timed"]]]
    order = list(range(len(rows)))
    warmup = order[:PROTOCOL["warmup"]]
    sequences = F003Sequences(rows, 224)
    frames = {k: [sequences.frames(i, k).unsqueeze(0) for i in order] for k in OPERATING_POINTS}
    models = {scheme: build(scheme) for scheme in SCHEMES}
    sync = lambda: None  # noqa: E731

    def batched(model, k):
        def run(i):
            with torch.inference_mode():
                model(frames[k][i])
        return run

    def per_frame(model, k):
        def run(i):
            with torch.inference_mode():
                model.classify(encode_per_frame(model, frames[k][i][0]).unsqueeze(0))
        return run

    configs = {}
    for k in OPERATING_POINTS:
        configs[f"fp32_batched_k{k}"] = (batched(models["fp32"], k), k)
        configs[f"int8_dynamic_batched_k{k}"] = (batched(models["int8_dynamic"], k), k)
        configs[f"int8_dynamic_per_frame_k{k}"] = (per_frame(models["int8_dynamic"], k), k)
    names = list(configs)
    repeats = {name: [] for name in names}
    for r in range(PROTOCOL["repeats"]):
        for name in names[r % len(names):] + names[:r % len(names)]:
            repeats[name].append(timed_pass(configs[name][0], order, warmup, sync))
        print(f"round {r + 1}/{PROTOCOL['repeats']} done", flush=True)
    summary = {name: pipe.summarize_timing(repeats[name], k) | {"raw": repeats[name]}
               for name, (_, k) in configs.items()}
    for name in names:
        print(f"{name}: {summary[name]['median_latency_ms']:.1f} ms", flush=True)
    pipe.write_json(out, {"configs": summary, "protocol": PROTOCOL, "threads": args.threads,
                          "quantized_engine": engine, "device": "cpu",
                          "request_ids": [f"{e}/{z}" for e, z in requests["timed"][:PROTOCOL["timed"]]],
                          "inputs": "preloaded K-frame tensors (model only)", "environment": env})
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--threads", type=int, default=8)
    evaluate.set_defaults(func=cmd_evaluate)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--threads", type=int, default=4)
    bench.set_defaults(func=cmd_benchmark)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
