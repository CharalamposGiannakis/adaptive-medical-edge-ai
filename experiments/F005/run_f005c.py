"""F005c (T011) commands: static INT8 quantization of the per-frame encoder in ONNX Runtime.

prepare (export, calibrate, quantize, choose the calibration method on train
rows) → evaluate (validation, every K) → benchmark (CPU). ``--model`` is
``d026`` (the F004 family) or ``nested_s<seed>_w<width>`` (an F005b width,
physically pruned). Writes to ignored artifacts/F005/f005c/<model>/. No command
reads T006 test rows.

Static quantization fixes every activation scale on calibration data before
deployment, so a frame's INT8 features do not depend on the other frames in a
request and the F004 anytime cache stays exact; it also skips the run-time
range computation that made F005a's dynamic INT8 slower (lecture 3, part 2).
The pooling over K and the 320→1 head stay FP32 in the caller.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, benchmark_request_ids, key
from experiments.F003.adaptive import positions_in_k16
from experiments.F003.run_f003 import auroc, auroc_ci, bootstrap, mcnemar, paired_delta
from experiments.F004.family import OPERATING_POINTS, load_family
from experiments.F004.run_f004 import prediction, timed_pass
from experiments.F005 import cache
from experiments.F005.prune import physical, set_width
from experiments.F005.run_f005b import cache_path, load_trained, train_subsets

ART = ROOT / "artifacts" / "F005" / "f005c"
SOURCES = ("experiments/F005/run_f005c.py", "experiments/F005/prune.py", "experiments/F005/run_f005b.py")
# A third run (--reduce-range) repeats version 2 with 7-bit weights, to test whether
# U8S8 saturation on this AVX2 CPU without VNNI explains version 2's quality loss.
# Version 2, fixed after the first attempt (job 111079): quantizing every operator
# with MinMax ranges on 64 calibration rows gave train BCE 0.548 at K16 against
# 0.287 in FP32, and Percentile calibration ran out of 32 GB keeping every
# activation of 1,024 frames. Version 2 quantizes only the matrix operators
# (lecture 3: keep sensitive operators in higher precision) on 16 rows.
CONFIG = {
    "version": 2, "op_types_to_quantize": ["Conv", "MatMul", "Gemm"],
    "format": "QDQ", "per_channel": True, "weight_type": "QInt8", "activation_type": "QUInt8",
    "reduce_range": False, "opset": 17, "calibration_rows": 16, "calibration_source":
        "first 16 rows of the F005b Taylor train subset (16 frames each, disjoint from the zero-shot rows)",
    "methods": ["MinMax", "Percentile"], "percentile": 99.99,
    "rule": "keep the calibration method with the lower duplicate-weighted train BCE at K16 on the 300 "
            "F005b zero-shot train rows",
    "head": "FP32 mean over K and 320→1 linear head, outside the graph",
    "pre_process": "onnxruntime quant_pre_process with skip_symbolic_shape=True",
}
PROTOCOL = {"warmup": 5, "timed": 50, "repeats": 5}


def environment() -> dict:
    record = pipe.environment_record()
    record["f005c_source_sha256"] = {s: pipe.sha256_file(ROOT / s) for s in SOURCES}
    return record


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(name: str):
    """(encoder nn.Module in standard timm form, head weight (1×D), head bias) for a model name."""
    if name == "d026":
        family = load_family()
        encoder, head = family.encoder, family.head[-1]
    elif name.startswith("nested_s"):
        seed, width = name[len("nested_s"):].split("_w")
        model = load_trained("nested", int(seed))
        encoder, head = physical(model.encoder, float(width)), model.head[-1]
        set_width(model.encoder, 1.0)
    else:
        raise ValueError(f"Unknown model {name}")
    return encoder.eval(), head.weight.detach().numpy().astype(np.float32), head.bias.detach().numpy().astype(np.float32)


def session(path: Path, threads: int):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def encode(sess, frames: np.ndarray, per_frame: bool) -> np.ndarray:
    if per_frame:
        return np.concatenate([sess.run(None, {"frames": frames[i:i + 1]})[0] for i in range(len(frames))])
    return sess.run(None, {"frames": frames})[0]


def logits_per_k(features: np.ndarray, frame_count: int, weight, bias) -> dict[int, float]:
    return {k: float(features[positions_in_k16(frame_count, k)].mean(0) @ weight[0] + bias[0])
            for k in OPERATING_POINTS}


def model_dir(args) -> Path:
    return ART / (args.model + ("_reduce_range" if args.reduce_range else ""))


def cmd_prepare(args) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    out = model_dir(args)
    config = CONFIG | {"reduce_range": args.reduce_range}
    if (out / "prepare.json").exists():
        raise SystemExit(f"{out} is already prepared")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    encoder, weight, bias = build_model(args.model)
    fp32, pre = out / "encoder_fp32.onnx", out / "encoder_fp32_pre.onnx"
    torch.onnx.export(encoder, (torch.randn(4, 3, 224, 224),), fp32, input_names=["frames"],
                      output_names=["features"], dynamic_axes={"frames": {0: "n"}, "features": {0: "n"}},
                      opset_version=CONFIG["opset"], dynamo=False)
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)  # symbolic inference fails on TinyViT
    inputs = pipe.load_inputs()
    rows, weights = inputs["train"], inputs["train_weights"]
    frames = cache.load(cache_path("train"), rows)
    subsets = train_subsets(rows)
    calibration_rows = subsets["taylor"][:CONFIG["calibration_rows"]]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter([{"frames": cache.normalize(frames[i]).numpy()} for i in calibration_rows])

        def get_next(self):
            return next(self.batches, None)
    started = time.time()
    choices = {}
    for method in CONFIG["methods"]:
        path = out / f"encoder_int8_{method.lower()}.onnx"
        extra = {"CalibPercentile": CONFIG["percentile"]} if method == "Percentile" else {}
        quantize_static(str(pre), str(path), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        reduce_range=config["reduce_range"], activation_type=QuantType.QUInt8,
                        op_types_to_quantize=CONFIG["op_types_to_quantize"],
                        weight_type=QuantType.QInt8, calibrate_method=getattr(CalibrationMethod, method),
                        extra_options=extra)
        sess = session(path, args.threads)
        total, weight_sum = 0.0, 0.0
        for i in subsets["zero_shot"]:
            features = encode(sess, cache.normalize(frames[i]).numpy(), per_frame=False)
            logit = logits_per_k(features, int(rows[i]["frame_count"]), weight, bias)[16]
            target = float(rows[i]["binary_target"])
            bce = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logit), torch.tensor(target)))
            total += weights[key(rows[i])] * bce
            weight_sum += weights[key(rows[i])]
        choices[method] = total / weight_sum
        print(method, choices[method], flush=True)
    chosen = min(choices, key=choices.get)
    np.savez(out / "head.npz", weight=weight, bias=bias)
    pipe.write_json(out / "prepare.json", {
        "model": args.model, "config": config, "train_bce_k16_by_method": choices, "chosen": chosen,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": pipe.sha256_file(p)} for p in sorted(out.glob("*.onnx"))},
        "seconds": time.time() - started, "environment": environment()})
    print("chosen", chosen)


def cmd_evaluate(args) -> None:
    """Validation at every K: ORT FP32 and the chosen ORT INT8, frames encoded together and one by one."""
    out = model_dir(args)
    prepared = read(out / "prepare.json")
    head = np.load(out / "head.npz")
    weight, bias = head["weight"], head["bias"]
    sessions = {"fp32": session(out / "encoder_fp32_pre.onnx", args.threads),
                "int8": session(out / f"encoder_int8_{prepared['chosen'].lower()}.onnx", args.threads)}
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    frames = cache.load(cache_path("validation"), rows)
    logits = {(v, k): [] for v in sessions for k in OPERATING_POINTS}
    batch_gap = {v: 0.0 for v in sessions}
    started = time.time()
    for i, row in enumerate(rows):
        x = cache.normalize(frames[i]).numpy()
        for variant, sess in sessions.items():
            together = encode(sess, x, per_frame=False)
            alone = encode(sess, x, per_frame=True)
            batch_gap[variant] = max(batch_gap[variant], float(np.abs(together - alone).max()))
            for k, value in logits_per_k(alone, int(row["frame_count"]), weight, bias).items():
                logits[(variant, k)].append(value)
        if i % 50 == 0:
            print(f"validation {i}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)
    results, units = {}, {}
    for (variant, k), values in logits.items():
        preds = [prediction(r, unit_of, k, v) for r, v in zip(rows, values)]
        evaluation = aggregate_validation(rows, preds, unit_of)
        pipe.write_predictions(out / "predictions" / f"{variant}_k{k}.csv", evaluation)
        units[(variant, k)] = evaluation["units"]
        results.setdefault(variant, {})[str(k)] = {
            "duplicate_aware": evaluation["duplicate_aware"], "macro_f1_ci95": bootstrap(evaluation["units"]),
            "auroc": auroc(evaluation["units"]), "auroc_ci95": auroc_ci(evaluation["units"])}
    comparison = {str(k): paired_delta(units[("int8", k)], units[("fp32", k)]) | {
        "mcnemar": mcnemar(units[("int8", k)], units[("fp32", k)]),
        "max_abs_logit_difference": max(abs(a - b) for a, b in zip(logits[("int8", k)], logits[("fp32", k)])),
        "median_abs_logit_difference": statistics.median(abs(a - b) for a, b in zip(logits[("int8", k)], logits[("fp32", k)]))}
        for k in OPERATING_POINTS}
    pipe.write_json(out / "evaluation.json", {
        "model": args.model, "chosen": prepared["chosen"], "results": results, "int8_minus_fp32": comparison,
        "together_vs_one_by_one_max_abs_feature_difference": batch_gap, "partition": "validation",
        "test_image_inference": False, "seconds": time.time() - started, "environment": environment()})
    for k in OPERATING_POINTS:
        print(f"K{k}", round(results["fp32"][str(k)]["duplicate_aware"]["macro_f1"], 4),
              round(results["int8"][str(k)]["duplicate_aware"]["macro_f1"], 4), flush=True)
    print("together vs one by one", batch_gap)


def cmd_benchmark(args) -> None:
    """CPU model-only timing, ORT FP32 vs ORT INT8, K frames as one request, round-robin."""
    env = environment()
    if not env.get("slurm_job_id"):
        raise SystemExit("F005c timing must run as a SLURM job")
    out = model_dir(args)
    path = out / f"benchmark_cpu_{env['hostname']}.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    prepared = read(out / "prepare.json")
    head = np.load(out / "head.npz")
    weight, bias = head["weight"], head["bias"]
    sessions = {"fp32": session(out / "encoder_fp32_pre.onnx", args.threads),
                "int8": session(out / f"encoder_int8_{prepared['chosen'].lower()}.onnx", args.threads)}
    inputs = pipe.load_inputs()
    by_key = {key(r): r for r in inputs["validation"]}
    requests = [by_key[node] for node in benchmark_request_ids(inputs["validation"])["timed"][:PROTOCOL["timed"]]]
    frames = cache.load(cache_path("validation"), inputs["validation"])
    index = {key(r): i for i, r in enumerate(inputs["validation"])}
    data = {k: [cache.normalize(frames[index[key(r)]][positions_in_k16(int(r["frame_count"]), k)]).numpy()
                for r in requests] for k in OPERATING_POINTS}
    order, warmup = list(range(len(requests))), list(range(PROTOCOL["warmup"]))

    def run(sess, k):
        def once(i):
            features = sess.run(None, {"frames": data[k][i]})[0]
            features.mean(0) @ weight[0] + bias[0]
        return once
    configs = {f"{v}_k{k}": (run(s, k), k) for k in OPERATING_POINTS for v, s in sessions.items()}
    names = list(configs)
    repeats = {n: [] for n in names}
    for r in range(PROTOCOL["repeats"]):
        for name in names[r % len(names):] + names[:r % len(names)]:
            repeats[name].append(timed_pass(configs[name][0], order, warmup, lambda: None))
        print(f"round {r + 1}/{PROTOCOL['repeats']}", flush=True)
    summary = {n: pipe.summarize_timing(repeats[n], k) | {"raw": repeats[n]} for n, (_, k) in configs.items()}
    for n in names:
        print(n, round(summary[n]["median_latency_ms"], 1), flush=True)
    pipe.write_json(path, {"configs": summary, "protocol": PROTOCOL, "threads": args.threads,
                           "chosen": prepared["chosen"], "environment": env})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("prepare", cmd_prepare), ("evaluate", cmd_evaluate), ("benchmark", cmd_benchmark)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--reduce-range", action="store_true",
                       help="7-bit weights: ONNX Runtime's remedy for U8S8 saturation on CPUs without VNNI")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
