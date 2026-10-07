"""F004 (T010) commands: the shared-weight frame-count family and its evidence.

F004a, the frozen D026 family: flops → weights → evaluate → benchmark (GPU on
GTX 1080 Ti nodes, CPU on a CPU node). F004b, exploratory multi-K fine-tune:
train --loss multik / --loss k16 (control) at seeds 2026–2028 → compare.
Everything writes to ignored artifacts/F004/. Figures are made separately by
create_figures.py from the saved results. No command builds a loader for the
T006 test partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, benchmark_request_ids, key, optimization_loss
from experiments.F002c.model import F002cSequences, SequenceDataset, TinyViTSequence, build_optimizer
from experiments.F003.adaptive import (CHECKPOINT, CHECKPOINT_SHA256, F003Sequences, count_flops,
                                       positions_in_k16)
from experiments.F003.run_f003 import auroc, auroc_ci, bootstrap, mcnemar, paired_delta
from experiments.F004.family import (FEATURES, OPERATING_POINTS, AnytimeClip,
                                     Upgrade, load_family, read_parameters, tensor_sha256)

ART = ROOT / "artifacts" / "F004"
F003_ART = ROOT / "artifacts" / "F003"
SOURCES = ("experiments/F004/family.py", "experiments/F004/run_f004.py",
           "experiments/F003/adaptive.py", "experiments/F002c/model.py")
# The one-step chain, then the larger jumps a budget change can ask for.
UPGRADES = ((1, 2), (2, 4), (4, 8), (8, 12), (12, 16), (8, 16), (4, 16), (1, 16))
SEEDS = (2026, 2027, 2028)

# F004b, fixed before any F004b training run. Starts from the D026 checkpoint;
# the two runs differ only in which K enter the loss. TRAIN_KS is the K set of
# the loss and of selection; K12 joined the family later and is only evaluated.
TRAIN_KS = (1, 2, 4, 8, 16)
TRAIN_CONFIG = {
    "experiment": "F004b", "init": "D026 checkpoint", "init_sha256": CHECKPOINT_SHA256,
    "seed": 2026, "max_epochs": 10, "min_epochs": 3, "patience": 3, "warmup_epochs": 1,
    "backbone_lr": 2.5e-5, "head_lr": 2.5e-4, "min_lr_ratio": 0.05, "weight_decay": 0.01,
    "gradient_clip_norm": 1.0, "microbatch_sequences": 2, "effective_batch_sequences": 8,
    "precision": "float32", "batchnorm_running_stats": "frozen",
    "drop_path_rate": 0.1, "head_dropout": 0.1,
    "frames": "frozen K16 selection per train row; every K is a nested subset of it",
    "augmentation": "train-only sequence-consistent horizontal flip p=0.5",
    "loss": {"multik": "mean over K in {1,2,4,8,16} of BCEWithLogits on the K-frame mean",
             "k16": "BCEWithLogits at K16 only (control: same recipe, no extra K)"},
    "selection": "validation duplicate-aware macro-F1 at 0.5 averaged over K in {1,2,4,8,16}, "
                 "then lower mean BCE, then earlier epoch; epoch 0 (D026) is not selectable",
    "threshold": 0.5, "workers": 2,
}
# Adopting the multi-K weights would replace the D026 weights of the family, so
# it needs a team decision; this rule only says whether to propose it. It is
# applied to seed 2026, the seed fixed before running; seeds 2027–2028 were
# added afterwards to show how much the result moves between seeds.
ADOPTION_RULE = {
    "min_mean_f1_gain_over_d026": 0.01, "min_mean_f1_gain_over_k16_control": 0.01,
    "max_k16_f1_loss_vs_d026": 0.01,
    "text": "Propose the multi-K checkpoint only if its mean macro-F1 over the five K "
            "exceeds both D026 and the K16-only control by at least 0.01, and its K16 "
            "macro-F1 is at most 0.01 below D026's. Otherwise the family keeps D026.",
}
# Timing protocols. GPU is F002 §9; the CPU run keeps its structure with fewer
# requests so a K16 pass at about 0.4 s per request fits one job.
PROTOCOLS = {"cuda": {"warmup": 50, "timed": 200, "repeats": 5, "switching_requests": 1000},
             "cpu": {"warmup": 10, "timed": 100, "repeats": 5, "switching_requests": 300}}


def environment() -> dict:
    record = pipe.environment_record()
    record["f004_source_sha256"] = {s: pipe.sha256_file(ROOT / s) for s in SOURCES}
    return record


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- flops

def cmd_flops(args) -> None:
    """fvcore counts per operating point and per upgrade, F003 convention."""
    model = load_family()
    points = {k: count_flops(model, k) for k in OPERATING_POINTS}
    upgrades = {}
    for small, large in UPGRADES:
        record = count_flops(Upgrade(model, small), large - small)
        upgrades[f"k{small}_to_k{large}"] = record | {"from_k": small, "to_k": large}
    check = {}
    if (F003_ART / "flops.json").exists():
        f003 = read(F003_ART / "flops.json")["configs"]
        check = {k: f003[f"final_k{k}_r224"]["flops"] == points[k]["flops"] for k in OPERATING_POINTS}
        if not all(check.values()):
            raise RuntimeError(f"Family FLOPs differ from F003: {check}")
    pipe.write_json(ART / "flops.json", {
        "points": {str(k): v for k, v in points.items()}, "upgrades": upgrades,
        "equal_to_f003": {str(k): v for k, v in check.items()}, "environment": environment()})
    for k, v in points.items():
        print(f"K{k}: {v['flops'] / 1e9:.3f} GFLOPs")
    for name, v in upgrades.items():
        print(f"{name}: {v['flops'] / 1e9:.3f} GFLOPs")


# ---------------------------------------------------------------- weights

def cmd_weights(args) -> None:
    """Which tensors each operating point reads, compared with the checkpoint file.

    The checks that can fail: every parameter is read at every K, and every
    parameter and BN buffer equals the checkpoint. Storage identity across K
    holds by construction (one module) and is recorded, not tested.
    """
    model = load_family()
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    reads = {k: read_parameters(model, k) for k in OPERATING_POINTS}
    names = sorted(dict(model.named_parameters()))
    layers = []
    for name in names:
        tensors = [reads[k].get(name) for k in OPERATING_POINTS]
        if any(t is None for t in tensors):
            raise RuntimeError(f"{name} is not read at every operating point")
        layers.append({
            "parameter": name, "shape": list(tensors[0].shape), "elements": tensors[0].numel(),
            "read_at_every_k": True,
            "same_storage_all_k": len({t.data_ptr() for t in tensors}) == 1,
            "equal_to_checkpoint": bool(torch.equal(tensors[0], saved[name])),
            "sha256": tensor_sha256(tensors[0])})
    state = model.state_dict()  # persistent buffers too (BN statistics)
    buffers_equal = state.keys() == saved.keys() and all(torch.equal(state[n], saved[n]) for n in state)
    if not all(l["equal_to_checkpoint"] for l in layers) or not buffers_equal:
        raise RuntimeError("An operating point reads weights that differ from the shared checkpoint")
    parameters = sum(l["elements"] for l in layers)
    size = CHECKPOINT.stat().st_size
    record = {
        "operating_points": list(OPERATING_POINTS), "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": size,
        "parameters": parameters, "parameter_tensors": len(layers),
        "parameter_tensors_read_per_k": {str(k): len(v) for k, v in reads.items()},
        "buffers_equal_to_checkpoint": buffers_equal,
        "separate_models_bytes_for_comparison": size * len(OPERATING_POINTS),
        "layers": layers, "environment": environment()}
    pipe.write_json(ART / "weights.json", record)
    print(f"{len(layers)} parameter tensors, {parameters:,} parameters, all read at "
          f"K{', K'.join(map(str, OPERATING_POINTS))} and equal to the {size:,}-byte checkpoint")


# ---------------------------------------------------------------- evaluate

def unit_metrics(units: list[dict]) -> dict:
    return {"macro_f1_ci95": bootstrap(units), "auroc": auroc(units), "auroc_ci95": auroc_ci(units)}


def prediction(row: dict, unit_of: dict, k: int, logit: float, **extra) -> dict:
    return {"exam_id": row["exam_id"], "zone_id": row["zone_id"],
            "binary_target": int(row["binary_target"]), "device": row["device"],
            "unit_id": unit_of[key(row)], "k": k, "logit": logit,
            "probability": 1 / (1 + math.exp(-logit)), **extra}


def f003_logits(k: int) -> dict | None:
    path = F003_ART / "predictions" / f"final_k{k}_r224.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return {(r["exam_id"], r["zone_id"]): float(r["logit"]) for r in csv.DictReader(handle)}


def k16_loader(frames16: torch.Tensor, selection: list[int]):
    """``AnytimeClip`` loader over a row's preloaded K16 frames, by clip frame number."""
    return lambda numbers: frames16[[selection.index(x) for x in numbers]]


def cmd_evaluate(args) -> None:
    """Validation through the family module: direct at each K, and anytime up and down."""
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    sequences = F003Sequences(rows, 224)
    model = load_family(device=device)
    direct = {k: [] for k in OPERATING_POINTS}
    anytime = {k: [] for k in OPERATING_POINTS}
    encoded_up, encoded_down, worst_up, worst_down = {}, {}, 0.0, 0.0
    started = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            frames16 = sequences.frames(i, 16).to(device)
            n = int(row["frame_count"])
            selection = [int(x) for x in sequences.identity(i)["frame_indices"].split()]
            logits = {}
            for k in OPERATING_POINTS:
                logits[k] = float(model(frames16[positions_in_k16(n, k)].unsqueeze(0))[0])
                direct[k].append(prediction(row, unit_of, k, logits[k]))
            clip = AnytimeClip(model, n, k16_loader(frames16, selection))
            for k in OPERATING_POINTS:  # budget rises: each step encodes only the added frames
                logit, encoded = clip.predict(k)
                encoded_up.setdefault(k, []).append(encoded)
                worst_up = max(worst_up, abs(float(logit) - logits[k]))
                anytime[k].append(prediction(row, unit_of, k, float(logit), encoded_frames=encoded))
            for k in reversed(OPERATING_POINTS):  # budget falls: nothing new to encode
                logit, encoded = clip.predict(k)
                encoded_down.setdefault(k, []).append(encoded)
                worst_down = max(worst_down, abs(float(logit) - logits[k]))
            if i % 50 == 0:
                print(f"validation {i}/{len(rows)}", flush=True)
    results, checks = {}, {}
    for k in OPERATING_POINTS:
        evaluation = aggregate_validation(rows, direct[k], unit_of)
        pipe.write_predictions(ART / "predictions" / f"family_k{k}.csv", evaluation)
        pipe.write_predictions(ART / "predictions" / f"anytime_k{k}.csv",
                               aggregate_validation(rows, anytime[k], unit_of))
        results[str(k)] = {"duplicate_aware": evaluation["duplicate_aware"],
                           "row_level": evaluation["row_level"]} | unit_metrics(evaluation["units"])
        reference = f003_logits(k)
        if reference is not None:
            ours = {(p["exam_id"], p["zone_id"]): p["logit"] for p in direct[k]}
            checks[str(k)] = {
                "max_abs_logit_difference_vs_f003": max(abs(ours[r] - reference[r]) for r in ours),
                "decision_flips_vs_f003": sum((ours[r] >= 0) != (reference[r] >= 0) for r in ours)}
    units = {k: read(ART / "predictions" / f"family_k{k}_units.json") for k in OPERATING_POINTS}
    paired = {f"k{k}_minus_k16": paired_delta(units[k], units[16]) | {"mcnemar": mcnemar(units[k], units[16])}
              for k in OPERATING_POINTS[:-1]}
    record = {
        "results": results, "paired": paired, "reproduces_f003": checks,
        "anytime": {"max_abs_logit_difference_rising": worst_up,
                    "max_abs_logit_difference_falling": worst_down,
                    "frames_encoded_rising": {str(k): sorted(set(v)) for k, v in encoded_up.items()},
                    "frames_encoded_falling": {str(k): sorted(set(v)) for k, v in encoded_down.items()}},
        "rows": len(rows), "units": len(units[16]), "threshold": 0.5, "partition": "validation",
        "test_image_inference": False, "seconds": time.time() - started, "environment": environment()}
    pipe.write_json(ART / "evaluation.json", record)
    for k in OPERATING_POINTS:
        m = results[str(k)]
        print(f"K{k}: macro-F1 {m['duplicate_aware']['macro_f1']:.4f} AUROC {m['auroc']:.4f}",
              checks.get(str(k), ""))
    print("anytime max |Δlogit| rising", worst_up, "falling", worst_down)


def cmd_all_frames(args) -> None:
    """Reference point: the D026 weights on every retained frame of each validation clip.

    Not an operating point (its cost varies with clip length); it shows what the
    family gives up by sampling at most 16 frames. Cost per clip = K1 FLOPs plus
    one frame's encoder FLOPs per further frame, from the F004 counts.
    """
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    sequences = F003Sequences(rows, 224)
    model = load_family(device=device)
    flops = read(ART / "flops.json")["points"]
    per_frame = (flops["16"]["flops"] - flops["1"]["flops"]) / 15
    predictions, costs, counts = [], [], []
    started = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            n = int(row["frame_count"])
            features = torch.cat([model.encode(sequences.frames_at(i, range(start, min(n, start + 64))).to(device))
                                  for start in range(0, n, 64)])
            logit = float(model.classify(features.unsqueeze(0))[0])
            predictions.append(prediction(row, unit_of, n, logit))
            costs.append(flops["1"]["flops"] + (n - 1) * per_frame)
            counts.append(n)
            if i % 50 == 0:
                print(f"validation {i}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)
    evaluation = aggregate_validation(rows, predictions, unit_of)
    pipe.write_predictions(ART / "predictions" / "all_frames.csv", evaluation)
    k16 = read(ART / "predictions" / "family_k16_units.json")
    record = {"result": {"duplicate_aware": evaluation["duplicate_aware"], "row_level": evaluation["row_level"]}
              | unit_metrics(evaluation["units"]),
              "minus_k16": paired_delta(evaluation["units"], k16) | {"mcnemar": mcnemar(evaluation["units"], k16)},
              "gflops_per_clip": {"mean": statistics.fmean(costs) / 1e9, "median": statistics.median(costs) / 1e9,
                                  "min": min(costs) / 1e9, "max": max(costs) / 1e9},
              "frames_per_clip": {"mean": statistics.fmean(counts), "median": statistics.median(counts),
                                  "min": min(counts), "max": max(counts)},
              "per_frame_flops": per_frame, "rows": len(rows), "partition": "validation",
              "test_image_inference": False, "seconds": time.time() - started, "environment": environment()}
    pipe.write_json(ART / "all_frames.json", record)
    print(json.dumps({"macro_f1": evaluation["duplicate_aware"]["macro_f1"], "auroc": record["result"]["auroc"],
                      "gflops": record["gflops_per_clip"], "minus_k16": record["minus_k16"]["delta"]}, indent=1))


def cmd_extras(args) -> None:
    """Per-device results per K and tail latency, from saved F004 results only."""
    from experiments.F003.run_f003 import device_table, unit_records
    inputs = pipe.load_inputs()
    units = unit_records(inputs["validation"], inputs["unit_of"])
    devices = {str(k): device_table(read(ART / "predictions" / f"family_k{k}_units.json"), units)
               for k in OPERATING_POINTS}

    def p99(repeats):
        return statistics.median(sorted(r["latencies_ms"])[math.ceil(0.99 * len(r["latencies_ms"])) - 1]
                                 for r in repeats)
    tails = {}
    for path in sorted((ART / "benchmark").glob("*.json")):
        bench = read(path)
        tails[path.stem] = {name: {"median_ms": c["median_latency_ms"], "p95_ms": c["p95_latency_ms"],
                                   "p99_ms": p99(c["raw"])} for name, c in bench["configs"].items()
                            if name.startswith(("point_", "pipeline_", "f002b_"))}
    pipe.write_json(ART / "extras.json", {"per_device": devices, "tail_latency": tails,
                                          "environment": environment()})
    for k in OPERATING_POINTS:
        print(f"K{k}", {d: (v["units"], round(v["normal_recall"], 3), round(v["abnormal_recall"], 3))
                        for d, v in devices[str(k)].items()})


def cmd_memory(args) -> None:
    """GPU memory per K: weights, and the extra memory one warm request allocates."""
    device = torch.device("cuda")
    pipe.set_benchmark_flags()
    env = environment()
    if not env.get("slurm_job_id") or not env.get("gpu_model"):
        raise SystemExit("Run as a SLURM GPU job")
    model = load_family(device=device)
    weights = torch.cuda.memory_allocated(device)
    record = {"weights_bytes_on_device": weights, "points": {}}
    for k in OPERATING_POINTS:
        frames = torch.randn(1, k, 3, 224, 224, device=device)
        with torch.inference_mode():
            for _ in range(5):  # warm-up: cuDNN tuning and allocator caching happen here
                model(frames)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            model(frames)
            torch.cuda.synchronize(device)
        record["points"][str(k)] = {"input_bytes": frames.numel() * 4,
                                    "peak_extra_bytes_one_request": torch.cuda.max_memory_allocated(device) - before}
        del frames
    record["environment"] = env
    pipe.write_json(ART / "memory.json", record)
    print(json.dumps({k: round(v["peak_extra_bytes_one_request"] / 2**20, 1) for k, v in record["points"].items()}))


def cmd_export_onnx(args) -> None:
    """The family as ONNX: one graph with a variable frame axis, and the per-frame encoder."""
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(args.threads)
    model = load_family()
    out = ART / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    family_path, encoder_path = out / "tinyvit_family.onnx", out / "tinyvit_frame_encoder.onnx"
    torch.onnx.export(model, (torch.randn(1, 4, 3, 224, 224),), family_path, input_names=["frames"],
                      output_names=["logit"], dynamic_axes={"frames": {1: "k"}}, opset_version=17,
                      dynamo=False)
    torch.onnx.export(model.encoder, (torch.randn(4, 3, 224, 224),), encoder_path, input_names=["frames"],
                      output_names=["features"], dynamic_axes={"frames": {0: "n"}, "features": {0: "n"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    family = ort.InferenceSession(str(family_path), options, providers=["CPUExecutionProvider"])
    encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    torch.manual_seed(2026)
    record = {"points": {}, "threads": args.threads}
    for k in OPERATING_POINTS:
        frames = torch.randn(1, k, 3, 224, 224)
        with torch.inference_mode():
            expected = model(frames).numpy()
            features = model.encode(frames[0]).numpy()
        got = family.run(None, {"frames": frames.numpy()})[0]
        got_features = encoder.run(None, {"frames": frames[0].numpy()})[0]

        def median_ms(fn, repeats=20):
            fn()
            times = []
            for _ in range(repeats):
                began = time.perf_counter()
                fn()
                times.append((time.perf_counter() - began) * 1000)
            return statistics.median(times)
        with torch.inference_mode():
            torch_ms = median_ms(lambda: model(frames))
        record["points"][str(k)] = {
            "max_abs_logit_difference": float(np.abs(got - expected).max()),
            "max_abs_feature_difference": float(np.abs(got_features - features).max()),
            "onnxruntime_median_ms": median_ms(lambda: family.run(None, {"frames": frames.numpy()})),
            "pytorch_median_ms": torch_ms}
        print(f"K{k}", {n: round(v, 6) if "difference" in n else round(v, 1)
                        for n, v in record["points"][str(k)].items()}, flush=True)
    worst = max(max(v["max_abs_logit_difference"], v["max_abs_feature_difference"])
                for v in record["points"].values())
    if worst > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch by {worst}")
    record |= {"files": {p.name: {"bytes": p.stat().st_size, "sha256": pipe.sha256_file(p)}
                         for p in (family_path, encoder_path)},
               "opset": 17, "onnxruntime": ort.__version__, "environment": environment()}
    pipe.write_json(out / "export.json", record)


# ---------------------------------------------------------------- benchmark

def timed_pass(run_once, requests, warmup, sync) -> dict:
    """One repeat: warm-up requests, then synchronized timed requests."""
    for request in warmup:
        run_once(request)
    sync()
    latencies = []
    start = time.perf_counter()
    for request in requests:
        began = time.perf_counter()
        run_once(request)
        sync()
        latencies.append((time.perf_counter() - began) * 1000.0)
    return {"latencies_ms": latencies, "wall_seconds": time.perf_counter() - start}


def cmd_benchmark(args) -> None:
    """Timing of points, upgrades, switching and F002b in one job on one device.

    Configurations are timed round-robin: each of the repeats times every
    configuration once, in an order rotated by one per repeat, so drift during
    the job reaches all configurations alike. The first request of each
    configuration in the first round is recorded before its warm-up (cold
    start, including cuDNN autotuning on CUDA).
    """
    device = torch.device(args.device)
    env = environment()
    if not env.get("slurm_job_id") or (device.type == "cuda" and not env.get("gpu_model")):
        raise SystemExit("F004 timing must run as a SLURM job (a GPU job for CUDA)")
    out = ART / "benchmark" / f"{device.type}_{env['hostname']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to mix timing sessions")
    protocol = PROTOCOLS[device.type]
    if device.type == "cuda":
        pipe.set_benchmark_flags()
        sync = lambda: torch.cuda.synchronize(device)  # noqa: E731
    else:
        torch.set_num_threads(args.threads)
        sync = lambda: None  # noqa: E731
    inputs = pipe.load_inputs()
    requests = benchmark_request_ids(inputs["validation"])
    by_key = {key(r): r for r in inputs["validation"]}
    rows = [by_key[node] for node in requests["timed"][:protocol["timed"]]]
    order = list(range(len(rows)))
    warmup = order[:protocol["warmup"]]
    sequences = F003Sequences(rows, 224)
    counts = [int(r["frame_count"]) for r in rows]
    model = load_family(device=device)
    frames16 = [sequences.frames(i, 16).to(device) for i in order]
    index = [{k: torch.tensor(positions_in_k16(n, k), device=device) for k in OPERATING_POINTS}
             for n in counts]
    with torch.inference_mode():
        cached = {small: [model.encode(frames16[i].index_select(0, index[i][small])).unsqueeze(0)
                          for i in order] for small in {s for s, _ in UPGRADES}}
    added = {(s, l): [torch.tensor(sorted(set(positions_in_k16(counts[i], l)) -
                                         set(positions_in_k16(counts[i], s))), device=device)
                      for i in order] for s, l in UPGRADES}
    f002b_run, f002b = pipe.load_checkpoint(ROOT / "artifacts/F002/f002b/mobilenet_v2_seed2026", device)
    f002b_sequences = pipe.Sequences(rows, 16)
    f002b_frames = [f002b_sequences.frames(i).unsqueeze(0).to(device) for i in order]
    for i in order:  # untimed warm-cache decode pass (every K's frames are in K16)
        sequences.frames(i, 16)

    configs = {}

    def point(k):
        def run(i):
            with torch.inference_mode():
                model(frames16[i].index_select(0, index[i][k]).unsqueeze(0))
        return run

    def pipeline(k):
        def run(i):
            with torch.inference_mode():
                model(sequences.frames(i, k).unsqueeze(0).to(device))
        return run

    def upgrade(small, large):
        def run(i):
            with torch.inference_mode():
                new = model.encode(frames16[i].index_select(0, added[(small, large)][i])).unsqueeze(0)
                model.classify(torch.cat([cached[small][i], new], dim=1))
        return run

    def f002b_model(i):
        with torch.inference_mode():
            f002b(f002b_frames[i])

    def f002b_pipeline(i):
        with torch.inference_mode():
            f002b(f002b_sequences.frames(i).unsqueeze(0).to(device))

    for k in OPERATING_POINTS:
        configs[f"point_k{k}"] = (point(k), k)
        configs[f"pipeline_k{k}"] = (pipeline(k), k)
    for small, large in UPGRADES:
        configs[f"upgrade_k{small}_to_k{large}"] = (upgrade(small, large), large - small)
    configs["f002b_model_k16"] = (f002b_model, 16)
    configs["f002b_pipeline_k16"] = (f002b_pipeline, 16)

    names = list(configs)
    repeats = {name: [] for name in names}
    first_call = {}
    for r in range(protocol["repeats"]):
        for name in names[r % len(names):] + names[:r % len(names)]:
            run, _ = configs[name]
            if r == 0:
                sync()
                began = time.perf_counter()
                run(0)
                sync()
                first_call[name] = (time.perf_counter() - began) * 1000.0
            repeats[name].append(timed_pass(run, order, warmup, sync))
        print(f"round {r + 1}/{protocol['repeats']} done", flush=True)
    summary = {name: pipe.summarize_timing(repeats[name], frames) | {"raw": repeats[name]}
               for name, (_, frames) in configs.items()}
    for name in names:
        print(f"{name}: {summary[name]['median_latency_ms']:.2f} ms", flush=True)
    record = {"configs": summary, "first_call_ms_round_one": first_call, "round_one_order": names,
              "switching": switching_trace(model, frames16, index, protocol, sync),
              "request_ids": [f"{e}/{z}" for e, z in requests["timed"][:protocol["timed"]]],
              "protocol": protocol, "device": device.type,
              "torch_threads": torch.get_num_threads(), "attention": "unfused", "precision": "float32",
              "f002b_checkpoint_sha256": f002b_run["checkpoint_sha256"],
              "frame_slicing": "each request selects its K frames from preloaded K16 inputs with "
                               "index_select inside the timed region, for points, upgrades and switching",
              "environment": env}
    if device.type == "cuda":
        record["benchmark_flags"] = dict(pipe.PRIMARY_BENCHMARK_FLAGS)
    pipe.write_json(out, record)
    print("wrote", out)


def switching_trace(model, frames16, index, protocol, sync) -> dict:
    """The same requests with K drawn per request, and the same requests grouped by K.

    Repeated with the order of the two passes alternated (interleaved first on
    even repeats, grouped first on odd ones), so drift does not favour either.
    """
    rng = random.Random(2026)
    requests = [(i % len(frames16), rng.choice(OPERATING_POINTS)) for i in range(protocol["switching_requests"])]
    grouped_requests = sorted(requests, key=lambda r: OPERATING_POINTS.index(r[1]))

    def run(sequence):
        for row, k in sequence[:protocol["warmup"]]:
            with torch.inference_mode():
                model(frames16[row].index_select(0, index[row][k]).unsqueeze(0))
        sync()
        latencies = []
        for row, k in sequence:
            began = time.perf_counter()
            with torch.inference_mode():
                model(frames16[row].index_select(0, index[row][k]).unsqueeze(0))
            sync()
            latencies.append((time.perf_counter() - began) * 1000.0)
        return latencies
    passes = []
    for r in range(protocol["repeats"]):
        first = "interleaved" if r % 2 == 0 else "grouped"
        result = {}
        for name in ([first, "grouped" if first == "interleaved" else "interleaved"]):
            result[name] = run(requests if name == "interleaved" else grouped_requests)
        passes.append(result | {"first": first})
    by_k = {}
    for k in OPERATING_POINTS:
        per_repeat = []
        for p in passes:
            a = statistics.median(t for (_, kk), t in zip(requests, p["interleaved"]) if kk == k)
            b = statistics.median(t for (_, kk), t in zip(grouped_requests, p["grouped"]) if kk == k)
            per_repeat.append({"interleaved_median_ms": a, "grouped_median_ms": b, "difference_ms": a - b})
        differences = [x["difference_ms"] for x in per_repeat]
        by_k[str(k)] = {"requests": sum(kk == k for _, kk in requests), "per_repeat": per_repeat,
                        "median_difference_ms": statistics.median(differences),
                        "min_difference_ms": min(differences), "max_difference_ms": max(differences)}
    return {"requests": [[r, k] for r, k in requests], "grouped_requests": [[r, k] for r, k in grouped_requests],
            "passes": passes, "by_k": by_k, "seed": 2026,
            "switches_interleaved": sum(a[1] != b[1] for a, b in zip(requests, requests[1:]))}


# ---------------------------------------------------------------- F004b training

def train_dir(loss: str, seed: int) -> Path:
    return ART / "train" / f"{loss}_seed{seed}"


def per_k_logits(model: TinyViTSequence, frames: torch.Tensor, positions: list[dict],
                 ks) -> dict[int, torch.Tensor]:
    """Encode each row's K16 frames once; K-frame logits from nested subsets of them."""
    batch = frames.shape[0]
    features = model.encoder(frames.reshape(batch * 16, 3, 224, 224)).reshape(batch, 16, FEATURES)
    return {k: model.head(torch.stack([features[b, positions[b][k]].mean(dim=0)
                                       for b in range(batch)])).squeeze(-1) for k in ks}


def validate_all_k(model, sequences, unit_of, device) -> dict:
    """Duplicate-aware validation at every operating point from one K16 pass per row."""
    model.eval()
    rows = sequences.rows
    positions = [{k: positions_in_k16(int(r["frame_count"]), k) for k in OPERATING_POINTS} for r in rows]
    predictions = {k: [] for k in OPERATING_POINTS}
    with torch.inference_mode():
        for i, row in enumerate(rows):
            logits = per_k_logits(model, sequences.frames(i).unsqueeze(0).to(device), [positions[i]],
                                  OPERATING_POINTS)
            for k, logit in logits.items():
                if not torch.isfinite(logit).all():
                    raise FloatingPointError("Nonfinite validation logit")
                predictions[k].append(prediction(row, unit_of, k, float(logit[0])))
    return {k: aggregate_validation(rows, p, unit_of) for k, p in predictions.items()}


def summarize_epoch(evaluations: dict) -> dict:
    """Per-K metrics at every point; selection means over the pre-specified TRAIN_KS."""
    per_k = {str(k): e["duplicate_aware"] for k, e in evaluations.items()}
    return {"per_k": per_k,
            "mean_macro_f1": statistics.fmean(per_k[str(k)]["macro_f1"] for k in TRAIN_KS),
            "mean_loss": statistics.fmean(per_k[str(k)]["loss"] for k in TRAIN_KS)}


def better(new: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if new["mean_macro_f1"] != best["mean_macro_f1"]:
        return new["mean_macro_f1"] > best["mean_macro_f1"]
    return new["mean_loss"] < best["mean_loss"]


def lr_factor(step: int, steps_per_epoch: int) -> float:
    warmup = TRAIN_CONFIG["warmup_epochs"] * steps_per_epoch
    total = TRAIN_CONFIG["max_epochs"] * steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    floor = TRAIN_CONFIG["min_lr_ratio"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def cmd_train(args) -> None:
    """F004b: fine-tune D026 with the multi-K loss or the K16-only control."""
    device = torch.device(args.device)
    smoke = args.max_batches is not None
    out = (ART / "train_smoke" / args.loss) if smoke else train_dir(args.loss, args.seed)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite or resume")
    if not smoke and (device.type != "cuda" or args.workers < 1):
        # Workers draw the flips from the seeded generator, so both losses see the
        # same flips; with workers=0 they would share the global RNG with dropout.
        raise SystemExit("Full F004b training requires CUDA and --workers >= 1")
    config = TRAIN_CONFIG | {"seed": args.seed}
    ks = TRAIN_KS if args.loss == "multik" else (16,)
    out.mkdir(parents=True)
    started = time.time()
    run = {"status": "running", "loss": args.loss, "loss_ks": list(ks), "seed": args.seed, "config": config,
           "adoption_rule": ADOPTION_RULE, "environment": environment(), "history": [],
           "max_batches": args.max_batches, "workers": args.workers, "test_image_inference": False}
    pipe.write_json(out / "run.json", run)
    try:
        pipe.set_training_determinism(args.seed)
        inputs = pipe.load_inputs()
        train_rows = inputs["train"]
        val_rows = inputs["validation"][:args.max_val_rows] if args.max_val_rows else inputs["validation"]
        train_seq = F002cSequences(train_rows, 16, augment=True)
        val_seq = F002cSequences(val_rows, 16)
        positions = [{k: positions_in_k16(int(r["frame_count"]), k) for k in ks} for r in train_rows]
        model = TinyViTSequence(pretrained=False, drop_path_rate=config["drop_path_rate"],
                                head_dropout=config["head_dropout"])
        if pipe.sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("D026 checkpoint hash differs from the frozen record")
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        model.to(device)
        epoch0 = summarize_epoch(validate_all_k(model, val_seq, inputs["unit_of"], device))
        run["epoch0_d026"] = epoch0
        print(json.dumps({"epoch": 0, "mean_macro_f1": epoch0["mean_macro_f1"]}), flush=True)
        optimizer = build_optimizer(model, config["backbone_lr"], config["head_lr"], config["weight_decay"])
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        gen = torch.Generator().manual_seed(args.seed)
        batches = DataLoader(SequenceDataset(train_seq, inputs["train_weights"]),
                             batch_size=config["microbatch_sequences"], shuffle=True,
                             generator=gen, num_workers=args.workers,
                             pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
        accumulate = config["effective_batch_sequences"] // config["microbatch_sequences"]
        limit = len(batches) if args.max_batches is None else min(args.max_batches, len(batches))
        steps_per_epoch = math.ceil(limit / accumulate)
        best, stale, update = None, 0, 0
        for epoch in range(1, config["max_epochs"] + 1):
            epoch_started = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            weighted = {k: 0. for k in ks}
            weight_sum, max_norm = 0., 0.
            for batch_index, (frames, targets, weights, indices) in enumerate(batches):
                if batch_index >= limit:
                    break
                if batch_index % accumulate == 0:
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr * lr_factor(update, steps_per_epoch)
                frames, targets, weights = (t.to(device, non_blocking=True)
                                            for t in (frames, targets, weights))
                logits = per_k_logits(model, frames, [positions[i] for i in indices.tolist()], ks)
                per_k = {k: torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[k], targets, reduction="none") for k in ks}
                losses = torch.stack(list(per_k.values())).mean(dim=0)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError(f"Nonfinite train loss in epoch {epoch}")
                optimization_loss(losses, weights).backward()
                for k in ks:
                    weighted[k] += float((per_k[k].detach() * weights).sum())
                weight_sum += float(weights.sum())
                if (batch_index + 1) % accumulate == 0 or batch_index + 1 == limit:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"],
                                                         error_if_nonfinite=True)
                    max_norm = max(max_norm, float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
            evaluations = validate_all_k(model, val_seq, inputs["unit_of"], device)
            entry = {"epoch": epoch, **summarize_epoch(evaluations),
                     "train_loss_per_k": {str(k): v / weight_sum for k, v in weighted.items()},
                     "max_gradient_norm_before_clip": max_norm,
                     "learning_rates": [g["lr"] for g in optimizer.param_groups],
                     "seconds": time.time() - epoch_started}
            run["history"].append(entry)
            if better(entry, best):
                best, stale = {**entry, "epoch": epoch}, 0
                torch.save(model.state_dict(), out / "checkpoint.pt")
                for k, evaluation in evaluations.items():
                    pipe.write_predictions(out / "predictions" / f"k{k}.csv", evaluation)
            else:
                stale += 1
            run.update(selected={"epoch": best["epoch"], "mean_macro_f1": best["mean_macro_f1"],
                                 "mean_loss": best["mean_loss"]},
                       elapsed_seconds=time.time() - started)
            pipe.write_json(out / "run.json", run)
            print(json.dumps({"epoch": epoch, "mean_macro_f1": entry["mean_macro_f1"],
                              "k1": entry["per_k"]["1"]["macro_f1"],
                              "k16": entry["per_k"]["16"]["macro_f1"],
                              "seconds": round(entry["seconds"])}), flush=True)
            if epoch >= config["min_epochs"] and stale >= config["patience"]:
                break
        run.update(status="completed", checkpoint_sha256=pipe.sha256_file(out / "checkpoint.pt"),
                   training_flags=pipe.training_flags(), elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
        raise


# ---------------------------------------------------------------- compare

def layer_changes(checkpoint: Path) -> dict:
    """Per encoder stage: largest absolute weight change from D026."""
    before = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    after = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if before.keys() != after.keys():
        raise ValueError("Checkpoints have different tensors")
    stages: dict[str, dict] = {}
    for name, tensor in before.items():
        if not tensor.is_floating_point():
            continue
        parts = name.split(".")
        stage = ".".join(parts[:3]) if parts[1] == "stages" else ".".join(parts[:2])
        entry = stages.setdefault(stage, {"tensors": 0, "changed_tensors": 0, "max_abs_change": 0.})
        change = float((after[name] - tensor).abs().max()) if tensor.numel() else 0.
        entry["tensors"] += 1
        entry["changed_tensors"] += change > 0
        entry["max_abs_change"] = max(entry["max_abs_change"], change)
    return stages


def adoption(mean: dict, k16: dict) -> dict:
    """The pre-specified rule on mean macro-F1 over TRAIN_KS and on K16 macro-F1."""
    gains = {"over_d026": mean["multik"] - mean["d026"], "over_k16_control": mean["multik"] - mean["k16"],
             "k16_change_vs_d026": k16["multik"] - k16["d026"]}
    propose = (gains["over_d026"] >= ADOPTION_RULE["min_mean_f1_gain_over_d026"] and
               gains["over_k16_control"] >= ADOPTION_RULE["min_mean_f1_gain_over_k16_control"] and
               gains["k16_change_vs_d026"] >= -ADOPTION_RULE["max_k16_f1_loss_vs_d026"])
    return {"gains": gains, "propose_multik": propose}


def selected_entry(run: dict) -> dict:
    return next(e for e in run["history"] if e["epoch"] == run["selected"]["epoch"])


def cmd_compare(args) -> None:
    """F004b against D026 at every K, the pre-specified rule (seed 2026), and the seed spread."""
    runs = {}
    for seed in SEEDS:
        for loss in ("multik", "k16"):
            path = train_dir(loss, seed) / "run.json"
            if not path.exists():
                continue
            run = read(path)
            if (run["status"] != "completed" or run["config"] != TRAIN_CONFIG | {"seed": seed}
                    or run["max_batches"] is not None):
                raise SystemExit(f"F004b {loss} seed {seed} is not a completed full run with the fixed config")
            runs[(loss, seed)] = run
    if ("multik", 2026) not in runs or ("k16", 2026) not in runs:
        raise SystemExit("The pre-specified seed-2026 runs are missing")
    evaluation = read(ART / "evaluation.json")["results"]
    family = {k: read(ART / "predictions" / f"family_k{k}_units.json") for k in OPERATING_POINTS}
    d026 = {k: evaluation[str(k)]["duplicate_aware"] for k in OPERATING_POINTS}

    def units(loss, seed, k):
        return read(train_dir(loss, seed) / "predictions" / f"k{k}_units.json")

    table = {}
    for k in OPERATING_POINTS:
        row = {"d026": {"macro_f1": d026[k]["macro_f1"], "accuracy": d026[k]["accuracy"],
                        "auroc": evaluation[str(k)]["auroc"]}}
        for loss in ("multik", "k16"):
            u = units(loss, 2026, k)
            selected = selected_entry(runs[(loss, 2026)])["per_k"][str(k)]
            row[loss] = {"macro_f1": selected["macro_f1"], "accuracy": selected["accuracy"], "auroc": auroc(u),
                         "macro_f1_ci95": bootstrap(u), "auroc_ci95": auroc_ci(u)}
        row["multik_minus_d026"] = paired_delta(units("multik", 2026, k), family[k]) | {
            "mcnemar": mcnemar(units("multik", 2026, k), family[k])}
        row["multik_minus_k16"] = paired_delta(units("multik", 2026, k), units("k16", 2026, k))
        table[str(k)] = row
    mean = {name: statistics.fmean(table[str(k)][name]["macro_f1"] for k in TRAIN_KS)
            for name in ("d026", "multik", "k16")}
    rule = adoption(mean, {name: table["16"][name]["macro_f1"] for name in ("d026", "multik", "k16")})
    seeds = {}
    for (loss, seed), run in runs.items():
        entry = selected_entry(run)
        seeds.setdefault(loss, {})[str(seed)] = {
            "selected_epoch": entry["epoch"], "mean_macro_f1": entry["mean_macro_f1"],
            "per_k_macro_f1": {str(k): entry["per_k"][str(k)]["macro_f1"] for k in OPERATING_POINTS},
            "per_k_accuracy": {str(k): entry["per_k"][str(k)]["accuracy"] for k in OPERATING_POINTS},
            "per_k_auroc": {str(k): auroc(units(loss, seed, k)) for k in OPERATING_POINTS},
            "checkpoint_sha256": run["checkpoint_sha256"]}
    reproduction = {}
    for loss in ("multik", "k16"):  # first execution, before --seed existed, against its rerun
        original = ART / "train" / loss / "run.json"
        if original.exists():
            old, new = read(original), runs[(loss, 2026)]
            pairs = list(zip(old["history"], new["history"]))
            reproduction[loss] = {
                "epochs_compared": len(pairs), "same_selected_epoch": old["selected"]["epoch"] == new["selected"]["epoch"],
                "max_abs_mean_macro_f1_difference": max(abs(a["mean_macro_f1"] - b["mean_macro_f1"]) for a, b in pairs),
                "same_checkpoint_sha256": old["checkpoint_sha256"] == new["checkpoint_sha256"]}
    record = {"per_k": table, "mean_macro_f1": mean, **rule, "adoption_rule": ADOPTION_RULE,
              "rule_seed": 2026, "seeds": seeds, "reproduction_of_first_execution": reproduction,
              "weight_change_vs_d026": {loss: layer_changes(train_dir(loss, 2026) / "checkpoint.pt")
                                        for loss in ("multik", "k16")},
              "environment": environment()}
    pipe.write_json(ART / "comparison.json", record)
    print(json.dumps({"mean_macro_f1": mean, **rule, "reproduction": reproduction,
                      "seed_means": {loss: {s: round(v["mean_macro_f1"], 4) for s, v in d.items()}
                                     for loss, d in seeds.items()}}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("flops").set_defaults(func=cmd_flops)
    sub.add_parser("weights").set_defaults(func=cmd_weights)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate.set_defaults(func=cmd_evaluate)
    every = sub.add_parser("all-frames")
    every.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    every.set_defaults(func=cmd_all_frames)
    sub.add_parser("extras").set_defaults(func=cmd_extras)
    sub.add_parser("memory").set_defaults(func=cmd_memory)
    onnx = sub.add_parser("export-onnx")
    onnx.add_argument("--threads", type=int, default=4)
    onnx.set_defaults(func=cmd_export_onnx)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    bench.add_argument("--threads", type=int, default=4, help="CPU only: torch intra-op threads")
    bench.set_defaults(func=cmd_benchmark)
    train = sub.add_parser("train")
    train.add_argument("--loss", choices=("multik", "k16"), required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, default=2026)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--workers", type=int, default=TRAIN_CONFIG["workers"])
    train.add_argument("--max-batches", type=int, help="smoke only: train batches per epoch")
    train.add_argument("--max-val-rows", type=int, help="smoke only: validation rows")
    train.set_defaults(func=cmd_train)
    sub.add_parser("compare").set_defaults(func=cmd_compare)
    args = parser.parse_args()
    if getattr(args, "max_val_rows", None) and args.max_batches is None:
        parser.error("--max-val-rows is for smoke runs with --max-batches")
    if args.command != "benchmark":
        torch.set_num_threads(2)
    args.func(args)


if __name__ == "__main__":
    main()
