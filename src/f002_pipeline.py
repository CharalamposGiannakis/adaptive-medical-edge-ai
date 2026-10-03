"""T008 execution of the D024 F002 protocol: data, training, evaluation, timing.

The scientific rules live in `docs/f002_methodology.md`; the frozen helpers in
`f002_protocol.py` and `f002_flops.py` are reused here, never re-implemented.
No function in this module reads the T006 test partition for inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

from .f002_protocol import (Key, PRIMARY_BENCHMARK_FLAGS, _runtime_advantage,
                            aggregate_validation, benchmark_request_ids, key,
                            nested_frame_indices, optimization_loss,
                            recording_units, reported_weighted_train_loss,
                            training_weights, verify_t006)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "f002_protocol.json"
T006 = ROOT / "data" / "openpocus" / "t006"
RAW = ROOT / "data" / "openpocus" / "raw"
CANDIDATES = ("mobilenet_v2", "resnet18", "deit_tiny")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SOURCE_FILES = ("src/f002_protocol.py", "src/f002_flops.py", "src/f002_pipeline.py",
                "experiments/F002/run_f002.py", "configs/f002_protocol.json")

sys.path.insert(0, str(ROOT / "data" / "openpocus"))
from cohort_loader import retained_frames  # noqa: E402  (T006 path-hash check)


# ---------------------------------------------------------------- provenance

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_record() -> dict:
    """Software, hardware and SLURM provenance for one process."""
    import fvcore  # noqa: F401
    import numpy
    import PIL
    import sklearn
    import timm
    import torch
    import torchvision
    from importlib.metadata import version

    record = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in
                     Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor())
        if Path("/proc/cpuinfo").exists() else platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "fvcore": version("fvcore"),
        "pillow": PIL.__version__, "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git_head": _command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES
                          if (ROOT / name).exists()},
        "protocol_sha256": sha256_file(PROTOCOL),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record.update({
            "gpu_model": props.name, "gpu_memory_mb": props.total_memory // 2**20,
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_driver": _command(["nvidia-smi", "--query-gpu=driver_version",
                                    "--format=csv,noheader", "-i", "0"]),
            "node_features": _slurm_features(),
        })
    return record


def _slurm_features() -> str | None:
    node = os.environ.get("SLURMD_NODENAME")
    out = _command(["scontrol", "show", "node", node]) if node else None
    if not out:
        return None
    for token in out.split():
        if token.startswith("AvailableFeatures="):
            return token.split("=", 1)[1]
    return None


# ---------------------------------------------------------------- data

def load_inputs() -> dict:
    """Verify T006 (fail closed) and build the frozen evaluation units."""
    partitions = verify_t006(PROTOCOL, T006)
    with (T006 / "shared_recording_edges.csv").open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    with (T006 / "exclusions.csv").open(newline="", encoding="utf-8") as handle:
        exclusions = list(csv.DictReader(handle))
    unit_of = recording_units(partitions, edges, exclusions)
    return {"train": partitions["train"], "validation": partitions["validation"],
            "unit_of": unit_of,
            "train_weights": training_weights(partitions["train"], unit_of)}


def preprocess_frame(path: Path):
    """Common contract: RGB, full-square bilinear 224, to_tensor, ImageNet norm."""
    from PIL import Image
    from torchvision.transforms import functional as F

    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return F.normalize(F.to_tensor(image), MEAN, STD)


class Sequences:
    """Fixed-K nested frame selection for manifest rows; paths resolved once."""

    def __init__(self, rows: list[dict], k: int, raw: Path = RAW):
        self.rows, self.k, self.raw = rows, k, Path(raw)
        self.selected = []
        for row in rows:
            frames = retained_frames(row, self.raw)
            indices = nested_frame_indices(len(frames), k)
            self.selected.append((indices, tuple(frames[i] for i in indices)))

    def __len__(self) -> int:
        return len(self.rows)

    def frames(self, index: int):
        import torch
        return torch.stack([preprocess_frame(path) for path in self.selected[index][1]])

    def identity(self, index: int) -> dict:
        indices, paths = self.selected[index]
        return {"frame_indices": " ".join(map(str, indices)),
                "frame_files": ";".join(path.relative_to(self.raw).as_posix() for path in paths)}


# ---------------------------------------------------------------- determinism

def set_training_determinism(seed: int) -> None:
    """§5 flags. CUBLAS_WORKSPACE_CONFIG must already be set before CUDA init."""
    import numpy
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def training_flags() -> dict:
    import torch
    return {"torch.use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch.backends.cudnn.deterministic": torch.backends.cudnn.deterministic,
            "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
            "torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch.backends.cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def set_benchmark_flags() -> None:
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if training_flags() | {"CUBLAS_WORKSPACE_CONFIG": None} != PRIMARY_BENCHMARK_FLAGS | {"CUBLAS_WORKSPACE_CONFIG": None}:
        raise RuntimeError("Benchmark flags not applied")


# ---------------------------------------------------------------- validation

def predict(model, sequences: Sequences, unit_of: dict, device) -> list[dict]:
    """FP32 eval-mode predictions for every row, one sequence per forward."""
    import torch

    model.eval()
    rows = []
    with torch.inference_mode():
        for index, row in enumerate(sequences.rows):
            logit = model(sequences.frames(index).unsqueeze(0).to(device)).float().item()
            if not math.isfinite(logit):
                raise FloatingPointError(f"Nonfinite validation logit for {key(row)}")
            rows.append({"exam_id": row["exam_id"], "zone_id": row["zone_id"],
                         "binary_target": int(row["binary_target"]),
                         "split_group": row["split_group"], "device": row["device"],
                         "unit_id": unit_of[key(row)], "k": sequences.k,
                         "logit": logit, "probability": 1.0 / (1.0 + math.exp(-logit)),
                         **sequences.identity(index)})
    return rows


def write_predictions(path: Path, evaluation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = evaluation["rows"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    units = [{**unit, "aliases": [f"{e}/{z}" for e, z in unit["aliases"]]}
             for unit in evaluation["units"]]
    write_json(path.with_name(path.stem + "_units.json"), units)


def checkpoint_better(new: dict, best: dict | None) -> bool:
    """§5: higher macro-F1, then lower loss; earlier epoch keeps exact ties."""
    if best is None:
        return True
    if new["macro_f1"] != best["macro_f1"]:
        return new["macro_f1"] > best["macro_f1"]
    return new["loss"] < best["loss"]


def rescue_triggered(history: list[dict]) -> bool:
    """§5 optimization failure: best epoch-2..4 loss not ≥1% below epoch 1."""
    if len(history) < 4:
        return False
    first = history[0]["train_loss"]
    return min(entry["train_loss"] for entry in history[1:4]) > 0.99 * first


def is_valid(metrics: dict, all_normal_macro_f1: float) -> bool:
    return (metrics["macro_f1"] > all_normal_macro_f1 and
            metrics["normal_recall"] > 0 and metrics["abnormal_recall"] > 0)


def all_normal_reference(inputs: dict) -> dict:
    rows = inputs["validation"]
    preds = [{"exam_id": r["exam_id"], "zone_id": r["zone_id"],
              "binary_target": int(r["binary_target"]), "probability": 0.0} for r in rows]
    return aggregate_validation(rows, preds, inputs["unit_of"])


def metadata_reference(inputs: dict) -> dict:
    """Train-fitted device/zone logistic regression; non-image shortcut check."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    def features(rows):
        return [[r["device"], r["zone_id"]] for r in rows]

    train, validation = inputs["train"], inputs["validation"]
    encoder = OneHotEncoder(handle_unknown="ignore").fit(features(train))
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", penalty="l2",
                               fit_intercept=True, class_weight=None, random_state=42)
    model.fit(encoder.transform(features(train)), [int(r["binary_target"]) for r in train],
              sample_weight=[inputs["train_weights"][key(r)] for r in train])
    probabilities = model.predict_proba(encoder.transform(features(validation)))[:, 1]
    preds = [{"exam_id": r["exam_id"], "zone_id": r["zone_id"],
              "binary_target": int(r["binary_target"]), "device": r["device"],
              "unit_id": inputs["unit_of"][key(r)], "probability": float(p)}
             for r, p in zip(validation, probabilities)]
    result = aggregate_validation(validation, preds, inputs["unit_of"])
    result["coefficients"] = dict(zip(encoder.get_feature_names_out(["device", "zone"]).tolist(),
                                      model.coef_[0].tolist()))
    result["intercept"] = float(model.intercept_[0])
    return result


# ---------------------------------------------------------------- training

def train_once(candidate: str, seed: int, learning_rate: float, out: Path,
               inputs: dict, train_seq: Sequences, val_seq: Sequences, device,
               epochs_max: int = 12, pretrained: bool = True) -> dict:
    """One §5 training attempt. Returns history, selection and failure trigger."""
    import torch
    import torch.nn.functional as F
    from .f002_flops import build_sequence_model

    set_training_determinism(seed)
    model = build_sequence_model(candidate, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=1e-4)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    weights = [inputs["train_weights"][key(row)] for row in train_seq.rows]
    n = len(train_seq)
    history, best, stale, trigger = [], None, 0, None
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs_max + 1):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator).tolist()
        microbatches = [order[i:i + 2] for i in range(0, n, 2)]
        epoch_losses, epoch_weights, skipped_steps = [], [], 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(microbatches):
            frames = torch.stack([train_seq.frames(i) for i in batch]).to(device)
            targets = torch.tensor([float(train_seq.rows[i]["binary_target"]) for i in batch],
                                   device=device)
            batch_weights = torch.tensor([weights[i] for i in batch], device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(frames)
            losses = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
            if not torch.isfinite(losses).all():
                trigger = f"nonfinite_train_loss_epoch_{epoch}"
                break
            scaler.scale(optimization_loss(losses, batch_weights)).backward()
            epoch_losses += losses.detach().cpu().tolist()
            epoch_weights += batch_weights.cpu().tolist()
            if (step + 1) % 4 == 0 or step + 1 == len(microbatches):
                scale_before = scaler.get_scale() if amp else None
                scaler.step(optimizer)
                scaler.update()
                if amp and scaler.get_scale() < scale_before:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
        if trigger:
            break

        evaluation = aggregate_validation(val_seq.rows, predict(model, val_seq, inputs["unit_of"], device),
                                          inputs["unit_of"])
        metrics = evaluation["duplicate_aware"]
        entry = {"epoch": epoch,
                 "train_loss": reported_weighted_train_loss(epoch_losses, epoch_weights),
                 "amp_skipped_steps": skipped_steps, "seconds": round(time.time() - started, 1),
                 "validation": metrics, "validation_row_level": evaluation["row_level"]}
        history.append(entry)
        write_predictions(out / "epochs" / f"validation_epoch{epoch:02d}.csv", evaluation)
        if not math.isfinite(entry["train_loss"]) or not math.isfinite(metrics["loss"]):
            trigger = f"nonfinite_loss_epoch_{epoch}"
            break
        if checkpoint_better(metrics, best):
            best, stale = {**metrics, "epoch": epoch}, 0
            torch.save(model.state_dict(), out / "checkpoint.pt")
            write_predictions(out / "validation_selected.csv", evaluation)
        else:
            stale += 1
        print(json.dumps({"candidate": candidate, "seed": seed, "lr": learning_rate,
                          "epoch": epoch, "train_loss": entry["train_loss"],
                          "val_macro_f1": metrics["macro_f1"], "val_loss": metrics["loss"],
                          "seconds": entry["seconds"]}), flush=True)
        if epoch == 4 and rescue_triggered(history):
            trigger = "train_loss_not_1pct_below_epoch1_by_epoch4"
            break
        if stale >= 3:
            break

    result = {"candidate": candidate, "seed": seed, "learning_rate": learning_rate,
              "history": history, "selected": best, "trigger": trigger,
              "epochs_completed": len(history),
              "early_stopped": trigger is None and len(history) < epochs_max,
              "training_flags": training_flags(), "amp": amp, "device": str(device)}
    if trigger is None and best is not None:
        result["checkpoint_sha256"] = sha256_file(out / "checkpoint.pt")
        result["checkpoint_bytes"] = (out / "checkpoint.pt").stat().st_size
        result["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(out / "attempt.json", result)
    return result


def train_candidate(candidate: str, seed: int, out: Path, device, all_normal_macro_f1: float,
                    inputs: dict | None = None, train_rows=None, val_rows=None,
                    epochs_max: int = 12, pretrained: bool = True) -> dict:
    """Primary 1e-4 attempt plus the single preregistered 1e-5 rescue."""
    inputs = inputs or load_inputs()
    train_seq = Sequences(train_rows or inputs["train"], 16)
    val_seq = Sequences(val_rows or inputs["validation"], 16)
    attempts = []
    for learning_rate, name in ((1e-4, "lr1e-4"), (1e-5, "rescue_lr1e-5")):
        attempt = train_once(candidate, seed, learning_rate, out / name, inputs,
                             train_seq, val_seq, device, epochs_max, pretrained)
        attempts.append(name)
        if attempt["trigger"] is None:
            break
    final = attempt
    status = "optimization_failed" if final["trigger"] else (
        "valid" if is_valid(final["selected"], all_normal_macro_f1) else "invalid_validity_gate")
    run = {"candidate": candidate, "seed": seed, "k": 16, "status": status,
           "attempts": attempts, "rescued": len(attempts) == 2,
           "selected_attempt": attempts[-1], "selected": final["selected"],
           "checkpoint": _repo_path(out / attempts[-1] / "checkpoint.pt")
           if final["trigger"] is None else None,
           "checkpoint_sha256": final.get("checkpoint_sha256"),
           "checkpoint_bytes": final.get("checkpoint_bytes"),
           "parameters": final.get("parameters"),
           "all_normal_macro_f1": all_normal_macro_f1,
           "environment": environment_record()}
    write_json(out / "run.json", run)
    return run


def _repo_path(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load_checkpoint(run_dir: Path, device, fused_attn: bool = False):
    import torch
    from .f002_flops import build_sequence_model

    run = json.loads((run_dir / "run.json").read_text())
    require_frozen_f002b(run, run_dir)
    path = ROOT / run["checkpoint"]
    if sha256_file(path) != run["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash changed: {path}")
    model = build_sequence_model(run["candidate"], pretrained=False, fused_attn=fused_attn)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return run, model.to(device).eval()


def require_frozen_f002b(run: dict, run_dir: Path) -> None:
    """Refuse to consume a run that is not the accepted frozen F002b reference."""
    if Path(run_dir).parent.name != "f002b":
        return
    if (Path(run_dir).name != "mobilenet_v2_seed2026" or
            run.get("status") != "valid" or run.get("candidate") != "mobilenet_v2" or
            run.get("seed") != 2026 or run.get("backbone_decision") != "D025" or
            not run.get("checkpoint_sha256") or not run.get("checkpoint")):
        raise ValueError("F002b requires valid MobileNetV2 seed 2026, D025 and a checkpoint")
    checkpoint = Path(run["checkpoint"]).as_posix()
    if not checkpoint.startswith("artifacts/F002/f002b/mobilenet_v2_seed2026/"):
        raise ValueError("F002b checkpoint is outside the frozen run")
    if sha256_file(ROOT / checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("F002b checkpoint hash differs from run.json")


# ---------------------------------------------------------------- benchmark

def _timed_requests(run_once, requests, warmup, sync) -> list[dict]:
    """5 repeats of: 50 warm-up requests, then 200 synchronized timed requests."""
    repeats = []
    for _ in range(5):
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
        repeats.append({"latencies_ms": latencies, "wall_seconds": time.perf_counter() - start})
    return repeats


def _percentile(values, q):
    import numpy
    return float(numpy.percentile(values, q))


def summarize_timing(repeats: list[dict], k: int) -> dict:
    import statistics
    medians = [statistics.median(r["latencies_ms"]) for r in repeats]
    p95s = [_percentile(r["latencies_ms"], 95) for r in repeats]
    seq_rates = [len(r["latencies_ms"]) / r["wall_seconds"] for r in repeats]
    return {"median_latency_ms": statistics.median(medians),
            "p95_latency_ms": statistics.median(p95s),
            "sequence_throughput_s": statistics.median(seq_rates),
            "frame_throughput_s": statistics.median([rate * k for rate in seq_rates]),
            "repeat_median_latency_ms": medians, "repeat_p95_latency_ms": p95s,
            "repeat_sequence_throughput_s": seq_rates}


def benchmark_checkpoint(run_dir: Path, k: int, device) -> dict:
    """§9 model-only (primary) and JPEG pipeline timing on fixed requests."""
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Primary D024 benchmark requires CUDA; CPU timing is not DAS-5 evidence")
    set_benchmark_flags()
    env = environment_record()
    if (not env.get("slurm_job_id") or not env.get("slurm_node") or
            not env.get("node_features") or not env.get("gpu_model")):
        raise ValueError("Primary D024 benchmark requires DAS-5 SLURM GPU/node provenance")
    run, model = load_checkpoint(run_dir, device)
    inputs = load_inputs()
    request_ids = benchmark_request_ids(inputs["validation"])
    by_key = {key(row): row for row in inputs["validation"]}
    sequences = Sequences([by_key[node] for node in request_ids["timed"]], k)
    tensors = [sequences.frames(i).unsqueeze(0).to(device) for i in range(len(sequences))]
    sync = lambda: torch.cuda.synchronize(device)  # noqa: E731

    def model_only(index):
        with torch.inference_mode():
            model(tensors[index])

    def pipeline(index):
        with torch.inference_mode():
            frames = sequences.frames(index).unsqueeze(0).to(device)
            model(frames)

    timed, warmup = list(range(200)), list(range(50))
    paths = {}
    deit = run["candidate"] == "deit_tiny"
    equivalence = None
    if deit:
        from .f002_flops import set_deit_fused_attention, verify_deit_fused_equivalence
        try:
            equivalence = verify_deit_fused_equivalence(model, tensors)
            if (equivalence.get("equivalent") is not True or equivalence.get("requests") != 200 or
                    equivalence.get("k") != k or equivalence.get("tolerance") != 1e-5 or
                    not math.isfinite(equivalence.get("max_abs_logit_difference", float("nan"))) or
                    equivalence["max_abs_logit_difference"] > 1e-5):
                raise ValueError(f"DeiT fused equivalence failed: {equivalence}")
        except Exception as error:
            write_json(run_dir / f"benchmark_equivalence_failure_k{k}.json", {
                "candidate": run["candidate"], "seed": run["seed"], "k": k,
                "checkpoint_sha256": run["checkpoint_sha256"],
                "equivalence": equivalence, "error": str(error)})
            raise ValueError("DeiT fused equivalence failed; primary benchmark stopped") from error
        finally:
            set_deit_fused_attention(model, False)
    attention_paths = (["fused", "unfused"] if deit else [None])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for attention in attention_paths:
        if deit:
            set_deit_fused_attention(model, attention == "fused")
        label = f"model_only_{attention}" if attention else "model_only"
        repeats = _timed_requests(model_only, timed, warmup, sync)
        paths[label] = summarize_timing(repeats, k) | {"raw": repeats}
    for path in sequences.selected:  # untimed warm-cache read/decode pass
        for frame in path[1]:
            preprocess_frame(frame)
    attention = "fused" if deit else None
    if deit:
        set_deit_fused_attention(model, attention == "fused")
    label = f"pipeline_{attention}" if attention else "pipeline"
    repeats = _timed_requests(pipeline, timed, warmup, sync)
    paths[label] = summarize_timing(repeats, k) | {"raw": repeats}
    if deit:
        set_deit_fused_attention(model, False)
    primary = "model_only_fused" if deit else "model_only"
    return {"candidate": run["candidate"], "seed": run["seed"], "stage": run_dir.parent.name,
            "k": k,
            "checkpoint_sha256": run["checkpoint_sha256"],
            "request_ids": [f"{e}/{z}" for e, z in request_ids["timed"]],
            "warmup_ids": [f"{e}/{z}" for e, z in request_ids["warmup"]],
            "warmup_per_repeat": 50, "timed_requests": 200, "repeats": 5,
            "primary_path": primary, "fused_runtime_valid": equivalence["equivalent"] if deit else None,
            "deit_equivalence": equivalence, "paths": paths,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "precision": "float32", "execution_device": str(device),
            "batch_semantics": "one_sequence_request_K_frames_internal_batch",
            "transfer_boundary": "model_only: tensors already on device; pipeline: JPEG path list to device output",
            "timing_api": "time.perf_counter with torch.cuda.synchronize after each request",
            "benchmark_flags": dict(PRIMARY_BENCHMARK_FLAGS),
            "benchmark_platform": "DAS-5", "environment": env}


# ---------------------------------------------------------------- gates

def validate_benchmark_evidence(bench: dict, run: dict, *, stage: str, candidate: str,
                                seed: int, k: int, request_ids: dict,
                                hardware: tuple[str, str] | None = None) -> tuple[str, str]:
    """Check a saved primary timing artifact before any D024 comparison gate.

    Older recorded artifacts omit ``stage``. Their selected checkpoint path
    supplies it, while the checkpoint SHA binds the benchmark to that run.
    """
    checkpoint = run.get("checkpoint")
    prefix = f"artifacts/F002/{stage}/{candidate}_seed{seed}/"
    if (run.get("candidate") != candidate or run.get("seed") != seed or
            run.get("status") not in ("valid", "invalid_validity_gate") or
            not isinstance(checkpoint, str) or not checkpoint.startswith(prefix) or
            not run.get("checkpoint_sha256")):
        raise ValueError("Benchmark run identity or selected checkpoint is invalid")
    if stage == "f002b" and (run.get("status") != "valid" or
                             candidate != "mobilenet_v2" or seed != 2026 or
                             run.get("backbone_decision") != "D025"):
        raise ValueError("Benchmark does not belong to valid frozen F002b")
    if (bench.get("candidate") != candidate or bench.get("seed") != seed or
            bench.get("k") != k or bench.get("stage", stage) != stage or
            bench.get("checkpoint_sha256") != run["checkpoint_sha256"]):
        raise ValueError("Benchmark candidate/seed/stage/K/checkpoint mismatch")
    execution_device = bench.get("execution_device", "")
    if (bench.get("benchmark_platform") != "DAS-5" or
            not (execution_device == "cuda" or
                 (isinstance(execution_device, str) and execution_device.startswith("cuda:") and
                  execution_device[5:].isdigit()))):
        raise ValueError("Primary benchmark must be DAS-5 CUDA evidence")
    if bench.get("benchmark_flags") != PRIMARY_BENCHMARK_FLAGS:
        raise ValueError("Benchmark flags differ from D024")
    timed = [f"{exam}/{zone}" for exam, zone in request_ids["timed"]]
    warmup = [f"{exam}/{zone}" for exam, zone in request_ids["warmup"]]
    if (len(timed) != 200 or len(warmup) != 50 or warmup != timed[:50] or
            bench.get("request_ids") != timed or bench.get("warmup_ids") != warmup or
            bench.get("timed_requests") != 200 or bench.get("warmup_per_repeat") != 50 or
            bench.get("repeats") != 5):
        raise ValueError("Benchmark request IDs, order or repetitions differ from D024")
    env = bench.get("environment")
    if not isinstance(env, dict):
        raise ValueError("Benchmark environment provenance is missing")
    pair = (env.get("gpu_model"), env.get("node_features"))
    if (not all(pair) or not env.get("hostname") or not env.get("slurm_job_id") or
            not env.get("slurm_node") or (hardware is not None and pair != hardware)):
        raise ValueError("Benchmark GPU model or node class differs")
    expected_primary = "model_only_fused" if candidate == "deit_tiny" else "model_only"
    primary_name = bench.get("primary_path")
    if primary_name != expected_primary or not isinstance(bench.get("paths"), dict):
        raise ValueError("Benchmark primary path is invalid")
    primary = bench["paths"].get(primary_name)
    required = ("median_latency_ms", "p95_latency_ms", "sequence_throughput_s",
                "frame_throughput_s")
    if (not isinstance(primary, dict) or any(
            not isinstance(primary.get(field), (int, float)) or
            not math.isfinite(primary[field]) or primary[field] <= 0 for field in required)):
        raise ValueError("Benchmark primary path has invalid timings")
    raw = primary.get("raw")
    if (not isinstance(raw, list) or len(raw) != 5 or any(
            not isinstance(repeat, dict) or
            not isinstance(repeat.get("latencies_ms"), list) or
            len(repeat["latencies_ms"]) != 200 or
            not isinstance(repeat.get("wall_seconds"), (int, float)) or
            not math.isfinite(repeat["wall_seconds"]) or repeat["wall_seconds"] <= 0 or
            any(not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0
                for ms in repeat["latencies_ms"]) for repeat in raw)):
        raise ValueError("Benchmark primary path lacks five valid 200-request repeats")
    if candidate == "deit_tiny":
        equivalence = bench.get("deit_equivalence") or {}
        if (bench.get("fused_runtime_valid") is not True or
                equivalence.get("equivalent") is not True or equivalence.get("requests") != 200 or
                equivalence.get("k") != k or equivalence.get("tolerance") != 1e-5 or
                not isinstance(equivalence.get("max_abs_logit_difference"), (int, float)) or
                not math.isfinite(equivalence["max_abs_logit_difference"]) or
                equivalence["max_abs_logit_difference"] > 1e-5):
            raise ValueError("DeiT fused primary path lacks accepted equivalence")
    return pair

def k16_gate_row(run: dict, bench: dict, flops: dict) -> dict:
    primary = bench["paths"][bench["primary_path"]]
    env = bench["environment"]
    return {"id": run["candidate"], "macro_f1": run["selected"]["macro_f1"],
            "accuracy": run["selected"]["accuracy"],
            "normal_recall": run["selected"]["normal_recall"],
            "abnormal_recall": run["selected"]["abnormal_recall"],
            "flops": flops["flops"],
            "model_only_median_latency_ms": primary["median_latency_ms"],
            "model_only_sequence_throughput_s": primary["sequence_throughput_s"],
            "benchmark_platform": bench["benchmark_platform"],
            "execution_device": bench["execution_device"],
            "gpu_model": env.get("gpu_model"), "node_class": env.get("node_features"),
            "hostname": env["hostname"], "benchmark_flags": bench["benchmark_flags"]}


def final_dominance(by_candidate: dict[str, dict[int, dict]]) -> str | None:
    """§6 two-finalist rule over K4/8/16; returns the automatic winner or None.

    Each evidence row needs macro_f1, flops and the primary model-only
    latency/throughput. A wins iff at every K its F1 is no worse and FLOPs no
    higher, at least one comparison is strict, and B has no >10% model-only
    runtime advantage at any K.
    """
    if len(by_candidate) == 1:
        return next(iter(by_candidate))
    if len(by_candidate) != 2:
        raise ValueError("Final dominance compares one or two finalists")
    (a_id, a), (b_id, b) = sorted(by_candidate.items())
    if set(a) != {4, 8, 16} or set(b) != {4, 8, 16}:
        raise ValueError("Final dominance needs K4/K8/K16 evidence for both")

    def dominates(x, y):
        no_worse = all(x[k]["macro_f1"] >= y[k]["macro_f1"] and x[k]["flops"] <= y[k]["flops"]
                       for k in (4, 8, 16))
        strict = any(x[k]["macro_f1"] > y[k]["macro_f1"] or x[k]["flops"] < y[k]["flops"]
                     for k in (4, 8, 16))
        contradiction = any(_runtime_advantage(y[k], x[k]) for k in (4, 8, 16))
        return no_worse and strict and not contradiction

    if dominates(a, b):
        return a_id
    if dominates(b, a):
        return b_id
    return None
