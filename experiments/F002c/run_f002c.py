"""One exploratory TinyViT run; the frozen F002 implementation is imported only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, key, optimization_loss
from experiments.F002c.model import (MODEL_NAME, PRETRAINED_REVISION, F002cSequences, SequenceDataset,
                                     TinyViTSequence, build_optimizer)

ART = ROOT / "artifacts" / "F002c"
CONFIG = {
    "experiment": "F002c", "model": MODEL_NAME, "seed": 2026, "k": 16,
    "pretrained_revision": PRETRAINED_REVISION,
    "max_epochs": 20, "min_epochs": 6, "patience": 5, "warmup_epochs": 2,
    "backbone_lr": 5e-5, "head_lr": 5e-4, "min_lr_ratio": 0.05,
    "weight_decay": 0.01, "gradient_clip_norm": 1.0,
    "microbatch_sequences": 2, "effective_batch_sequences": 8,
    "precision": "float32", "batchnorm_running_stats": "frozen",
    "drop_path_rate": 0.1, "head_dropout": 0.1,
    "preprocess": "full-square RGB 224 bicubic; ImageNet mean/std",
    "augmentation": "train-only sequence-consistent horizontal flip p=0.5",
    "loss": "BCEWithLogits; duplicate weights; fixed step denominator 8; no class weights",
    "selection": "validation duplicate-aware macro-F1 at 0.5, then BCE, then earliest",
    "threshold": 0.5, "workers": 2,
    "automatic_k_response_min_f1": 0.84,
    "automatic_k_response_min_abnormal_recall": 0.65,
    "rescue": "none automatic; at most one separately documented optimization rescue",
}


def output_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Run name must contain only letters, digits, underscores or hyphens")
    root = ART.resolve()
    result = root / name
    if result.resolve().parent != root:
        raise ValueError("Output must remain inside artifacts/F002c")
    return result


def frozen_snapshot() -> dict:
    paths = [ROOT / "configs/f002_protocol.json", ROOT / "docs/f002_methodology.md",
             ROOT / "docs/decision_log.md", ROOT / "data/openpocus/labels.csv",
             ROOT / "data/openpocus/split_groups.csv"]
    for directory, pattern in (("src", "f002_*.py"), ("experiments/F002", "*"),
                               ("data/openpocus/t006", "*")):
        paths += [p for p in (ROOT / directory).glob(pattern) if p.is_file()]
    baseline = ROOT / "artifacts/F002/f002b/mobilenet_v2_seed2026/lr1e-4/checkpoint.pt"
    if baseline.exists():
        paths.append(baseline)
    result = {p.relative_to(ROOT).as_posix(): pipe.sha256_file(p) for p in paths}
    if baseline.exists() and result[baseline.relative_to(ROOT).as_posix()] != (
            "5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823"):
        raise ValueError("Frozen F002b checkpoint differs from the recorded reference")
    return result


def environment() -> dict:
    record = pipe.environment_record()
    names = ("experiments/F002c/run_f002c.py", "experiments/F002c/model.py",
             "experiments/F002c/das5_f002c.sbatch", "data/openpocus/cohort_loader.py")
    record["f002c_source_sha256"] = {
        name: pipe.sha256_file(ROOT / name) for name in names if (ROOT / name).exists()}
    record["config_sha256"] = hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    return record


def loader(sequences, *, weights=None, shuffle=False, workers=0, generator=None):
    return DataLoader(SequenceDataset(sequences, weights),
                      batch_size=CONFIG["microbatch_sequences"] if shuffle else 1,
                      shuffle=shuffle, generator=generator, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def evaluate(model, sequences, unit_of, device, workers=0) -> dict:
    if not sequences.rows or any(r["partition"] != "validation" for r in sequences.rows):
        raise ValueError("F002c inference is validation-only")
    if sequences.augment:
        raise ValueError("Validation must not use augmentation")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for frames, targets, _, indices in loader(sequences, workers=workers):
            logits = model(frames.to(device, non_blocking=True)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite validation logit")
            for index, logit, prob in zip(indices.tolist(), logits.tolist(),
                                           logits.sigmoid().tolist()):
                row = sequences.rows[index]
                predictions.append({"exam_id": row["exam_id"], "zone_id": row["zone_id"],
                    "binary_target": int(row["binary_target"]), "split_group": row["split_group"],
                    "device": row["device"], "unit_id": unit_of[key(row)], "k": sequences.k,
                    "logit": logit, "probability": prob, **sequences.identity(index)})
    return aggregate_validation(sequences.rows, predictions, unit_of, threshold=0.5)


def lr_factor(step: int, steps_per_epoch: int) -> float:
    warmup = CONFIG["warmup_epochs"] * steps_per_epoch
    total = CONFIG["max_epochs"] * steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    floor = CONFIG["min_lr_ratio"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def memory_record(device) -> dict:
    if device.type != "cuda":
        return {"device": str(device), "gpu_memory_verified": False}
    return {"device": str(device),
            "gpu_memory_verified": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20}


def smoke(device, out: Path, pretrained=True) -> dict:
    """Real train backward + full validation path inventory; no model selection."""
    out.mkdir(parents=True, exist_ok=False)
    before = frozen_snapshot()
    pipe.set_training_determinism(CONFIG["seed"])
    inputs = pipe.load_inputs()  # Test CSV is hashed/verified, never passed to an image loader.
    print("Preflight: verifying train and validation frame inventories", flush=True)
    train_seq = F002cSequences(inputs["train"], 16, augment=True)
    val_seq = F002cSequences(inputs["validation"], 16)
    # Both labels, two sequences, exactly the intended training microbatch.
    indices = [next(i for i, r in enumerate(train_seq.rows) if int(r["binary_target"]) == y)
               for y in (0, 1)]
    model = TinyViTSequence(pretrained=pretrained).to(device)
    print("Preflight: two K16 forward/backward/Adam steps", flush=True)
    optimizer = build_optimizer(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    frames = torch.stack([train_seq.frames(i) for i in indices]).to(device)
    targets = torch.tensor([0., 1.], device=device)
    weights = torch.tensor([inputs["train_weights"][key(train_seq.rows[i])] for i in indices],
                           device=device)
    for _ in range(2):
        # The second step measures backward with Adam moments already allocated.
        logits = model(frames)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        optimization_loss(losses, weights).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    memory = memory_record(device)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Nonfinite smoke loss")
    # All 265 predictions use synthetic probabilities to verify 254-unit aggregation.
    aggregation = pipe.all_normal_reference(inputs)
    if len(aggregation["rows"]) != 265 or len(aggregation["units"]) != 254:
        raise ValueError("Validation evaluation units changed")
    small = F002cSequences(inputs["validation"][:2], 16)
    evaluation = evaluate(model, small, inputs["unit_of"], device)
    pipe.write_predictions(out / "validation_smoke.csv", evaluation)
    result = {"status": "passed", "pretrained": pretrained, "config": CONFIG,
              "environment": environment(), "memory": memory,
              "two_forward_backward_steps_seconds": backward_seconds,
              "gradient_norm_before_clip": float(grad_norm), "losses": losses.tolist(),
              "parameters": sum(p.numel() for p in model.parameters()),
              "train_sequences_verified": len(train_seq), "validation_sequences_verified": len(val_seq),
              "validation_units": len(aggregation["units"]),
              "test_image_inference": False, "test_manifest_metadata_verified": True,
              "frozen_inputs_sha256": before}
    if before != frozen_snapshot():
        raise RuntimeError("Frozen inputs changed during smoke")
    pipe.write_json(out / "smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_selected(out: Path, device):
    run = json.loads((out / "run.json").read_text())
    if run["status"] != "completed" or run["config"] != CONFIG:
        raise ValueError("Require a completed F002c run with the same configuration")
    checkpoint = out / "checkpoint.pt"
    if pipe.sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    if run["frozen_inputs_sha256"] != frozen_snapshot():
        raise ValueError("Frozen inputs differ from training")
    if run["environment"]["f002c_source_sha256"] != environment()["f002c_source_sha256"]:
        raise ValueError("F002c implementation differs from training")
    model = TinyViTSequence(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, run


def k_response(out: Path, device, ks=(4, 8, 16), workers=0) -> dict:
    model, run = load_selected(out, device)
    inputs = pipe.load_inputs()
    results = {}
    for k in ks:
        sequences = F002cSequences(inputs["validation"], k)
        evaluation = evaluate(model, sequences, inputs["unit_of"], device, workers)
        pipe.write_predictions(out / "k_response" / f"validation_k{k}.csv", evaluation)
        results[str(k)] = {"duplicate_aware": evaluation["duplicate_aware"],
                           "row_level": evaluation["row_level"]}
        print(json.dumps({"k": k, **evaluation["duplicate_aware"]}), flush=True)
    record = {"checkpoint_sha256": run["checkpoint_sha256"], "threshold": 0.5,
              "results": results, "retrained": False, "test_image_inference": False,
              "environment": environment()}
    pipe.write_json(out / "k_response.json", record)
    return record


def train(device, out: Path, workers=2):
    if device.type != "cuda":
        raise ValueError("The full evening run requires a CUDA GPU; use smoke for CPU checks")
    # Same-device smoke is mandatory and separate from the fresh training initialization.
    smoke_out = output_dir(out.name + "_preflight")
    if smoke_out.exists():
        evidence = json.loads((smoke_out / "smoke.json").read_text())
        if (evidence["status"] != "passed" or not evidence["pretrained"] or
                evidence["config"] != CONFIG or
                not evidence["memory"].get("gpu_memory_verified", False) or
                not evidence["memory"]["device"].startswith("cuda") or
                evidence["frozen_inputs_sha256"] != frozen_snapshot() or
                evidence["environment"]["gpu_model"] != torch.cuda.get_device_name(device) or
                evidence["environment"]["f002c_source_sha256"] != environment()["f002c_source_sha256"]):
            raise ValueError("Existing preflight does not match this training execution")
    else:
        smoke(device, smoke_out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    before = frozen_snapshot()
    run = {"status": "running", "config": CONFIG, "environment": environment(),
           "workers": workers, "frozen_inputs_sha256": before, "history": [],
           "test_image_inference": False, "test_manifest_metadata_verified": True}
    pipe.write_json(out / "run.json", run)
    try:
        pipe.set_training_determinism(CONFIG["seed"])
        inputs = pipe.load_inputs()
        train_seq = F002cSequences(inputs["train"], 16, augment=True)
        val_seq = F002cSequences(inputs["validation"], 16)
        model = TinyViTSequence(pretrained=True).to(device)
        # Save the exact loaded pretrained state for offline reconstruction, including new head.
        torch.save(model.state_dict(), out / "initial_state.pt")
        run["initial_state_sha256"] = pipe.sha256_file(out / "initial_state.pt")
        run["pretrained_cfg"] = model.encoder.pretrained_cfg
        optimizer = build_optimizer(model)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        gen = torch.Generator().manual_seed(CONFIG["seed"])
        batches = loader(train_seq, weights=inputs["train_weights"], shuffle=True,
                         workers=workers, generator=gen)
        steps_per_epoch = math.ceil(len(batches) / 4)
        best, stale, update = None, 0, 0
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, CONFIG["max_epochs"] + 1):
            epoch_started = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            weighted_sum, weight_sum, max_grad_norm = 0., 0., 0.
            for batch_index, (frames, targets, weights, _) in enumerate(batches):
                if batch_index % 4 == 0:
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr * lr_factor(update, steps_per_epoch)
                frames, targets, weights = (t.to(device, non_blocking=True)
                                             for t in (frames, targets, weights))
                logits = model(frames)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets,
                                                                               reduction="none")
                if not torch.isfinite(losses).all():
                    raise FloatingPointError(f"Nonfinite train loss in epoch {epoch}")
                optimization_loss(losses, weights).backward()
                weighted_sum += float((losses.detach() * weights).sum())
                weight_sum += float(weights.sum())
                if (batch_index + 1) % 4 == 0 or batch_index + 1 == len(batches):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                         error_if_nonfinite=True)
                    max_grad_norm = max(max_grad_norm, float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
            evaluation = evaluate(model, val_seq, inputs["unit_of"], device, workers)
            metrics = evaluation["duplicate_aware"]
            predicted_abnormal = (metrics["tp"] + metrics["fp"]) / len(evaluation["units"])
            entry = {"epoch": epoch, "train_loss": weighted_sum / weight_sum,
                     "validation": metrics, "validation_row_level": evaluation["row_level"],
                     "predicted_abnormal_fraction": predicted_abnormal,
                     "max_gradient_norm_before_clip": max_grad_norm,
                     "learning_rates": [g["lr"] for g in optimizer.param_groups],
                     "seconds": time.time() - epoch_started}
            run["history"].append(entry)
            pipe.write_predictions(out / "epochs" / f"validation_epoch{epoch:02d}.csv", evaluation)
            if pipe.checkpoint_better(metrics, best):
                best, stale = {**metrics, "epoch": epoch}, 0
                torch.save(model.state_dict(), out / "checkpoint.pt")
                pipe.write_predictions(out / "validation_selected.csv", evaluation)
            else:
                stale += 1
            run.update(selected=best, elapsed_seconds=time.time() - started, memory=memory_record(device))
            pipe.write_json(out / "run.json", run)
            print(json.dumps(entry), flush=True)
            if epoch >= CONFIG["min_epochs"] and stale >= CONFIG["patience"]:
                break
        if before != frozen_snapshot():
            raise RuntimeError("Frozen inputs changed during training")
        run.update(status="completed", checkpoint_sha256=pipe.sha256_file(out / "checkpoint.pt"),
                   checkpoint_bytes=(out / "checkpoint.pt").stat().st_size,
                   parameters=sum(p.numel() for p in model.parameters()),
                   training_flags=pipe.training_flags(), elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
        raise
    if (best["macro_f1"] >= CONFIG["automatic_k_response_min_f1"] and
            best["abnormal_recall"] >= CONFIG["automatic_k_response_min_abnormal_recall"]):
        # Release training state before loading the selected checkpoint.
        del model, optimizer, batches, frames, logits, losses
        torch.cuda.empty_cache()
        k_response(out, device, workers=workers)
    print(json.dumps({"status": run["status"], "selected": best,
                      "checkpoint_sha256": run["checkpoint_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-weights", "smoke", "train", "evaluate"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run", default="tinyvit5m_seed2026")
    parser.add_argument("--workers", type=int, default=CONFIG["workers"])
    parser.add_argument("--random-init", action="store_true", help="Smoke only; never a training fallback")
    args = parser.parse_args()
    if args.random_init and args.command != "smoke":
        parser.error("--random-init is only allowed for a construction smoke")
    if not 0 <= args.workers <= 4:
        parser.error("--workers must be between 0 and 4")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.command == "prepare-weights":
        model = TinyViTSequence(pretrained=True)
        print(json.dumps({"model": MODEL_NAME, "pretrained_cfg": model.encoder.pretrained_cfg,
                          "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    out = output_dir(args.run)
    if args.command == "smoke":
        smoke(device, out, pretrained=not args.random_init)
    elif args.command == "train":
        if out.exists():
            parser.error("Run directory already exists; refusing to overwrite or implicitly resume")
        train(device, out, args.workers)
    else:
        k_response(out, device, workers=args.workers)


if __name__ == "__main__":
    main()
