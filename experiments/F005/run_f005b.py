"""F005b (T011) commands: nested structured pruning of the F004 family with distillation.

cache (train, validation) → importance → train (nested and single-width
control, three seeds each) → evaluate → benchmark. Writes to ignored
artifacts/F005/. Train rows only for gradients, importance and BN calibration;
validation for epoch selection and reporting; no command builds a loader for the
T006 test partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
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
from torch import nn
from torch.utils.data import DataLoader

from src import f002_pipeline as pipe
from src.f002_protocol import aggregate_validation, key, optimization_loss
from experiments.F002c.model import TinyViTSequence, build_optimizer
from experiments.F003.adaptive import CHECKPOINT, CHECKPOINT_SHA256, count_flops, positions_in_k16
from experiments.F003.run_f003 import auroc, auroc_ci, bootstrap, mcnemar, paired_delta
from experiments.F004.family import OPERATING_POINTS, AdaptiveTinyViT, load_family
from experiments.F004.run_f004 import TRAIN_KS, per_k_logits, prediction
from experiments.F005 import cache
from experiments.F005.prune import (WIDTHS, groups, hidden_size, l1_scores, make_nested, physical,
                                    recalibrate, set_width, TaylorRecorder)

ART = ROOT / "artifacts" / "F005"
F004_ART = ROOT / "artifacts" / "F004"
SOURCES = ("experiments/F005/prune.py", "experiments/F005/cache.py", "experiments/F005/run_f005b.py",
           "experiments/F004/family.py", "experiments/F004/run_f004.py", "experiments/F002c/model.py")
SEEDS = (2026, 2027, 2028)

# Fixed before the importance and training jobs ran.
IMPORTANCE_CONFIG = {
    "train_rows_shuffle_seed": 2026, "taylor_rows": 256, "calibration_rows": 32, "zero_shot_rows": 300,
    "criteria": ["l1", "taylor"], "zero_shot_k": 16, "zero_shot_widths": [0.75, 0.5, 0.25],
    "rule": "choose the criterion with the lower mean duplicate-weighted train BCE over the zero-shot "
            "widths at K16 on the zero-shot rows (disjoint from Taylor and calibration rows)",
    "sensitivity": "each stage alone at width 0.5, all others at 1.0; descriptive only",
}
TRAIN_CONFIG = {
    "experiment": "F005b", "init": "D026 checkpoint, channels reordered by the chosen importance",
    "init_sha256": CHECKPOINT_SHA256, "teacher": "frozen D026 (F004 family), same frames and flips",
    "max_epochs": 10, "min_epochs": 3, "patience": 3, "warmup_epochs": 1,
    "backbone_lr": 2.5e-5, "head_lr": 2.5e-4, "min_lr_ratio": 0.05, "weight_decay": 0.01,
    "gradient_clip_norm": 1.0, "microbatch_sequences": 2, "effective_batch_sequences": 8,
    "precision": "float32", "batchnorm": "frozen; conv3 BN statistics per width, recalibrated on train "
                                         "calibration rows before training and before every validation",
    "drop_path_rate": 0.1, "head_dropout": 0.1, "augmentation": "train-only sequence flip p=0.5",
    "loss_ks": list(TRAIN_KS), "kd_alpha": 0.5, "kd_temperature": 2.0,
    "loss": "per width: mean over loss_ks of (1-α)·BCE(label) + α·τ²·BCE(σ(student/τ), σ(teacher/τ))",
    "modes": {"nested": "sandwich rule: widths 1.0 and 0.25 every step plus one of 0.75/0.5 at random "
                        "(seeded); loss averaged over the step's widths",
              "single": "control: width 0.5 only, same loss"},
    "selection": {"nested": "validation macro-F1 at 0.5 averaged over the 4 widths × 5 loss_ks",
                  "single": "validation macro-F1 at 0.5 averaged over the 5 loss_ks at width 0.5",
                  "ties": "lower mean BCE, then earlier epoch; epoch 0 not selectable"},
    "workers": 2,
}
# Added after the main F005b results, to answer whether the importance order matters
# after recovery: the same nested training with one random channel order per group
# (generator seed 2026, the same order for every training seed).
RANDOM_ORDER_CONTROL = {"mode": "nested_random", "order": "torch.randperm per group, generator seed 2026",
                        "training": "identical to nested (TRAIN_CONFIG)"}
# Added after the main F005b results: recalibrate every BN downstream of a pruned
# layer per width (moment matching, as for conv3), post hoc on the trained seeds.
FULL_BN_RULE = {"text": "Adopt full per-width BN recalibration if the three-seed mean macro-F1 over widths "
                        "0.75/0.5/0.25 × the five loss K rises by at least 0.005; width 1.0 is unchanged by "
                        "construction.", "min_gain": 0.005}
ADOPTION_RULE = {
    "text": "Propose the nested model as T011's compressed family if, averaged over the three seeds, "
            "(a) its width-1.0 mean macro-F1 over the five loss K is at least D026's minus 0.01 and "
            "(b) its width-0.5 mean (29% fewer FLOPs per frame) is at least D026's minus 0.02. "
            "The full width × K grid is reported either way.",
    "max_loss_width_1": 0.01, "max_loss_width_05": 0.02,
}


def environment() -> dict:
    record = pipe.environment_record()
    record["f005b_source_sha256"] = {s: pipe.sha256_file(ROOT / s) for s in SOURCES}
    return record


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cache_path(partition: str) -> Path:
    return ART / "cache" / f"{partition}_k16_uint8.pt"


def train_subsets(rows: list[dict]) -> dict[str, list[int]]:
    """Disjoint train-row index sets for Taylor scores, BN calibration and zero-shot checks."""
    order = list(range(len(rows)))
    random.Random(IMPORTANCE_CONFIG["train_rows_shuffle_seed"]).shuffle(order)
    a, c, z = (IMPORTANCE_CONFIG[k] for k in ("taylor_rows", "calibration_rows", "zero_shot_rows"))
    return {"taylor": order[:a], "calibration": order[a:a + c], "zero_shot": order[a + c:a + c + z]}


def random_orders() -> dict[str, torch.Tensor]:
    """One random channel order per prunable group (RANDOM_ORDER_CONTROL)."""
    generator = torch.Generator().manual_seed(2026)
    return {name: torch.randperm(hidden_size(m), generator=generator)
            for name, m in groups(load_family().encoder)}


def calibration_batches(frames: torch.Tensor, indices: list[int], device) -> list[torch.Tensor]:
    return [cache.normalize(frames[i]).to(device) for i in indices]


def positions_of(rows: list[dict], ks=OPERATING_POINTS) -> list[dict]:
    return [{k: positions_in_k16(int(r["frame_count"]), k) for k in ks} for r in rows]


def load_student(orders: dict, device) -> TinyViTSequence:
    if pipe.sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("D026 checkpoint hash differs from the frozen record")
    model = TinyViTSequence(pretrained=False, drop_path_rate=TRAIN_CONFIG["drop_path_rate"],
                            head_dropout=TRAIN_CONFIG["head_dropout"])
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    make_nested(model.encoder, orders)
    return model.to(device)


# ---------------------------------------------------------------- cache

def cmd_cache(args) -> None:
    out = cache_path(args.partition)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rows = pipe.load_inputs()[args.partition]
    started = time.time()
    record = cache.build(rows) | {"partition": args.partition, "environment": environment()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out)
    print(f"wrote {out}: {tuple(record['frames'].shape)} in {time.time() - started:.0f} s")


# ---------------------------------------------------------------- importance

@torch.no_grad()
def train_bce(model, frames, rows, indices, weights, device, k=16) -> float:
    """Duplicate-weighted mean BCE at K on the given train rows."""
    total, weight_sum = 0.0, 0.0
    for i in indices:
        logit = per_k_logits(model, cache.normalize(frames[i]).unsqueeze(0).to(device),
                             [{k: positions_in_k16(int(rows[i]["frame_count"]), k)}], (k,))[k]
        target = torch.tensor([float(rows[i]["binary_target"])], device=device)
        w = weights[key(rows[i])]
        total += w * float(nn.functional.binary_cross_entropy_with_logits(logit, target))
        weight_sum += w
    return total / weight_sum


def cmd_importance(args) -> None:
    """Channel orders by L1 and Taylor; zero-shot choice between them; stage sensitivity."""
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    rows, weights = inputs["train"], inputs["train_weights"]
    frames = cache.load(cache_path("train"), rows)
    subsets = train_subsets(rows)
    calibration = calibration_batches(frames, subsets["calibration"], device)
    family = load_family(device=device)
    orders = {"l1": {n: torch.argsort(l1_scores(m), descending=True) for n, m in groups(family.encoder)}}
    # Taylor: gradients of the D026 K16 BCE with respect to the hidden activations.
    model = load_family(device=device)
    recorder = TaylorRecorder(model.encoder)
    caches = [m for m in model.modules() if hasattr(m, "attention_bias_cache")]
    for i in subsets["taylor"]:
        for module in caches:  # timm caches attention biases in eval mode, graph included
            module.attention_bias_cache = {}
        x = cache.normalize(frames[i]).unsqueeze(0).to(device)
        logit = per_k_logits(model, x, [{16: list(range(16))}], (16,))[16]
        target = torch.tensor([float(rows[i]["binary_target"])], device=device)
        model.zero_grad(set_to_none=True)
        nn.functional.binary_cross_entropy_with_logits(logit, target).backward()
    recorder.remove()
    orders["taylor"] = {n: torch.argsort(s, descending=True) for n, s in recorder.scores.items()}
    zero_shot, baseline = {}, train_bce(family, frames, rows, subsets["zero_shot"], weights, device)
    for criterion, order in orders.items():
        model = load_family(device=device)
        make_nested(model.encoder, order)
        recalibrate(model.encoder, calibration)
        zero_shot[criterion] = {}
        for width in IMPORTANCE_CONFIG["zero_shot_widths"]:
            set_width(model.encoder, width)
            zero_shot[criterion][str(width)] = train_bce(model, frames, rows, subsets["zero_shot"], weights, device)
        print(criterion, zero_shot[criterion], flush=True)
    chosen = min(orders, key=lambda c: statistics.fmean(zero_shot[c].values()))
    # Stage sensitivity with the chosen order: one stage at width 0.5, the rest full.
    model = load_family(device=device)
    make_nested(model.encoder, orders[chosen])
    recalibrate(model.encoder, calibration)
    sensitivity = {}
    for stage in range(4):
        set_width(model.encoder, 1.0)
        for name, module in groups(model.encoder):
            if name.startswith(f"stages.{stage}."):
                module.active, module.key = round(0.5 * hidden_size(module)), "050"
        sensitivity[str(stage)] = train_bce(model, frames, rows, subsets["zero_shot"], weights, device)
    set_width(model.encoder, 1.0)
    torch.save({"orders": orders, "chosen": chosen}, ART / "importance.pt")
    pipe.write_json(ART / "importance.json", {
        "config": IMPORTANCE_CONFIG, "subsets": subsets, "baseline_train_bce_k16": baseline,
        "zero_shot_train_bce_k16": zero_shot, "chosen": chosen,
        "stage_sensitivity_train_bce_k16_width05": sensitivity, "environment": environment()})
    print(json.dumps({"baseline": baseline, "zero_shot": zero_shot, "chosen": chosen,
                      "sensitivity": sensitivity}, indent=1))


# ---------------------------------------------------------------- train

def kd_loss(student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row (1-α)·BCE(label) + α·τ²·BCE against the teacher's softened probability."""
    alpha, tau = TRAIN_CONFIG["kd_alpha"], TRAIN_CONFIG["kd_temperature"]
    hard = nn.functional.binary_cross_entropy_with_logits(student, target, reduction="none")
    soft = nn.functional.binary_cross_entropy_with_logits(student / tau, torch.sigmoid(teacher / tau),
                                                          reduction="none")
    return (1 - alpha) * hard + alpha * tau ** 2 * soft


@torch.no_grad()
def validate_grid(model, frames, rows, unit_of, device, widths) -> dict:
    """Validation predictions at every (width, K) from one K16 pass per row and width."""
    model.eval()
    positions = positions_of(rows)
    predictions = {(w, k): [] for w in widths for k in OPERATING_POINTS}
    for width in widths:
        set_width(model.encoder, width)
        for i, row in enumerate(rows):
            logits = per_k_logits(model, cache.normalize(frames[i]).unsqueeze(0).to(device), [positions[i]],
                                  OPERATING_POINTS)
            for k, logit in logits.items():
                predictions[(width, k)].append(prediction(row, unit_of, k, float(logit[0]), width=width))
    set_width(model.encoder, 1.0)
    return {cell: aggregate_validation(rows, p, unit_of) for cell, p in predictions.items()}


def summarize(evaluations: dict, widths) -> dict:
    cells = {f"w{w}_k{k}": e["duplicate_aware"] for (w, k), e in evaluations.items()}
    chosen = [f"w{w}_k{k}" for w in widths for k in TRAIN_KS]
    return {"cells": cells, "mean_macro_f1": statistics.fmean(cells[c]["macro_f1"] for c in chosen),
            "mean_loss": statistics.fmean(cells[c]["loss"] for c in chosen)}


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


def run_dir(mode: str, seed: int) -> Path:
    return ART / "train" / f"{mode}_seed{seed}"


def cmd_train(args) -> None:
    device = torch.device(args.device)
    smoke = args.max_batches is not None
    out = (ART / "train_smoke" / f"{args.mode}") if smoke else run_dir(args.mode, args.seed)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if not smoke and (device.type != "cuda" or args.workers < 1):
        raise SystemExit("Full F005b training requires CUDA and --workers >= 1")
    importance = torch.load(ART / "importance.pt", map_location="cpu", weights_only=False)
    if args.mode == "nested_random":
        criterion = "random"
        orders = random_orders()
    else:
        criterion = importance["chosen"]
        orders = importance["orders"][criterion]
    nested_mode = args.mode in ("nested", "nested_random")
    eval_widths = WIDTHS if nested_mode else (0.5,)
    out.mkdir(parents=True)
    started = time.time()
    run = {"status": "running", "mode": args.mode, "seed": args.seed, "config": TRAIN_CONFIG,
           "importance": importance["chosen"], "adoption_rule": ADOPTION_RULE, "history": [],
           "max_batches": args.max_batches, "workers": args.workers, "test_image_inference": False,
           "environment": environment()}
    pipe.write_json(out / "run.json", run)
    try:
        pipe.set_training_determinism(args.seed)
        inputs = pipe.load_inputs()
        train_rows = inputs["train"]
        val_rows = inputs["validation"][:args.max_val_rows] if args.max_val_rows else inputs["validation"]
        train_frames = cache.load(cache_path("train"), train_rows)
        val_frames = cache.load(cache_path("validation"), inputs["validation"])[:len(val_rows)]
        calibration = calibration_batches(train_frames, train_subsets(train_rows)["calibration"], device)
        positions = positions_of(train_rows, TRAIN_KS)
        student = load_student(orders, device)
        teacher = load_family(device=device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        run["recalibration_epoch0"] = recalibrate(student.encoder, calibration)
        epoch0 = summarize(validate_grid(student, val_frames, val_rows, inputs["unit_of"], device, eval_widths),
                           eval_widths)
        run["epoch0"] = epoch0
        print(json.dumps({"epoch": 0, "mean_macro_f1": epoch0["mean_macro_f1"]}), flush=True)
        optimizer = build_optimizer(student, TRAIN_CONFIG["backbone_lr"], TRAIN_CONFIG["head_lr"],
                                    TRAIN_CONFIG["weight_decay"])
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        gen = torch.Generator().manual_seed(args.seed)
        width_rng = random.Random(args.seed)
        dataset = cache.CachedSequences(train_frames, train_rows, inputs["train_weights"], augment=True)
        batches = DataLoader(dataset, batch_size=TRAIN_CONFIG["microbatch_sequences"], shuffle=True,
                             generator=gen, num_workers=args.workers, pin_memory=device.type == "cuda",
                             persistent_workers=args.workers > 0)
        accumulate = TRAIN_CONFIG["effective_batch_sequences"] // TRAIN_CONFIG["microbatch_sequences"]
        limit = len(batches) if args.max_batches is None else min(args.max_batches, len(batches))
        steps_per_epoch = math.ceil(limit / accumulate)
        best, stale, update = None, 0, 0
        for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
            epoch_started = time.time()
            student.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, weight_sum, max_norm = 0.0, 0.0, 0.0
            for batch_index, (frames, targets, weights, indices) in enumerate(batches):
                if batch_index >= limit:
                    break
                if batch_index % accumulate == 0:
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr * lr_factor(update, steps_per_epoch)
                frames = cache.normalize(frames.to(device, non_blocking=True))
                targets, weights = targets.to(device), weights.to(device)
                rows_pos = [positions[i] for i in indices.tolist()]
                with torch.no_grad():
                    teacher_logits = per_k_logits(teacher, frames, rows_pos, TRAIN_KS)
                widths = ([1.0, 0.25, width_rng.choice([0.75, 0.5])] if nested_mode else [0.5])
                for width in widths:
                    set_width(student.encoder, width)
                    logits = per_k_logits(student, frames, rows_pos, TRAIN_KS)
                    losses = torch.stack([kd_loss(logits[k], teacher_logits[k], targets)
                                          for k in TRAIN_KS]).mean(0)
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError(f"Nonfinite loss in epoch {epoch}")
                    (optimization_loss(losses, weights) / len(widths)).backward()
                    loss_sum += float((losses.detach() * weights).sum()) / len(widths)
                set_width(student.encoder, 1.0)
                weight_sum += float(weights.sum())
                if (batch_index + 1) % accumulate == 0 or batch_index + 1 == limit:
                    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), TRAIN_CONFIG["gradient_clip_norm"],
                                                         error_if_nonfinite=True)
                    max_norm = max(max_norm, float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
            recalibration = recalibrate(student.encoder, calibration)
            evaluations = validate_grid(student, val_frames, val_rows, inputs["unit_of"], device, eval_widths)
            entry = {"epoch": epoch, **summarize(evaluations, eval_widths), "train_loss": loss_sum / weight_sum,
                     "max_gradient_norm_before_clip": max_norm, "recalibration": recalibration,
                     "learning_rates": [g["lr"] for g in optimizer.param_groups],
                     "seconds": time.time() - epoch_started}
            run["history"].append(entry)
            if better(entry, best):
                best, stale = entry, 0
                torch.save(student.state_dict(), out / "checkpoint.pt")
                for (w, k), evaluation in evaluations.items():
                    pipe.write_predictions(out / "predictions" / f"w{w}_k{k}.csv", evaluation)
            else:
                stale += 1
            run.update(selected={"epoch": best["epoch"], "mean_macro_f1": best["mean_macro_f1"]},
                       elapsed_seconds=time.time() - started)
            pipe.write_json(out / "run.json", run)
            print(json.dumps({"epoch": epoch, "mean_macro_f1": round(entry["mean_macro_f1"], 4),
                              "cells": {c: round(v["macro_f1"], 3) for c, v in entry["cells"].items()
                                        if c.endswith(("_k1", "_k16"))},
                              "seconds": round(entry["seconds"])}), flush=True)
            if epoch >= TRAIN_CONFIG["min_epochs"] and stale >= TRAIN_CONFIG["patience"]:
                break
        run.update(status="completed", checkpoint_sha256=pipe.sha256_file(out / "checkpoint.pt"),
                   orders_criterion=criterion, elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time() - started)
        pipe.write_json(out / "run.json", run)
        raise


# ---------------------------------------------------------------- evaluate

class _Holder(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head


def load_trained(mode: str, seed: int, device="cpu") -> TinyViTSequence:
    run = read(run_dir(mode, seed) / "run.json")
    path = run_dir(mode, seed) / "checkpoint.pt"
    if run["status"] != "completed" or pipe.sha256_file(path) != run["checkpoint_sha256"]:
        raise ValueError(f"{mode} seed {seed} is not a completed run with a matching checkpoint")
    importance = torch.load(ART / "importance.pt", map_location="cpu", weights_only=False)
    model = TinyViTSequence(pretrained=False)
    make_nested(model.encoder, None)  # buffers for every width; weights already reordered
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    expected = "random" if mode == "nested_random" else importance["chosen"]
    if run["orders_criterion"] != expected:
        raise ValueError("Checkpoint and importance file disagree")
    return model.to(device).eval()


def cmd_evaluate(args) -> None:
    """Grid metrics with intervals, FLOPs per width, Pareto front, adoption rule."""
    inputs = pipe.load_inputs()
    family = {k: read(F004_ART / "predictions" / f"family_k{k}_units.json") for k in OPERATING_POINTS}
    d026 = read(F004_ART / "evaluation.json")["results"]
    d026_mean = statistics.fmean(d026[str(k)]["duplicate_aware"]["macro_f1"] for k in TRAIN_KS)
    model = load_family()
    holder = load_trained("nested", SEEDS[0]) if run_dir("nested", SEEDS[0]).exists() else None
    flops = {}
    if holder is not None:
        for width in WIDTHS:
            phys = AdaptiveTinyViT(_Holder(physical(holder.encoder, width), holder.head)).eval()
            flops[str(width)] = {str(k): count_flops(phys, k)["flops"] for k in OPERATING_POINTS}
            flops[str(width)]["parameters"] = sum(p.numel() for p in phys.parameters())
    del model
    results = {}
    for mode in ("nested", "single", "nested_random"):
        widths = WIDTHS if mode != "single" else (0.5,)
        for seed in SEEDS:
            directory = run_dir(mode, seed)
            if not (directory / "run.json").exists():
                continue
            run = read(directory / "run.json")
            cells = {}
            for w in widths:
                for k in OPERATING_POINTS:
                    units = read(directory / "predictions" / f"w{w}_k{k}_units.json")
                    selected = next(e for e in run["history"] if e["epoch"] == run["selected"]["epoch"])
                    metrics = selected["cells"][f"w{w}_k{k}"]
                    cells[f"w{w}_k{k}"] = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                                           "normal_recall": metrics["normal_recall"],
                                           "abnormal_recall": metrics["abnormal_recall"],
                                           "auroc": auroc(units), "macro_f1_ci95": bootstrap(units),
                                           "auroc_ci95": auroc_ci(units),
                                           "minus_d026_same_k": paired_delta(units, family[k])["delta"],
                                           "mcnemar_vs_d026_same_k": mcnemar(units, family[k])}
            results.setdefault(mode, {})[str(seed)] = {"selected_epoch": run["selected"]["epoch"], "cells": cells}
    summary = {}
    for mode, per_seed in results.items():
        names = next(iter(per_seed.values()))["cells"].keys()
        summary[mode] = {n: {m: statistics.fmean(s["cells"][n][m] for s in per_seed.values())
                             for m in ("macro_f1", "accuracy", "auroc")} for n in names}
    adoption = {}
    if "nested" in summary and len(results["nested"]) == len(SEEDS):
        mean = {w: statistics.fmean(summary["nested"][f"w{w}_k{k}"]["macro_f1"] for k in TRAIN_KS) for w in WIDTHS}
        adoption = {"d026_mean": d026_mean, "nested_mean_by_width": {str(w): v for w, v in mean.items()},
                    "propose": (mean[1.0] >= d026_mean - ADOPTION_RULE["max_loss_width_1"] and
                                mean[0.5] >= d026_mean - ADOPTION_RULE["max_loss_width_05"])}
    points = []
    if "nested" in summary and flops:
        for w in WIDTHS:
            for k in OPERATING_POINTS:
                points.append({"name": f"w{w}_k{k}", "width": w, "k": k, "flops": flops[str(w)][str(k)],
                               "macro_f1": summary["nested"][f"w{w}_k{k}"]["macro_f1"],
                               "auroc": summary["nested"][f"w{w}_k{k}"]["auroc"]})
    front = [p["name"] for p in points if not any(
        q["macro_f1"] >= p["macro_f1"] and q["flops"] <= p["flops"] and
        (q["macro_f1"] > p["macro_f1"] or q["flops"] < p["flops"]) for q in points)]
    order_effect = {}
    if "nested_random" in summary and "nested" in summary:
        order_effect = {str(w): {mode: statistics.fmean(summary[mode][f"w{w}_k{k}"]["macro_f1"] for k in TRAIN_KS)
                                 for mode in ("nested", "nested_random")} for w in WIDTHS}
    record = {"results": results, "seed_mean": summary, "flops": flops, "points": points,
              "taylor_vs_random_order_mean_macro_f1": order_effect, "random_order_control": RANDOM_ORDER_CONTROL,
              "pareto_front_macro_f1": front, "adoption": adoption, "adoption_rule": ADOPTION_RULE,
              "d026": {str(k): {m: d026[str(k)]["duplicate_aware"][m] for m in ("macro_f1", "accuracy")} |
                       {"auroc": d026[str(k)]["auroc"]} for k in OPERATING_POINTS},
              "rows": len(inputs["validation"]), "test_image_inference": False, "environment": environment()}
    pipe.write_json(ART / "f005b_evaluation.json", record)
    print(json.dumps({"adoption": adoption, "front": front,
                      "seed_mean_macro_f1": {m: {n: round(v["macro_f1"], 3) for n, v in s.items()}
                                             for m, s in summary.items()}}, indent=1))


def cmd_benchmark(args) -> None:
    """Model-only latency per width and K: physically pruned copies and the sliced nested file.

    Seed 2026 (the rule seed) is the deployable checkpoint. Round-robin over
    configurations as in F004; GPU uses all K, CPU K1/K4/K16.
    """
    from experiments.F004.run_f004 import timed_pass
    device = torch.device(args.device)
    env = environment()
    if not env.get("slurm_job_id") or (device.type == "cuda" and not env.get("gpu_model")):
        raise SystemExit("F005b timing must run as a SLURM job (a GPU job for CUDA)")
    out = ART / "benchmark" / f"f005b_{device.type}_{env['hostname']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if device.type == "cuda":
        pipe.set_benchmark_flags()
        sync = lambda: torch.cuda.synchronize(device)  # noqa: E731
        protocol, ks = {"warmup": 50, "timed": 200, "repeats": 5}, OPERATING_POINTS
    else:
        torch.set_num_threads(args.threads)
        sync = lambda: None  # noqa: E731
        protocol, ks = {"warmup": 5, "timed": 50, "repeats": 5}, (1, 4, 16)
    nested = load_trained("nested", 2026, device)
    physical_models = {w: AdaptiveTinyViT(_Holder(physical(nested.encoder, w), nested.head)).to(device).eval()
                       for w in WIDTHS}
    inputs = pipe.load_inputs()
    from src.f002_protocol import benchmark_request_ids
    by_key = {key(r): r for r in inputs["validation"]}
    requests = [by_key[n] for n in benchmark_request_ids(inputs["validation"])["timed"][:protocol["timed"]]]
    frames = cache.load(cache_path("validation"), inputs["validation"])
    index = {key(r): i for i, r in enumerate(inputs["validation"])}
    data = {k: [cache.normalize(frames[index[key(r)]][positions_in_k16(int(r["frame_count"]), k)]).unsqueeze(0).to(device)
                for r in requests] for k in ks}
    order, warmup = list(range(len(requests))), list(range(protocol["warmup"]))

    def physical_run(width, k):
        def once(i):
            with torch.inference_mode():
                physical_models[width](data[k][i])
        return once

    def sliced_run(width, k):
        def once(i):
            set_width(nested.encoder, width)
            with torch.inference_mode():
                frames_k = data[k][i]
                nested.head(nested.encoder(frames_k[0]).mean(0, keepdim=True))
        return once
    configs = {}
    for w in WIDTHS:
        for k in ks:
            configs[f"physical_w{w}_k{k}"] = (physical_run(w, k), k)
            configs[f"sliced_w{w}_k{k}"] = (sliced_run(w, k), k)
    names = list(configs)
    repeats = {n: [] for n in names}
    for r in range(protocol["repeats"]):
        for name in names[r % len(names):] + names[:r % len(names)]:
            repeats[name].append(timed_pass(configs[name][0], order, warmup, sync))
        print(f"round {r + 1}/{protocol['repeats']}", flush=True)
    set_width(nested.encoder, 1.0)
    summary = {n: pipe.summarize_timing(repeats[n], k) | {"raw": repeats[n]} for n, (_, k) in configs.items()}
    for n in names:
        print(n, round(summary[n]["median_latency_ms"], 2), flush=True)
    pipe.write_json(out, {"configs": summary, "protocol": protocol, "device": device.type,
                          "threads": torch.get_num_threads(), "checkpoint": "nested seed 2026",
                          "environment": env})


def ece(units: list[dict], bins: int = 10) -> float:
    """Expected calibration error of the abnormal-class probability, equal-width bins."""
    total, error = len(units), 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        chosen = [u for u in units if low <= u["probability"] < high or (b == bins - 1 and u["probability"] == 1)]
        if chosen:
            confidence = statistics.fmean(u["probability"] for u in chosen)
            frequency = statistics.fmean(u["binary_target"] for u in chosen)
            error += len(chosen) / total * abs(confidence - frequency)
    return error


def cmd_extras(args) -> None:
    """Per width (seed 2026): parameters, density, file sizes, ONNX export, calibration error."""
    import io
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(args.threads)
    nested = load_trained("nested", 2026)
    out = ART / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    def size(module) -> int:
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        return buffer.getbuffer().nbytes
    full_parameters = sum(p.numel() for p in physical(nested.encoder, 1.0).parameters()) + \
        sum(p.numel() for p in nested.head.parameters())
    inputs = pipe.load_inputs()
    rows = inputs["validation"]
    frames = cache.load(cache_path("validation"), rows)
    sample = torch.cat([cache.normalize(frames[i]) for i in range(2)])  # 32 real validation frames
    record = {"nested_file_bytes": size(nested), "widths": {}}
    for width in WIDTHS:
        model = AdaptiveTinyViT(_Holder(physical(nested.encoder, width), nested.head)).eval()
        parameters = sum(p.numel() for p in model.parameters())
        path = out / f"tinyvit_width{int(width * 100)}.onnx"
        torch.onnx.export(model.encoder, (torch.randn(4, 3, 224, 224),), path, input_names=["frames"],
                          output_names=["features"], dynamic_axes={"frames": {0: "n"}, "features": {0: "n"}},
                          opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        set_width(nested.encoder, width)
        with torch.no_grad():
            expected = nested.encoder(sample).numpy()
        set_width(nested.encoder, 1.0)
        got = session.run(None, {"frames": sample.numpy()})[0]
        calibration = {}
        for k in OPERATING_POINTS:
            units = read(run_dir("nested", 2026) / "predictions" / f"w{width}_k{k}_units.json")
            calibration[str(k)] = ece(units)
        record["widths"][str(width)] = {
            "parameters": parameters, "density": parameters / full_parameters,
            "physical_copy_bytes": size(model), "onnx_bytes": path.stat().st_size,
            "onnx_max_abs_feature_difference_vs_sliced": float(np.abs(got - expected).max()),
            "ece_validation": calibration}
        print(width, parameters, f"{parameters / full_parameters:.3f}", path.stat().st_size,
              float(np.abs(got - expected).max()), {k: round(v, 3) for k, v in calibration.items()}, flush=True)
    record["four_physical_copies_bytes"] = sum(v["physical_copy_bytes"] for v in record["widths"].values())
    record["d026_ece_validation"] = {str(k): ece(read(F004_ART / "predictions" / f"family_k{k}_units.json"))
                                     for k in OPERATING_POINTS}
    record["training_seconds"] = {f"{m}_seed{s}": read(run_dir(m, s) / "run.json")["elapsed_seconds"]
                                  for m in ("nested", "single") for s in SEEDS}
    record["environment"] = environment()
    pipe.write_json(ART / "f005b_extras.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("widths", "environment")}, indent=1))


def cmd_weights(args) -> None:
    """Layer-by-layer weight sharing of the deployable nested file (seed 2026).

    For every width: each parameter of the physically pruned copy must equal a
    prefix slice of the nested file's tensor (pruned groups) or the whole tensor
    (everything else), and a forward at every K must read only the nested
    file's tensors. Buffers that differ per width are listed.
    """
    import hashlib
    nested = load_trained("nested", 2026)
    full = dict(nested.encoder.named_parameters())
    layers, per_width = {}, {}
    for width in WIDTHS:
        copy = physical(nested.encoder, width)
        equal, sliced = 0, 0
        for name, tensor in copy.named_parameters():
            source = full[name]
            index = tuple(slice(0, n) for n in tensor.shape)
            same = torch.equal(source[index], tensor.detach())
            if not same:
                raise RuntimeError(f"width {width}: {name} is not a prefix of the nested tensor")
            equal += 1
            sliced += tensor.shape != source.shape
            entry = layers.setdefault(name, {"full_shape": list(source.shape), "shape_per_width": {}})
            entry["shape_per_width"][str(width)] = list(tensor.shape)
        per_width[str(width)] = {"parameter_tensors": equal, "sliced_tensors": sliced,
                                 "parameters": sum(t.numel() for t in copy.parameters())}
        print(width, per_width[str(width)], flush=True)
    width_buffers = sorted(n for n, _ in nested.encoder.named_buffers() if "bn3_" in n)
    pipe.write_json(ART / "f005b_weights.json", {
        "checkpoint": "nested seed 2026", "checkpoint_sha256": read(run_dir("nested", 2026) / "run.json")["checkpoint_sha256"],
        "per_width": per_width, "layers": layers,
        "per_width_buffers": width_buffers,
        "head_shared": True, "head_sha256": hashlib.sha256(nested.head[-1].weight.detach().numpy().tobytes()).hexdigest(),
        "environment": environment()})
    print(len(layers), "parameter tensors checked;", len(width_buffers), "per-width BN buffers", flush=True)


def cmd_full_bn(args) -> None:
    """Post hoc: every downstream BN gets per-width statistics; validation grid per seed."""
    from experiments.F005.prune import make_all_bn_width_aware, recalibrate_all
    device = torch.device(args.device)
    inputs = pipe.load_inputs()
    train_rows, val_rows = inputs["train"], inputs["validation"]
    calibration = calibration_batches(cache.load(cache_path("train"), train_rows),
                                      train_subsets(train_rows)["calibration"], device)
    val_frames = cache.load(cache_path("validation"), val_rows)
    before = read(ART / "f005b_evaluation.json")["results"]["nested"]
    record = {"rule": FULL_BN_RULE, "seeds": {}}
    for seed in SEEDS:
        model = load_trained("nested", seed, device)
        make_all_bn_width_aware(model.encoder)
        report = recalibrate_all(model.encoder, calibration)
        evaluations = validate_grid(model, val_frames, val_rows, inputs["unit_of"], device, WIDTHS)
        cells = {f"w{w}_k{k}": e["duplicate_aware"] | {"auroc": auroc(e["units"])} for (w, k), e in evaluations.items()}
        for (w, k), e in evaluations.items():
            pipe.write_predictions(ART / "full_bn" / f"seed{seed}" / f"w{w}_k{k}.csv", e)
        record["seeds"][str(seed)] = {"cells": cells, "recalibration": report}
        print(seed, {n: round(c["macro_f1"], 3) for n, c in cells.items() if n.endswith(("_k1", "_k16"))}, flush=True)
    def mean(source, widths):
        return statistics.fmean(source[str(seed)]["cells"][f"w{w}_k{k}"]["macro_f1"]
                                for seed in SEEDS for w in widths for k in TRAIN_KS)
    narrow = (0.75, 0.5, 0.25)
    old = {str(s_): {"cells": before[str(s_)]["cells"]} for s_ in SEEDS}
    record["mean_macro_f1_narrow_widths"] = {"conv3_only": mean(old, narrow), "all_bn": mean(record["seeds"], narrow)}
    record["mean_macro_f1_by_width"] = {str(w): {"conv3_only": mean(old, (w,)), "all_bn": mean(record["seeds"], (w,))}
                                        for w in WIDTHS}
    gain = record["mean_macro_f1_narrow_widths"]["all_bn"] - record["mean_macro_f1_narrow_widths"]["conv3_only"]
    record["gain"], record["adopt"] = gain, gain >= FULL_BN_RULE["min_gain"]
    record["environment"] = environment()
    pipe.write_json(ART / "f005d_full_bn.json", record)
    print(json.dumps({k: v for k, v in record.items() if k in ("mean_macro_f1_by_width", "gain", "adopt")}, indent=1))


def cmd_switching(args) -> None:
    """Interleaved width × K switching of the sliced nested file (seed 2026), F004 method.

    The same requests with (width, K) drawn per request, and grouped by
    (width, K); 5 repeats with the two orders alternating. Requests slice their
    K frames from preloaded K16 inputs inside the timed region.
    """
    import random as _random
    device = torch.device(args.device)
    env = environment()
    if not env.get("slurm_job_id") or (device.type == "cuda" and not env.get("gpu_model")):
        raise SystemExit("Timing must run as a SLURM job (a GPU job for CUDA)")
    out = ART / "benchmark" / f"switching_{device.type}_{env['hostname']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if device.type == "cuda":
        pipe.set_benchmark_flags()
        sync = lambda: torch.cuda.synchronize(device)  # noqa: E731
        n_requests, warm, ks = 1000, 50, OPERATING_POINTS
    else:
        torch.set_num_threads(args.threads)
        sync = lambda: None  # noqa: E731
        n_requests, warm, ks = 240, 10, (1, 4, 16)
    model = load_trained("nested", 2026, device)
    from src.f002_protocol import benchmark_request_ids
    by_key = {key(r): r for r in pipe.load_inputs()["validation"]}
    validation = pipe.load_inputs()["validation"]
    rows = [by_key[n] for n in benchmark_request_ids(validation)["timed"][:50]]
    frames = cache.load(cache_path("validation"), validation)
    index = {key(r): i for i, r in enumerate(validation)}
    data = [cache.normalize(frames[index[key(r)]]).to(device) for r in rows]
    positions = [{k: torch.tensor(positions_in_k16(int(r["frame_count"]), k), device=device) for k in ks} for r in rows]
    rng = _random.Random(2026)
    requests = [(i % len(rows), rng.choice(WIDTHS), rng.choice(ks)) for i in range(n_requests)]
    grouped = sorted(requests, key=lambda r: (WIDTHS.index(r[1]), ks.index(r[2])))

    def serve(row, width, k):
        set_width(model.encoder, width)
        with torch.inference_mode():
            model.head(model.encoder(data[row].index_select(0, positions[row][k])).mean(0, keepdim=True))

    def run(sequence):
        for request in sequence[:warm]:
            serve(*request)
        sync()
        latencies = []
        for request in sequence:
            began = time.perf_counter()
            serve(*request)
            sync()
            latencies.append((time.perf_counter() - began) * 1000.0)
        return latencies
    passes = []
    for r in range(5):
        first = "interleaved" if r % 2 == 0 else "grouped"
        result = {}
        for name in (first, "grouped" if first == "interleaved" else "interleaved"):
            result[name] = run(requests if name == "interleaved" else grouped)
        passes.append(result)
        print(f"repeat {r + 1}/5", flush=True)
    by_cell = {}
    for w in WIDTHS:
        for k in ks:
            per = []
            for p_ in passes:
                a = statistics.median(t for (_, ww, kk), t in zip(requests, p_["interleaved"]) if (ww, kk) == (w, k))
                b = statistics.median(t for (_, ww, kk), t in zip(grouped, p_["grouped"]) if (ww, kk) == (w, k))
                per.append({"interleaved_median_ms": a, "grouped_median_ms": b, "difference_ms": a - b})
            diffs = [x["difference_ms"] for x in per]
            by_cell[f"w{w}_k{k}"] = {"median_difference_ms": statistics.median(diffs), "min_difference_ms": min(diffs),
                                     "max_difference_ms": max(diffs),
                                     "grouped_median_ms": statistics.median(x["grouped_median_ms"] for x in per),
                                     "per_repeat": per}
    set_width(model.encoder, 1.0)
    switches = sum((a[1], a[2]) != (b[1], b[2]) for a, b in zip(requests, requests[1:]))
    pipe.write_json(out, {"by_cell": by_cell, "requests": requests, "passes": passes, "switches": switches,
                          "device": device.type, "threads": torch.get_num_threads(), "environment": env})
    worst = max(abs(c["max_difference_ms"]) / c["grouped_median_ms"] for c in by_cell.values())
    print(f"{switches} switches; worst |interleaved - grouped| = {100 * worst:.1f}% of latency", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--partition", choices=("train", "validation"), required=True)
    c.set_defaults(func=cmd_cache)
    imp = sub.add_parser("importance")
    imp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    imp.set_defaults(func=cmd_importance)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=("nested", "single", "nested_random"), required=True)
    t.add_argument("--seed", type=int, choices=SEEDS, default=2026)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--workers", type=int, default=TRAIN_CONFIG["workers"])
    t.add_argument("--max-batches", type=int)
    t.add_argument("--max-val-rows", type=int)
    t.set_defaults(func=cmd_train)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    extras = sub.add_parser("extras")
    extras.add_argument("--threads", type=int, default=8)
    extras.set_defaults(func=cmd_extras)
    sub.add_parser("weights").set_defaults(func=cmd_weights)
    full = sub.add_parser("full-bn")
    full.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    full.set_defaults(func=cmd_full_bn)
    sw = sub.add_parser("switching")
    sw.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sw.add_argument("--threads", type=int, default=4)
    sw.set_defaults(func=cmd_switching)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    bench.add_argument("--threads", type=int, default=4)
    bench.set_defaults(func=cmd_benchmark)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.func(args)


if __name__ == "__main__":
    main()
