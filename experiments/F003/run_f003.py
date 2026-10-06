"""F003 (T009) commands: cost, cached features, validation analysis, runtime.

Run order: flops → features (train/validation at 224; validation at 160/192)
→ analyze → benchmark(-reference) → features-temporal → analyze-extra →
analyze-robustness → report. Everything writes to ignored artifacts/F003/.
Figures are made separately by create_figures.py from the saved results.
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
                                       RESOLUTIONS, TEMPORAL_K, WINDOW_FRACTIONS,
                                       ExitSequenceModel, F003Sequences, Trunk, count_flops,
                                       dense_indices, fit_exit_head, load_backbone,
                                       positions_in_k16, set_fused_attention, window_indices)

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


# ---------------------------------------------------------------- extra analyses

def temporal_configs(frame_count: int) -> dict[str, tuple[int, ...]]:
    """Every scan-duration (window) and native-rate (dense) frame selection."""
    configs = {f"window{int(w * 100)}_k{k}": window_indices(frame_count, w, k)
               for w in WINDOW_FRACTIONS for k in TEMPORAL_K}
    configs |= {f"dense_k{k}": dense_indices(frame_count, k) for k in TEMPORAL_K}
    return configs


class TemporalFrames(Dataset):
    def __init__(self, sequences: F003Sequences):
        self.sequences = sequences
        self.needed = [sorted(set().union(*temporal_configs(int(r["frame_count"])).values()))
                       for r in sequences.rows]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences.frames_at(index, self.needed[index]), index


def cmd_features_temporal(args) -> None:
    """Final-exit features for every frame a temporal configuration needs."""
    out = ART / "features" / "validation_temporal.pt"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    sequences = F003Sequences(inputs["validation"])
    dataset = TemporalFrames(sequences)
    trunk = Trunk(load_backbone(device).encoder).eval()
    features, started = [None] * len(dataset), time.time()
    with torch.inference_mode():
        for frames, index in DataLoader(dataset, batch_size=1, num_workers=args.workers):
            features[index.item()] = trunk(frames[0].to(device))["final"].float().cpu()
    record = {"features": features, "frame_numbers": dataset.needed,
              "keys": [key(r) for r in sequences.rows], "seconds": time.time() - started,
              "frames": sum(len(n) for n in dataset.needed), "environment": environment()}
    torch.save(record, out)
    print(f"wrote {out}: {record['frames']} frames in {record['seconds']:.0f} s", flush=True)


def unit_records(rows: list[dict], unit_of: dict) -> dict:
    """Unit → split group and device(s); aliases always share a group (T006)."""
    units = {}
    for row in rows:
        unit = units.setdefault(unit_of[key(row)], {"groups": set(), "devices": set()})
        unit["groups"].add(row["split_group"])
        unit["devices"].add(row["device"])
    for unit_id, unit in units.items():
        if len(unit["groups"]) != 1:
            raise ValueError(f"Unit {unit_id} spans dependency groups")
    return units


def group_folds(units: dict, folds: int = 5, seed: int = 2026) -> dict:
    groups = sorted({next(iter(u["groups"])) for u in units.values()})
    random.Random(seed).shuffle(groups)
    fold_of_group = {g: i % folds for i, g in enumerate(groups)}
    return {u: fold_of_group[next(iter(rec["groups"]))] for u, rec in units.items()}


def cross_validated_cascade(cheap: list[dict], full: list[dict], score: dict, fold_of: dict,
                            cheap_flops: float, full_flops: float, reused: float,
                            tolerance: float = 0.01) -> dict:
    """Pick the escalation threshold on 4 group folds, score it on the 5th.

    Rule fixed before looking at held-out folds: the cheapest threshold whose
    training-fold macro-F1 is at least training-fold K16 macro-F1 − tolerance.
    Higher ``score`` means more reason to escalate.
    """
    from src.f002_protocol import binary_metrics
    cheap_by, full_by = {u["unit_id"]: u for u in cheap}, {u["unit_id"]: u for u in full}

    def f1(ids, escalate):
        chosen = [full_by[i] if i in escalate else cheap_by[i] for i in ids]
        return binary_metrics([u["binary_target"] for u in chosen],
                              [u["probability"] for u in chosen])["macro_f1"]
    held_out_escalated, per_fold = set(), []
    for fold in sorted(set(fold_of.values())):
        train = [u for u in cheap_by if fold_of[u] != fold]
        test = [u for u in cheap_by if fold_of[u] == fold]
        target = f1(train, set(train)) - tolerance
        ranked = sorted(train, key=lambda u: -score[u])
        threshold = math.inf
        for n in range(len(ranked) + 1):
            if f1(train, set(ranked[:n])) >= target:
                threshold = math.inf if n == 0 else score[ranked[n - 1]]
                break
        escalated = {u for u in test if score[u] >= threshold}
        held_out_escalated |= escalated
        per_fold.append({"fold": fold, "threshold": threshold, "train_units": len(train),
                         "held_out_units": len(test),
                         "held_out_escalation_rate": len(escalated) / len(test)})
    ids = sorted(cheap_by)
    chosen = [full_by[i] if i in held_out_escalated else cheap_by[i] for i in ids]
    rate = len(held_out_escalated) / len(ids)
    return {"held_out": binary_metrics([u["binary_target"] for u in chosen],
                                       [u["probability"] for u in chosen]),
            "escalation_rate": rate,
            "expected_flops": cheap_flops + rate * (full_flops - reused),
            "rule": f"cheapest threshold with training-fold macro-F1 >= K16 - {tolerance}",
            "folds": per_fold}


def device_table(evaluation_units: list[dict], units: dict) -> dict:
    from src.f002_protocol import binary_metrics
    table = {}
    for device in sorted({d for u in units.values() for d in u["devices"]} | {"mixed"}):
        chosen = [u for u in evaluation_units if
                  (device == "mixed" and len(units[u["unit_id"]]["devices"]) > 1) or
                  units[u["unit_id"]]["devices"] == {device}]
        if chosen:
            metrics = binary_metrics([u["binary_target"] for u in chosen],
                                     [u["probability"] for u in chosen])
            table[device] = {"units": len(chosen),
                             "abnormal_units": sum(u["binary_target"] for u in chosen),
                             **{m: metrics[m] for m in ("accuracy", "normal_recall",
                                                        "abnormal_recall", "tn", "fp", "fn", "tp")}}
    return table


def cmd_analyze_extra(args) -> None:
    """Temporal axis, cross-validated cascades, disagreement signal, per device."""
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    backbone = load_backbone("cpu")
    flops = read(ART / "flops.json")["configs"]
    analysis = read(ART / "analysis.json")
    val = load_features("validation", 224)
    temporal = torch.load(ART / "features" / "validation_temporal.pt", weights_only=False)
    if temporal["keys"] != [key(r) for r in rows] or val["keys"] != temporal["keys"]:
        raise ValueError("Temporal features do not match the frozen validation rows")
    if temporal["environment"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Temporal features belong to another checkpoint")
    units = unit_records(rows, unit_of)

    def score_rows(logits, name, k):
        preds = [{"exam_id": r["exam_id"], "zone_id": r["zone_id"],
                  "binary_target": int(r["binary_target"]), "device": r["device"],
                  "unit_id": unit_of[key(r)], "k": k, "config": name, "logit": float(l),
                  "probability": float(torch.sigmoid(l))} for r, l in zip(rows, logits)]
        return aggregate_validation(rows, preds, unit_of)

    # 1. Temporal axis: same head, same cost per K, different frames.
    evaluations, results = {}, {}
    k16_full = None
    for name in temporal_configs(int(rows[0]["frame_count"])):
        k = int(name.rsplit("_k", 1)[1])
        pooled_rows = []
        for i, row in enumerate(rows):
            wanted = temporal_configs(int(row["frame_count"]))[name]
            position = {f: p for p, f in enumerate(temporal["frame_numbers"][i])}
            pooled_rows.append(temporal["features"][i][[position[f] for f in wanted]].mean(dim=0))
        with torch.no_grad():
            logits = backbone.head(torch.stack(pooled_rows)).squeeze(-1).double()
        evaluation = score_rows(logits, name, k)
        evaluations[name] = evaluation
        pipe.write_predictions(ART / "predictions" / f"temporal_{name}.csv", evaluation)
        results[name] = {"k": k, "duplicate_aware": evaluation["duplicate_aware"],
                         "macro_f1_ci95": bootstrap(evaluation["units"]),
                         "flops": flops[f"final_k{k}_r224"]["flops"]}
    # The full-clip window must reproduce the main K configurations.
    reproduction = {}
    for k in TEMPORAL_K:
        main = analysis["configs"][f"final_k{k}_r224"]["duplicate_aware"]["macro_f1"]
        reproduction[f"k{k}"] = abs(results[f"window100_k{k}"]["duplicate_aware"]["macro_f1"] - main)
    if max(reproduction.values()) > 1e-9:
        raise ValueError(f"window100 does not reproduce the main K results: {reproduction}")
    k16_units = read_units("final_k16_r224")
    for name in evaluations:
        results[name]["paired_vs_k16_full_clip"] = paired_delta(evaluations[name]["units"], k16_units)
        d = results[name]["duplicate_aware"]
        print(f"{name:16s} F1 {d['macro_f1']:.4f} n {d['normal_recall']:.3f} a {d['abnormal_recall']:.3f}", flush=True)

    # 2–3. Escalation signals for K → K16 cascades, descriptive and cross-validated.
    folds = group_folds(units)
    per_frame = (flops["final_k16_r224"]["flops"] - flops["final_k8_r224"]["flops"]) / 8
    signals, cv = {}, {}
    with torch.no_grad():
        frame_probs = torch.sigmoid(backbone.head(val["features"]["final"]).squeeze(-1))
    for k in (1, 2, 4, 8):
        cheap = read_units(f"final_k{k}_r224")
        confidence = {u["unit_id"]: -abs(u["probability"] - 0.5) for u in cheap}
        row_spread = {}
        if k > 1:
            for i, row in enumerate(rows):
                probs = frame_probs[i, val["positions"][k][i]]
                row_spread.setdefault(unit_of[key(row)], []).append(float(probs.std(unbiased=False)))
        disagreement = {u: sum(v) / len(v) for u, v in row_spread.items()}
        cost = dict(cheap_flops=flops[f"final_k{k}_r224"]["flops"],
                    full_flops=flops["final_k16_r224"]["flops"], reused=k * per_frame)
        for signal_name, score in (("confidence", confidence), ("disagreement", disagreement)):
            if not score:
                continue
            name = f"final_k{k}_to_k16_{signal_name}"
            cv[name] = cross_validated_cascade(cheap, k16_units, score, folds, **cost)
            ranked = sorted(score, key=lambda u: -score[u])
            curve = []
            for rate in [i / 20 for i in range(21)]:
                escalate = set(ranked[:round(rate * len(ranked))])
                chosen = [u if u["unit_id"] not in escalate else
                          next(f for f in k16_units if f["unit_id"] == u["unit_id"]) for u in cheap]
                from src.f002_protocol import binary_metrics
                m = binary_metrics([u["binary_target"] for u in chosen],
                                   [u["probability"] for u in chosen])
                curve.append({"escalation_rate": len(escalate) / len(ranked),
                              "expected_flops": cost["cheap_flops"] + len(escalate) / len(ranked) *
                              (cost["full_flops"] - cost["reused"]), **m})
            signals[name] = curve
            held = cv[name]["held_out"]
            print(f"CV {name:32s} F1 {held['macro_f1']:.4f} esc {cv[name]['escalation_rate']:.2f} "
                  f"GF {cv[name]['expected_flops'] / 1e9:.2f}", flush=True)

    # 4. Per-device K curve (TinyViT) and the static MobileNetV2 reference.
    devices = {f"final_k{k}_r224": device_table(read_units(f"final_k{k}_r224"), units)
               for k in K_VALUES}
    if F002B_UNITS.exists():
        devices["f002b_mobilenet_v2_k16"] = device_table(read(F002B_UNITS), units)

    pipe.write_json(ART / "analysis_extra.json", {
        "temporal": results, "window100_reproduction_abs_diff": reproduction,
        "cascade_curves_by_escalation_rate": signals, "cascade_cross_validated": cv,
        "group_folds": {"folds": 5, "seed": 2026, "units_per_fold":
                        [sum(1 for f in folds.values() if f == i) for i in range(5)]},
        "per_device": devices, "threshold": 0.5,
        "notes": ["temporal windows are shares of each clip; frame rate is unknown",
                  "cross-validated cascades pick thresholds on 4 group folds and score the 5th"],
        "environment": environment()})
    print(json.dumps({d: {k: (v["units"], v["abnormal_units"], round(v["abnormal_recall"], 3))
                          for k, v in t.items()} for d, t in devices.items()}, indent=0))


def read_units(config: str) -> list[dict]:
    return read(ART / "predictions" / f"{config}_units.json")


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
    print("wrote", ART / "summary.json")


# ---------------------------------------------------------------- robustness

def auroc(units: list[dict]) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([u["binary_target"] for u in units], [u["probability"] for u in units]))


def auroc_ci(units: list[dict], reps: int = 1000, seed: int = 2026) -> list[float]:
    """Unit bootstrap 95% interval of AUROC."""
    from sklearn.metrics import roc_auc_score
    rng, values = random.Random(seed), []
    for _ in range(reps):
        sample = [units[rng.randrange(len(units))] for _ in units]
        targets = [u["binary_target"] for u in sample]
        if len(set(targets)) == 2:
            values.append(roc_auc_score(targets, [u["probability"] for u in sample]))
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def paired_auroc_delta(a: list[dict], b: list[dict], reps: int = 2000, seed: int = 2026) -> dict:
    """Paired unit bootstrap of AUROC(a) − AUROC(b), threshold-free."""
    from sklearn.metrics import roc_auc_score
    pa, pb = {u["unit_id"]: u for u in a}, {u["unit_id"]: u for u in b}
    if pa.keys() != pb.keys():
        raise ValueError("Paired comparison needs identical evaluation units")
    ids, rng, deltas = sorted(pa), random.Random(seed), []
    for _ in range(reps):
        chosen = [ids[rng.randrange(len(ids))] for _ in ids]
        targets = [pa[i]["binary_target"] for i in chosen]
        if len(set(targets)) == 2:
            deltas.append(roc_auc_score(targets, [pa[i]["probability"] for i in chosen]) -
                          roc_auc_score(targets, [pb[i]["probability"] for i in chosen]))
    deltas.sort()
    return {"delta": auroc(a) - auroc(b), "ci95": [deltas[int(0.025 * len(deltas))],
                                                    deltas[int(0.975 * len(deltas)) - 1]]}


def mcnemar(a: list[dict], b: list[dict], ids=None) -> dict:
    """Exact McNemar on correctness at 0.5: units only a, or only b, gets right."""
    from scipy.stats import binomtest
    pa, pb = {u["unit_id"]: u for u in a}, {u["unit_id"]: u for u in b}
    ids = sorted(pa) if ids is None else ids
    right = lambda u: (u["probability"] >= 0.5) == bool(u["binary_target"])  # noqa: E731
    only_a = sum(right(pa[i]) and not right(pb[i]) for i in ids)
    only_b = sum(right(pb[i]) and not right(pa[i]) for i in ids)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "exact_p": float(p), "units": len(ids)}


def cmd_analyze_robustness(args) -> None:
    """Threshold-free and paired checks of the F003 claims; CPU only."""
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    units = unit_records(rows, unit_of)
    names = sorted(p.name[:-len("_units.json")] for p in (ART / "predictions").glob("*_units.json"))
    preds = {n: read_units(n) for n in names}
    if F002B_UNITS.exists():
        preds["f002b_mobilenet_v2_k16"] = read(F002B_UNITS)
    result = {"auroc": {n: auroc(u) for n, u in preds.items()},
              "auroc_ci95": {n: auroc_ci(u) for n, u in preds.items()},
              "f002b_macro_f1_ci95": bootstrap(preds["f002b_mobilenet_v2_k16"])}
    pairs = [("final_k16_r224", "f002b_mobilenet_v2_k16"), ("final_k4_r224", "f002b_mobilenet_v2_k16"),
             ("final_k1_r224", "final_k16_r224"), ("final_k2_r224", "final_k16_r224"),
             ("final_k4_r224", "final_k16_r224"), ("final_k8_r224", "final_k16_r224"),
             ("final_k2_r192", "final_k2_r224"), ("final_k8_r192", "final_k8_r224"),
             ("final_k16_r192", "final_k16_r224"), ("stage2_mid_k1_r224", "final_k1_r224"),
             ("stage2_k4_r224", "final_k4_r224"), ("stage2_k16_r224", "final_k8_r224"),
             ("temporal_window25_k4", "final_k4_r224"), ("temporal_window25_k16", "final_k16_r224"),
             ("temporal_dense_k4", "final_k4_r224"), ("temporal_dense_k8", "final_k8_r224"),
             ("temporal_dense_k16", "final_k16_r224")]
    result["paired"] = {f"{a}_vs_{b}": {"auroc": paired_auroc_delta(preds[a], preds[b]),
                                       "mcnemar_at_0.5": mcnemar(preds[a], preds[b])}
                        for a, b in pairs}
    device_tests = {}
    tiny, mobile = preds["final_k4_r224"], preds["f002b_mobilenet_v2_k16"]
    for device in ("0", "4"):
        for target, label in ((0, "normal"), (1, "abnormal")):
            ids = sorted(u["unit_id"] for u in tiny
                         if units[u["unit_id"]]["devices"] == {device} and u["binary_target"] == target)
            device_tests[f"device{device}_{label}_tinyvit_k4_vs_f002b"] = mcnemar(tiny, mobile, ids)
    result["per_device_mcnemar"] = device_tests
    # How much can any K -> K16 escalation rule gain, and how stable is the CV estimate?
    right = lambda u: (u["probability"] >= 0.5) == bool(u["binary_target"])  # noqa: E731
    k16 = {u["unit_id"]: u for u in preds["final_k16_r224"]}
    flops = read(ART / "flops.json")["configs"]
    per_frame = (flops["final_k16_r224"]["flops"] - flops["final_k8_r224"]["flops"]) / 8
    val = load_features("validation", 224)
    backbone = load_backbone("cpu")
    with torch.no_grad():
        frame_probs = torch.sigmoid(backbone.head(val["features"]["final"]).squeeze(-1))
    ceiling, sensitivity = {}, {}
    for cheap_k in (2, 4):
        cheap = {u["unit_id"]: u for u in preds[f"final_k{cheap_k}_r224"]}
        ceiling[f"k{cheap_k}_to_k16"] = {
            "cheap_wrong_k16_right": sum(right(k16[i]) and not right(cheap[i]) for i in cheap),
            "cheap_right_k16_wrong": sum(right(cheap[i]) and not right(k16[i]) for i in cheap)}
        spread = {}
        for i, row in enumerate(rows):
            spread.setdefault(unit_of[key(row)], []).append(
                float(frame_probs[i, val["positions"][cheap_k][i]].std(unbiased=False)))
        scores = {"confidence": {u: -abs(cheap[u]["probability"] - 0.5) for u in cheap},
                  "disagreement": {u: sum(v) / len(v) for u, v in spread.items()}}
        cost = dict(cheap_flops=flops[f"final_k{cheap_k}_r224"]["flops"],
                    full_flops=flops["final_k16_r224"]["flops"], reused=cheap_k * per_frame)
        for signal, score in scores.items():
            for tolerance in (0.0, 0.005, 0.01, 0.02):
                runs = [cross_validated_cascade(preds[f"final_k{cheap_k}_r224"], preds["final_k16_r224"],
                                                score, group_folds(units, seed=seed),
                                                tolerance=tolerance, **cost)
                        for seed in (2026, 1, 2, 3, 4)]
                sensitivity[f"k{cheap_k}_{signal}_tol{tolerance}"] = [
                    {"fold_seed": seed, "macro_f1": r["held_out"]["macro_f1"],
                     "escalation_rate": r["escalation_rate"], "expected_flops": r["expected_flops"]}
                    for seed, r in zip((2026, 1, 2, 3, 4), runs)]
    result["escalation_ceiling"] = ceiling
    result["k4_to_k16_ceiling"] = {"k4_wrong_k16_right": ceiling["k4_to_k16"]["cheap_wrong_k16_right"],
                                   "k4_right_k16_wrong": ceiling["k4_to_k16"]["cheap_right_k16_wrong"]}
    result["cascade_cv_sensitivity"] = sensitivity
    result["environment"] = environment()
    pipe.write_json(ART / "analysis_robustness.json", result)
    for name in ("final_k1_r224", "final_k4_r224", "final_k16_r224", "f002b_mobilenet_v2_k16"):
        print(f"AUROC {name:24s} {result['auroc'][name]:.4f}")
    for name, test in result["paired"].items():
        a = test["auroc"]
        print(f"{name:48s} dAUROC {a['delta']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] "
              f"McNemar {test['mcnemar_at_0.5']['only_a_correct']}/{test['mcnemar_at_0.5']['only_b_correct']} "
              f"p={test['mcnemar_at_0.5']['exact_p']:.3f}")
    print(json.dumps(result["per_device_mcnemar"], indent=0))
    print(result["escalation_ceiling"])


# ---------------------------------------------------------------- MobileNetV2 frame count

F002B_RUN = ROOT / "artifacts/F002/f002b/mobilenet_v2_seed2026"
MOBILENET_MATCHED = (("final_k1_r224", 4), ("final_k2_r224", 8), ("final_k4_r224", 16))


def count_sequence_flops_any_k(model, k: int) -> dict:
    """F002 strict fvcore count for a K the frozen F002 counter does not allow.

    Same handlers, bookkeeping allowlist and head check as
    ``src.f002_flops.count_sequence_flops``; only the K restriction is lifted.
    """
    from fvcore.nn import FlopCountAnalysis
    from src.f002_flops import BOOKKEEPING_ALLOWLIST, CUSTOM_HANDLERS

    class Strict(FlopCountAnalysis):
        def _should_ignore_node(self, node):
            if node.kind() in BOOKKEEPING_ALLOWLIST:
                return True
            if super()._should_ignore_node(node):
                raise RuntimeError(f"Unapproved fvcore-ignored operator: {node.kind()}")
            return False

    model.eval()
    analysis = Strict(model, torch.zeros(1, k, 3, 224, 224)).set_op_handle(**CUSTOM_HANDLERS)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = int(analysis.total())
    if analysis.unsupported_ops():
        raise RuntimeError(f"Uncounted operators: {dict(analysis.unsupported_ops())}")
    head = int(analysis.by_module_and_operator().get("head", {}).get("linear", 0))
    if head != model.head.in_features or analysis.by_operator().get("mean_pool", 0) <= 0:
        raise RuntimeError("Incomplete sequence encoder/pool/head FLOPs trace")
    return {"flops": total, "by_operator": dict(analysis.by_operator()), "k": k}


def cmd_mobilenet_k(args) -> None:
    """Frozen F002b MobileNetV2 at every F003 K on validation; read-only.

    F002 bilinear preprocessing (the model's own), F003 nested frame sets, one
    cached K16 pass of per-frame encoder features. Writes only to
    artifacts/F003/mobilenet_k/.
    """
    out = ART / "mobilenet_k"
    if (out / "mobilenet_k.json").exists():
        raise SystemExit(f"{out} already has results; refusing to overwrite")
    device = torch.device(args.device)
    pipe.set_training_determinism(2026)
    inputs = pipe.load_inputs()
    rows, unit_of = inputs["validation"], inputs["unit_of"]
    run, model = pipe.load_checkpoint(F002B_RUN, device)
    sequences = pipe.Sequences(rows, 16)
    width = model.head.in_features
    features, direct = torch.zeros(len(rows), 16, width), torch.zeros(len(rows))
    started = time.time()
    with torch.inference_mode():
        for i in range(len(rows)):
            frames = sequences.frames(i).to(device)
            features[i] = model.encoder(frames).float().cpu()
            direct[i] = model(frames.unsqueeze(0)).float().cpu()[0]
    seconds = time.time() - started
    model = model.cpu()
    positions = {k: [positions_in_k16(int(r["frame_count"]), k) for r in rows] for k in K_VALUES}
    results, units = {}, {}
    for k in K_VALUES:
        pooled_k = torch.stack([features[i, positions[k][i]].mean(dim=0) for i in range(len(rows))])
        with torch.no_grad():
            logits = model.head(pooled_k).squeeze(-1).double()
        if k == 16:
            cache_error = float((logits.float() - direct).abs().max())
            if cache_error > 1e-4:
                raise ValueError(f"Cached MobileNetV2 features do not reproduce direct logits ({cache_error})")
        preds = [{"exam_id": r["exam_id"], "zone_id": r["zone_id"],
                  "binary_target": int(r["binary_target"]), "device": r["device"],
                  "unit_id": unit_of[key(r)], "k": k, "logit": float(l),
                  "probability": float(torch.sigmoid(l))} for r, l in zip(rows, logits)]
        evaluation = aggregate_validation(rows, preds, unit_of)
        pipe.write_predictions(out / "predictions" / f"mobilenet_k{k}.csv", evaluation)
        units[k] = evaluation["units"]
        results[k] = {"duplicate_aware": evaluation["duplicate_aware"],
                      "row_level": evaluation["row_level"],
                      "macro_f1_ci95": bootstrap(evaluation["units"]),
                      "auroc": auroc(evaluation["units"]), "auroc_ci95": auroc_ci(evaluation["units"]),
                      "flops": count_sequence_flops_any_k(model, k)}
        print(f"MobileNetV2 K{k}: macro-F1 {evaluation['duplicate_aware']['macro_f1']:.4f} "
              f"AUROC {results[k]['auroc']:.4f} GFLOPs {results[k]['flops']['flops'] / 1e9:.2f}", flush=True)
    recorded = run["selected"]["macro_f1"]
    if abs(results[16]["duplicate_aware"]["macro_f1"] - recorded) > 1e-12:
        raise ValueError("K16 does not reproduce the recorded F002b macro-F1")
    comparisons = {}
    for tiny_name, mobile_k in MOBILENET_MATCHED:
        tiny = read_units(tiny_name)
        comparisons[f"{tiny_name}_vs_mobilenet_k{mobile_k}"] = {
            "macro_f1": paired_delta(tiny, units[mobile_k]),
            "auroc": paired_auroc_delta(tiny, units[mobile_k]),
            "mcnemar_at_0.5": mcnemar(tiny, units[mobile_k])}
    for k in K_VALUES:
        tiny = read_units(f"final_k{k}_r224")
        comparisons[f"same_k{k}_tinyvit_vs_mobilenet"] = {
            "macro_f1": paired_delta(tiny, units[k]), "auroc": paired_auroc_delta(tiny, units[k])}
    pipe.write_json(out / "mobilenet_k.json", {
        "checkpoint_sha256": run["checkpoint_sha256"], "preprocessing": "F002 bilinear (model's own)",
        "frame_sets": "F003 nested K1 ⊂ K2 ⊂ K4 ⊂ K8 ⊂ K12 ⊂ K16",
        "results": {str(k): v for k, v in results.items()}, "comparisons": comparisons,
        "cache_max_abs_logit_error": cache_error, "feature_pass_seconds": seconds,
        "test_image_inference": False, "environment": environment()})
    for name, c in comparisons.items():
        if "same_k" in name:
            continue
        print(f"{name:44s} dF1 {c['macro_f1']['delta']:+.4f} [{c['macro_f1']['ci95'][0]:+.3f}, "
              f"{c['macro_f1']['ci95'][1]:+.3f}] dAUROC {c['auroc']['delta']:+.4f} "
              f"[{c['auroc']['ci95'][0]:+.3f}, {c['auroc']['ci95'][1]:+.3f}] McNemar "
              f"{c['mcnemar_at_0.5']['only_a_correct']}/{c['mcnemar_at_0.5']['only_b_correct']}", flush=True)


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
    temporal = sub.add_parser("features-temporal")
    temporal.add_argument("--device", default="cuda")
    temporal.add_argument("--workers", type=int, default=4)
    temporal.set_defaults(func=cmd_features_temporal)
    sub.add_parser("analyze-extra").set_defaults(func=cmd_analyze_extra)
    sub.add_parser("analyze-robustness").set_defaults(func=cmd_analyze_robustness)
    mobile = sub.add_parser("mobilenet-k")
    mobile.add_argument("--device", default="cuda")
    mobile.set_defaults(func=cmd_mobilenet_k)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--configs", nargs="*", help="subset of config names; default all")
    bench.set_defaults(func=cmd_benchmark)
    sub.add_parser("benchmark-reference").set_defaults(func=cmd_benchmark_reference)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
