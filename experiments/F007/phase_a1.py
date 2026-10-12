"""F007 Phase A1: guarded validation requests, routing and unit aggregation.

This module has no held-out execution command. A caller asking to evaluate the
test partition must pass the protocol gate before any predictor or image loader
is constructed. Phase A2 uses these validation request and unit records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Callable

from experiments.F006a_2 import run_analysis as f006a_2
from experiments.F006b import controller
from src.f002_protocol import binary_metrics, key, recording_units, verify_t006
from src.f007_protocol import (PROTOCOL, ROOT, load_protocol,
                               rational, require_test_authorization, validate_structure,
                               sha256_file, verify_inputs)


@dataclass(frozen=True)
class Plan:
    system: str
    resource: Fraction
    permitted_flops: Fraction
    checkpoint_key: str
    configuration: str | None
    width: float | None
    k: int | None
    full_flops: int
    deferral_reason: str | None

    @property
    def deferred(self) -> bool:
        return self.configuration is None


def state_string(resource: Fraction) -> str:
    return (format(Decimal(resource.numerator) / Decimal(resource.denominator), ".2f")
            if resource else "0")


def exact_string(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def route(system: str, resource: Fraction | str, protocol: dict,
          points: tuple[controller.OperatingPoint, ...],
          baseline: f006a_2.Quality) -> Plan:
    """Apply one exact allowance to static and adaptive systems alike."""
    if system not in protocol["systems"]:
        raise ValueError(f"Unknown F007 system: {system}")
    b = controller.resource_fraction(resource)
    capacity = rational(protocol["resource"]["ccapacity_flops"], "capacity")
    limit = b * capacity
    definition = protocol["systems"][system]
    checkpoint_key = definition["checkpoint"]
    if system in "ABC":
        full = definition["flops"]
        if system == "B" and full != controller.C_MAX:
            raise ValueError("Original TinyViT static FLOPs changed")
        if system == "C":
            matches = [p for p in points if p.width == float(definition["static_width"])
                       and p.k == definition["static_k"]]
            if len(matches) != 1 or matches[0].flops != full:
                raise ValueError("Canonical nested C configuration/cost changed")
        reason = "resource_unavailable" if b == 0 else "fixed_point_over_budget"
        if full > limit:
            return Plan(system, b, limit, checkpoint_key, None, None, None, 0, reason)
        width = float(definition["static_width"]) if system == "C" else None
        return Plan(system, b, limit, checkpoint_key,
                    {"A": "mobilenet_v2_k16", "B": "original_tinyvit_k16",
                     "C": "w0.5_k8"}[system], width, definition["static_k"], full, None)
    selection = controller.select(points, baseline, b, definition["policy"],
                                  capacity_flops=capacity)
    if selection.permitted_flops != limit:
        raise AssertionError("Controller used the wrong F007 capacity")
    point = selection.point
    if point is None:
        return Plan(system, b, limit, checkpoint_key, None, None, None, 0,
                    selection.reason)
    if system == "D" and point.name != definition["fixed_point"]:
        raise ValueError("Minimum-quality controller no longer selects D's fixed point")
    return Plan(system, b, limit, checkpoint_key, point.name, point.width,
                point.k, point.flops, None)


def mixed_assignments(rows: list[dict], unit_of: dict[tuple[str, str], str],
                      protocol: dict) -> tuple[dict[str, Fraction], list[Fraction]]:
    """Unit-hash schedule, independent of labels and original row ordering."""
    validate_structure(protocol)
    _require_complete_units(rows, unit_of)
    schedule = protocol["heterogeneous_schedule"]
    salt = schedule["salt"]
    units = set()
    for row in rows:
        units.add(unit_of[key(row)])
    ordered = sorted(units, key=lambda unit: (hashlib.sha256(
        (salt + unit).encode("utf-8")).digest(), unit))
    states = tuple(Fraction(state) for state in protocol["resource_states"])
    by_unit = {unit: states[index % len(states)] for index, unit in enumerate(ordered)}
    return by_unit, [by_unit[unit_of[key(row)]] for row in rows]


def _checkpoint_sha(protocol: dict, key_name: str) -> str:
    record = protocol["checkpoints"][key_name]
    return record.get("sha256", record.get("checkpoint_sha256"))


def _require_complete_units(rows: list[dict], unit_of: dict[tuple[str, str], str]) -> None:
    if any(key(row) not in unit_of for row in rows):
        raise ValueError("Unmapped shared-recording alias")
    all_sizes = Counter(unit_of.values())
    requested_sizes = Counter(unit_of[key(row)] for row in rows)
    if any(size != all_sizes[unit] for unit, size in requested_sizes.items()):
        raise ValueError("F007 request omits a retained alias of a recording unit")


def evaluate_requests(rows: list[dict], unit_of: dict[tuple[str, str], str],
                      systems: tuple[str, ...], states_by_row: list[Fraction],
                      protocol: dict, predictor: Callable[[dict, Plan], tuple[float, dict]],
                      *, partition: str = "validation", schedule_id: str = "factorial",
                      root: Path = ROOT) -> list[dict]:
    """Produce one record per alias × system, failing before test image access."""
    validate_structure(protocol)
    if partition == "test":
        require_test_authorization(protocol, root=root)
        verify_inputs(protocol, root=root, check_checkpoints=True)
    elif partition != "validation":
        raise ValueError("F007 Phase A1 accepts only validation or gated test requests")
    if len(rows) != len(states_by_row) or not rows:
        raise ValueError("Rows and resource assignments must be nonempty and aligned")
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate alias request")
    if set(systems) - set(protocol["systems"]) or len(systems) != len(set(systems)):
        raise ValueError("Invalid F007 system list")
    if any(row.get("partition") != partition for row in rows):
        raise ValueError("Row partition disagrees with the guarded execution path")
    if any(key(row) not in unit_of for row in rows):
        raise ValueError("Unmapped shared-recording alias")
    _require_complete_units(rows, unit_of)
    by_unit: dict[str, set[Fraction]] = defaultdict(set)
    for row, state in zip(rows, states_by_row):
        if Fraction(state) not in {Fraction(s) for s in protocol["resource_states"]}:
            raise ValueError("Non-F007 resource state")
        by_unit[unit_of[key(row)]].add(Fraction(state))
    if any(len(values) != 1 for values in by_unit.values()):
        raise ValueError("Shared-recording aliases received different resource states")
    baseline = f006a_2.read_baseline()
    points = controller.load_policy_points()
    records = []
    protocol_sha = sha256_file(root / "configs" / "f007_protocol.json")
    for system in systems:
        for row, state in zip(rows, states_by_row):
            plan = route(system, state, protocol, points, baseline)
            if plan.full_flops > plan.permitted_flops:
                raise ValueError("F007 exact budget violation")
            probability, details = (None, {}) if plan.deferred else predictor(row, plan)
            if (not isinstance(details, dict) or
                    set(details) - {"frame_indices", "frame_files", "new_encoded_frames"} or
                    any(not isinstance(details.get(name), str) for name in
                        set(details) & {"frame_indices", "frame_files"})):
                raise ValueError("Invalid F007 frame diagnostic fields")
            details = dict(details)
            new_frames = details.pop("new_encoded_frames", None)
            if new_frames is not None and (type(new_frames) is not int or
                                           not 0 <= new_frames <= plan.k):
                raise ValueError("Invalid F007 cache diagnostic")
            if not plan.deferred and (type(probability) not in (float, int) or
                                      not math.isfinite(probability) or
                                      not 0 <= probability <= 1):
                raise ValueError("Invalid model probability")
            records.append({
                "system_id": system, "schedule_id": schedule_id,
                "unit_id": unit_of[key(row)],
                "exam_id": row["exam_id"], "zone_id": row["zone_id"],
                "binary_target": int(row["binary_target"]),
                "split_group": row["split_group"], "device": row.get("device", ""),
                "partition": partition, "resource_state": state_string(plan.resource),
                "permitted_flops_exact": exact_string(plan.permitted_flops),
                "configuration": plan.configuration or "DEFER",
                "width": plan.width, "k": plan.k,
                "selection_policy": protocol["systems"][system].get("policy"),
                "deferred": plan.deferred, "deferral_reason": plan.deferral_reason,
                "full_flops": plan.full_flops, "probability": probability,
                "cache_mode": "no_cache_deferred" if plan.deferred else "fresh_per_alias_request",
                "new_encoded_frames": 0 if plan.deferred else new_frames,
                "checkpoint_key": plan.checkpoint_key,
                "checkpoint_sha256": _checkpoint_sha(protocol, plan.checkpoint_key),
                "protocol_sha256": protocol_sha, "preprocessing":
                    protocol["checkpoints"][plan.checkpoint_key]["input_path"],
                **details,
            })
    return records


def aggregate_requests(rows: list[dict], unit_of: dict[tuple[str, str], str],
                       records: list[dict], system: str, *,
                       schedule_id: str = "factorial") -> dict:
    """Duplicate-aware completed-unit metrics with explicit deferral coverage."""
    if not rows:
        raise ValueError("F007 unit aggregation requires nonempty requests")
    expected = {key(row): row for row in rows}
    _require_complete_units(rows, unit_of)
    selected = [record for record in records if record["system_id"] == system
                and record["schedule_id"] == schedule_id]
    observed = {key(record): record for record in selected}
    if len(expected) != len(rows) or len(selected) != len(observed) or \
            expected.keys() != observed.keys():
        raise ValueError("F007 alias records must cover requested rows exactly once")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for alias, row in expected.items():
        record = observed[alias]
        if (record["unit_id"] != unit_of[alias] or
                int(record["binary_target"]) != int(row["binary_target"]) or
                record["split_group"] != row["split_group"] or
                record["partition"] != row["partition"]):
            raise ValueError("F007 alias identity/target/group mismatch")
        grouped[record["unit_id"]].append(record)
    units = []
    for unit_id, aliases in sorted(grouped.items()):
        targets = {alias["binary_target"] for alias in aliases}
        groups = {alias["split_group"] for alias in aliases}
        states = {alias["resource_state"] for alias in aliases}
        deferred = {alias["deferred"] for alias in aliases}
        signatures = {(alias["configuration"], alias["full_flops"],
                       alias["permitted_flops_exact"], alias["checkpoint_key"],
                       alias["checkpoint_sha256"], alias["protocol_sha256"],
                       alias["deferral_reason"]) for alias in aliases}
        if (len(targets) != 1 or len(groups) != 1 or len(states) != 1 or
                len(deferred) != 1 or len(signatures) != 1):
            raise ValueError("F007 shared unit has conflicting alias evidence")
        completed = not next(iter(deferred))
        probability = (sum(alias["probability"] for alias in aliases) / len(aliases)
                       if completed else None)
        if completed and (not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("F007 unit probability is invalid")
        devices = {alias["device"] for alias in aliases}
        units.append({"unit_id": unit_id, "aliases": [[alias["exam_id"], alias["zone_id"]]
                                               for alias in aliases],
                      "system_id": system, "schedule_id": schedule_id,
                      "binary_target": next(iter(targets)),
                      "split_group": next(iter(groups)), "resource_state": next(iter(states)),
                      "permitted_flops_exact": aliases[0]["permitted_flops_exact"],
                      "configuration": aliases[0]["configuration"],
                      "checkpoint_key": aliases[0]["checkpoint_key"],
                      "checkpoint_sha256": aliases[0]["checkpoint_sha256"],
                      "protocol_sha256": aliases[0]["protocol_sha256"],
                      "device": next(iter(devices)) if len(devices) == 1 else "mixed",
                      "completed": completed, "probability": probability,
                      "predicted_class": int(probability >= 0.5) if completed else None,
                      "full_flops": sum(alias["full_flops"] for alias in aliases),
                      "deferral_reason": None if completed else aliases[0]["deferral_reason"]})
    completed = [unit for unit in units if unit["completed"]]
    targets = [unit["binary_target"] for unit in completed]
    probabilities = [unit["probability"] for unit in completed]
    if completed:
        metrics = binary_metrics(targets, probabilities, 0.5)
        if 0 not in targets:
            metrics["normal_recall"] = None
        if 1 not in targets:
            metrics["abnormal_recall"] = None
        if len(set(targets)) != 2:
            metrics["macro_f1"] = None
            metrics["auroc"] = None
        else:
            from sklearn.metrics import roc_auc_score
            metrics["auroc"] = float(roc_auc_score(targets, probabilities))
    else:
        metrics = {name: None for name in ("accuracy", "macro_f1", "normal_recall",
                                          "abnormal_recall", "auroc")}
        metrics.update(tn=0, fp=0, fn=0, tp=0)
    total_flops = sum(record["full_flops"] for record in selected)
    completed_rows = sum(not record["deferred"] for record in selected)
    return {"system_id": system, "schedule_id": schedule_id, "units": units,
            "summary": {**metrics, "requested_units": len(units),
                        "completed_units": len(completed), "deferred_units": len(units) - len(completed),
                        "requested_rows": len(rows), "completed_rows": completed_rows,
                        "deferred_rows": len(rows) - completed_rows,
                        "coverage": len(completed) / len(units),
                        "full_flops": total_flops,
                        "mean_full_flops_per_completed_row_exact":
                            exact_string(Fraction(total_flops, completed_rows)) if completed_rows else None,
                        "mean_full_flops_per_requested_row_exact":
                            exact_string(Fraction(total_flops, len(rows)))}}


class CanonicalModels:
    """Lazy frozen loaders; a new sequence/clip cache is made per served row."""

    def __init__(self, protocol: dict, *, partition: str = "validation", device="cpu"):
        if protocol != load_protocol():
            raise ValueError("F007 predictor protocol differs from the local protocol file")
        if partition == "test":
            require_test_authorization(protocol)
        elif partition != "validation":
            raise ValueError("Only validation or authorized test inference is permitted")
        self.verification = (verify_inputs(protocol, check_checkpoints=True)
                             if partition == "validation" else None)
        from src import f002_pipeline as pipe
        self.partition, self.device, self.models = partition, device, {}
        if partition == "validation":
            self.allowed_rows = {key(row): row for row in pipe.load_inputs()["validation"]}
        else:
            self.allowed_rows = {key(row): row for row in verify_t006(
                ROOT / "configs/f002_protocol.json", ROOT / "data/openpocus/t006")["test"]}

    def __call__(self, row: dict, plan: Plan) -> tuple[float, dict]:
        if (row.get("partition") != self.partition or plan.deferred or
                self.allowed_rows.get(key(row)) != row):
            raise ValueError("Invalid image request for canonical predictor")
        import torch
        from experiments.F003.adaptive import F003Sequences
        from experiments.F004.family import load_family
        from experiments.F004.run_f004 import load_static_f002b
        from experiments.F005.prune import NestedAnytimeClip
        from experiments.F005.run_f005b import load_deployable
        from src import f002_pipeline as pipe

        if plan.checkpoint_key not in self.models:
            loader = {"f002b_mobilenetv2": load_static_f002b,
                      "f002c_original_tinyvit": load_family,
                      "f005b_nested_tinyvit": load_deployable}[plan.checkpoint_key]
            self.models[plan.checkpoint_key] = loader(device=self.device)
        model = self.models[plan.checkpoint_key]
        with torch.inference_mode():
            if plan.system == "A":
                sequence = pipe.Sequences([row], 16)
                logit = model(sequence.frames(0).unsqueeze(0).to(self.device)).float().item()
                identity = sequence.identity(0)
                new_frames = plan.k
            else:
                sequence = F003Sequences([row], 224)
                if plan.checkpoint_key == "f002c_original_tinyvit":
                    logit = model(sequence.frames(0, plan.k).unsqueeze(0).to(self.device)).float().item()
                    new_frames = plan.k
                else:
                    clip = NestedAnytimeClip(model, int(row["frame_count"]),
                        lambda numbers: sequence.frames_at(0, numbers).to(self.device))
                    nested_logit, new_frames = clip.predict(plan.width, plan.k)
                    if new_frames != plan.k:
                        raise ValueError("Primary F007 request reused a clip cache")
                    logit = nested_logit.float().item()
                from experiments.F003.adaptive import frame_indices
                identity = {"frame_indices": " ".join(map(str, frame_indices(
                    int(row["frame_count"]), plan.k)))}
            identity["new_encoded_frames"] = new_frames
        if not math.isfinite(logit):
            raise FloatingPointError("Nonfinite F007 logit")
        probability = (1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else
                       math.exp(logit) / (1.0 + math.exp(logit)))
        return probability, identity


def validation_context(protocol: dict, *, full_hashes: bool = True
                       ) -> tuple[list[dict], dict, dict]:
    """Structural verification and frozen recording units; no image decoding."""
    verification = verify_inputs(protocol, check_checkpoints=full_hashes)
    partitions = verify_t006(ROOT / "configs/f002_protocol.json", ROOT / "data/openpocus/t006")
    directory = ROOT / "data/openpocus/t006"
    with (directory / "shared_recording_edges.csv").open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    with (directory / "exclusions.csv").open(newline="", encoding="utf-8") as handle:
        exclusions = list(csv.DictReader(handle))
    return partitions["validation"], recording_units(partitions, edges, exclusions), verification


def write_evidence(directory: Path, requests: list[dict], results: list[dict],
                   provenance: dict) -> None:
    """Create one ignored attempt directory and never overwrite prior evidence."""
    directory.mkdir(parents=True, exist_ok=False)
    for name, records in (("requests.jsonl", requests),
                          ("units.jsonl", [unit for result in results for unit in result["units"]])):
        with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    summaries: dict[str, dict] = defaultdict(dict)
    for result in results:
        schedule, system = result["schedule_id"], result["system_id"]
        if system in summaries[schedule]:
            raise ValueError("Duplicate F007 schedule/system summary")
        summaries[schedule][system] = result["summary"]
    (directory / "summary.json").write_text(json.dumps(
        summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "run.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)
                                              + "\n", encoding="utf-8")


def smoke_validation(output: Path, *, rows_per_class: int = 1, device="cpu") -> dict:
    """A bounded real-validation run; no test partition image path is accepted."""
    if not output.resolve().is_relative_to((ROOT / "artifacts" / "F007").resolve()):
        raise ValueError("F007 generated evidence must stay inside ignored artifacts/F007")
    protocol = load_protocol()
    rows, unit_of, _ = validation_context(protocol, full_hashes=False)
    counts = Counter(unit_of[key(row)] for row in rows)
    chosen = []
    for target in (0, 1):
        singleton = [row for row in rows if int(row["binary_target"]) == target
                     and counts[unit_of[key(row)]] == 1]
        if len(singleton) < rows_per_class:
            raise ValueError("Not enough singleton validation rows for bounded smoke")
        chosen.extend(singleton[:rows_per_class])
    chosen.sort(key=lambda row: rows.index(row))
    predictor = CanonicalModels(protocol, device=device)
    verification = predictor.verification
    systems = tuple(protocol["systems"])
    states = [Fraction(1)] * len(chosen)
    served = evaluate_requests(chosen, unit_of, systems, states, protocol, predictor,
                               schedule_id="factorial_b1")
    deferred = evaluate_requests(chosen, unit_of, systems, [Fraction(0)] * len(chosen),
                                 protocol, predictor, schedule_id="factorial_b0")
    requests = served + deferred
    results = [aggregate_requests(chosen, unit_of, records, system, schedule_id=schedule)
               for schedule, records in (("factorial_b1", served), ("factorial_b0", deferred))
               for system in systems]
    provenance = {"experiment": "F007", "phase": "A1_validation_smoke",
                  "status": "completed", "partition": "validation", "systems": list(systems),
                  "resource_states": ["1.00", "0"], "rows": len(chosen),
                  "protocol_sha256": sha256_file(PROTOCOL),
                  "frozen_inputs": verification,
                  "checkpoints": protocol["checkpoints"],
                  "code_sha256": {"phase_a1.py": sha256_file(Path(__file__)),
                                  "f007_protocol.py": sha256_file(ROOT / "src/f007_protocol.py"),
                                  "controller.py": sha256_file(ROOT / "experiments/F006b/controller.py")},
                  "request_ids": [[row["exam_id"], row["zone_id"]] for row in chosen],
                  "test_inference_authorized": False,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
    write_evidence(output, requests, results, provenance)
    return {"output": str(output), "systems": list(systems), "rows": len(chosen),
            "requests": len(requests),
            "completed_requests": sum(not record["deferred"] for record in requests),
            "summaries": {schedule: {result["system_id"]: result["summary"]
                                    for result in results if result["schedule_id"] == schedule}
                          for schedule in ("factorial_b1", "factorial_b0")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("smoke-validation", nargs="?", default="smoke-validation",
                        choices=("smoke-validation",))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts" / "F007" / f"phase_a1_validation_{stamp}"
    print(json.dumps(smoke_validation(output, device=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
