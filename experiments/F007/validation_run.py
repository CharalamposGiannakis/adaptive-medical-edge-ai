"""Produce validation evidence and share guarded F007 request orchestration."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from experiments.F007.phase_a1 import (CanonicalModels, Plan, aggregate_requests,
                                        evaluate_requests, exact_string, mixed_assignments,
                                        state_string, validation_context, write_evidence)
from experiments.F007.provenance import assert_unchanged, capture
from src.f002_protocol import key
from src.f007_protocol import (ROOT, load_protocol, rational,
                               require_test_authorization, sha256_file)


def _select(rows: list[dict], unit_of: dict, limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("Diagnostic unit limit must be positive")
    units = list(dict.fromkeys(unit_of[key(row)] for row in rows))[:limit]
    chosen = set(units)
    return [row for row in rows if unit_of[key(row)] in chosen]


def _supplementary(rows: list[dict], unit_of: dict, protocol: dict,
                   predictor: CanonicalModels, *, system: str, state: str,
                   k: int, width: float | None, flops: int,
                   checkpoint_key: str, configuration: str,
                   partition: str = "validation") -> list[dict]:
    """Use the A1 predictor and record shape for two prespecified diagnostics."""
    from experiments.F007.phase_a1 import _checkpoint_sha

    if partition == "test":
        require_test_authorization(protocol)
    elif partition != "validation":
        raise ValueError("Invalid F007 supplementary partition")

    expected = {
        "f002c_original_k4": ("0.25", 4, None, 5149017395,
                               "f002c_original_tinyvit", "original_tinyvit_k4"),
        "f005b_nested_w1.0_k16": ("1.00", 16, 1.0, 20596068563,
                                   "f005b_nested_tinyvit", "w1.0_k16"),
    }
    if expected.get(system) != (state, k, width, flops, checkpoint_key, configuration):
        raise ValueError("Unknown F007 supplementary comparison")
    resource = Fraction(state)
    budget = resource * rational(protocol["resource"]["ccapacity_flops"], "capacity")
    if flops > budget:
        raise ValueError("Supplementary F007 configuration exceeds exact budget")
    plan = Plan(system, resource, budget, checkpoint_key, configuration,
                width, k, flops, None)
    schedule = f"factorial_b{state}"
    protocol_sha = sha256_file(ROOT / "configs/f007_protocol.json")
    records = []
    for row in rows:
        probability, details = predictor(row, plan)
        new_frames = details.pop("new_encoded_frames", None)
        records.append({"system_id": system, "schedule_id": schedule,
                        "unit_id": unit_of[key(row)], "exam_id": row["exam_id"],
                        "zone_id": row["zone_id"], "binary_target": int(row["binary_target"]),
                        "split_group": row["split_group"], "device": row.get("device", ""),
                        "partition": partition, "resource_state": state_string(resource),
                        "permitted_flops_exact": exact_string(budget),
                        "configuration": configuration, "width": width, "k": k,
                        "selection_policy": None, "deferred": False,
                        "deferral_reason": None, "full_flops": flops,
                        "probability": probability, "cache_mode": "fresh_per_alias_request",
                        "new_encoded_frames": new_frames, "checkpoint_key": checkpoint_key,
                        "checkpoint_sha256": _checkpoint_sha(protocol, checkpoint_key),
                        "protocol_sha256": protocol_sha,
                        "preprocessing": protocol["checkpoints"][checkpoint_key]["input_path"],
                        **details})
    return records


def assert_identical_predictions(records: list[dict]) -> int:
    """Independent fresh requests for one frozen point must agree bit for bit."""
    seen = {}
    checked = 0
    for record in records:
        if record["deferred"]:
            continue
        identity = (record["exam_id"], record["zone_id"],
                    record["checkpoint_key"], record["configuration"])
        evidence = (record["checkpoint_sha256"], record["preprocessing"],
                    record["width"], record["k"], record["frame_indices"],
                    record["probability"], record["full_flops"])
        if identity in seen:
            checked += 1
            if seen[identity] != evidence:
                raise RuntimeError("Identical F007 configuration produced unequal fresh "
                                   f"predictions: {identity}")
        else:
            seen[identity] = evidence
    return checked


def collect_requests(rows: list[dict], unit_of: dict, protocol: dict,
                     predictor: CanonicalModels, *, states: tuple[str, ...],
                     partition: str) -> tuple[list[dict], list[dict], int]:
    """Run the frozen factorial, mixed and supplementary requests for one split."""
    if partition == "test":
        require_test_authorization(protocol)
    elif partition != "validation":
        raise ValueError("Invalid F007 request partition")
    requests, results = [], []
    systems = tuple(protocol["systems"])
    for state in states:
        schedule = f"factorial_b{state}"
        records = evaluate_requests(rows, unit_of, systems,
                                    [Fraction(state)] * len(rows), protocol,
                                    predictor, partition=partition, schedule_id=schedule)
        requests.extend(records)
        results.extend(aggregate_requests(rows, unit_of, records, system,
                                         schedule_id=schedule) for system in systems)
    _, mixed_states = mixed_assignments(rows, unit_of, protocol)
    mixed = evaluate_requests(rows, unit_of, systems, mixed_states, protocol,
                              predictor, partition=partition,
                              schedule_id="heterogeneous")
    requests.extend(mixed)
    results.extend(aggregate_requests(rows, unit_of, mixed, system,
                                      schedule_id="heterogeneous") for system in systems)
    extras = (("f002c_original_k4", "0.25", 4, None, 5149017395,
               "f002c_original_tinyvit", "original_tinyvit_k4"),
              ("f005b_nested_w1.0_k16", "1.00", 16, 1.0, 20596068563,
               "f005b_nested_tinyvit", "w1.0_k16"))
    for system, state, k, width, flops, checkpoint, config in extras:
        if state not in states:
            continue
        records = _supplementary(rows, unit_of, protocol, predictor,
                                 system=system, state=state, k=k, width=width,
                                 flops=flops, checkpoint_key=checkpoint,
                                 configuration=config, partition=partition)
        requests.extend(records)
        results.append(aggregate_requests(rows, unit_of, records, system,
                                          schedule_id=f"factorial_b{state}"))
    return requests, results, assert_identical_predictions(requests)


def run(*, unit_limit: int | None, diagnostic_states: tuple[str, ...] | None,
        output: Path, device: str = "cpu") -> dict:
    if not output.resolve().is_relative_to((ROOT / "artifacts/F007").resolve()):
        raise ValueError("F007 validation evidence belongs in ignored artifacts/F007")
    if unit_limit is None and diagnostic_states is not None:
        raise ValueError("State subsets are permitted only in diagnostic mode")
    protocol = load_protocol()
    states = tuple(protocol["resource_states"] if diagnostic_states is None else diagnostic_states)
    if not states or len(states) != len(set(states)) or set(states) - set(protocol["resource_states"]):
        raise ValueError("Invalid F007 validation diagnostic states")
    if device == "cpu":
        import torch
        torch.set_num_threads(1)  # fixed serial CPU reductions for exact repeat identity
    all_rows, unit_of, verification = validation_context(protocol, full_hashes=False)
    rows = _select(all_rows, unit_of, unit_limit)
    provenance = capture(protocol, verification, official=unit_limit is None,
                         execution_device=device, parameters={
                             "unit_limit": unit_limit, "diagnostic_states": diagnostic_states,
                             "resource_states": states, "output": str(output),
                             "partition": "validation"})
    predictor = CanonicalModels(protocol, device=device)
    provenance["frozen_inputs"] = predictor.verification
    systems = tuple(protocol["systems"])
    requests, results, identical_checks = collect_requests(
        rows, unit_of, protocol, predictor, states=states, partition="validation")
    assert_unchanged(provenance)
    run_record = {"experiment": "F007", "phase": "A2_validation_evidence",
                  "status": "completed", "partition": "validation",
                  "diagnostic_only": unit_limit is not None, "device": device,
                  "states": states, "systems": systems,
                  "rows": len(rows), "units": len({unit_of[key(row)] for row in rows}),
                  "request_ids": [[row["exam_id"], row["zone_id"]] for row in rows],
                  "group_count": len({row["split_group"] for row in rows}),
                  "class_counts": dict(Counter(str(row["binary_target"]) for row in rows)),
                  "protocol_sha256": sha256_file(ROOT / "configs/f007_protocol.json"),
                  "frozen_inputs": predictor.verification,
                  "checkpoints": protocol["checkpoints"],
                  "provenance": provenance,
                  "identical_fresh_predictions_checked": identical_checks,
                  "test_inference_authorized": False,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
    write_evidence(output, requests, results, run_record)
    return {"output": str(output), "diagnostic_only": unit_limit is not None,
            "rows": len(rows), "units": run_record["units"], "requests": len(requests),
            "completed_requests": sum(not record["deferred"] for record in requests),
            "schedules": len(states) + 1, "systems": len(systems)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all-validation-units", action="store_true")
    choice.add_argument("--diagnostic-units", type=int)
    parser.add_argument("--diagnostic-states", nargs="+",
                        help="optional subset of frozen states; diagnostic mode only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts/F007" / f"validation_a2_{stamp}"
    print(json.dumps(run(unit_limit=None if args.all_validation_units else args.diagnostic_units,
                         diagnostic_states=tuple(args.diagnostic_states) if
                         args.diagnostic_states is not None else None,
                         output=output, device=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
