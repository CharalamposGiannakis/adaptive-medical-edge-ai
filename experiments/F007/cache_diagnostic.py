"""F007 separate within-clip cache diagnostic on validation aliases only."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from experiments.F003.adaptive import F003Sequences
from experiments.F005.run_f005b import load_deployable
from experiments.F006a_2 import run_analysis as f006a_2
from experiments.F006b import controller
from experiments.F006b.runtime import NestedClipRuntime, direct_predict
from experiments.F007.phase_a1 import route, state_string, validation_context
from experiments.F007.provenance import assert_unchanged, capture
from src.f002_protocol import key
from src.f007_protocol import ROOT, load_protocol, sha256_file

TOLERANCE = 1e-4
POLICIES = ("D", "E", "F")
ORDERS = ("increasing", "decreasing")


def select_validation_units(rows: list[dict], unit_of: dict, limit: int | None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("Diagnostic unit limit must be positive")
    ordered = list(dict.fromkeys(unit_of[key(row)] for row in rows))
    chosen = set(ordered if limit is None else ordered[:limit])
    return [row for row in rows if unit_of[key(row)] in chosen]


def diagnose(rows: list[dict], unit_of: dict, protocol: dict, model,
             points: tuple[controller.OperatingPoint, ...], baseline) -> list[dict]:
    """Fresh NestedAnytimeClip for every alias × policy × order."""
    if not rows or any(row.get("partition") != "validation" for row in rows):
        raise PermissionError("F007 cache diagnostic accepts validation rows only")
    states = [Fraction(value) for value in protocol["resource_states"]]
    records = []
    for row in rows:
        sequence = F003Sequences([row], 224)
        frame_count = int(row["frame_count"])
        loader = lambda numbers: sequence.frames_at(0, numbers).to("cpu")
        for system in POLICIES:
            for order in ORDERS:
                runtime = NestedClipRuntime(model, frame_count, loader, chunk_size=4)
                previous = None
                for state in (states if order == "increasing" else list(reversed(states))):
                    plan = route(system, state, protocol, points, baseline)
                    choice = plan.configuration or "DEFER"
                    switch = ("none" if previous is None or previous == choice else
                              "from_defer" if previous == "DEFER" else
                              "to_defer" if choice == "DEFER" else "active_to_active")
                    record = {"partition": "validation", "system_id": system,
                              "order": order, "exam_id": row["exam_id"],
                              "zone_id": row["zone_id"], "unit_id": unit_of[key(row)],
                              "resource_state": state_string(state),
                              "configuration": choice, "deferral_reason": plan.deferral_reason,
                              "switch": switch, "full_independent_flops": plan.full_flops,
                              "checkpoint_sha256": protocol["checkpoints"]["f005b_nested_tinyvit"][
                                  "checkpoint_sha256"]}
                    if plan.deferred:
                        record.update(new_encoded_frames=0, cached_frames=0,
                                      incremental_cached_flops=0, logit=None,
                                      direct_logit=None, abs_logit_difference=None,
                                      cache_entries=len(runtime._cache.features))
                    else:
                        cached = runtime.predict(plan.width, plan.k)
                        direct = direct_predict(model, frame_count, loader,
                                                plan.width, plan.k, chunk_size=4)
                        difference = abs(cached.logit - direct)
                        if (not math.isfinite(difference) or difference > TOLERANCE or
                                (cached.logit >= 0) != (direct >= 0)):
                            raise RuntimeError("F007 cached/direct logit or class mismatch: "
                                               f"{row['exam_id']}/{row['zone_id']} "
                                               f"{system} {order} {choice}: {difference}")
                        incremental = controller.incremental_flops(
                            points, plan.width, cached.new_frames)
                        if incremental > plan.full_flops:
                            raise ValueError("Cached incremental FLOPs exceed independent work")
                        record.update(new_encoded_frames=cached.new_frames,
                                      cached_frames=cached.cached_frames,
                                      incremental_cached_flops=incremental,
                                      logit=cached.logit, direct_logit=direct,
                                      abs_logit_difference=difference,
                                      cache_entries=cached.cache_entries,
                                      elapsed_ms_diagnostic=cached.elapsed_ms)
                    records.append(record)
                    previous = choice
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No F007 cache diagnostic records")
    return {"alias_rows": len({(r["exam_id"], r["zone_id"]) for r in records}),
            "requests": len(records), "served": sum(r["configuration"] != "DEFER" for r in records),
            "deferred": sum(r["configuration"] == "DEFER" for r in records),
            "switches": dict(sorted(Counter(r["switch"] for r in records).items())),
            "new_encoded_frames": sum(r["new_encoded_frames"] for r in records),
            "full_independent_flops": sum(r["full_independent_flops"] for r in records),
            "incremental_cached_flops": sum(r["incremental_cached_flops"] for r in records),
            "maximum_abs_logit_difference": max(
                (r["abs_logit_difference"] or 0.0) for r in records),
            "method": "within-clip cache, separate from primary independent-request FLOPs"}


def run(*, unit_limit: int | None, output: Path) -> dict:
    if not output.resolve().is_relative_to((ROOT / "artifacts/F007").resolve()):
        raise ValueError("F007 cache evidence belongs in ignored artifacts/F007")
    protocol = load_protocol()
    rows, unit_of, verification = validation_context(protocol, full_hashes=True)
    chosen = select_validation_units(rows, unit_of, unit_limit)
    provenance = capture(protocol, verification, official=unit_limit is None,
                         execution_device="cpu", parameters={
                             "unit_limit": unit_limit, "output": str(output),
                             "partition": "validation", "policies": POLICIES,
                             "orders": ORDERS})
    points = controller.load_policy_points()
    baseline = f006a_2.read_baseline()
    model = load_deployable(device="cpu")
    records = diagnose(chosen, unit_of, protocol, model, points, baseline)
    summary = summarize(records)
    assert_unchanged(provenance)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "requests.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)
                                          + "\n", encoding="utf-8")
    (output / "run.json").write_text(json.dumps({
        "experiment": "F007", "phase": "A2_cache_diagnostic", "partition": "validation",
        "status": "completed", "diagnostic_only": unit_limit is not None,
        "requested_ids": [[r["exam_id"], r["zone_id"]] for r in chosen],
        "frozen_inputs": verification, "protocol_sha256": sha256_file(
            ROOT / "configs/f007_protocol.json"),
        "checkpoints": protocol["checkpoints"], "provenance": provenance,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_inference_authorized": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"output": str(output), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all-validation-aliases", action="store_true")
    choice.add_argument("--diagnostic-units", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    limit = None if args.all_validation_aliases else args.diagnostic_units
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts/F007" / f"cache_validation_{stamp}"
    print(json.dumps(run(unit_limit=limit, output=output), indent=2), flush=True)


if __name__ == "__main__":
    main()
