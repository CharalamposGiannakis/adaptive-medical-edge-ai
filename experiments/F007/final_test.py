"""Guarded execution of the frozen F007 held-out comparison.

The authorization gate runs before constructing a test predictor or requesting
an image. The currently proposed protocol deliberately denies this command.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.F007 import analysis
from experiments.F007.phase_a1 import (CanonicalModels, partition_context,
                                        write_evidence)
from experiments.F007.provenance import assert_unchanged, capture
from experiments.F007.validation_run import collect_requests
from src.f002_protocol import key
from src.f007_protocol import ROOT, load_protocol, require_test_authorization, sha256_file


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_attempt(path: Path, record: dict) -> None:
    # Only the status record inside this freshly created attempt is updated.
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(*, output: Path, device: str = "cpu") -> dict:
    """Execute all frozen test requests only after committed authorization."""
    output = Path(output)
    base = (ROOT / "artifacts/F007").resolve()
    if output.resolve() == base or not output.resolve().is_relative_to(base):
        raise ValueError("F007 test evidence belongs in a new artifacts/F007 attempt")
    if device != "cpu":
        raise ValueError("F007 final-test runner currently uses the fixed CPU inference path")
    output.mkdir(parents=True, exist_ok=False)
    attempt_path = output / "attempt.json"
    attempt = {"experiment": "F007", "partition": "test", "status": "started",
               "started_utc": _utc(), "output": str(output), "device": device}
    _write_attempt(attempt_path, attempt)
    try:
        protocol = load_protocol()
        # No test manifest, image loader, predictor, or inference exists before
        # this gate. It also verifies committed approval and all frozen hashes.
        approval = require_test_authorization(protocol)
        attempt.update(status="authorized_preflight", authorization=approval)
        _write_attempt(attempt_path, attempt)
        import torch
        torch.set_num_threads(1)
        rows, unit_of, _ = partition_context(protocol, "test", full_hashes=False)
        verification = approval["frozen_inputs"]
        provenance = capture(protocol, verification, official=True,
                             execution_device=device,
                             parameters={"partition": "test", "output": str(output),
                                         "states": tuple(protocol["resource_states"]),
                                         "systems": tuple(protocol["systems"]),
                                         "heterogeneous_schedule": True,
                                         "supplementary_comparisons": True})
        attempt.update(status="running", provenance=provenance,
                       rows=len(rows), units=len({unit_of[key(row)] for row in rows}),
                       request_ids=[[row["exam_id"], row["zone_id"]] for row in rows])
        _write_attempt(attempt_path, attempt)
        predictor = CanonicalModels(protocol, partition="test", device=device)
        states = tuple(protocol["resource_states"])
        systems = tuple(protocol["systems"])
        requests, results, identical_checks = collect_requests(
            rows, unit_of, protocol, predictor, states=states, partition="test")
        assert_unchanged(provenance)
        evidence_dir = output / "evidence"
        report_dir = output / "report"
        run_record = {"experiment": "F007", "phase": "final_test_predictions",
                      "status": "completed", "partition": "test",
                      "diagnostic_only": False, "device": device,
                      "states": states, "systems": systems,
                      "rows": len(rows), "units": len({unit_of[key(row)] for row in rows}),
                      "request_ids": [[row["exam_id"], row["zone_id"]] for row in rows],
                      "group_count": len({row["split_group"] for row in rows}),
                      "class_counts": dict(Counter(str(row["binary_target"]) for row in rows)),
                      "protocol_sha256": sha256_file(ROOT / "configs/f007_protocol.json"),
                      "frozen_inputs": verification, "checkpoints": protocol["checkpoints"],
                      "authorization": approval, "provenance": provenance,
                      "identical_fresh_predictions_checked": identical_checks,
                      "test_inference_authorized": True, "created_utc": _utc()}
        write_evidence(evidence_dir, requests, results, run_record)
        cells = defaultdict(lambda: defaultdict(list))
        for result in results:
            cells[result["schedule_id"]][result["system_id"]].extend(result["units"])
        report, draws = analysis.analyze(cells, requests, protocol,
                                         require_complete=True, partition="test")
        report["authorization"] = approval
        report["provenance"] = provenance
        report["source_evidence_sha256"] = {
            name: sha256_file(evidence_dir / name)
            for name in ("run.json", "requests.jsonl", "units.jsonl")}
        assert_unchanged(provenance)
        analysis.write_report(report_dir, report, draws, source=evidence_dir)
        assert_unchanged(provenance)
        attempt.update(status="completed", completed_utc=_utc(),
                       evidence=str(evidence_dir), report=str(report_dir),
                       request_count=len(requests), unit_count=run_record["units"],
                       authorization=approval, provenance=provenance)
        _write_attempt(attempt_path, attempt)
        return {"output": str(output), "status": "completed",
                "rows": len(rows), "units": run_record["units"],
                "requests": len(requests), "cells": len(report["cells"]),
                "contrasts": len(report["contrasts"])}
    except Exception as exc:
        attempt.update(status="failed", failed_utc=_utc(),
                       error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, attempt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        help="new ignored artifacts/F007 attempt directory")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts/F007" / f"final_test_{stamp}"
    print(json.dumps(run(output=output, device=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
