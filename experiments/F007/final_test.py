"""Guarded execution of the frozen F007 held-out comparison.

The authorization gate runs before constructing a test predictor or requesting
an image. The currently proposed protocol deliberately denies this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.F007 import analysis
from experiments.F007.phase_a1 import (CanonicalModels, partition_context,
                                        preflight_test_inventories, write_evidence)
from experiments.F007.provenance import SOURCE_MODULES, assert_unchanged, capture
from experiments.F007.validation_run import collect_requests
from src.f002_protocol import key
from src.f007_protocol import (ROOT, git_blob, load_protocol,
                               require_test_authorization, sha256_file)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_attempt(path: Path, record: dict) -> None:
    # Only the status record inside this freshly created attempt is updated.
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempt_directory(output: Path) -> Path:
    output = Path(output)
    base = (ROOT / "artifacts/F007").resolve()
    if output.resolve() == base or not output.resolve().is_relative_to(base):
        raise ValueError("F007 evidence belongs in a new artifacts/F007 attempt")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _source_evidence_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name)
            for name in ("run.json", "requests.jsonl", "units.jsonl", "summary.json")}


def _verify_original_provenance(source_run: dict, approval: dict) -> None:
    """Authenticate original clean source against its Git commit, not current code."""
    recorded = source_run.get("provenance", {})
    commit = recorded.get("commit_sha")
    if (recorded.get("official") is not True or
            recorded.get("working_tree_dirty") is not False or
            not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("F007 source run lacks clean official Git provenance")
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                       cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("F007 source commit is not in repository ancestry") from exc
    sources = recorded.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_MODULES):
        raise ValueError("F007 source module hashes are absent")
    for name, hashes in sources.items():
        if (not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts or
                not isinstance(hashes, dict) or
                not isinstance(hashes.get("committed_blob_sha256"), str) or
                not isinstance(hashes.get("working_tree_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", hashes["working_tree_sha256"])):
            raise ValueError("F007 source module provenance is malformed")
        committed = hashlib.sha256(git_blob(ROOT, commit, name)).hexdigest()
        if committed != hashes["committed_blob_sha256"]:
            raise ValueError(f"F007 original source Git hash mismatch: {name}")
        # The original working bytes may have CRLF; its clean-tree assertion
        # and the independently checked committed blob anchor that record.
    authorization = source_run.get("authorization", {})
    for field in ("protocol_commit", "accepted_protocol_sha256",
                  "approval_receipt_sha256", "independent_review_ref",
                  "charalampos_signoff_ref", "decision_record_ref"):
        if authorization.get(field) != approval.get(field):
            raise ValueError(f"F007 source approval provenance changed: {field}")
    if (source_run.get("frozen_inputs") != approval["frozen_inputs"] or
            recorded.get("frozen_inputs") != source_run.get("frozen_inputs") or
            recorded.get("frozen_checkpoints") != source_run.get("checkpoints") or
            source_run.get("protocol_sha256") != recorded.get("protocol_sha256")):
        raise ValueError("F007 source frozen-input or protocol provenance changed")


def run(*, output: Path, device: str = "cpu") -> dict:
    """Execute all frozen test requests only after committed authorization."""
    if device != "cpu":
        raise ValueError("F007 final-test runner currently uses the fixed CPU inference path")
    output = _attempt_directory(output)
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
        preflight = preflight_test_inventories(rows, protocol)
        attempt["inventory_preflight"] = preflight
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
                      "inventory_preflight": preflight,
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
        report["source_evidence_sha256"] = _source_evidence_hashes(evidence_dir)
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
    except (KeyboardInterrupt, SystemExit) as exc:
        attempt.update(status="interrupted", interrupted_utc=_utc(),
                       error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, attempt)
        raise
    except Exception as exc:
        attempt.update(status="failed", failed_utc=_utc(),
                       error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, attempt)
        raise


def recover_report(*, evidence: Path, output: Path) -> dict:
    """Analyze complete authorized predictions without decoding or inference."""
    evidence = Path(evidence).resolve()
    base = (ROOT / "artifacts/F007").resolve()
    if not evidence.is_relative_to(base) or evidence.name != "evidence":
        raise ValueError("F007 recovery requires a prior final-test evidence directory")
    output = _attempt_directory(output)
    attempt_path = output / "attempt.json"
    attempt = {"experiment": "F007", "partition": "test",
               "operation": "analysis_recovery", "status": "started",
               "started_utc": _utc(), "source_evidence": str(evidence),
               "output": str(output)}
    _write_attempt(attempt_path, attempt)
    try:
        protocol = load_protocol()
        approval = require_test_authorization(protocol)
        attempt.update(status="authorized_preflight", authorization=approval)
        _write_attempt(attempt_path, attempt)
        cells, requests, source_run = analysis.load_test_evidence(evidence)
        _verify_original_provenance(source_run, approval)
        source_hashes = _source_evidence_hashes(evidence)
        provenance = capture(protocol, approval["frozen_inputs"], official=True,
                             execution_device="cpu_analysis",
                             parameters={"operation": "analysis_recovery",
                                         "source_evidence": str(evidence),
                                         "source_evidence_sha256": source_hashes,
                                         "output": str(output),
                                         "bootstrap_seed": protocol["bootstrap"]["seed"]})
        attempt.update(status="running", source_evidence_sha256=source_hashes,
                       source_run_commit=source_run["provenance"]["commit_sha"],
                       provenance=provenance)
        _write_attempt(attempt_path, attempt)
        report, draws = analysis.analyze(cells, requests, protocol,
                                         require_complete=True, partition="test")
        if _source_evidence_hashes(evidence) != source_hashes:
            raise RuntimeError("F007 source evidence changed during recovery")
        report.update(authorization=approval, provenance=provenance,
                      source_evidence_sha256=source_hashes,
                      recovery={"source_evidence": str(evidence),
                                "source_run_commit": source_run["provenance"]["commit_sha"],
                                "attempt": str(output)})
        assert_unchanged(provenance)
        report_dir = output / "report"
        analysis.write_report(report_dir, report, draws, source=evidence)
        assert_unchanged(provenance)
        if _source_evidence_hashes(evidence) != source_hashes:
            raise RuntimeError("F007 source evidence changed during report writing")
        attempt.update(status="completed", completed_utc=_utc(), report=str(report_dir),
                       cells=len(report["cells"]), contrasts=len(report["contrasts"]))
        _write_attempt(attempt_path, attempt)
        return {"output": str(output), "status": "completed",
                "source_evidence": str(evidence), "cells": len(report["cells"]),
                "contrasts": len(report["contrasts"])}
    except (KeyboardInterrupt, SystemExit) as exc:
        attempt.update(status="interrupted", interrupted_utc=_utc(),
                       error_type=type(exc).__name__, error=str(exc))
        _write_attempt(attempt_path, attempt)
        raise
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
    parser.add_argument("--recover-evidence", type=Path,
                        help="guarded analysis-only recovery from complete final-test evidence")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.recover_evidence is not None:
        output = args.output or ROOT / "artifacts/F007" / f"final_report_recovery_{stamp}"
        result = recover_report(evidence=args.recover_evidence, output=output)
    else:
        output = args.output or ROOT / "artifacts/F007" / f"final_test_{stamp}"
        result = run(output=output, device=args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
