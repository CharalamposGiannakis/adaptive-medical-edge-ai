"""F007 final runner checks with synthetic requests and no held-out images."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.F007 import analysis, final_test, phase_a1, validation_run
from src import f007_protocol
from tests.test_f007_phase_a1 import BASELINE, points_from_screen


ROOT = f007_protocol.ROOT
ARTIFACTS = ROOT / "artifacts/F007"


class ApprovalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposed = f007_protocol.load_protocol()

    def test_unaccepted_protocol_and_missing_receipt_deny(self):
        with self.assertRaisesRegex(PermissionError, "not authorized"):
            f007_protocol.require_test_authorization(self.proposed)
        accepted = copy.deepcopy(self.proposed)
        accepted.update(status=f007_protocol.ACCEPTED, authorizes_test_inference=True)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            with self.assertRaisesRegex(PermissionError, "receipt"):
                f007_protocol.require_test_authorization(accepted, root=Path(folder))

    def _accepted_fixture(self, folder: str, *, review: str | None = None,
                          decision_status: str = "Accepted"):
        root = Path(folder)
        (root / "configs").mkdir()
        protocol = copy.deepcopy(self.proposed)
        protocol.update(status=f007_protocol.ACCEPTED, authorizes_test_inference=True)
        protocol_bytes = (json.dumps(protocol, sort_keys=True) + "\n").encode()
        (root / "configs/f007_protocol.json").write_bytes(protocol_bytes)
        review = review or "https://github.com/CharalamposGiannakis/adaptive-medical-edge-ai/pull/29"
        receipt = {"protocol_commit": "a" * 40, "independent_review_ref": review,
                   "charalampos_signoff_ref": "docs/decision_log.md#d030",
                   "decision_record_ref": "docs/decision_log.md#d030"}
        receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        (root / "configs/f007_approval.json").write_bytes(receipt_bytes)
        decision = (f"### D030\n**Status:** {decision_status} by Charalampos\n"
                    f"Independent review: {review}\n---\n").encode()

        def blob(_root, _revision, name):
            return {"configs/f007_approval.json": receipt_bytes,
                    "configs/f007_protocol.json": protocol_bytes,
                    "docs/decision_log.md": decision}[name]
        return root, protocol, blob

    def test_invalid_review_signoff_and_frozen_hash_deny(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root, protocol, blob = self._accepted_fixture(
                folder, review="https://example.invalid/review")
            with patch.object(f007_protocol, "git_blob", side_effect=blob), \
                    patch.object(f007_protocol.subprocess, "run"):
                with self.assertRaisesRegex(PermissionError, "review reference"):
                    f007_protocol.require_test_authorization(protocol, root=root)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root, protocol, blob = self._accepted_fixture(folder, decision_status="Proposed")
            with patch.object(f007_protocol, "git_blob", side_effect=blob), \
                    patch.object(f007_protocol.subprocess, "run"):
                with self.assertRaisesRegex(PermissionError, "sign-off"):
                    f007_protocol.require_test_authorization(protocol, root=root)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root, protocol, blob = self._accepted_fixture(folder, decision_status="Proposed")
            def unrelated_acceptance(_root, revision, name):
                content = blob(_root, revision, name)
                return (content.replace(b"---\n", b"Accepted by Charalampos elsewhere\n---\n")
                        if name == "docs/decision_log.md" else content)
            with patch.object(f007_protocol, "git_blob", side_effect=unrelated_acceptance), \
                    patch.object(f007_protocol.subprocess, "run"):
                with self.assertRaisesRegex(PermissionError, "sign-off"):
                    f007_protocol.require_test_authorization(protocol, root=root)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root, protocol, blob = self._accepted_fixture(folder)
            with patch.object(f007_protocol, "git_blob", side_effect=blob), \
                    patch.object(f007_protocol.subprocess, "run"), \
                    patch.object(f007_protocol, "verify_inputs",
                                 side_effect=ValueError("Frozen checkpoint hash mismatch")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    f007_protocol.require_test_authorization(protocol, root=root)

    def test_accepted_fixture_returns_auditable_receipt_identity(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root, protocol, blob = self._accepted_fixture(folder)
            with patch.object(f007_protocol, "git_blob", side_effect=blob), \
                    patch.object(f007_protocol.subprocess, "run"), \
                    patch.object(f007_protocol, "verify_inputs",
                                 return_value={"checkpoint_hashes_checked": True}):
                authorization = f007_protocol.require_test_authorization(protocol, root=root)
            self.assertEqual(authorization["protocol_commit"], "a" * 40)
            self.assertTrue(authorization["frozen_inputs"]["checkpoint_hashes_checked"])
            self.assertEqual(len(authorization["approval_receipt_sha256"]), 64)


class FinalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = f007_protocol.load_protocol()

    def test_denial_occurs_before_test_manifest_or_predictor(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            output = Path(folder) / "denied"
            with patch.object(final_test, "partition_context",
                              side_effect=AssertionError("test manifest opened")), \
                    patch.object(final_test, "CanonicalModels",
                                 side_effect=AssertionError("predictor constructed")):
                with self.assertRaises(PermissionError):
                    final_test.run(output=output)
            attempt = json.loads((output / "attempt.json").read_text())
            self.assertEqual(attempt["status"], "failed")
            self.assertFalse((output / "evidence").exists())
            with self.assertRaises(FileExistsError):
                final_test.run(output=output)

    def test_partition_context_denial_precedes_manifest_verification(self):
        with patch.object(phase_a1, "require_test_authorization",
                          side_effect=PermissionError("denied")), \
                patch.object(phase_a1, "verify_inputs",
                             side_effect=AssertionError("manifest accessed")):
            with self.assertRaises(PermissionError):
                phase_a1.partition_context(self.protocol, "test")
        with self.assertRaises(PermissionError):
            analysis.analyze({}, [{"partition": "test"}], self.protocol,
                             partition="test")

    def test_inventory_preflight_failure_precedes_model_construction(self):
        row = {"exam_id": "S", "zone_id": "z1", "binary_target": "0",
               "split_group": "G", "partition": "test"}
        approval = {"frozen_inputs": {"checkpoint_hashes_checked": True}}
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            output = Path(folder) / "preflight_failed"
            with patch.object(final_test, "require_test_authorization",
                              return_value=approval), \
                    patch.object(final_test, "partition_context",
                                 return_value=([row], {("S", "z1"): "singleton:S/z1"},
                                               approval["frozen_inputs"])), \
                    patch.object(final_test, "capture", return_value={"official": True}), \
                    patch.object(final_test, "preflight_test_inventories",
                                 side_effect=ValueError("inventory changed")), \
                    patch.object(final_test, "CanonicalModels",
                                 side_effect=AssertionError("model constructed")):
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    final_test.run(output=output)
            self.assertEqual(json.loads((output / "attempt.json").read_text())["status"],
                             "failed")

    def test_interruptions_preserve_terminal_status_and_original_exit(self):
        approval = {"frozen_inputs": {"checkpoint_hashes_checked": True}}
        for interruption in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(type=type(interruption).__name__), \
                    tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
                output = Path(folder) / "interrupted"
                with patch.object(final_test, "require_test_authorization",
                                  return_value=approval), \
                        patch.object(final_test, "partition_context",
                                     side_effect=interruption):
                    with self.assertRaises(type(interruption)):
                        final_test.run(output=output)
                attempt = json.loads((output / "attempt.json").read_text())
                self.assertEqual(attempt["status"], "interrupted")
                self.assertEqual(attempt["error_type"], type(interruption).__name__)

    def test_recovery_authenticates_original_source_commit_and_receipt(self):
        digest = hashlib.sha256(b"original committed source").hexdigest()
        source = {name: {"committed_blob_sha256": digest,
                         "working_tree_sha256": digest}
                  for name in final_test.SOURCE_MODULES}
        approval = {field: "same" for field in
                    ("protocol_commit", "accepted_protocol_sha256",
                     "approval_receipt_sha256", "independent_review_ref",
                     "charalampos_signoff_ref", "decision_record_ref")}
        approval["frozen_inputs"] = {"checkpoint_hashes_checked": True}
        run = {"provenance": {"official": True, "working_tree_dirty": False,
                              "commit_sha": "a" * 40, "source_sha256": source,
                              "protocol_sha256": "protocol",
                              "frozen_inputs": approval["frozen_inputs"],
                              "frozen_checkpoints": {"f002b": "frozen"}},
               "authorization": approval, "frozen_inputs": approval["frozen_inputs"],
               "checkpoints": {"f002b": "frozen"}, "protocol_sha256": "protocol"}
        with patch.object(final_test, "git_blob", return_value=b"original committed source"), \
                patch.object(final_test.subprocess, "run"):
            final_test._verify_original_provenance(run, approval)
            changed = copy.deepcopy(run)
            changed["provenance"]["source_sha256"][final_test.SOURCE_MODULES[0]][
                "committed_blob_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source Git hash mismatch"):
                final_test._verify_original_provenance(changed, approval)
            changed = copy.deepcopy(run)
            changed["authorization"]["approval_receipt_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "approval provenance changed"):
                final_test._verify_original_provenance(changed, approval)
            changed = copy.deepcopy(run)
            changed["provenance"]["frozen_checkpoints"] = {"f002b": "changed"}
            with self.assertRaisesRegex(ValueError, "source frozen-input"):
                final_test._verify_original_provenance(changed, approval)

    def test_synthetic_full_dispatch_and_report_without_images(self):
        rows = [{"exam_id": exam, "zone_id": "z1", "binary_target": str(target),
                 "split_group": group, "partition": "test", "device": "synthetic"}
                for exam, target, group in (("S1", 0, "G1"), ("S2", 1, "G2"))]
        unit_of = {(row["exam_id"], "z1"): f"singleton:{row['exam_id']}/z1"
                   for row in rows}
        approval = {"protocol_commit": "a" * 40,
                    "frozen_inputs": {"checkpoint_hashes_checked": True}}
        fake_predictor = lambda row, plan: (0.2 if row["binary_target"] == "0" else 0.8,
                                            {"frame_indices": "synthetic"})
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            output = Path(folder) / "synthetic_attempt"
            with patch.object(final_test, "require_test_authorization", return_value=approval), \
                    patch.object(phase_a1, "require_test_authorization", return_value=approval), \
                    patch.object(validation_run, "require_test_authorization", return_value=approval), \
                    patch.object(analysis, "require_test_authorization", return_value=approval), \
                    patch.object(phase_a1, "verify_inputs", return_value=approval["frozen_inputs"]), \
                    patch.object(final_test, "partition_context",
                                 return_value=(rows, unit_of, approval["frozen_inputs"])), \
                    patch.object(final_test, "preflight_test_inventories",
                                 return_value={"rows": 2, "decoded_images": 0}), \
                    patch.object(final_test, "CanonicalModels", return_value=fake_predictor), \
                    patch.object(final_test, "capture", return_value={"official": True,
                                                                       "commit_sha": "a" * 40}), \
                    patch.object(final_test, "assert_unchanged"), \
                    patch.object(phase_a1.controller, "load_policy_points",
                                 return_value=points_from_screen()), \
                    patch.object(phase_a1.f006a_2, "read_baseline", return_value=BASELINE), \
                    patch.object(analysis, "bootstrap_indices",
                                 return_value=np.array([[0, 1], [1, 0]], dtype=np.int32)):
                result = final_test.run(output=output)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["cells"], 38)
            self.assertEqual(result["contrasts"], 24)
            run = json.loads((output / "evidence/run.json").read_text())
            report = json.loads((output / "report/report.json").read_text())
            self.assertEqual(run["partition"], "test")
            self.assertEqual(run["identical_fresh_predictions_checked"] > 0, True)
            self.assertEqual(report["partition"], "test")
            self.assertTrue((output / "report/bootstrap_indices.npy").exists())
            requests = (output / "evidence/requests.jsonl").read_text().splitlines()
            self.assertEqual(len(requests), result["requests"])
            self.assertEqual({json.loads(line)["system_id"] for line in requests},
                             set(self.protocol["systems"]) |
                             {"f002c_original_k4", "f005b_nested_w1.0_k16"})

    def test_recovery_reuses_complete_predictions_after_report_failure(self):
        rows = [{"exam_id": exam, "zone_id": "z1", "binary_target": str(target),
                 "split_group": group, "partition": "test", "device": "synthetic"}
                for exam, target, group in (("R1", 0, "G1"), ("R2", 1, "G2"))]
        unit_of = {(row["exam_id"], "z1"): f"singleton:{row['exam_id']}/z1"
                   for row in rows}
        approval = {"protocol_commit": "a" * 40,
                    "frozen_inputs": {"checkpoint_hashes_checked": True}}
        predictor = lambda row, plan: (0.2 if row["binary_target"] == "0" else 0.8,
                                       {"frame_indices": "synthetic"})
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            first, recovery = Path(folder) / "first", Path(folder) / "recovery"
            with patch.object(final_test, "require_test_authorization", return_value=approval), \
                    patch.object(phase_a1, "require_test_authorization", return_value=approval), \
                    patch.object(validation_run, "require_test_authorization", return_value=approval), \
                    patch.object(analysis, "require_test_authorization", return_value=approval), \
                    patch.object(phase_a1, "verify_inputs", return_value=approval["frozen_inputs"]), \
                    patch.object(final_test, "partition_context",
                                 return_value=(rows, unit_of, approval["frozen_inputs"])), \
                    patch.object(phase_a1, "partition_context",
                                 return_value=(rows, unit_of, approval["frozen_inputs"])), \
                    patch.object(final_test, "preflight_test_inventories",
                                 return_value={"rows": 2, "decoded_images": 0}), \
                    patch.object(final_test, "CanonicalModels", return_value=predictor), \
                    patch.object(final_test, "capture", return_value={"official": True,
                                                                       "commit_sha": "a" * 40}), \
                    patch.object(final_test, "assert_unchanged"), \
                    patch.object(phase_a1.controller, "load_policy_points",
                                 return_value=points_from_screen()), \
                    patch.object(phase_a1.f006a_2, "read_baseline", return_value=BASELINE), \
                    patch.object(analysis, "bootstrap_indices",
                                 return_value=np.array([[0, 1], [1, 0]], dtype=np.int32)):
                with patch.object(analysis, "write_report",
                                  side_effect=RuntimeError("report failed")):
                    with self.assertRaisesRegex(RuntimeError, "report failed"):
                        final_test.run(output=first)
                self.assertEqual(json.loads((first / "attempt.json").read_text())["status"],
                                 "failed")
                self.assertTrue((first / "evidence/requests.jsonl").exists())
                with patch.object(final_test, "_verify_original_provenance"), \
                        patch.object(final_test, "CanonicalModels",
                                     side_effect=AssertionError("model constructed")), \
                        patch("PIL.Image.open", side_effect=AssertionError("image decoded")):
                    result = final_test.recover_report(evidence=first / "evidence",
                                                       output=recovery)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["cells"], 38)
                self.assertTrue((recovery / "report/report.json").exists())
                with self.assertRaises(FileExistsError):
                    final_test.recover_report(evidence=first / "evidence", output=recovery)
                (first / "evidence/units.jsonl").write_text("{}\n", encoding="utf-8")
                with patch.object(final_test, "_verify_original_provenance"):
                    with self.assertRaises(ValueError):
                        final_test.recover_report(evidence=first / "evidence",
                                                  output=Path(folder) / "corrupt")


if __name__ == "__main__":
    unittest.main()
