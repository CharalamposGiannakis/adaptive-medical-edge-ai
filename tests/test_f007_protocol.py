"""Focused checks for the accepted F007 protocol's cross-platform hash pin."""

import copy
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

from src.f002_protocol import recording_units
from src.f007_protocol import require_test_authorization


ROOT = Path(__file__).resolve().parents[1]
F007 = ROOT / "configs" / "f007_protocol.json"
F002 = ROOT / "configs" / "f002_protocol.json"


class F007ProtocolTests(unittest.TestCase):
    def test_canonical_f002_hash_matches_git_and_local_content(self):
        protocol = json.loads(F007.read_text(encoding="utf-8"))
        frozen = protocol["frozen_t006"]
        self.assertEqual(frozen["canonical_protocol"], "configs/f002_protocol.json")
        self.assertEqual(frozen["canonical_protocol_sha256_basis"],
                         "SHA256_of_canonical_Git_LF_blob_bytes")

        git_blob = subprocess.check_output(
            ["git", "show", "HEAD:configs/f002_protocol.json"], cwd=ROOT)
        self.assertNotIn(b"\r", git_blob)
        self.assertEqual(hashlib.sha256(git_blob).hexdigest(),
                         frozen["canonical_protocol_sha256"])

        # Existing Windows worktrees may still have CRLF until Git rewrites them.
        local_lf = F002.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(local_lf, git_blob)
        simulated_windows_copy = git_blob.replace(b"\n", b"\r\n")
        self.assertEqual(simulated_windows_copy.replace(b"\r\n", b"\n"),
                         git_blob)
        attribute = subprocess.check_output(
            ["git", "check-attr", "eol", "--", "configs/f002_protocol.json"],
            cwd=ROOT, text=True).strip()
        self.assertEqual(attribute, "configs/f002_protocol.json: eol: lf")

    def test_accepted_protocol_and_synthetic_denial(self):
        protocol = json.loads(F007.read_text(encoding="utf-8"))
        self.assertEqual(protocol["status"], "accepted_final_protocol")
        self.assertEqual(protocol["accepted_status_value"],
                         "accepted_final_protocol")
        self.assertIs(protocol["authorizes_test_inference"], True)
        self.assertIs(protocol["execution_gate"]
                      ["requires_authorizes_test_inference_true"], True)
        self.assertEqual(protocol["methodology_details"],
                         "experiments/F007/README.md")
        unauthorized = copy.deepcopy(protocol)
        unauthorized.update(status="proposed_pending_team_approval",
                            authorizes_test_inference=False)
        with self.assertRaisesRegex(PermissionError, "not authorized"):
            require_test_authorization(unauthorized)

    def test_schedule_unit_ids_follow_recording_units(self):
        shared_rows = [
            {"exam_id": exam, "zone_id": "z1", "binary_target": "1",
             "split_group": "G", "partition": "test"}
            for exam in ("ExamA", "ExamB")
        ]
        singleton = {"exam_id": "ExamS", "zone_id": "z2",
                     "binary_target": "0", "split_group": "S",
                     "partition": "test"}
        edges = [
            {"exam_a": "ExamA", "zone_a": "z1",
             "exam_b": "ExamB", "zone_b": "z1"},
            {"exam_a": "ExamB", "zone_a": "z1",
             "exam_b": "ExamC", "zone_b": "z1"},
        ]
        unavailable_alias = {"exam_id": "ExamC", "zone_id": "z1",
                             "exclusion_reason": "t004:unavailable_label"}
        unit_of = recording_units({"test": shared_rows + [singleton]},
                                  edges, [unavailable_alias])
        full_component = "ExamA/z1|ExamB/z1|ExamC/z1"
        shared_id = "shared:" + hashlib.sha256(
            full_component.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(unit_of[("ExamA", "z1")], shared_id)
        self.assertEqual(unit_of[("ExamB", "z1")], shared_id)
        self.assertEqual(unit_of[("ExamS", "z2")], "singleton:ExamS/z2")


if __name__ == "__main__":
    unittest.main()
