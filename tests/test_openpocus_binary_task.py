"""T004 semantics and manifest checks; no download, Pillow or training required."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("binary_task", ROOT / "data/openpocus/build_binary_task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class BinaryTaskTests(unittest.TestCase):
    def test_exact_mapping_and_exclusions(self):
        expected = {
            "0": (0, None), "1": (1, None), "2": (1, None), "3": (1, None),
            "4": (None, "indeterminate"), "5": (None, "undefined_code_5"),
            "": (None, "unavailable_label"), None: (None, "unavailable_label"),
            "  ": (None, "unavailable_label"),
        }
        for source, result in expected.items():
            with self.subTest(source=source):
                self.assertEqual(task.map_label(source), result)
        with self.assertRaisesRegex(ValueError, "Unexpected source zone label"):
            task.map_label("6")

    def test_sequence_units_provenance_and_deterministic_output(self):
        task.ARTIFACTS.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task.ARTIFACTS) as directory:
            root = Path(directory)
            raw = root / "raw"
            # Distinct IDs must survive even when future grouping might join them.
            rows = [{"patient": exam, "ai_id": exam[2:], **dict.fromkeys(task.ZONES, "")}
                    for exam in ("Pt146B", "Pt146A")]
            rows[0].update(z1="0", z2="1", z3="2", z4="3", z5="4", z6="5")
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["patient", "ai_id", *task.ZONES])
                writer.writeheader()
                writer.writerows(rows)
            before = labels.read_bytes()
            for row in rows:
                (raw / row["patient"]).mkdir(parents=True)
            names = ["Z01_frame_10.jpg", "z01_frame_2 2.jpg", "z01_frame_2.jpg",
                     "z03_frame_0.jpg", "z04_frame_0.jpg", "z05_frame_0.jpg",
                     "z06_frame_0.jpg", "z07_frame_0.jpg"]
            for name in names:
                # Enumeration does not need to decode images or adjudicate quality.
                (raw / "Pt146B" / name).write_bytes(b"synthetic frame placeholder")
            raw_before = {p: p.read_bytes() for p in raw.rglob("*.jpg")}
            records, digest = task.build_records(labels, raw)
            summary = task.summarize(records, digest)
            task.write_outputs(records, summary, root / "first")
            records2, digest2 = task.build_records(labels, raw)
            task.write_outputs(records2, task.summarize(records2, digest2), root / "second")
            for name in ("zones.jsonl", "summary.json"):
                self.assertEqual((root / "first" / name).read_bytes(),
                                 (root / "second" / name).read_bytes())
            self.assertEqual(labels.read_bytes(), before)
            self.assertEqual(digest, hashlib.sha256(before).hexdigest())
            self.assertEqual({p: p.read_bytes() for p in raw.rglob("*.jpg")}, raw_before)
            self.assertEqual(len(records), 24)
            self.assertEqual(len({(r["exam_id"], r["zone_id"]) for r in records}), 24)
            self.assertEqual({r["exam_id"] for r in records}, {"Pt146A", "Pt146B"})
            zone = next(r for r in records if r["exam_id"] == "Pt146B" and r["zone_id"] == "z1")
            self.assertEqual(zone["frame_paths"], ["Pt146B/z01_frame_2 2.jpg",
                                                   "Pt146B/z01_frame_2.jpg", "Pt146B/Z01_frame_10.jpg"])
            self.assertEqual(zone["source_label"], "0")
            self.assertEqual(zone["ai_id"], "146B")
            missing_frames = next(r for r in records if r["exam_id"] == "Pt146B" and r["zone_id"] == "z2")
            self.assertEqual(missing_frames["binary_target"], 1)
            self.assertTrue(missing_frames["label_eligible"])
            self.assertFalse(missing_frames["image_backed_candidate"])
            for r in records:
                if r["source_label"] in {"", "4", "5"}:
                    self.assertIsNone(r["binary_target"])
                    self.assertFalse(r["image_backed_candidate"])
                    self.assertIsNotNone(r["label_exclusion_reason"])
            self.assertEqual(summary["label_eligible_count"], 4)
            self.assertEqual(summary["image_backed_candidate_class_counts"], {"normal": 1, "abnormal": 2})
            self.assertEqual(summary["available_frame_files"], 8)
            self.assertEqual(len(summary["label_eligible_without_frames"]), 1)
            self.assertEqual(len((root / "first/zones.jsonl").read_text().splitlines()), 24)
            (raw / "Pt146B/bad_name.jpg").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Unmapped frame file"):
                task.build_records(labels, raw)

    def test_outputs_cannot_write_source_directory(self):
        with self.assertRaisesRegex(ValueError, "ignored artifacts"):
            task.write_outputs([], {}, ROOT / "data/openpocus")

    def test_local_snapshot_reconciles_with_t002(self):
        """Optional integration check against locally regenerated T002 evidence."""
        audit_dir = task.ARTIFACTS / "T002_openpocus_audit"
        if not all(p.exists() for p in (task.DATA / "raw", audit_dir / "audit.json", audit_dir / "zones.csv")):
            self.skipTest("Local raw dataset and T002 audit artifacts are required")
        before = (task.DATA / "labels.csv").read_bytes()
        audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
        records, digest = task.build_records(task.DATA / "labels.csv", task.DATA / "raw")
        self.assertEqual(digest, audit["source_sha256"]["data/openpocus/labels.csv"],
                         "Source changed; rerun T002 and review snapshot counts")
        summary = task.summarize(records, digest)
        with (audit_dir / "zones.csv").open(newline="", encoding="utf-8") as stream:
            audited = {(r["exam"], r["zone"]): (r["label"], int(r["frames"]))
                       for r in csv.DictReader(stream)}
        observed = {(r["exam_id"], r["zone_id"]): (r["source_label"], r["frame_count"])
                    for r in records}
        self.assertEqual(observed, audited)
        counts = audit["image_backed_zone_labels"]
        expected = {"normal": counts["0"], "abnormal": sum(counts[c] for c in ("1", "2", "3"))}
        self.assertEqual(summary["image_backed_candidate_class_counts"], expected)
        self.assertEqual(summary["image_backed_candidate_count"], sum(expected.values()))
        self.assertEqual(summary["available_frame_files"], audit["scale"]["frames"])
        self.assertEqual((task.DATA / "labels.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
