"""Safety checks for the T002b split-group generator; no dataset download needed."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_recordings", ROOT / "data/openpocus/find_shared_recordings.py")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


class InventorySafetyTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.raw = self.data / "raw"
        self.raw.mkdir()
        with (self.data / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient", "device", "z1"])
            writer.writeheader()
            writer.writerows([{"patient": "Pt1", "device": "0", "z1": "0"},
                              {"patient": "Pt2", "device": "0", "z1": "0"}])
        self.groups = self.data / "split_groups.csv"
        self.original_groups = b"patient,split_group\nPt1,linked\nPt2,linked\n"
        self.groups.write_bytes(self.original_groups)

    def frame(self, exam):
        folder = self.raw / exam
        folder.mkdir(exist_ok=True)
        path = folder / f"{exam}_z01_frame_000000.jpg"
        path.write_bytes(b"fixture; image decoding is not reached")
        return path

    def invoke_with_expected_failure(self, message):
        output = self.data / "search_output"
        argv = [str(shared.DATA / "find_shared_recordings.py"), "--workers", "1",
                "--output", str(output)]
        with mock.patch.object(shared, "DATA", self.data), \
             mock.patch.object(shared, "RAW", self.raw), \
             mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, message):
                shared.main()
        self.assertEqual(self.groups.read_bytes(), self.original_groups)
        self.assertFalse(output.exists())

    def test_missing_exam_folder_fails_before_overwriting_groups(self):
        self.frame("Pt1")
        self.invoke_with_expected_failure(r"missing=\['Pt2'\]")

    def test_unlabelled_exam_folder_fails_before_overwriting_groups(self):
        self.frame("Pt1")
        self.frame("Pt2")
        self.frame("Pt3")
        self.invoke_with_expected_failure(r"unlabelled=\['Pt3'\]")

    def test_empty_exam_folder_fails_before_overwriting_groups(self):
        self.frame("Pt1")
        (self.raw / "Pt2").mkdir()
        self.invoke_with_expected_failure("No JPEG frames in exam folder: Pt2")

    def test_valid_inventory_preserves_exam_and_zone_keys(self):
        first = self.frame("Pt1")
        second = self.frame("Pt2")
        files, clips = shared.inventory_frames({"Pt1": {}, "Pt2": {}}, self.raw)
        self.assertEqual(files, [first, second])
        self.assertEqual(dict(clips), {("Pt1", 1): [(0, 0)], ("Pt2", 1): [(0, 1)]})


if __name__ == "__main__":
    unittest.main()
