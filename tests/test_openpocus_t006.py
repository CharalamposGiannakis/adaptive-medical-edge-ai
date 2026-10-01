"""T006 cohort policy, split integrity and architecture-independent loader checks."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/openpocus"
sys.path.insert(0, str(DATA))
import build_cohort as cohort
import cohort_loader as loader


def record(exam, zone, label, target, paths):
    return {"exam_id": exam, "zone_id": zone, "source_label": label,
            "binary_target": target, "target_name": {0: "normal", 1: "abnormal"}.get(target),
            "label_eligible": target is not None,
            "label_exclusion_reason": "unavailable_label" if target is None else None,
            "frame_count": len(paths), "frame_paths": paths,
            "image_backed_candidate": target is not None and bool(paths)}


class CohortPolicyTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def fixture(self):
        records = [record("Pt147", "z11", "", None, ["Pt147/z11_frame_0.jpg"]),
                   record("Pt163", "z11", "1", 1, ["Pt163/z11_frame_0.jpg"]),
                   record("Pt178", "z7", "3", 1, ["Pt178/z07_frame_0.jpg"]),
                   record("Pt2", "z1", "0", 0, ["Pt2/z01_frame_0.jpg"]),
                   record("Pt3", "z1", "0", 0, ["Pt3/z01_frame_0.jpg"]),
                   record("Pt45", "z1", "0", 0,
                          ["Pt45/Pt45_z01_frame_2 2.jpg",
                           "Pt45/image_001_Pt45_z01_frame_2.jpg"])]
        groups = {"Pt147": "linked", "Pt163": "linked", "Pt178": "linked",
                  "Pt2": "same", "Pt3": "same", "Pt45": "Pt45"}
        edges = [{"exam_a": "Pt147", "zone_a": "z11", "exam_b": "Pt163", "zone_b": "z11"},
                 {"exam_a": "Pt163", "zone_a": "z11", "exam_b": "Pt178", "zone_b": "z7"},
                 {"exam_a": "Pt2", "zone_a": "z1", "exam_b": "Pt3", "zone_b": "z1"}]
        path = self.path / "edges.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=cohort.EDGE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({**row, "verified_frames": "2", "sampled_frames": "2"} for row in edges)
        return records, groups, path

    def test_conflict_removes_only_affected_recording_and_pt45_keeps_masked_frames(self):
        records, groups, edges = self.fixture()
        conflict, same, components, count = cohort.shared_components(edges, records, groups)
        self.assertEqual(count, 2)
        self.assertEqual(conflict, {("Pt147", "z11"), ("Pt163", "z11"), ("Pt178", "z7")})
        self.assertEqual(same, {("Pt2", "z1"), ("Pt3", "z1")})
        self.assertEqual(len(components), 1)
        metadata = {exam: {"device": "0"} for exam in groups}
        retained, excluded, pt45 = cohort.make_cohort(records, metadata, groups, conflict, same)
        self.assertEqual({r["exam_id"] for r in retained}, {"Pt2", "Pt3", "Pt45"})
        self.assertEqual(pt45["alternate_frames_omitted"], 1)
        masked = next(r for r in retained if r["exam_id"] == "Pt45")
        self.assertEqual(masked["frame_count"], 1)
        self.assertIn("pt45_masked_primary", masked["quality_flags"])
        self.assertEqual(masked["frame_paths_sha256"], cohort.path_digest(
            ["Pt45/image_001_Pt45_z01_frame_2.jpg"]))
        self.assertEqual({r["exam_id"] for r in excluded if r["exclusion_reason"].startswith("t006:")},
                         {"Pt163", "Pt178"})
        self.assertEqual(next(r for r in excluded if r["exam_id"] == "Pt147")["exclusion_reason"],
                         "t004:unavailable_label")

    def test_cross_group_shared_edge_fails(self):
        records, groups, edges = self.fixture()
        groups["Pt178"] = "different"
        with self.assertRaisesRegex(ValueError, "crosses known groups"):
            cohort.shared_components(edges, records, groups)

    def test_group_allocation_does_not_depend_on_input_row_order(self):
        rows = [{"split_group": f"g{i:02}", "binary_target": target}
                for i in range(40) for target in ([0] * 5 + [1] * 3)]
        percentages = {"train": 70, "validation": 15, "test": 15}
        first = cohort.allocate_groups(rows, percentages, 42)
        self.assertEqual(first, cohort.allocate_groups(list(reversed(rows)), percentages, 42))
        self.assertEqual(set(first), {f"g{i:02}" for i in range(40)})

    def test_loader_verifies_retained_inventory_and_accepts_injected_processing(self):
        raw = self.path / "raw"
        folder = raw / "Pt45"
        folder.mkdir(parents=True)
        primary = folder / "image_001_Pt45_z01_frame_000002.jpg"
        alternate = folder / "Pt45_z01_frame_000002 2.jpg"
        primary.write_bytes(b"primary fixture")
        alternate.write_bytes(b"alternate fixture")
        metadata = self.path / "metadata"
        metadata.mkdir()
        relative = primary.relative_to(raw).as_posix()
        row = {"exam_id": "Pt45", "zone_id": "z1", "source_label": "0",
               "binary_target": 0, "target_name": "normal", "split_group": "Pt45",
               "partition": "train", "device": "0", "t004_frame_count": 2, "frame_count": 1,
               "frame_paths_sha256": cohort.path_digest([relative]),
               "quality_flags": "pt45_masked_primary"}
        manifest = metadata / "train.csv"
        manifest.write_bytes(cohort.csv_bytes([row], cohort.MANIFEST_FIELDS))
        (metadata / "summary.json").write_text(json.dumps({"fingerprints": {
            "train.csv": hashlib.sha256(manifest.read_bytes()).hexdigest()}}), encoding="utf-8")
        dataset = loader.SequenceDataset("train", raw, metadata,
                                         sampler=lambda paths: paths[:1], transform=lambda p: p.name)
        sample = dataset[0]
        self.assertEqual(sample["frames"], (primary.name,))
        self.assertEqual(sample["retained_frame_count"], 1)
        (folder / "image_002_Pt45_z01_frame_000003.jpg").write_bytes(b"new frame")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            dataset[0]
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest fingerprint changed"):
            loader.SequenceDataset("train", raw, metadata)

    def test_source_fingerprint_and_raw_output_guards(self):
        labels, groups, edges = (self.path / name for name in
                                 ("labels.csv", "groups.csv", "edges.csv"))
        for path in (labels, groups, edges):
            path.write_bytes(b"fixture")
        config = {"schema_version": 1,
                  "task": "T006_known_family_shared_recording_group_disjoint_cohort",
                  "seed": 42, "partition_percentages": {"train": 70, "validation": 15, "test": 15},
                  "split_sanity_limits": {
                      "minimum_sequences_per_class_per_partition": 20,
                      "maximum_sequence_share_deviation_percentage_points": 5,
                      "maximum_abnormal_share_deviation_percentage_points": 10},
                  "expected_sha256": {"labels_csv": "wrong"}}
        with self.assertRaisesRegex(ValueError, "Source fingerprint mismatch"):
            cohort.validate_sources(config, labels, groups, edges, [])
        config_path = self.path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        raw = self.path / "raw"
        raw.mkdir()
        with self.assertRaisesRegex(ValueError, "must not overwrite raw"):
            cohort.build(labels, raw, groups, edges, config_path, raw / "output")

    def test_group_fingerprint_is_checkout_line_ending_independent(self):
        groups = self.path / "groups.csv"
        groups.write_bytes(b"patient,split_group\nPt1,Pt1\n")
        expected = cohort.canonical_lf_sha256(groups)
        groups.write_bytes(b"patient,split_group\r\nPt1,Pt1\r\n")
        self.assertEqual(cohort.canonical_lf_sha256(groups), expected)

    def test_local_snapshot_regenerates_tracked_metadata_byte_for_byte(self):
        if not (DATA / "raw").is_dir() or not (DATA / "t006/summary.json").is_file():
            self.skipTest("Local raw snapshot and tracked T006 metadata required")
        sources = [DATA / "labels.csv", DATA / "split_groups.csv"]
        before = [cohort.sha256(path) for path in sources]
        output = self.path / "regenerated"
        summary = cohort.build(sources[0], DATA / "raw", sources[1],
                               DATA / "t006/shared_recording_edges.csv",
                               DATA / "t006/config.json", output)
        self.assertEqual(summary["t004_image_backed_candidates"], 1765)
        self.assertEqual(summary["retained_sequences"], 1759)
        self.assertEqual(summary["retained_class_counts"], {"normal": 1225, "abnormal": 534})
        self.assertEqual(summary["source_dependency_groups"], 212)
        self.assertEqual(summary["retained_dependency_groups"], 211)
        self.assertEqual(summary["source_groups_without_retained_sequences"], ["ED4"])
        for name in ("train.csv", "validation.csv", "test.csv", "exclusions.csv",
                     "group_assignments.csv", "summary.json"):
            self.assertEqual((output / name).read_bytes(), (DATA / "t006" / name).read_bytes(), name)
        group_sides = {r["split_group"]: r["partition"] for r in
                       cohort.read_csv(output / "group_assignments.csv", cohort.GROUP_FIELDS)}
        exam_group = {r["patient"]: r["split_group"] for r in
                      cohort.read_csv(DATA / "split_groups.csv", ("patient", "split_group"))}
        self.assertEqual(len(group_sides), 212)
        for edge in cohort.read_csv(DATA / "t006/shared_recording_edges.csv", cohort.EDGE_FIELDS):
            self.assertEqual(group_sides[exam_group[edge["exam_a"]]],
                             group_sides[exam_group[edge["exam_b"]]])
        all_keys = set()
        for partition in cohort.PARTITIONS:
            dataset = loader.SequenceDataset(partition, DATA / "raw", output)
            for i in range(len(dataset)):
                item = dataset[i]
                key = (item["exam_id"], item["zone_id"])
                self.assertNotIn(key, all_keys)
                all_keys.add(key)
                self.assertTrue(all(path.is_file() for path in item["frames"]))
                if item["exam_id"] == "Pt45":
                    self.assertFalse(any(path.name.endswith(" 2.jpg") for path in item["frames"]))
        self.assertEqual(len(all_keys), 1759)
        for component in summary["conflicting_shared_recording_components"]:
            for alias in component:
                self.assertNotIn((alias["exam_id"], alias["zone_id"]), all_keys)
        self.assertEqual([cohort.sha256(path) for path in sources], before)


if __name__ == "__main__":
    unittest.main()
