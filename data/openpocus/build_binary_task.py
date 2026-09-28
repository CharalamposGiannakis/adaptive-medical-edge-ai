"""Derive T004 exam × zone records without modifying labels or selecting splits.

Uses only the standard library. Run from the existing project environment:
    python data/openpocus/build_binary_task.py
Outputs are restricted to the repository's ignored artifacts/ directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/openpocus"
ARTIFACTS = ROOT / "artifacts"
ZONES = tuple(f"z{i}" for i in range(1, 13))
ZONE_TOKEN = re.compile(r"(?:^|_)z(\d{1,2})_", re.I)
FRAME_TOKEN = re.compile(r"frame_(\d+)(.*?)\.[^.]+$", re.I)


def map_label(source_label):
    """Return (target, exclusion reason); unexpected codes require review."""
    code = source_label.strip() if source_label is not None else ""
    if code == "0":
        return 0, None
    if code in {"1", "2", "3"}:
        return 1, None
    reasons = {"4": "indeterminate", "5": "undefined_code_5", "": "unavailable_label"}
    if code not in reasons:
        raise ValueError(f"Unexpected source zone label: {source_label!r}")
    return None, reasons[code]


def build_records(labels_path, raw_path):
    """Enumerate all label slots and all matching JPEG paths, without decoding.

    Frame index, then relative path, supplies a deterministic enumeration order.
    Ties/alternate renditions are retained; this does not establish chronology.
    """
    labels_bytes = labels_path.read_bytes()
    reader = csv.DictReader(io.StringIO(labels_bytes.decode("utf-8-sig")))
    required = {"patient", "ai_id", *ZONES}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("labels.csv must contain patient, ai_id and z1 through z12")
    labels = list(reader)
    exams = [row["patient"] for row in labels]
    if not exams or len(exams) != len(set(exams)):
        raise ValueError("Expected nonempty labels with unique exam folder IDs")
    if any(not e or e in {".", ".."} or re.search(r"[/\\:]", e) for e in exams):
        raise ValueError("Each patient value must be a single exam folder name")
    if not raw_path.is_dir():
        raise ValueError(f"Raw directory does not exist: {raw_path}")

    frames = defaultdict(list)
    exam_ids = set(exams)
    for path in raw_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        relative = path.relative_to(raw_path)
        zone_match = ZONE_TOKEN.search(path.name)
        frame_match = FRAME_TOKEN.search(path.name)
        if (len(relative.parts) != 2 or relative.parts[0] not in exam_ids
                or zone_match is None or frame_match is None):
            raise ValueError(f"Unmapped frame file: {relative}")
        zone = f"z{int(zone_match[1])}"
        if zone not in ZONES:
            raise ValueError(f"Invalid zone in frame file: {relative}")
        frames[relative.parts[0], zone].append((int(frame_match[1]), relative.as_posix()))

    records = []
    for row in sorted(labels, key=lambda r: r["patient"]):
        for zone in ZONES:
            target, reason = map_label(row[zone])
            paths = [name for _, name in sorted(frames[row["patient"], zone])]
            records.append({
                "exam_id": row["patient"], "ai_id": row["ai_id"], "zone_id": zone,
                "source_label": row[zone], "binary_target": target,
                "target_name": {0: "normal", 1: "abnormal"}.get(target),
                "label_eligible": target is not None, "label_exclusion_reason": reason,
                "frame_count": len(paths), "frame_paths": paths,
                "data_availability": "frames_present" if paths else "no_frames",
                "image_backed_candidate": target is not None and bool(paths),
            })
    return records, hashlib.sha256(labels_bytes).hexdigest()


def summarize(records, labels_sha256):
    candidates = [r for r in records if r["image_backed_candidate"]]
    def class_counts(rows):
        return {name: sum(r["binary_target"] == target for r in rows)
                for target, name in ((0, "normal"), (1, "abnormal"))}
    return {
        "schema_version": 1,
        "task": "T004_binary_normal_vs_definite_abnormal",
        "prediction_unit": "exam_x_lung_zone_sequence",
        "labels_sha256": labels_sha256,
        "frame_order": "numeric frame index, then case-sensitive relative POSIX path",
        "candidate_rule": "label_eligible and frame_count > 0; before quality/split decisions",
        "exam_count": len({r["exam_id"] for r in records}),
        "record_count": len(records),
        "observed_exam_zones": sum(r["frame_count"] > 0 for r in records),
        "available_frame_files": sum(r["frame_count"] for r in records),
        "source_label_counts": dict(Counter(r["source_label"] or "missing" for r in records)),
        "label_eligible_count": sum(r["label_eligible"] for r in records),
        "label_eligible_class_counts": class_counts(records),
        "label_exclusion_counts": dict(Counter(r["label_exclusion_reason"] for r in records
                                              if not r["label_eligible"])),
        "label_eligible_without_frames": [
            {k: r[k] for k in ("exam_id", "zone_id", "source_label", "binary_target")}
            for r in records if r["label_eligible"] and not r["frame_count"]],
        "image_backed_candidate_count": len(candidates),
        "image_backed_candidate_class_counts": class_counts(candidates),
        "candidate_frame_files": sum(r["frame_count"] for r in candidates),
    }


def write_outputs(records, summary, output):
    output = output.resolve()
    destinations = [output / name for name in ("zones.jsonl", "summary.json")]
    if not all(p.resolve().is_relative_to(ARTIFACTS.resolve()) for p in destinations):
        raise ValueError("Output must stay inside the repository's ignored artifacts/ directory")
    output.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8 and LF, with no timestamps or host-specific absolute paths.
    with destinations[0].open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {**summary, "manifest_sha256": hashlib.sha256(destinations[0].read_bytes()).hexdigest()}
    destinations[1].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    parser.add_argument("--raw", type=Path, default=DATA / "raw")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "T004_binary_task")
    args = parser.parse_args()
    # Custom input paths must remain inputs even if located under artifacts/.
    output = args.output.resolve()
    if output == args.raw.resolve() or output.is_relative_to(args.raw.resolve()):
        parser.error("Output must not be inside the raw input directory")
    if args.labels.resolve() in {(output / n).resolve() for n in ("zones.jsonl", "summary.json")}:
        parser.error("Output must not overwrite the source labels")
    records, labels_sha256 = build_records(args.labels, args.raw)
    summary = summarize(records, labels_sha256)
    write_outputs(records, summary, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
