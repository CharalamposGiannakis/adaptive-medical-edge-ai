"""Read-only T002 inventory, label reconciliation, and exact-duplicate audit.

Run with the existing project Python environment; requires Pillow.
Writes only to artifacts/, never to raw/ or labels.csv. No split or task is chosen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
from itertools import chain
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/openpocus"
RAW = DATA / "raw"
ZONES = [f"z{i}" for i in range(1, 13)]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    """Numeric equivalence for Excel IDs; suffixes remain distinct."""
    value = re.sub(r"^pt", "", value.replace(" ", ""), flags=re.I).upper()
    if value.startswith("ED"):
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([ABC]?)", value)
    if not match:
        raise ValueError(f"Unrecognized ID: {value!r}")
    # Workbook XML contains binary-float artifacts (e.g. 33.159999999999997).
    # Match the existing builder's three-decimal ID normalization, not a patient merge.
    return str(Decimal(match[1]).quantize(Decimal("0.001")).normalize()) + match[2]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def quantile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"n": len(values), "min": values[0], "q1": quantile(.25),
            "median": statistics.median(values), "q3": quantile(.75),
            "max": values[-1], "mean": statistics.mean(values)}


def workbook_check(labels):
    """Compare every committed data cell, then 94 cached workbook controls.

    Read OOXML directly to preserve strings, empty cells, and cached values.
    This does not recalculate formulas or write the workbook.
    """
    with zipfile.ZipFile(RAW / "labels.xlsx") as archive:
        strings = ["".join(t.itertext()) for t in
                   ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(NS + "si")]
        sheets = {}
        for name in sorted(archive.namelist()):
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                cells = {}
                for cell in ET.fromstring(archive.read(name)).iter(NS + "c"):
                    value = cell.find(NS + "v")
                    if value is not None:
                        cells[cell.get("r")] = (strings[int(value.text)]
                                                if cell.get("t") == "s" else value.text)
                sheets[name] = cells
        sheet_names = [s.attrib for s in ET.fromstring(
            archive.read("xl/workbook.xml")).iter(NS + "sheet")]
    cells = sheets["xl/worksheets/sheet1.xml"]
    source = []
    row_index = 2
    while cells.get(f"A{row_index}"):
        source.append([cells.get(f"{letter}{row_index}", "") for letter in "ABCDEFGHIJKLMNOPQ"])
        row_index += 1
    assert len(source) == len(labels), "Workbook/CSV row counts differ"
    expected_fields = ["ai_id", "device", "covid", "abnormal", *ZONES, "frames"]
    source_by_id = {identity(row[0]): row for row in source}
    assert len(source_by_id) == len(source), "Duplicate workbook IDs"
    for row in labels:
        source_row = source_by_id[identity(row["ai_id"])]
        assert identity(row["patient"]) == identity(row["ai_id"])
        assert [row[k] for k in expected_fields[1:]] == source_row[1:], row["patient"]
    def count(field, value):
        return sum(row[field] == value for row in labels)
    checks = [("exams", len(labels), "B251"),
              ("covid", count("covid", "1"), "C233"),
              ("abnormal", count("abnormal", "1"), "D233"),
              ("frames", sum(int(r["frames"]) for r in labels), f"Q{row_index}")]
    checks += [(f"device {i}", count("device", str(i)), f"B{245+i}") for i in range(6)]
    for zone, letter in zip(ZONES, "EFGHIJKLMNOP"):
        checks += [(f"{zone} filled", sum(bool(r[zone]) for r in labels), f"{letter}233"),
                   (f"{zone} defined", sum(r[zone] in {"0", "1", "2", "3", "4"}
                                            for r in labels), f"{letter}241")]
        checks += [(f"{zone} code {i}", count(zone, str(i)), f"{letter}{236+i}")
                   for i in range(5)]
    failures = [(name, actual, cells.get(address)) for name, actual, address in checks
                if str(actual) != cells.get(address)]
    assert not failures, failures
    return {"csv_cells_match": True, "cached_summary_checks_passed": len(checks),
            "sheet_names": sheet_names,
            "data_region_equal_between_sheets": {
                name: all(other.get(f"{c}{r}", "") == cells.get(f"{c}{r}", "")
                          for r in range(1, row_index) for c in "ABCDEFGHIJKLMNOPQ")
                for name, other in sheets.items()}}


def inspect_image(path):
    data = path.read_bytes()
    result = {"path": path.relative_to(RAW).as_posix(), "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # full decoding, not just header inspection
            result.update(format=image.format, width=image.width, height=image.height,
                          mode=image.mode, info_keys=sorted(image.info),
                          exif_keys=sorted(str(k) for k in image.getexif()))
            pixels = image.tobytes()
            prefix = f"{image.mode}:{image.width}:{image.height}:".encode()
            result["pixel_sha256"] = hashlib.sha256(prefix + pixels).hexdigest()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_summary(groups, output, name, labels):
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    pairs = Counter()
    cross = []
    cross_zone = []
    label_conflicts = []
    conflicts_by_exam = Counter()
    for paths in duplicates:
        exams = sorted({p.split("/")[0] for p in paths})
        zones = {int(re.search(r"z(\d+)_", p, re.I)[1]) for p in paths}
        inherited_labels = {
            labels[p.split("/")[0]]["z" + str(int(re.search(r"z(\d+)_", p, re.I)[1]))]
            for p in paths
        }
        if len(zones) > 1:
            cross_zone.append(paths)
        if len(inherited_labels) > 1:
            label_conflicts.append({"files": paths, "labels": sorted(inherited_labels)})
            conflicts_by_exam.update(exams)
        if len(exams) > 1:
            cross.append(paths)
            for i, left in enumerate(exams):
                for right in exams[i+1:]:
                    pairs[left, right] += 1
    detail = [{"left": a, "right": b, "shared_distinct_hashes": n,
               "covid": [labels[a]["covid"], labels[b]["covid"]],
               "device": [labels[a]["device"], labels[b]["device"]]}
              for (a, b), n in sorted(pairs.items())]
    (output / f"{name}_duplicate_groups.json").write_text(
        json.dumps(duplicates, indent=2) + "\n", encoding="utf-8")
    return {"duplicate_hash_groups": len(duplicates),
            "files_in_duplicate_groups": sum(map(len, duplicates)),
            "redundant_copies": sum(len(g)-1 for g in duplicates),
            "cross_exam_hash_groups": len(cross),
            "files_in_cross_exam_groups": sum(map(len, cross)), "cross_exam_pairs": detail,
            "cross_zone_hash_groups": len(cross_zone),
            "label_conflict_hash_groups": len(label_conflicts),
            "files_with_conflicting_inherited_labels": sum(len(g["files"]) for g in label_conflicts),
            "label_conflict_groups_by_exam": dict(conflicts_by_exam),
            "label_conflict_examples": label_conflicts[:3]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/T002_openpocus_audit")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or args.workers < 1:
        parser.error("Output must be under the ignored artifacts/ directory; workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with (DATA / "labels.csv").open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    by_exam = {r["patient"]: r for r in labels}
    folders = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    assert len(by_exam) == len(labels), "Duplicate CSV exam identifiers"
    assert set(folders) == set(by_exam), "Folder/label coverage mismatch"
    workbook = workbook_check(labels)
    inputs = [DATA / "labels.csv", RAW / "labels.xlsx", DATA / "build_labels.py",
              DATA / "download_openpocus.sh", DATA / "LICENSE.md", Path(__file__),
              ROOT / "docs/OpenPOCUS_paper_Kumar2026.pdf",
              ROOT / "docs/OpenPOCUS_supplementary_S1.pdf"]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    inventory = []
    non_images = []
    images = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    shape_counts, info_counts, exif_counts = Counter(), Counter(), Counter()
    frame_labels, zone_counts, exam_counts = Counter(), Counter(), Counter()
    series, internal_ids = defaultdict(list), Counter()
    image_errors, bad_names, internal_mismatches = [], [], Counter()
    manifest_fields = ["path", "bytes", "sha256", "pixel_sha256", "format", "width", "height", "mode"]
    with (output / "frame_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            batches = (pool.map(inspect_image, images[start:start+1000])
                       for start in range(0, len(images), 1000))
            for i, item in enumerate(chain.from_iterable(batches), 1):
                writer.writerow(item)
                relative = item["path"]
                inventory.append({k: item[k] for k in ("path", "bytes", "sha256")})
                byte_groups[item["sha256"]].append(relative)
                exam, name = relative.split("/", 1)
                exam_counts[exam] += 1
                if "error" in item:
                    image_errors.append(item)
                else:
                    pixel_groups[item["pixel_sha256"]].append(relative)
                    shape_counts[f'{item["format"]}:{item["width"]}x{item["height"]}:{item["mode"]}'] += 1
                    info_counts.update(item["info_keys"])
                    exif_counts.update(item["exif_keys"])
                zone_match = re.search(r"(?:^|_)z(\d{1,2})_", name, re.I)
                frame_match = re.search(r"frame_(\d+)(.*?)\.[^.]+$", name, re.I)
                if not zone_match or not frame_match:
                    bad_names.append(relative)
                    continue
                zone = f"z{int(zone_match[1])}"
                if zone not in ZONES:
                    bad_names.append(relative)
                    continue
                zone_counts[exam, zone] += 1
                frame_labels[by_exam[exam][zone] or "missing"] += 1
                family = re.sub(r"^image_\d+_", "", name[:frame_match.start()], flags=re.I).lower()
                family += frame_match[2]
                series[exam, zone, family].append(int(frame_match[1]))
                patient_match = re.search(r"(?:^|_)(pt\d+(?:\.\d+)?[ABC]?|ED\d+)_", name, re.I)
                if patient_match:
                    internal_ids[patient_match[1]] += 1
                    if identity(patient_match[1]) != identity(exam):
                        internal_mismatches[exam, patient_match[1]] += 1
                if i % 25000 == 0:
                    print(f"Decoded and hashed {i:,}/{len(images):,} frames", flush=True)
    image_paths = set(images)
    csv_summaries = []
    for path in files:
        if path in image_paths:
            continue
        relative = path.relative_to(RAW).as_posix()
        inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
        non_images.append(relative)
        if path.suffix.lower() != ".csv":
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        summary = {"path": relative, "rows": len(rows), "fields": fields}
        filename_key = "filename" if "filename" in fields else "File name"
        expected = [r[filename_key] for r in rows]
        actual = {p.name for p in path.parent.iterdir() if p.suffix.lower() == ".jpg"}
        normalized = {str(Path(n).with_suffix(".jpg")) for n in expected}
        summary.update(exact_name_matches=sum(n in actual for n in expected),
                       jpg_name_matches=sum(str(Path(n).with_suffix(".jpg")) in actual for n in expected),
                       jpgs_without_metadata=len(actual - normalized),
                       duplicate_names=len(expected) - len(set(expected)))
        for key in fields:
            if key != filename_key and key not in {"image_number", "file_size", "timestamp"}:
                summary[key] = dict(Counter(r[key] for r in rows))
        if "timestamp" in fields:
            stamps = [r["timestamp"] for r in rows]
            summary["timestamp"] = {"unique": len(set(stamps)), "min": min(stamps), "max": max(stamps)}
        if "annotation" in fields:
            annotation_labels = Counter()
            for name in expected:
                zone = "z" + str(int(re.search(r"z(\d+)", name, re.I)[1]))
                annotation_labels[by_exam[path.parent.name][zone] or "missing"] += 1
            summary["zone_labels"] = dict(annotation_labels)
        csv_summaries.append(summary)
    write_csv(output / "raw_inventory.csv", sorted(inventory, key=lambda r: r["path"]), ["path", "bytes", "sha256"])
    zone_rows = [{"exam": r["patient"], "zone": z, "label": r[z],
                  "frames": zone_counts[r["patient"], z]} for r in labels for z in ZONES]
    write_csv(output / "zones.csv", zone_rows, ["exam", "zone", "label", "frames"])
    series_rows = [{"exam": e, "zone": z, "family": f, "frames": len(v),
                    "unique_indices": len(set(v)), "first_index": min(v), "last_index": max(v),
                    "gaps_between_first_last": max(v)-min(v)+1-len(set(v))}
                   for (e, z, f), v in sorted(series.items())]
    write_csv(output / "series.csv", series_rows, list(series_rows[0]))
    grouped = defaultdict(list)
    for exam in folders:
        grouped[re.sub(r"[ABC]$", "", exam, flags=re.I)].append(exam)
    groups = [v for v in grouped.values() if len(v) > 1]
    result = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "pillow": pillow_version},
        "source_sha256": source_hashes, "workbook_check": workbook,
        "scale": {"exam_folders": len(folders), "label_rows": len(labels), "raw_files": len(files),
                  "frames": len(images), "raw_bytes": sum(s[0] for s in before.values()),
                  "frame_bytes": sum(before[p][0] for p in images),
                  "extensions": dict(Counter(p.suffix.lower() for p in files)),
                  "observed_exam_zones": sum(r["frames"] > 0 for r in zone_rows),
                  "named_frame_series": len(series_rows)},
        "patient_labels": {k: dict(Counter(r[k] for r in labels)) for k in ("covid", "abnormal", "device")},
        "device_cross_tabs": {d: {k: dict(Counter(r[k] for r in labels if r["device"] == d))
                                   for k in ("covid", "abnormal")} for d in sorted({r["device"] for r in labels})},
        "zone_labels": dict(Counter(r["label"] or "missing" for r in zone_rows)),
        "image_backed_zone_labels": dict(Counter(r["label"] or "missing" for r in zone_rows if r["frames"])),
        "frame_labels": dict(frame_labels),
        "abnormal_without_definite_zone_findings": [r["patient"] for r in labels
            if r["abnormal"] == "1" and not any(r[z] in {"1", "2", "3"} for z in ZONES)],
        "normal_with_definite_zone_findings": [r["patient"] for r in labels
            if r["abnormal"] == "0" and any(r[z] in {"1", "2", "3"} for z in ZONES)],
        "zones_by_index": {z: {"labels": dict(Counter(r[z] or "missing" for r in labels)),
                                "observed_exams": sum(zone_counts[e, z] > 0 for e in folders)} for z in ZONES},
        "zones_with_label_without_frames": [r for r in zone_rows if r["label"] and not r["frames"]],
        "zones_with_frames_without_label": [r for r in zone_rows if not r["label"] and r["frames"]],
        "undefined_zone_codes": [r for r in zone_rows if r["label"] and r["label"] not in {"0", "1", "2", "3", "4"}],
        "frames_per_exam": distribution(exam_counts.values()),
        "frames_per_observed_zone": distribution(r["frames"] for r in zone_rows if r["frames"]),
        "observed_zone_count_per_exam": dict(Counter(sum(zone_counts[e,z] > 0 for z in ZONES) for e in folders)),
        "frame_count_mismatches": [{"exam": r["patient"], "spreadsheet": int(r["frames"]),
                                     "actual": exam_counts[r["patient"]],
                                     "delta": exam_counts[r["patient"]]-int(r["frames"])}
                                    for r in labels if int(r["frames"]) != exam_counts[r["patient"]]],
        "shapes": dict(shape_counts), "image_info_keys": dict(info_counts), "exif_keys": dict(exif_counts),
        "decode_errors": image_errors, "unparsed_filenames": bad_names,
        "internal_patient_id_mismatches": [{"folder": e, "filename_id": p, "frames": n}
                                            for (e,p),n in sorted(internal_mismatches.items())],
        "multiple_series_per_exam_zone": [{"exam": e, "zone": z, "series": [r for r in series_rows if r["exam"] == e and r["zone"] == z]}
                                          for e,z in sorted(zone_counts) if sum(r["exam"] == e and r["zone"] == z for r in series_rows) > 1],
        "series_with_index_gaps": [r for r in series_rows if r["gaps_between_first_last"] or r["frames"] != r["unique_indices"]],
        "suffix_groups_for_leakage_review": [{"exams": g, "labels": [by_exam[e] for e in g]} for g in groups],
        "groups_after_suffix_grouping": len(grouped), "non_image_files": non_images,
        "csv_metadata": csv_summaries,
        "exact_file_duplicates": duplicate_summary(byte_groups, output, "file", by_exam),
        "exact_pixel_duplicates": duplicate_summary(pixel_groups, output, "pixel", by_exam),
    }
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in RAW.rglob("*") if p.is_file()}
    assert before == after, "Raw file inventory/stat changed during audit"
    assert source_hashes == {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}, "Source changed during audit"
    result["raw_inventory_stat_unchanged"] = True
    result["source_hashes_unchanged"] = True
    result["inventory_sha256"] = sha256(output / "raw_inventory.csv")
    (output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scale": result["scale"], "zone_labels": result["zone_labels"],
                      "decode_errors": len(image_errors), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
