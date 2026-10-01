"""Verify offset-zero shared zone recordings within the committed dependency groups.

This is a targeted reproduction of T002b's full-resolution frame check, not a
new all-pairs discovery search. Pt157A/B's drifting-offset partial matches are
protected by their family split group; the audited shared zones have the same
T004 binary semantics. Run --check against the tracked small edge table, or
--write after reviewing a changed source snapshot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import io
from pathlib import Path

import numpy as np

from find_shared_recordings import CANDIDATE, FRAME_RE, ZONE_RE, full, same, thumb


DATA = Path(__file__).resolve().parent
EDGE_FILE = DATA / "t006/shared_recording_edges.csv"
FIELDS = ("exam_a", "zone_a", "exam_b", "zone_b", "verified_frames", "sampled_frames")


def read_groups(path: Path) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["patient", "split_group"]:
            raise ValueError("split_groups.csv schema changed")
        rows = list(reader)
    if not rows or len({r["patient"] for r in rows}) != len(rows):
        raise ValueError("split_groups.csv requires unique exam IDs")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if not row["patient"] or not row["split_group"]:
            raise ValueError("Empty exam or split-group ID")
        groups[row["split_group"]].append(row["patient"])
    return dict(groups)


def inventory_exam(raw: Path, exam: str) -> dict[tuple[int, int], Path]:
    folder = raw / exam
    if not folder.is_dir():
        raise ValueError(f"Missing exam folder: {exam}")
    frames = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        zone, index = ZONE_RE.search(path.name), FRAME_RE.search(path.name)
        if zone is None or index is None or not 1 <= int(zone[1]) <= 12:
            raise ValueError(f"Unrecognized frame: {path}")
        key = (int(zone[1]), int(index[1]))
        if key in frames:
            raise ValueError(f"Ambiguous frame index in {exam} zone {key[0]}: {key[1]}")
        frames[key] = path
    if not frames:
        raise ValueError(f"No frames in exam folder: {exam}")
    return frames


def derive_edges(groups: dict[str, list[str]], raw: Path) -> list[dict[str, str]]:
    members = sorted({exam for group in groups.values() if len(group) > 1 for exam in group})
    frames = {exam: inventory_exam(raw, exam) for exam in members}
    edges = []
    for group_id, exams in sorted(groups.items()):
        for a, b in combinations(sorted(exams), 2):
            left, right = frames[a], frames[b]
            left_zones = sorted({zone for zone, _ in left})
            right_zones = sorted({zone for zone, _ in right})
            for za in left_zones:
                left_indices = {index for zone, index in left if zone == za}
                for zb in right_zones:
                    common = sorted(left_indices & {index for zone, index in right if zone == zb})
                    if len(common) < 2:
                        continue
                    # Match the T002b 30-point aligned full-resolution check at offset 0.
                    sample = [common[int(i)] for i in np.linspace(0, len(common) - 1,
                                                                  min(30, len(common)))]
                    hits = 0
                    for index in sample:
                        pa, pb = left[(za, index)], right[(zb, index)]
                        ta, tb = thumb(pa).astype(np.float32), thumb(pb).astype(np.float32)
                        if np.sqrt(np.mean((ta - tb) ** 2)) < CANDIDATE and same(full(pa), full(pb)):
                            hits += 1
                    if hits >= 2:
                        edges.append(dict(zip(FIELDS, (a, f"z{za}", b, f"z{zb}",
                                                      str(hits), str(len(sample))))))
    return edges


def edge_bytes(edges: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(edges, key=lambda e: (e["exam_a"], int(e["zone_a"][1:]),
                                                e["exam_b"], int(e["zone_b"][1:]))))
    return stream.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, default=DATA / "split_groups.csv")
    parser.add_argument("--raw", type=Path, default=DATA / "raw")
    parser.add_argument("--edges", type=Path, default=EDGE_FILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the tracked edge table (default)")
    mode.add_argument("--write", action="store_true", help="update the small tracked edge table")
    args = parser.parse_args()
    if args.edges.resolve().is_relative_to(args.raw.resolve()):
        parser.error("Edge metadata cannot be written inside raw data")
    if args.edges.resolve() in {args.groups.resolve(), (DATA / "labels.csv").resolve(),
                                (DATA / "t006/config.json").resolve()}:
        parser.error("Edge metadata cannot overwrite source labels, groups or configuration")
    groups = read_groups(args.groups)
    edges = derive_edges(groups, args.raw)
    data = edge_bytes(edges)
    if args.write:
        args.edges.parent.mkdir(parents=True, exist_ok=True)
        args.edges.write_bytes(data)
    elif not args.edges.is_file() or args.edges.read_bytes() != data:
        raise ValueError("Tracked shared-recording edges differ from local raw evidence")
    print(f"verified {len(edges)} zone edges; sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
