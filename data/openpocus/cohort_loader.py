"""Architecture-independent access to the frozen T006 sequence manifests.

No frame count, sampling, image transform, model input size or normalization is
chosen here. A later caller can inject a sampler and a path-to-image transform.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from build_binary_task import DATA, FRAME_TOKEN, ZONE_TOKEN


T006 = DATA / "t006"
PARTITIONS = {"train", "validation", "test"}


def _digest(paths: tuple[Path, ...], raw: Path) -> str:
    return hashlib.sha256("".join(p.relative_to(raw).as_posix() + "\n" for p in paths)
                           .encode("utf-8")).hexdigest()


def retained_frames(row: dict, raw: Path) -> tuple[Path, ...]:
    """Re-enumerate exactly the retained raw paths and verify their manifest hash."""
    exam, zone = row["exam_id"], row["zone_id"]
    if not exam or exam in {".", ".."} or any(c in exam for c in "/\\:"):
        raise ValueError("Invalid exam ID in T006 manifest")
    if zone not in {f"z{i}" for i in range(1, 13)}:
        raise ValueError("Invalid zone ID in T006 manifest")
    folder = raw / exam
    if not folder.is_dir():
        raise ValueError(f"Missing raw exam folder: {exam}")
    frames = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        zm, fm = ZONE_TOKEN.search(path.name), FRAME_TOKEN.search(path.name)
        if zm is None or fm is None:
            raise ValueError(f"Unmapped JPEG in exam folder: {path.name}")
        if f"z{int(zm[1])}" != zone:
            continue
        if exam == "Pt45" and path.name.endswith(" 2.jpg"):
            continue
        frames.append((int(fm[1]), path.relative_to(raw).as_posix(), path))
    result = tuple(path for _, _, path in sorted(frames))
    if len(result) != int(row["frame_count"]) or _digest(result, raw) != row["frame_paths_sha256"]:
        raise ValueError(f"T006 retained-frame inventory changed: {exam} {zone}")
    return result


class SequenceDataset:
    """One sample is one exam × zone sequence, with caller-supplied processing."""

    def __init__(self, partition: str, raw: Path = DATA / "raw", metadata: Path = T006,
                 sampler: Callable[[tuple[Path, ...]], Iterable[Path]] | None = None,
                 transform: Callable[[Path], object] | None = None):
        if partition not in PARTITIONS:
            raise ValueError("partition must be train, validation or test")
        self.partition = partition
        self.raw = Path(raw)
        self.sampler = sampler
        self.transform = transform
        path = Path(metadata) / f"{partition}.csv"
        summary = json.loads((Path(metadata) / "summary.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != summary["fingerprints"][path.name]:
            raise ValueError(f"T006 {partition} manifest fingerprint changed")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or any(r["partition"] != partition or
                           r["target_name"] != {"0": "normal", "1": "abnormal"}.get(
                               r["binary_target"]) for r in rows):
            raise ValueError(f"Invalid T006 {partition} manifest rows")
        if len({(r["exam_id"], r["zone_id"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate zone sequence in T006 manifest")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        frames = retained_frames(row, self.raw)
        selected = tuple(self.sampler(frames)) if self.sampler is not None else frames
        if not selected or any(path not in frames for path in selected):
            raise ValueError("Sampler must return retained paths from this sequence")
        values = tuple(self.transform(path) for path in selected) if self.transform else selected
        return {"exam_id": row["exam_id"], "zone_id": row["zone_id"],
                "binary_target": int(row["binary_target"]), "target_name": row["target_name"],
                "split_group": row["split_group"], "partition": self.partition,
                "device": row["device"], "quality_flags": row["quality_flags"],
                "retained_frame_count": len(frames), "frames": values}
