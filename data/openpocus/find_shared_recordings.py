"""Find exam folders that contain the same recordings under different IDs.

audit_openpocus.py finds no exact (file or pixel hash) duplicates across folders, but some
folders hold the same recordings re-encoded, so their bytes differ. This script takes 5 evenly
spaced query frames from every exam x zone group, compares each at 32x32 grayscale against
every frame of every other folder, and verifies the closest match per other folder at full
512x512 resolution. Each confirmed zone pair is then checked along the whole clip, frame i
against frame i + offset. Linked folders are then merged with the A/B/C suffix families into
split groups.

Needs Pillow and numpy. Without installing anything into the project:
    uv run --no-project --with pillow --with numpy python data/openpocus/find_shared_recordings.py
Writes data/openpocus/split_groups.csv (committed, see decision D018) and the match evidence
under artifacts/. Never writes raw/ or labels.csv.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/openpocus"
RAW = DATA / "raw"
DEVICES = {"0": "Butterfly", "1": "EchoNous", "2": "SonoSite", "3": "Lumify", "4": "Vave", "5": "GE"}
ZONE_RE, FRAME_RE = re.compile(r"_z(\d+)_", re.I), re.compile(r"frame_(\d+)")
CONTENT = 0.005    # frames with fewer pixels above gray 30 than this are blank and match everything
CANDIDATE = 3.0    # RMS gray-level distance at 32x32 to count as a candidate
MAX_DIFF = 0.5     # full-resolution mean absolute difference for a confirmed match
MIN_CORR = 0.995   # full-resolution correlation for a confirmed match


def thumb(path):
    image = Image.open(path)
    image.draft("L", (64, 64))
    return np.asarray(image.convert("L").resize((32, 32)), dtype=np.uint8).ravel()


def full(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def same(a, b):
    """Both frames show content and are identical up to re-encoding."""
    if min((a > 30).mean(), (b > 30).mean()) < CONTENT:
        return False
    return np.abs(a - b).mean() < MAX_DIFF and np.corrcoef(a.ravel(), b.ravel())[0, 1] > MIN_CORR


def family(folder):
    return re.sub(r"(?<=\d)[A-C]$", "", folder)


def inventory_frames(labels, raw):
    """Require complete exam-folder coverage before deriving or writing groups."""
    if not raw.is_dir():
        raise ValueError(f"Raw dataset directory is missing: {raw}")
    folders = {path.name for path in raw.iterdir() if path.is_dir()}
    missing = sorted(set(labels) - folders)
    unlabelled = sorted(folders - set(labels))
    if missing or unlabelled:
        raise ValueError(f"Exam-folder mismatch: missing={missing}, unlabelled={unlabelled}")

    files, clips = [], defaultdict(list)
    for folder in sorted(labels):
        frame_paths = sorted((raw / folder).glob("*.jpg"))
        if not frame_paths:
            raise ValueError(f"No JPEG frames in exam folder: {folder}")
        for path in frame_paths:
            zone_match, frame_match = ZONE_RE.search(path.name), FRAME_RE.search(path.name)
            if not zone_match or not frame_match:
                raise ValueError(f"Unrecognised JPEG frame name: {path}")
            zone = int(zone_match[1])
            if not 1 <= zone <= 12:
                raise ValueError(f"Out-of-range lung zone in JPEG name: {path}")
            clips[(folder, zone)].append((int(frame_match[1]), len(files)))
            files.append(path)
    for key in clips:
        clips[key].sort()
    return files, clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/T002_cross_id_duplicates")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or args.workers < 1:
        parser.error("Output must be under the ignored artifacts/ directory; workers must be positive")
    with (DATA / "labels.csv").open(newline="", encoding="utf-8") as stream:
        label_rows = list(csv.DictReader(stream))
    if not label_rows or len({row["patient"] for row in label_rows}) != len(label_rows):
        raise ValueError("labels.csv must contain unique exam folder IDs")
    labels = {row["patient"]: row for row in label_rows}
    files, clips = inventory_frames(labels, RAW)
    output.mkdir(parents=True, exist_ok=True)
    index = {folder: i for i, folder in enumerate(sorted(labels))}
    folder_of = np.array([index[p.parent.name] for p in files])
    where = {i: (folder, zone, frame) for (folder, zone), v in clips.items() for frame, i in v}

    print(f"loading {len(files)} frames at 32x32")
    with ThreadPoolExecutor(args.workers) as pool:
        thumbs = np.stack(list(pool.map(thumb, files, chunksize=256)))
    bright = (thumbs > 30).mean(1) >= CONTENT

    queries = sorted({v[int(i)][1] for v in clips.values() for i in np.linspace(0, len(v) - 1, 5)})
    queries = np.array([q for q in queries if bright[q]])
    Q = thumbs[queries].astype(np.float32)
    q_sq = (Q ** 2).sum(1)
    print(f"{len(queries)} non-blank query frames from {len(clips)} exam x zone groups")

    # Closest candidate per (query, other folder), searched over every non-blank frame.
    best = {}
    refs = np.flatnonzero(bright)
    for start in range(0, len(refs), 4096):
        r = refs[start:start + 4096]
        R = thumbs[r].astype(np.float32)
        rms = np.sqrt(np.maximum(q_sq[:, None] + (R ** 2).sum(1)[None, :] - 2 * Q @ R.T, 0) / Q.shape[1])
        rms[folder_of[queries][:, None] == folder_of[r][None, :]] = np.inf
        for qi, rj in zip(*np.nonzero(rms < CANDIDATE)):
            key = (int(queries[qi]), int(folder_of[r[rj]]))
            if key not in best or rms[qi, rj] < best[key][0]:
                best[key] = (float(rms[qi, rj]), int(r[rj]))
    print(f"{len(best)} candidate (query, other folder) pairs at 32x32")

    matches, near, rejected = [], [], Counter()
    for (q, _), (_, ref) in sorted(best.items()):
        a, b = full(files[q]), full(files[ref])
        diff, corr = float(np.abs(a - b).mean()), float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if min((a > 30).mean(), (b > 30).mean()) < CONTENT:
            rejected["blank at full resolution"] += 1
        elif diff < MAX_DIFF and corr > MIN_CORR:
            matches.append((q, ref, diff))
        else:
            rejected[f"mean abs diff >= {MAX_DIFF}" if diff >= MAX_DIFF else f"corr <= {MIN_CORR}"] += 1
            if diff < 2 * MAX_DIFF:
                near.append((q, ref))
    print(f"confirmed at full resolution: {len(matches)}; rejected: {dict(rejected)}")

    def aligned(fa, za, fb, zb, offset):
        """Identical frames among up to 30 aligned pairs: frame n of fa/za vs n + offset of fb/zb."""
        ca, cb = dict(clips[(fa, za)]), dict(clips[(fb, zb)])
        both = [n for n in ca if n + offset in cb]
        both = [both[int(i)] for i in np.linspace(0, len(both) - 1, min(30, len(both)))]
        return sum(same(full(files[ca[n]]), full(files[cb[n + offset]])) for n in both), len(both)

    # Summarize per folder pair: which zones share frames, at which frame offset.
    pairs = defaultdict(lambda: defaultdict(Counter))
    rows = []
    for q, ref, diff in matches:
        (fa, za, na), (fb, zb, nb) = sorted([where[q], where[ref]])
        pairs[(fa, fb)][(za, zb)][nb - na] += 1
        rows.append({"folder_a": fa, "zone_a": za, "frame_a": na, "folder_b": fb, "zone_b": zb,
                     "frame_b": nb, "mean_abs_diff": f"{diff:.3f}"})
    with (output / "matches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["folder_a"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: tuple(x.values())))

    print("\nfolder pairs sharing recordings")
    for (fa, fb), zones in sorted(pairs.items()):
        la, lb = labels[fa], labels[fb]
        print(f"  {fa} / {fb}: {sum(sum(c.values()) for c in zones.values())} confirmed query frames; "
              f"devices {DEVICES[la['device']]} / {DEVICES[lb['device']]}")
        for (za, zb), offsets in sorted(zones.items()):
            offset = offsets.most_common(1)[0][0]
            print(f"    z{za} ({len(clips[(fa, za)])} frames, label {la[f'z{za}'] or '-'}) ~ "
                  f"z{zb} ({len(clips[(fb, zb)])} frames, label {lb[f'z{zb}'] or '-'}): "
                  f"{sum(offsets.values())} query matches, offsets {dict(offsets)}; "
                  f"aligned at offset {offset}: {'/'.join(map(str, aligned(fa, za, fb, zb, offset)))} frames identical")

    # Control: rejected near misses between unlinked folders, checked along the clip the same way.
    control = Counter()
    for q, ref in near:
        (fa, za, na), (fb, zb, nb) = sorted([where[q], where[ref]])
        if (fa, fb) not in pairs:
            hits, total = aligned(fa, za, fb, zb, nb - na)
            control[int(hits)] += 1
            if hits >= 2:
                print(f"  near miss {fa} z{za} ~ {fb} z{zb}, offset {nb - na}: {hits}/{total} frames identical")
    print(f"\ncontrol: {sum(control.values())} rejected near misses (mean abs diff < {2 * MAX_DIFF}) between "
          f"unlinked folders; identical aligned frames per near miss: {dict(sorted(control.items()))}")

    # Split groups: suffix families plus folders that share recordings.
    parent = {f: family(f) for f in labels}
    def root(f):
        while parent.get(f, f) != f:
            f = parent[f]
        return f
    for f in labels:
        parent.setdefault(family(f), family(f))
    for fa, fb in pairs:
        ra, rb = root(fa), root(fb)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for f in labels:
        groups[root(f)].append(f)
    with (DATA / "split_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["patient", "split_group"])
        writer.writerows(sorted((f, g) for g, members in groups.items() for f in members))
    print(f"\n{len(labels)} folders -> {len(groups)} split groups; groups with more than one folder:")
    for g, members in sorted(groups.items()):
        if len(members) > 1:
            print(f"  {g}: {', '.join(sorted(members))}")
    print(f"\nwrote {output.relative_to(ROOT)}/matches.csv and data/openpocus/split_groups.csv")


if __name__ == "__main__":
    main()
