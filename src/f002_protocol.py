"""Frozen F002 sampling and shared-recording evaluation units (no model code)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


Key = tuple[str, str]


def nested_frame_indices(frame_count: int, k: int) -> tuple[int, ...]:
    """Zero-based K=4/8/16 indices, nested and spread over the full sequence.

    Start with four equally spaced inclusive anchors floor(j*(N-1)/3).
    Repeatedly bisect the widest index gap (leftmost on ties), using the lower
    midpoint. Return the first K selected indices in chronological order.
    Integer arithmetic makes the result independent of Python/OS floating point.
    """
    if k not in (4, 8, 16) or frame_count < 16:
        raise ValueError("F002 requires K in {4,8,16} and at least 16 frames")
    selected = {j * (frame_count - 1) // 3 for j in range(4)}
    while len(selected) < k:
        ordered = sorted(selected)
        left, right = max(zip(ordered, ordered[1:]),
                          key=lambda pair: (pair[1] - pair[0], -pair[0]))
        midpoint = (left + right) // 2
        if midpoint in selected:
            raise ValueError("Cannot select distinct frames")
        selected.add(midpoint)
    return tuple(sorted(selected))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict) -> Key:
    return row["exam_id"], row["zone_id"]


def verify_t006(protocol_path: Path, t006_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Fail closed if any tracked T006 input differs from the preregistration."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    summary = json.loads((t006_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["fingerprints"].get("t004_manifest") != protocol["t006_expected"]["t004_manifest_fingerprint"]:
        raise ValueError("T004 manifest provenance mismatch")
    fingerprint_keys = {"config.json": "config_json", "labels.csv": "labels_csv",
                        "split_groups.csv": "split_groups_csv",
                        "shared_recording_edges.csv": "shared_recording_edges_csv"}
    for name, expected in protocol["t006_sha256"].items():
        if name in ("labels.csv", "split_groups.csv"):
            path = t006_dir.parent / name
        else:
            path = t006_dir / name
        content = path.read_bytes()
        if name == "split_groups.csv":
            content = content.replace(b"\r\n", b"\n")  # T006's canonical Git-text hash.
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"T006 fingerprint mismatch: {name}")
        if name != "summary.json" and summary["fingerprints"].get(
                fingerprint_keys.get(name, name)) != expected:
            raise ValueError(f"T006 summary fingerprint mismatch: {name}")
    partitions = {name: _rows(t006_dir / f"{name}.csv")
                  for name in ("train", "validation", "test")}
    for name, rows in partitions.items():
        if len(rows) != protocol["t006_expected"][name] or any(
                row["partition"] != name for row in rows):
            raise ValueError(f"T006 {name} count/partition changed")
    all_rows = [row for rows in partitions.values() for row in rows]
    if len(all_rows) != protocol["t006_expected"]["retained"] or len({key(row) for row in all_rows}) != len(all_rows):
        raise ValueError("T006 retained cohort count/uniqueness changed")
    if min(int(row["frame_count"]) for row in all_rows) < protocol["sampling"]["initial_k"]:
        raise ValueError("K=16 cannot be applied to every retained sequence")
    targets = [int(row["binary_target"]) for row in all_rows]
    if (targets.count(0) != protocol["t006_expected"]["normal"] or
            targets.count(1) != protocol["t006_expected"]["abnormal"]):
        raise ValueError("T006 target distribution changed")
    if len({row["split_group"] for row in all_rows}) != protocol["t006_expected"]["retained_groups"]:
        raise ValueError("T006 retained group count changed")
    if json.loads((t006_dir / "config.json").read_text(encoding="utf-8"))["seed"] != protocol["t006_expected"]["split_seed"]:
        raise ValueError("T006 split seed changed")
    return partitions


def recording_units(partitions: dict[str, list[dict]], edges: list[dict],
                    exclusions: list[dict]) -> dict[Key, str]:
    """Map retained rows to stable IDs of connected verified recordings.

    The graph includes unavailable and T006-excluded aliases, but only retained
    same-target aliases become an evaluation unit. A T006 conflict component
    entering any retained partition is an error.
    """
    retained: dict[Key, dict] = {}
    for partition, rows in partitions.items():
        for row in rows:
            node = key(row)
            if node in retained or row["partition"] != partition:
                raise ValueError("Duplicate or incorrectly partitioned T006 row")
            retained[node] = row

    parent: dict[Key, Key] = {}

    def root(node: Key) -> Key:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = root(parent[node])
        return parent[node]

    for edge in edges:
        a = edge["exam_a"], edge["zone_a"]
        b = edge["exam_b"], edge["zone_b"]
        if a == b:
            raise ValueError("Self-edge in verified recording graph")
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    components: dict[Key, set[Key]] = defaultdict(set)
    for node in parent:
        components[root(node)].add(node)
    conflict_nodes = {key(row) for row in exclusions if row["exclusion_reason"] ==
                      "t006:shared_recording_conflicting_t004_semantics"}
    mapping: dict[Key, str] = {}
    for nodes in components.values():
        members = sorted(nodes & retained.keys())
        if nodes & conflict_nodes and members:
            raise ValueError("T006 semantic-conflict component entered retained cohort")
        if not members:
            continue
        targets = {retained[node]["binary_target"] for node in members}
        groups = {retained[node]["split_group"] for node in members}
        split_names = {retained[node]["partition"] for node in members}
        if len(targets) != 1 or len(groups) != 1 or len(split_names) != 1:
            raise ValueError("Shared recording has conflicting target/group/partition")
        canonical = "|".join(f"{exam}/{zone}" for exam, zone in sorted(nodes))
        unit = "shared:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        for node in members:
            mapping[node] = unit
    for node in retained:
        mapping.setdefault(node, f"singleton:{node[0]}/{node[1]}")
    return mapping


def training_weights(rows: list[dict], unit_of: dict[Key, str]) -> dict[Key, float]:
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        sizes[unit_of[key(row)]] += 1
    return {key(row): 1.0 / sizes[unit_of[key(row)]] for row in rows}


def binary_metrics(targets: list[int], probabilities: list[float],
                   threshold: float = 0.5) -> dict:
    if not targets or len(targets) != len(probabilities):
        raise ValueError("Targets/probabilities must be nonempty and aligned")
    tn = fp = fn = tp = 0
    for y, probability in zip(targets, probabilities):
        if y not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid binary target or probability")
        pred = int(probability >= threshold)
        if y == 0 and pred == 0: tn += 1
        elif y == 0: fp += 1
        elif pred == 0: fn += 1
        else: tp += 1
    def f1(correct: int, false_pos: int, false_neg: int) -> float:
        denom = 2 * correct + false_pos + false_neg
        return 2 * correct / denom if denom else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tn + tp) / len(targets),
            "macro_f1": (f1(tn, fn, fp) + f1(tp, fp, fn)) / 2,
            "normal_recall": tn / (tn + fp) if tn + fp else 0.0,
            "abnormal_recall": tp / (tp + fn) if tp + fn else 0.0}


def aggregate_validation(rows: list[dict], predictions: list[dict],
                         unit_of: dict[Key, str], threshold: float = 0.5) -> dict:
    """Keep every row prediction; average probabilities within recording units."""
    expected = {key(row): row for row in rows}
    observed = {key(pred): pred for pred in predictions}
    if len(observed) != len(predictions) or expected.keys() != observed.keys():
        raise ValueError("Predictions must cover validation rows exactly once")
    grouped: dict[str, list[dict]] = defaultdict(list)
    raw_targets, raw_probs = [], []
    for node, row in expected.items():
        pred = observed[node]
        if int(pred["binary_target"]) != int(row["binary_target"]):
            raise ValueError("Prediction target differs from T006")
        probability = float(pred["probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        grouped[unit_of[node]].append(pred)
        raw_targets.append(int(row["binary_target"]))
        raw_probs.append(probability)
    units = []
    for unit_id, aliases in sorted(grouped.items()):
        targets = {int(row["binary_target"]) for row in aliases}
        if len(targets) != 1:
            raise ValueError("Shared recording targets disagree")
        units.append({"unit_id": unit_id, "aliases": [key(row) for row in aliases],
                      "binary_target": targets.pop(),
                      "probability": sum(float(row["probability"]) for row in aliases) / len(aliases)})
    unit_targets = [row["binary_target"] for row in units]
    unit_probs = [row["probability"] for row in units]
    eps = 1e-7
    loss = -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
                for y, p in zip(unit_targets, unit_probs)) / len(units)
    return {"rows": predictions, "units": units,
            "duplicate_aware": {**binary_metrics(unit_targets, unit_probs, threshold), "loss": loss},
            "row_level": binary_metrics(raw_targets, raw_probs, threshold)}
