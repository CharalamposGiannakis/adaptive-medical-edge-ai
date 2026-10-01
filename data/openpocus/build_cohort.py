"""Build the architecture-independent T006 cohort and group-disjoint partitions.

Reads immutable T004 sources, the committed T002b groups and verified shared
recording edges. Writes only small sequence-level metadata, never raw images.
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
import statistics

from build_binary_task import DATA, ROOT, ZONES, build_records


T006 = DATA / "t006"
PARTITIONS = ("train", "validation", "test")
MANIFEST_FIELDS = ("exam_id", "zone_id", "source_label", "binary_target", "target_name",
                   "split_group", "partition", "device", "t004_frame_count", "frame_count",
                   "frame_paths_sha256", "quality_flags")
EXCLUSION_FIELDS = ("exam_id", "zone_id", "source_label", "binary_target", "split_group",
                    "partition", "t004_frame_count", "retained_frame_count", "exclusion_reason",
                    "quality_flags")
GROUP_FIELDS = ("split_group", "partition", "retained_sequences", "normal", "abnormal")
EDGE_FIELDS = ("exam_a", "zone_a", "exam_b", "zone_b", "verified_frames", "sampled_frames")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    """Hash the Git-canonical LF form of a tracked text CSV."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("".join(p + "\n" for p in paths).encode("utf-8")).hexdigest()


def t004_digest(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_csv(path: Path, fields: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if fields is not None and tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"Unexpected CSV schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def validate_sources(config: dict, labels: Path, groups: Path, edges: Path,
                     records: list[dict]) -> dict[str, str]:
    if config.get("schema_version") != 1 or config.get("task") != "T006_known_family_shared_recording_group_disjoint_cohort":
        raise ValueError("Unsupported T006 configuration")
    percentages = config.get("partition_percentages")
    if tuple(percentages or ()) != PARTITIONS or sum(percentages.values()) != 100 or any(
            not isinstance(value, int) or value <= 0 for value in percentages.values()):
        raise ValueError("Partition percentages must be positive integers totaling 100")
    if not isinstance(config.get("seed"), int):
        raise ValueError("A fixed integer seed is required")
    limits = config.get("split_sanity_limits")
    if not isinstance(limits, dict) or set(limits) != {
            "minimum_sequences_per_class_per_partition",
            "maximum_sequence_share_deviation_percentage_points",
            "maximum_abnormal_share_deviation_percentage_points"} or any(
            not isinstance(value, int) or value <= 0 for value in limits.values()):
        raise ValueError("Invalid split sanity limits")
    observed = {"labels_csv": sha256(labels), "split_groups_csv": canonical_lf_sha256(groups),
                "shared_recording_edges_csv": sha256(edges), "t004_manifest": t004_digest(records)}
    if observed != config.get("expected_sha256"):
        raise ValueError(f"Source fingerprint mismatch: {observed}")
    return observed


def read_metadata(labels: Path, groups: Path, records: list[dict], expected_groups: int):
    label_rows = read_csv(labels)
    if not {"patient", "device", *ZONES}.issubset(label_rows[0]):
        raise ValueError("labels.csv lacks required T004/device fields")
    metadata = {row["patient"]: row for row in label_rows}
    if len(metadata) != len(label_rows) or any(row["device"] not in {"0", "1", "2", "3", "4", "5"}
                                                for row in label_rows):
        raise ValueError("Duplicate exam ID or unexpected device code")
    group_rows = read_csv(groups, ("patient", "split_group"))
    group_of = {row["patient"]: row["split_group"] for row in group_rows}
    if len(group_of) != len(group_rows) or set(group_of) != set(metadata) or any(not g for g in group_of.values()):
        raise ValueError("Split groups must cover each labeled exam exactly once")
    if len(set(group_of.values())) != expected_groups:
        raise ValueError("Known split-group count changed; review grouping evidence")
    if {r["exam_id"] for r in records} != set(metadata):
        raise ValueError("T004 exam coverage differs from labels/groups")
    families = defaultdict(set)
    for exam, group in group_of.items():
        families[re.sub(r"(?<=\d)[A-C]$", "", exam)].add(group)
    if any(len(ids) > 1 for ids in families.values()):
        raise ValueError("A known A/B/C family crosses split groups")
    return metadata, group_of


def shared_components(edges: Path, records: list[dict], group_of: dict[str, str]):
    by_key = {(r["exam_id"], r["zone_id"]): r for r in records}
    rows = read_csv(edges, EDGE_FIELDS)
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    seen = set()
    for row in rows:
        a = (row["exam_a"], row["zone_a"])
        b = (row["exam_b"], row["zone_b"])
        if (a, b) in seen or a >= b or a not in by_key or b not in by_key:
            raise ValueError(f"Invalid or duplicate shared-recording edge: {a}, {b}")
        seen.add((a, b))
        if group_of[a[0]] != group_of[b[0]] or a[0] == b[0]:
            raise ValueError(f"Shared-recording edge crosses known groups: {a}, {b}")
        if not by_key[a]["frame_count"] or not by_key[b]["frame_count"]:
            raise ValueError("Shared-recording edge references a zone without frames")
        verified, sampled = int(row["verified_frames"]), int(row["sampled_frames"])
        if verified < 2 or verified > sampled or sampled > 30:
            raise ValueError("Invalid shared-recording verification counts")
        parent[find(b)] = find(a)

    components = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)
    conflict_nodes, same_nodes, conflicts = set(), set(), []
    for nodes in components.values():
        nodes = sorted(nodes)
        targets = {by_key[node]["binary_target"] for node in nodes}
        if len(targets) > 1:
            conflict_nodes.update(nodes)
            conflicts.append([{"exam_id": exam, "zone_id": zone,
                               "source_label": by_key[exam, zone]["source_label"],
                               "binary_target": by_key[exam, zone]["binary_target"]}
                              for exam, zone in nodes])
        else:
            same_nodes.update(nodes)
    return conflict_nodes, same_nodes, sorted(conflicts, key=lambda c: (c[0]["exam_id"], c[0]["zone_id"])), len(components)


def make_cohort(records: list[dict], metadata: dict, group_of: dict,
                conflict_nodes: set, same_nodes: set):
    retained, excluded = [], []
    pt45_removed, pt45_zones = 0, set()
    for record in records:
        exam, zone = record["exam_id"], record["zone_id"]
        node = (exam, zone)
        paths = list(record["frame_paths"])
        flags = []
        if exam == "Pt45":
            alternate = [p for p in paths if p.endswith(" 2.jpg")]
            if alternate:
                paths = [p for p in paths if not p.endswith(" 2.jpg")]
                if not paths or any(not Path(p).name.startswith("image_") for p in paths):
                    raise ValueError(f"Pt45 masked primary rendition unavailable: {zone}")
                pt45_removed += len(alternate)
                pt45_zones.add(zone)
                flags.append("pt45_masked_primary")
        if node in conflict_nodes:
            flags.append("known_shared_recording_conflict")
        elif node in same_nodes:
            flags.append("known_shared_recording_same_target")
        flag_text = ";".join(flags)
        if not record["label_eligible"]:
            reason = "t004:" + record["label_exclusion_reason"]
        elif not record["frame_count"]:
            reason = "t004:no_frames"
        elif node in conflict_nodes:
            reason = "t006:shared_recording_conflicting_t004_semantics"
        elif not paths:
            reason = "t006:no_retained_frames"
        else:
            reason = None
        common = {"exam_id": exam, "zone_id": zone, "source_label": record["source_label"],
                  "binary_target": record["binary_target"], "split_group": group_of[exam],
                  "t004_frame_count": record["frame_count"], "quality_flags": flag_text}
        if reason:
            excluded.append({**common, "retained_frame_count": len(paths), "exclusion_reason": reason})
        else:
            retained.append({**common, "target_name": record["target_name"], "partition": "",
                             "device": metadata[exam]["device"], "frame_count": len(paths),
                             "frame_paths_sha256": path_digest(paths)})
    if not pt45_removed:
        raise ValueError("Expected Pt45 alternate rendition was not found")
    return retained, excluded, {"alternate_frames_omitted": pt45_removed,
                                "affected_zone_sequences": len(pt45_zones),
                                "zones": sorted(pt45_zones, key=lambda z: int(z[1:]))}


def allocate_groups(rows: list[dict], percentages: dict[str, int], seed: int) -> dict[str, str]:
    """Largest groups first; balance cumulative sequence and both class totals."""
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[row["split_group"]][int(row["binary_target"])] += 1
    order = sorted(groups, key=lambda g: (-sum(groups[g]),
                   hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), g))
    assigned = {p: [0, 0] for p in PARTITIONS}
    seen = [0, 0]
    allocation = {}
    for group in order:
        counts = groups[group]
        next_seen = [seen[c] + counts[c] for c in (0, 1)]
        best = None
        for index, partition in enumerate(PARTITIONS):
            projected = {p: assigned[p].copy() for p in PARTITIONS}
            projected[partition] = [projected[partition][c] + counts[c] for c in (0, 1)]
            score = 0.0
            for p in PARTITIONS:
                ratio = percentages[p] / 100
                for c in (0, 1):
                    target = ratio * next_seen[c]
                    score += (projected[p][c] - target) ** 2 / max(target, 1)
                total_target = ratio * sum(next_seen)
                score += 0.5 * (sum(projected[p]) - total_target) ** 2 / max(total_target, 1)
            candidate = (score, index, partition)
            if best is None or candidate < best:
                best = candidate
        chosen = best[2]
        allocation[group] = chosen
        assigned[chosen] = [assigned[chosen][c] + counts[c] for c in (0, 1)]
        seen = next_seen
    return allocation


def assign_empty_groups(allocation: dict[str, str], source_groups: set[str],
                        percentages: dict[str, int], seed: int) -> None:
    """Give source groups with no retained sequences a stable, non-influential side."""
    for group in sorted(source_groups - set(allocation)):
        bucket = int(hashlib.sha256(f"{seed}:empty:{group}".encode()).hexdigest(), 16) % 100
        cumulative = 0
        for partition in PARTITIONS:
            cumulative += percentages[partition]
            if bucket < cumulative:
                allocation[group] = partition
                break


def verify_split(rows: list[dict], allocation: dict, group_of: dict, percentages: dict,
                 limits: dict):
    keys = [(r["exam_id"], r["zone_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(allocation) != set(group_of.values()):
        raise ValueError("Duplicate sequence or incomplete group allocation")
    for row in rows:
        if row["partition"] != allocation[row["split_group"]] or group_of[row["exam_id"]] != row["split_group"]:
            raise ValueError("Sequence/group partition mismatch")
        if int(row["binary_target"]) not in (0, 1) or int(row["frame_count"]) < 1:
            raise ValueError("Invalid retained target or frame count")
    total = len(rows)
    global_abnormal = sum(int(r["binary_target"]) for r in rows) / total
    for partition in PARTITIONS:
        selected = [r for r in rows if r["partition"] == partition]
        counts = Counter(int(r["binary_target"]) for r in selected)
        if min(counts[0], counts[1]) < limits["minimum_sequences_per_class_per_partition"]:
            raise ValueError(f"Insufficient representation of a class in {partition}: {counts}")
        if abs(len(selected) / total - percentages[partition] / 100) > (
                limits["maximum_sequence_share_deviation_percentage_points"] / 100):
            raise ValueError(f"Pathological sequence proportion in {partition}: {len(selected)}/{total}")
        if abs(counts[1] / len(selected) - global_abnormal) > (
                limits["maximum_abnormal_share_deviation_percentage_points"] / 100):
            raise ValueError(f"Pathological class distribution in {partition}: {counts}")


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    return round(ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])
                 * (position - low), 2)


def partition_summary(rows: list[dict], total: int) -> dict:
    lengths = [int(r["frame_count"]) for r in rows]
    return {"sequences": len(rows), "sequence_fraction": round(len(rows) / total, 6),
            "normal": sum(int(r["binary_target"]) == 0 for r in rows),
            "abnormal": sum(int(r["binary_target"]) == 1 for r in rows),
            "groups": len({r["split_group"] for r in rows}),
            "exams": len({r["exam_id"] for r in rows}),
            "device_sequence_counts": dict(sorted(Counter(r["device"] for r in rows).items())),
            "retained_frame_count": {"min": min(lengths), "q1": percentile(lengths, .25),
                                     "median": percentile(lengths, .5), "q3": percentile(lengths, .75),
                                     "max": max(lengths), "mean": round(statistics.mean(lengths), 2)}}


def csv_bytes(rows: list[dict], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build(labels: Path, raw: Path, groups: Path, edges: Path, config_path: Path,
          output: Path) -> dict:
    output = output.resolve()
    inputs = {p.resolve() for p in (labels, groups, edges, config_path)}
    if output.is_relative_to(raw.resolve()) or any((output / name).resolve() in inputs
              for name in (*[p + ".csv" for p in PARTITIONS], "exclusions.csv",
                           "group_assignments.csv", "summary.json")):
        raise ValueError("T006 output must not overwrite raw data or source inputs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records, labels_digest = build_records(labels, raw)
    fingerprints = validate_sources(config, labels, groups, edges, records)
    if labels_digest != fingerprints["labels_csv"]:
        raise ValueError("T004 label fingerprint mismatch")
    metadata, group_of = read_metadata(labels, groups, records, config["expected_source_group_count"])
    conflict, same, conflict_components, component_count = shared_components(edges, records, group_of)
    retained, excluded, pt45 = make_cohort(records, metadata, group_of, conflict, same)
    allocation = allocate_groups(retained, config["partition_percentages"], config["seed"])
    assign_empty_groups(allocation, set(group_of.values()), config["partition_percentages"],
                        config["seed"])
    for row in retained:
        row["partition"] = allocation[row["split_group"]]
    for row in excluded:
        row["partition"] = allocation[row["split_group"]]
    verify_split(retained, allocation, group_of, config["partition_percentages"],
                 config["split_sanity_limits"])
    retained.sort(key=lambda r: (r["exam_id"], int(r["zone_id"][1:])))
    excluded.sort(key=lambda r: (r["exam_id"], int(r["zone_id"][1:])))
    selected = {p: [r for r in retained if r["partition"] == p] for p in PARTITIONS}
    outputs = {p + ".csv": csv_bytes(selected[p], MANIFEST_FIELDS) for p in PARTITIONS}
    outputs["exclusions.csv"] = csv_bytes(excluded, EXCLUSION_FIELDS)
    group_counts = defaultdict(lambda: [0, 0])
    for row in retained:
        group_counts[row["split_group"]][int(row["binary_target"])] += 1
    retained_group_ids = set(group_counts)
    empty_group_ids = sorted(set(allocation) - retained_group_ids)
    group_rows = [{"split_group": g, "partition": allocation[g],
                   "retained_sequences": sum(group_counts[g]), "normal": group_counts[g][0],
                   "abnormal": group_counts[g][1]} for g in sorted(allocation)]
    outputs["group_assignments.csv"] = csv_bytes(group_rows, GROUP_FIELDS)
    t004_candidates = sum(r["image_backed_candidate"] for r in records)
    t006_exclusion_counts = Counter(r["exclusion_reason"] for r in excluded
                                    if r["exclusion_reason"].startswith("t006:"))
    if t004_candidates != len(retained) + sum(t006_exclusion_counts.values()):
        raise ValueError("T004-to-T006 attrition does not reconcile")
    summary = {
        "schema_version": 1, "task": config["task"], "prediction_unit": "exam_x_lung_zone_sequence",
        "split_scope": "known_family_and_shared_recording_group_disjoint_not_verified_patient_disjoint",
        "seed": config["seed"], "requested_percentages": config["partition_percentages"],
        "split_sanity_limits": config["split_sanity_limits"],
        "t004_image_backed_candidates": t004_candidates, "retained_sequences": len(retained),
        "retained_class_counts": {"normal": sum(int(r["binary_target"]) == 0 for r in retained),
                                  "abnormal": sum(int(r["binary_target"]) == 1 for r in retained)},
        "source_dependency_groups": len(set(group_of.values())),
        "retained_dependency_groups": len(retained_group_ids),
        "source_groups_without_retained_sequences": empty_group_ids,
        "t004_exclusion_counts": dict(sorted(Counter(r["exclusion_reason"] for r in excluded
                                                    if r["exclusion_reason"].startswith("t004:")).items())),
        "t006_exclusion_counts_from_t004_candidates": dict(sorted(t006_exclusion_counts.items())),
        "attrition": {"t004_image_backed_candidates": t004_candidates,
                      "t006_excluded_candidates": sum(t006_exclusion_counts.values()),
                      "t006_retained": len(retained)},
        "pt45": pt45, "verified_shared_recording_edges": len(read_csv(edges, EDGE_FIELDS)),
        "shared_recording_components": component_count,
        "conflicting_shared_recording_components": conflict_components,
        "partitions": {p: partition_summary(selected[p], len(retained)) for p in PARTITIONS},
        "device_note": "Source exam metadata; some known shared recordings have conflicting device entries",
        "sequence_length_unit": "retained_frame_files_not_seconds_or_fps",
        "fingerprints": {**fingerprints, "config_json": sha256(config_path),
                         **{name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()}},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (output / name).write_bytes(data)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    parser.add_argument("--raw", type=Path, default=DATA / "raw")
    parser.add_argument("--groups", type=Path, default=DATA / "split_groups.csv")
    parser.add_argument("--edges", type=Path, default=T006 / "shared_recording_edges.csv")
    parser.add_argument("--config", type=Path, default=T006 / "config.json")
    parser.add_argument("--output", type=Path, default=T006)
    args = parser.parse_args()
    print(json.dumps(build(args.labels, args.raw, args.groups, args.edges, args.config, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
