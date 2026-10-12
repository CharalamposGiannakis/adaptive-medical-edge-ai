"""F007 Phase A2: dependency-group uncertainty and preselected comparisons.

This module analyses validation records only. It has no test-image path and no
test-evidence command. All resampling indices are made once and reused.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.f002_protocol import binary_metrics
from src.f007_protocol import ROOT, load_protocol, validate_structure
from experiments.F007.phase_a1 import exact_string

PREDICTIVE = ("accuracy", "macro_f1", "normal_recall", "abnormal_recall", "auroc")
INTERVAL_METRICS = PREDICTIVE + ("coverage", "mean_full_flops_per_requested_row")


def _auc(targets: list[int], probabilities: list[float]) -> float:
    """Exact tie-aware rank AUC (same binary definition as sklearn)."""
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    order = sorted(range(len(targets)), key=lambda index: probabilities[index])
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(targets[order[index]]
                                                 for index in range(start, end))
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _numeric_metrics(units: list[dict]) -> dict:
    """Metrics for a possibly resampled multiset of whole evaluation units."""
    if not units:
        raise ValueError("F007 statistics require at least one requested unit")
    complete = [unit for unit in units if unit["completed"]]
    targets = [int(unit["binary_target"]) for unit in complete]
    probabilities = [unit["probability"] for unit in complete]
    if complete:
        values = binary_metrics(targets, probabilities, 0.5)
        values["normal_recall"] = values["normal_recall"] if 0 in targets else None
        values["abnormal_recall"] = values["abnormal_recall"] if 1 in targets else None
        values["macro_f1"] = values["macro_f1"] if len(set(targets)) == 2 else None
        values["auroc"] = (_auc(targets, probabilities)
                           if len(set(targets)) == 2 else None)
    else:
        values = {name: None for name in PREDICTIVE}
        values.update(tn=0, fp=0, fn=0, tp=0)
    requested_rows = sum(len(unit["aliases"]) for unit in units)
    completed_rows = sum(len(unit["aliases"]) for unit in complete)
    full_flops = sum(int(unit["full_flops"]) for unit in units)
    values.update(normal_support=targets.count(0), abnormal_support=targets.count(1),
                  requested_units=len(units), completed_units=len(complete),
                  deferred_units=len(units) - len(complete),
                  requested_rows=requested_rows, completed_rows=completed_rows,
                  deferred_rows=requested_rows - completed_rows,
                  coverage=len(complete) / len(units), full_flops=full_flops,
                  mean_full_flops_per_requested_row=full_flops / requested_rows,
                  mean_full_flops_per_requested_row_exact=exact_string(
                      Fraction(full_flops, requested_rows)),
                  mean_full_flops_per_completed_row_exact=(exact_string(
                      Fraction(full_flops, completed_rows)) if completed_rows else None))
    return values


def _index(cells: dict[str, dict[str, list[dict]]]) -> tuple[tuple[str, ...], dict]:
    """Validate one common frozen unit universe across schedules and systems."""
    if not cells:
        raise ValueError("No F007 unit cells")
    indexed = {}
    reference = None
    group_of = {}
    for schedule, systems in cells.items():
        if not systems:
            raise ValueError(f"Empty F007 schedule: {schedule}")
        for system, units in systems.items():
            by_unit = {unit["unit_id"]: unit for unit in units}
            if not units or len(by_unit) != len(units):
                raise ValueError("Duplicate or absent F007 unit")
            if reference is None:
                reference = set(by_unit)
            elif set(by_unit) != reference:
                raise ValueError("F007 cells do not cover the same unit IDs")
            for unit in units:
                if unit["schedule_id"] != schedule or unit["system_id"] != system or \
                        not unit["aliases"] or len({tuple(a) for a in unit["aliases"]}) != len(unit["aliases"]):
                    raise ValueError("Invalid F007 unit cell or alias identity")
                if type(unit["full_flops"]) is not int or unit["full_flops"] < 0:
                    raise ValueError("F007 full FLOPs must be a nonnegative integer")
                identity = (unit["split_group"], int(unit["binary_target"]),
                            tuple(tuple(alias) for alias in unit["aliases"]))
                if unit["unit_id"] in group_of and group_of[unit["unit_id"]] != identity:
                    raise ValueError("F007 unit identity changed across cells")
                group_of[unit["unit_id"]] = identity
                if unit["completed"] != (unit["probability"] is not None):
                    raise ValueError("F007 completion/probability mismatch")
                if unit["completed"] and (not math.isfinite(unit["probability"]) or
                                          not 0 <= unit["probability"] <= 1):
                    raise ValueError("Invalid F007 probability")
                if unit["full_flops"] < 0 or (not unit["completed"] and unit["full_flops"]):
                    raise ValueError("Invalid F007 unit FLOPs")
            indexed[(schedule, system)] = by_unit
    groups = tuple(sorted({identity[0] for identity in group_of.values()}))
    if not groups or any(not group for group in groups):
        raise ValueError("Missing frozen dependency group")
    return groups, indexed


def bootstrap_indices(groups: tuple[str, ...], protocol: dict) -> np.ndarray:
    validate_structure(protocol)
    rule = protocol["bootstrap"]
    if (rule["group"] != "frozen_test_split_group" or
            rule["group_order"] != "sorted_dependency_group_ids" or
            rule["generator"] != "numpy_PCG64" or rule["seed"] != 13013 or
            rule["draws"] != 5000 or rule["minimum_valid_draws"] != 4000 or
            rule["percentiles"] != [2.5, 97.5] or
            rule["reuse_group_indices_across_systems_states_and_contrasts"] is not True):
        raise ValueError("F007 bootstrap protocol changed")
    if not groups or groups != tuple(sorted(set(groups))):
        raise ValueError("Dependency groups must be unique and sorted")
    return np.random.Generator(np.random.PCG64(13013)).integers(
        0, len(groups), size=(5000, len(groups)), dtype=np.int32)


def _resample(by_unit: dict[str, dict], groups: tuple[str, ...], draw: np.ndarray) -> list[dict]:
    members: dict[str, list[dict]] = defaultdict(list)
    for unit in by_unit.values():
        members[unit["split_group"]].append(unit)
    # Extend once per drawn group index. A repeated group keeps its multiplicity.
    return [unit for index in draw for unit in members[groups[int(index)]]]


def _interval(estimate, samples: list[float | None], minimum: int,
              *, structural: bool = False) -> dict:
    valid = [value for value in samples if value is not None and math.isfinite(value)]
    interval = ([float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]
                if len(valid) >= minimum and not structural else None)
    return {"estimate": estimate, "ci95": interval, "valid_draws": len(valid),
            "interval_suppressed": len(valid) < minimum or structural,
            "structural_no_interval": structural}


def cell_analysis(by_unit: dict[str, dict], groups: tuple[str, ...], draws: np.ndarray,
                  protocol: dict, *, factorial: bool) -> dict:
    point = _numeric_metrics(list(by_unit.values()))
    samples = {name: [] for name in INTERVAL_METRICS}
    for draw in draws:
        values = _numeric_metrics(_resample(by_unit, groups, draw))
        for name in samples:
            samples[name].append(values[name])
    minimum = protocol["bootstrap"]["minimum_valid_draws"]
    intervals = {name: _interval(point[name], values, minimum,
                 structural=factorial and name == "coverage")
                 for name, values in samples.items()}
    return {"point": point, "intervals": intervals,
            "deferral_reasons": dict(sorted(Counter(
                unit["deferral_reason"] for unit in by_unit.values()
                if not unit["completed"]).items())),
            "configurations": dict(sorted(Counter(
                unit["configuration"] for unit in by_unit.values()).items()))}


def _paired_values(left: list[dict], right: list[dict]) -> dict:
    if [u["unit_id"] for u in left] != [u["unit_id"] for u in right]:
        raise ValueError("Paired F007 draws lost unit alignment")
    common = [(a, b) for a, b in zip(left, right) if a["completed"] and b["completed"]]
    a_metrics = _numeric_metrics([a for a, _ in common]) if common else None
    b_metrics = _numeric_metrics([b for _, b in common]) if common else None
    all_a, all_b = _numeric_metrics(left), _numeric_metrics(right)
    rows = sum(len(a["aliases"]) for a, _ in common)
    common_flops = sum(a["full_flops"] - b["full_flops"] for a, b in common)
    result = {name: (a_metrics[name] - b_metrics[name]
                     if a_metrics and b_metrics and a_metrics[name] is not None
                     and b_metrics[name] is not None else None) for name in PREDICTIVE}
    result.update(common_completed_units=len(common), common_completed_rows=rows,
                  common_full_flops_difference=common_flops,
                  common_mean_flops_per_row_difference=(common_flops / rows if rows else None),
                  common_mean_flops_per_row_difference_exact=(exact_string(
                      Fraction(common_flops, rows)) if rows else None),
                  coverage_difference=all_a["coverage"] - all_b["coverage"],
                  left_coverage=all_a["coverage"], right_coverage=all_b["coverage"],
                  left_completed_units=all_a["completed_units"],
                  right_completed_units=all_b["completed_units"],
                  left_full_flops_all=all_a["full_flops"],
                  right_full_flops_all=all_b["full_flops"],
                  total_full_flops_difference=all_a["full_flops"] - all_b["full_flops"],
                  total_full_flops_ratio=(all_a["full_flops"] / all_b["full_flops"]
                                          if all_b["full_flops"] else None),
                  total_full_flops_reduction_fraction=(
                      1 - all_a["full_flops"] / all_b["full_flops"]
                      if all_b["full_flops"] else None),
                  common_full_flops_ratio=(sum(a["full_flops"] for a, _ in common) /
                                           sum(b["full_flops"] for _, b in common)
                                           if sum(b["full_flops"] for _, b in common) else None),
                  mean_full_flops_per_requested_row_difference=(
                      all_a["mean_full_flops_per_requested_row"] -
                      all_b["mean_full_flops_per_requested_row"]))
    return result


def paired_analysis(left: dict[str, dict], right: dict[str, dict],
                    groups: tuple[str, ...], draws: np.ndarray, protocol: dict,
                    *, identity_expected: bool = False) -> dict:
    ids = sorted(left)
    if ids != sorted(right):
        raise ValueError("Paired F007 comparison has different unit universes")
    left_rows, right_rows = [left[unit] for unit in ids], [right[unit] for unit in ids]
    point = _paired_values(left_rows, right_rows)
    if identity_expected:
        if any((a["completed"], a["configuration"], a["probability"], a["full_flops"]) !=
               (b["completed"], b["configuration"], b["probability"], b["full_flops"])
               for a, b in zip(left_rows, right_rows)):
            raise ValueError("F007 identity-by-construction pair disagrees")
        return {"point": point, "identity_by_construction": True,
                "intervals": {}, "bootstrap_tests_skipped": True}
    names = PREDICTIVE + (
        "coverage_difference", "common_full_flops_difference",
        "common_mean_flops_per_row_difference", "common_full_flops_ratio",
        "total_full_flops_difference", "total_full_flops_ratio",
        "total_full_flops_reduction_fraction",
        "mean_full_flops_per_requested_row_difference")
    samples = {name: [] for name in names}
    for draw in draws:
        selected = [unit["unit_id"] for unit in _resample(left, groups, draw)]
        values = _paired_values([left[unit] for unit in selected],
                                [right[unit] for unit in selected])
        for name in samples:
            samples[name].append(values[name])
    minimum = protocol["bootstrap"]["minimum_valid_draws"]
    return {"point": point, "identity_by_construction": False,
            "intervals": {name: _interval(point[name], values, minimum)
                          for name, values in samples.items()}}


def comparison_specs(protocol: dict) -> list[tuple[str, str, str, str, bool]]:
    """Read every prespecified pair from the JSON; no result-driven selection."""
    compare = protocol["comparisons"]
    specs = []
    for state, pairs in compare["adaptive_vs_static_by_state"].items():
        specs.extend((f"factorial_b{state}", *pair.split("-"), "adaptive_vs_static", False)
                     for pair in pairs)
    specs.extend(("heterogeneous", *pair.split("-"), "adaptive_vs_static", False)
                 for pair in compare["adaptive_vs_static_heterogeneous"])
    for state in compare["adaptive_policy_states"]:
        schedule = "heterogeneous" if state == "heterogeneous" else f"factorial_b{state}"
        for pair in compare["adaptive_policy_by_state"]:
            specs.append((schedule, *pair.split("-"), "adaptive_policy",
                          (state == "0.25" and pair in ("F-D", "F-E")) or
                          (state == "1.00" and pair == "F-E")))
    fixed = compare["fixed_cost_quality"]
    specs.append((f"factorial_b{fixed['state']}", *fixed["pair"].split("-"),
                  "fixed_cost_quality", False))
    near = compare["near_matched_flops"]
    specs.append((f"factorial_b{near['state']}", *near["pair"].split("-"),
                  "near_matched_flops", False))
    equal = compare["equal_flops_training_recipe"]
    specs.append((f"factorial_b{equal['state']}", *equal["pair"].split("-"),
                  "equal_flops_training_recipe", False))
    return specs


def _switches(requests: list[dict]) -> dict:
    result = {}
    grouped = defaultdict(list)
    for row in requests:
        grouped[(row["schedule_id"], row["system_id"])].append(row)
    for (schedule, system), rows in grouped.items():
        active, to_defer, from_defer = 0, 0, 0
        for old, new in zip(rows, rows[1:]):
            if old["configuration"] == new["configuration"]:
                continue
            if new["configuration"] == "DEFER":
                to_defer += 1
            elif old["configuration"] == "DEFER":
                from_defer += 1
            else:
                active += 1
        result[f"{schedule}:{system}"] = {"active_to_active": active,
                                             "to_defer": to_defer,
                                             "from_defer": from_defer}
    return result


def analyze(cells: dict[str, dict[str, list[dict]]], requests: list[dict],
            protocol: dict, *, require_complete: bool = True) -> tuple[dict, np.ndarray]:
    """Generate one analysis from the fixed unit universe and one draw matrix."""
    validate_structure(protocol)
    if not requests or any(row["partition"] != "validation" for row in requests):
        raise PermissionError("A2 reporting is validation-only before final approval")
    groups, indexed = _index(cells)
    draws = bootstrap_indices(groups, protocol)
    if require_complete:
        expected = {(f"factorial_b{state}", system) for state in protocol["resource_states"]
                    for system in protocol["systems"]}
        expected.update({("heterogeneous", system) for system in protocol["systems"]})
        expected.update({("factorial_b0.25", "f002c_original_k4"),
                         ("factorial_b1.00", "f005b_nested_w1.0_k16")})
        if not expected <= indexed.keys():
            raise ValueError(f"F007 report lacks prespecified cells: {sorted(expected - indexed.keys())}")
    cell_reports = {}
    for (schedule, system), units in indexed.items():
        cell_reports[f"{schedule}:{system}"] = cell_analysis(
            units, groups, draws, protocol, factorial=schedule.startswith("factorial_"))
    contrasts = {}
    omitted = []
    for schedule, left, right, category, identical in comparison_specs(protocol):
        if (schedule, left) not in indexed or (schedule, right) not in indexed:
            if require_complete:
                raise ValueError(f"F007 missing preselected pair {schedule}:{left}-{right}")
            omitted.append(f"{schedule}:{left}-{right}")
            continue
        name = f"{schedule}:{left}-{right}"
        contrasts[name] = {"category": category, **paired_analysis(
            indexed[(schedule, left)], indexed[(schedule, right)], groups, draws,
            protocol, identity_expected=identical)}
    per_device = {}
    for cell, units in indexed.items():
        devices = defaultdict(list)
        for unit in units.values():
            devices[unit["device"]].append(unit)
        per_device[f"{cell[0]}:{cell[1]}"] = {
            device: _numeric_metrics(members) for device, members in sorted(devices.items())}
    return {"experiment": "F007", "partition": "validation", "diagnostic_only":
            not require_complete, "bootstrap": {"seed": 13013, "generator": "numpy_PCG64",
            "draws": 5000, "group_count": len(groups), "sorted_groups": list(groups),
            "indices_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "interval": "approximate_unadjusted_per_comparison_percentile_95"},
            "cells": cell_reports, "contrasts": contrasts,
            "omitted_preselected_contrasts": omitted, "per_device": per_device,
            "switching": _switches(requests)}, draws


def load_validation_evidence(directory: Path) -> tuple[dict, list[dict], dict]:
    from experiments.F007.phase_a1 import aggregate_requests, validation_context
    from src.f002_protocol import key
    from src.f007_protocol import sha256_file

    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("partition") != "validation" or run.get("experiment") != "F007":
        raise PermissionError("F007 A2 accepts only validation evidence")
    protocol = load_protocol()
    if run.get("protocol_sha256") != sha256_file(ROOT / "configs/f007_protocol.json") or \
            run.get("checkpoints") != protocol["checkpoints"]:
        raise ValueError("F007 evidence protocol or frozen checkpoint identity changed")
    all_rows, unit_of, _ = validation_context(protocol, full_hashes=False)
    by_key = {key(row): row for row in all_rows}
    requested = [tuple(alias) for alias in run["request_ids"]]
    if len(requested) != len(set(requested)) or any(alias not in by_key for alias in requested):
        raise ValueError("F007 evidence requests are not canonical validation aliases")
    rows = [by_key[alias] for alias in requested]
    requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if not requests or any(row.get("partition") != "validation" for row in requests):
        raise PermissionError("F007 A2 refuses non-validation request records")
    units = [json.loads(line) for line in (directory / "units.jsonl").read_text(
        encoding="utf-8").splitlines()]
    by_cell = defaultdict(list)
    for record in requests:
        alias = key(record)
        if alias not in requested or record["protocol_sha256"] != run["protocol_sha256"]:
            raise ValueError("F007 request identity or protocol mismatch")
        expected_sha = protocol["checkpoints"][record["checkpoint_key"]]
        expected_sha = expected_sha.get("sha256", expected_sha.get("checkpoint_sha256"))
        if record["checkpoint_sha256"] != expected_sha:
            raise ValueError("F007 request checkpoint identity mismatch")
        by_cell[(record["schedule_id"], record["system_id"])].append(record)
    recomputed = []
    for (schedule, system), records in by_cell.items():
        recomputed.extend(aggregate_requests(rows, unit_of, records, system,
                                             schedule_id=schedule)["units"])
    if units != recomputed:
        raise ValueError("F007 unit records differ from canonical alias aggregation")
    cells = defaultdict(lambda: defaultdict(list))
    for unit in units:
        cells[unit["schedule_id"]][unit["system_id"]].append(unit)
    return cells, requests, run


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No F007 report rows for {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(directory: Path, report: dict, draws: np.ndarray, *, source: Path) -> None:
    if not directory.resolve().is_relative_to((ROOT / "artifacts/F007").resolve()):
        raise ValueError("F007 report evidence belongs in ignored artifacts/F007")
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "bootstrap_indices.npy", draws, allow_pickle=False)
    (directory / "report.json").write_text(json.dumps(
        {**report, "source_evidence": str(source)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    _write_csv(directory / "quality_coverage_cost.csv", [
        {"cell": name, **cell["point"],
         **{f"{metric}_ci95_{side}": interval["ci95"][index]
            if interval["ci95"] is not None else None
            for metric, interval in cell["intervals"].items()
            for side, index in (("low", 0), ("high", 1))},
         **{f"{metric}_valid_draws": interval["valid_draws"]
            for metric, interval in cell["intervals"].items()}}
        for name, cell in sorted(report["cells"].items())])
    if report["contrasts"]:
        _write_csv(directory / "paired_contrasts.csv", [
            {"contrast": name, "category": result["category"],
             "identity_by_construction": result["identity_by_construction"],
             **result["point"],
             **{f"{metric}_ci95_{side}": interval["ci95"][index]
                if interval["ci95"] is not None else None
                for metric, interval in result["intervals"].items()
                for side, index in (("low", 0), ("high", 1))},
             **{f"{metric}_valid_draws": interval["valid_draws"]
                for metric, interval in result["intervals"].items()}}
            for name, result in sorted(report["contrasts"].items())])
    heterogeneous = [
        {"system": name.split(":", 1)[1], **cell["point"]}
        for name, cell in sorted(report["cells"].items())
        if name.startswith("heterogeneous:")]
    if heterogeneous:
        _write_csv(directory / "heterogeneous.csv", heterogeneous)
    _write_csv(directory / "per_device.csv", [
        {"cell": cell, "device": device, **values}
        for cell, groups in sorted(report["per_device"].items())
        for device, values in groups.items()])
    _write_csv(directory / "configurations.csv", [
        {"cell": cell, "configuration": configuration, "units": count}
        for cell, value in sorted(report["cells"].items())
        for configuration, count in value["configurations"].items()])
    _write_csv(directory / "deferrals.csv", [
        {"cell": cell, "reason": reason, "units": count}
        for cell, value in sorted(report["cells"].items())
        for reason, count in value["deferral_reasons"].items()] or
        [{"cell": "none", "reason": "none", "units": 0}])
    _write_csv(directory / "configuration_switches.csv", [
        {"cell": cell, **counts} for cell, counts in sorted(report["switching"].items())])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.hashsalt"] = "F007-fixed-report"
    schedules = list(dict.fromkeys(name.split(":")[0] for name in sorted(report["cells"])))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False, sharex=True, sharey=True)
    colors = dict(zip(("A", "B", "C", "D", "E", "F"),
                      ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")))
    for index, schedule in enumerate(schedules):
        ax = axes[index // 3][index % 3]
        for name, value in sorted(report["cells"].items()):
            if not name.startswith(schedule + ":"):
                continue
            system = name.split(":", 1)[1]
            point = value["point"]
            if point["macro_f1"] is None:
                continue
            ax.scatter(point["mean_full_flops_per_requested_row"] / 1e9,
                       point["macro_f1"], s=28 + 60 * point["coverage"],
                       color=colors.get(system, "#333333"), label=system)
            ax.annotate(system, (point["mean_full_flops_per_requested_row"] / 1e9,
                                 point["macro_f1"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(schedule.replace("factorial_b", "b="), fontsize=10)
        ax.grid(alpha=.2)
    for index in range(len(schedules), 6):
        axes[index // 3][index % 3].axis("off")
    fig.supxlabel("Full FLOPs per requested alias row (billions)")
    fig.supylabel("Completed-unit macro-F1; point size reflects coverage")
    fig.suptitle("F007 validation diagnostic" if report["diagnostic_only"] else
                 "F007 validation quality, coverage and cost")
    fig.tight_layout()
    fig.savefig(directory / "quality_vs_flops.svg", metadata={"Date": None})
    fig.savefig(directory / "quality_vs_flops.png", dpi=180)
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true",
                        help="allow incomplete validation cells; label output diagnostic")
    args = parser.parse_args()
    protocol = load_protocol()
    cells, requests, _ = load_validation_evidence(args.evidence)
    report, draws = analyze(cells, requests, protocol, require_complete=not args.diagnostic)
    write_report(args.output, report, draws, source=args.evidence)
    print(json.dumps({"output": str(args.output), "cells": len(report["cells"]),
                      "contrasts": len(report["contrasts"]),
                      "diagnostic_only": report["diagnostic_only"]}))


if __name__ == "__main__":
    main()
