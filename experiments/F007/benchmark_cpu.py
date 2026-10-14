"""One-process, nine-point F007 validation CPU benchmark; never reads test images."""
from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from experiments.F003.adaptive import F003Sequences, positions_in_k16
from experiments.F004.run_f004 import load_static_f002b, timed_pass
from experiments.F004.family import load_family
from experiments.F005.prune import set_width
from experiments.F005.run_f005b import load_deployable
from experiments.F006a_2 import run_analysis as f006a_2
from experiments.F006b import controller
from experiments.F006b.benchmark_cpu import (Inputs as SwitchingInputs,
                                              cpu_model_name, timed_switching)
from experiments.F007.phase_a1 import route, validation_context
from experiments.F007.provenance import assert_unchanged, capture
from src import f002_pipeline as pipe
from src.f002_protocol import benchmark_request_ids, key
from src.f007_protocol import ROOT, load_protocol, rational, sha256_file

TOLERANCE = 1e-4
POINT_IDS = ("f002b_mobilenetv2_k16", "f002c_original_k16",
             "f002c_original_k4", "f005b_w0.5_k8", "f005b_w0.5_k2",
             "f005b_w0.75_k8", "f005b_w1.0_k8", "f005b_w1.0_k12",
             "f005b_w1.0_k16")
NESTED_SWITCH_IDS = ("w0.5_k2", "w0.5_k8", "w0.75_k8", "w1.0_k8",
                     "w1.0_k12", "w1.0_k16")


@dataclass(frozen=True)
class Point:
    name: str
    family: str
    k: int
    width: float | None
    flops: int
    receipt_system: str
    receipt_state: str


def nine_points(protocol: dict, grid: tuple[controller.OperatingPoint, ...]
                ) -> tuple[Point, ...]:
    by_name = {point.name: point for point in grid}
    if set(by_name) != {f"w{width}_k{k}" for width in (0.25, 0.5, 0.75, 1.0)
                        for k in (1, 2, 4, 8, 12, 16)}:
        raise ValueError("Frozen nested 24-point grid changed")
    definitions = (
        Point(POINT_IDS[0], "mobilenet", 16, None, protocol["systems"]["A"]["flops"], "A", "0.25"),
        Point(POINT_IDS[1], "original", 16, None, protocol["systems"]["B"]["flops"], "B", "1.00"),
        Point(POINT_IDS[2], "original", 4, None, 5149017395, "original_k4", "0.25"),
        Point(POINT_IDS[3], "nested", 8, 0.5, by_name["w0.5_k8"].flops, "C", "0.50"),
        Point(POINT_IDS[4], "nested", 2, 0.5, by_name["w0.5_k2"].flops, "D", "0.25"),
        Point(POINT_IDS[5], "nested", 8, 0.75, by_name["w0.75_k8"].flops, "E", "0.50"),
        Point(POINT_IDS[6], "nested", 8, 1.0, by_name["w1.0_k8"].flops, "F", "0.50"),
        Point(POINT_IDS[7], "nested", 12, 1.0, by_name["w1.0_k12"].flops, "F", "0.75"),
        Point(POINT_IDS[8], "nested", 16, 1.0, by_name["w1.0_k16"].flops, "F", "1.00"),
    )
    if tuple(point.name for point in definitions) != tuple(protocol["cpu_benchmark"]["points"]):
        raise ValueError("F007 nine benchmark point IDs changed")
    if tuple(name for name in NESTED_SWITCH_IDS) != tuple(
            protocol["cpu_benchmark"]["switching_nested_points"]):
        raise ValueError("F007 switching point set changed")
    if Fraction(abs(definitions[2].flops - definitions[0].flops),
                definitions[0].flops) > Fraction(1, 100):
        raise ValueError("Preselected near-FLOP pair exceeds 1%")
    if definitions[1].flops != definitions[8].flops:
        raise ValueError("Equal-FLOP comparison costs diverged")
    return definitions


@dataclass(frozen=True)
class Inputs:
    ids: tuple[str, ...]
    rows: tuple[dict, ...]
    bilinear16: tuple
    bicubic16: tuple
    positions: tuple[dict[int, object], ...]
    static_sequences: object
    tiny_sequences: object

    def selected(self, point: Point, index: int):
        source = self.bilinear16 if point.family == "mobilenet" else self.bicubic16
        return source[index], self.positions[index][point.k]


def prepare_inputs(protocol: dict, count: int, *, rows: list[dict] | None = None) -> Inputs:
    """Only first fixed validation IDs; predecode both canonical resize paths."""
    if not 1 <= count <= 200:
        raise ValueError("F007 CPU timing count must lie in 1..200")
    if rows is None:
        rows, _, _ = validation_context(protocol, full_hashes=True)
    fixed = benchmark_request_ids(rows)
    selected_ids = fixed["timed"][:count]
    by_key = {key(row): row for row in rows}
    selected = tuple(by_key[node] for node in selected_ids)
    static = pipe.Sequences(list(selected), 16)
    tiny = F003Sequences(list(selected), 224)
    bilinear = tuple(static.frames(i) for i in range(count))
    bicubic = tuple(tiny.frames(i, 16) for i in range(count))
    import torch
    ks = (1, 2, 4, 8, 12, 16)
    positions = tuple({k: torch.tensor(positions_in_k16(int(row["frame_count"]), k))
                       for k in ks} for row in selected)
    ids = tuple(f"{exam}/{zone}" for exam, zone in selected_ids)
    if ids != tuple(f"{exam}/{zone}" for exam, zone in fixed["timed"][:count]):
        raise ValueError("F007 timing IDs diverged from F002")
    return Inputs(ids, selected, bilinear, bicubic, positions, static, tiny)


def load_models() -> dict:
    return {"mobilenet": load_static_f002b(device="cpu"),
            "original": load_family(device="cpu"),
            "nested": load_deployable(device="cpu")}


def _infer_frames(model, point: Point, frames, *, chunk: int | None) -> tuple[float, float]:
    """Same frozen encoder/head; chunking changes only frame batch size."""
    import torch
    if chunk is not None and (type(chunk) is not int or not 1 <= chunk <= 4):
        raise ValueError("F007 primary encoder chunks must contain at most four frames")
    with torch.inference_mode():
        if point.family == "nested":
            set_width(model.encoder, point.width)
        features = (model.encoder(frames) if chunk is None else
                    torch.cat([model.encoder(part) for part in frames.split(chunk)], dim=0))
        logit = model.head(features.mean(0, keepdim=True)).squeeze(-1)[0]
        return float(logit), float(torch.sigmoid(logit))


def infer_preloaded(models: dict, point: Point, inputs: Inputs, index: int,
                    *, chunk: int | None) -> tuple[float, float]:
    frames16, positions = inputs.selected(point, index)
    return _infer_frames(models[point.family], point,
                         frames16.index_select(0, positions), chunk=chunk)


def verify_equivalence(models: dict, points: tuple[Point, ...], inputs: Inputs) -> dict:
    largest = {}
    for point in points:
        maximum = 0.0
        for index, identity in enumerate(inputs.ids):
            direct, _ = infer_preloaded(models, point, inputs, index, chunk=None)
            chunked, _ = infer_preloaded(models, point, inputs, index, chunk=4)
            difference = abs(direct - chunked)
            if not math.isfinite(difference) or difference > TOLERANCE or (direct >= 0) != (chunked >= 0):
                raise RuntimeError(f"F007 direct/chunk4 mismatch for {point.name} "
                                   f"{identity}: {difference}")
            maximum = max(maximum, difference)
        largest[point.name] = maximum
    return largest


def _timed_modes(models: dict, points: tuple[Point, ...], inputs: Inputs,
                 *, warmup: int, repeats: int, practical: bool,
                 protocol: dict | None = None, grid=None, baseline=None) -> dict:
    """Rotated point order, same IDs/warmups and raw five-repeat timings."""
    if not 1 <= warmup <= len(inputs.ids) or repeats < 1:
        raise ValueError("Invalid F007 CPU timing dimensions")
    raw = {point.name: {"chunk4": [], "direct": []} for point in points}
    requests = list(range(len(inputs.ids)))
    for repeat in range(repeats):
        rotated = points[repeat % len(points):] + points[:repeat % len(points)]
        for point in rotated:
            for mode, chunk in (("chunk4", 4), ("direct", None)):
                def serve(index, point=point, chunk=chunk):
                    if practical:
                        # Clock starts with the fixed-point budget receipt, before JPEG decode.
                        _receipt(point, protocol, grid, baseline)
                        frames = (inputs.static_sequences.frames(index) if point.family == "mobilenet"
                                  else inputs.tiny_sequences.frames(index, point.k))
                        _infer_frames(models[point.family], point, frames, chunk=chunk)
                    else:
                        infer_preloaded(models, point, inputs, index, chunk=chunk)
                raw[point.name][mode].append(timed_pass(serve, requests, requests[:warmup],
                                                        lambda: None))
        print(f"F007 {'JPEG' if practical else 'model-only'} repeat {repeat + 1}/{repeats}",
              flush=True)
    return {point.name: {mode: {**pipe.summarize_timing(raw[point.name][mode], point.k),
                                "raw": raw[point.name][mode]}
                         for mode in ("chunk4", "direct")}
            for point in points}


def _receipt(point: Point, protocol: dict, grid, baseline) -> None:
    """Validate a fixed diagnostic or route the primary system at its frozen state."""
    if point.receipt_system in protocol["systems"]:
        plan = route(point.receipt_system, point.receipt_state, protocol, grid, baseline)
        if plan.deferred or plan.full_flops != point.flops or plan.k != point.k or \
                plan.width != point.width:
            raise ValueError("F007 benchmark receipt no longer selects its frozen point")
    elif point.name == "f002c_original_k4" and point.receipt_state == "0.25":
        if point.flops > controller.resource_fraction("0.25") * rational(
                protocol["resource"]["ccapacity_flops"], "capacity"):
            raise ValueError("F007 original K4 diagnostic exceeds budget")
    else:
        raise ValueError("Unknown F007 benchmark receipt")


def _route_clocks(models: dict, points: tuple[Point, ...], inputs: Inputs,
                  protocol: dict, grid, baseline, *, warmup: int, repeats: int) -> dict:
    """Separate route-only, route-plus-model and early-DEFER clocks."""
    requests = list(range(len(inputs.ids)))
    route_only, served, deferred, served_k = {}, {}, {}, {}
    point_by_config = {point.name.removeprefix("f005b_"): point for point in points
                       if point.family == "nested"}
    point_by_system = {"A": points[0], "B": points[1], "C": points[3]}
    for system in protocol["systems"]:
        for state in protocol["resource_states"]:
            name = f"{system}@{state}"
            def route_once(_):
                return route(system, state, protocol, grid, baseline)
            route_only[name] = [timed_pass(route_once, requests, requests[:warmup],
                                           lambda: None) for _ in range(repeats)]
            plan = route_once(0)
            if plan.deferred:
                deferred[name] = route_only[name]
                continue
            point = (point_by_system[system] if system in point_by_system else
                     point_by_config[plan.configuration])
            served_k[name] = point.k
            def serve(index, point=point):
                selected = route_once(index)
                if selected.deferred or selected.full_flops != point.flops:
                    raise ValueError("F007 served policy route changed during CPU timing")
                infer_preloaded(models, point, inputs, index, chunk=4)
            served[name] = [timed_pass(serve, requests, requests[:warmup],
                                      lambda: None) for _ in range(repeats)]
    return {"route_only": {name: pipe.summarize_timing(raw, 0) | {"raw": raw}
                           for name, raw in route_only.items()},
            "route_plus_model": {name: pipe.summarize_timing(raw, served_k[name]) |
                                 {"raw": raw} for name, raw in served.items()},
            "early_defer": {name: pipe.summarize_timing(raw, 0) | {"raw": raw}
                            for name, raw in deferred.items()}}


def run(*, diagnostic_requests: int | None, output: Path) -> dict:
    """Official is fixed 200/50/5; diagnostics are visibly separate and smaller."""
    if not output.resolve().is_relative_to((ROOT / "artifacts/F007").resolve()):
        raise ValueError("F007 CPU evidence belongs in ignored artifacts/F007")
    protocol = load_protocol()
    cpu = protocol["cpu_benchmark"]
    if (cpu["validation_timed_requests"], cpu["warmup_requests"], cpu["repeats"],
            cpu["cpu_threads"], cpu["frame_encoder_chunk_max"]) != (200, 50, 5, 4, 4):
        raise ValueError("F007 official CPU protocol changed")
    if diagnostic_requests is not None and not 1 <= diagnostic_requests <= 50:
        raise ValueError("Diagnostic mode permits 1..50 validation requests")
    count = 200 if diagnostic_requests is None else diagnostic_requests
    warmup = 50 if diagnostic_requests is None else min(2, count)
    repeats = 5 if diagnostic_requests is None else 1
    import torch
    torch.set_num_threads(4)
    rows, _, verification = validation_context(protocol, full_hashes=True)
    provenance = capture(protocol, verification, official=diagnostic_requests is None,
                         execution_device="cpu", parameters={
                             "diagnostic_requests": diagnostic_requests,
                             "timed_requests": count, "warmup_requests": warmup,
                             "repeats": repeats, "threads": 4,
                             "points": POINT_IDS, "output": str(output),
                             "partition": "validation"})
    grid = controller.load_policy_points()
    baseline = f006a_2.read_baseline()
    points = nine_points(protocol, grid)
    for point in points:
        _receipt(point, protocol, grid, baseline)
    inputs = prepare_inputs(protocol, count, rows=rows)
    models = load_models()
    agreement = verify_equivalence(models, points, inputs)
    model_only = _timed_modes(models, points, inputs, warmup=warmup,
                              repeats=repeats, practical=False)
    switching_grid = tuple(next(p for p in grid if p.name == name)
                           for name in NESTED_SWITCH_IDS)
    switching_inputs = SwitchingInputs(inputs.ids, inputs.bicubic16,
        inputs.positions, "F007 fixed validation timing IDs; canonical bicubic K16")
    switching = timed_switching(models["nested"], switching_grid,
                                switching_inputs, warmup, repeats, 4)
    # Untimed full JPEG read pass before practical-pipeline warmup and timing.
    for index in range(count):
        inputs.static_sequences.frames(index)
        inputs.tiny_sequences.frames(index, 16)
    practical = _timed_modes(models, points, inputs, warmup=warmup,
                             repeats=repeats, practical=True, protocol=protocol,
                             grid=grid, baseline=baseline)
    policy_clocks = _route_clocks(models, points, inputs, protocol, grid, baseline,
                                  warmup=warmup, repeats=repeats)
    env = pipe.environment_record()
    env.update(cpu_model=cpu_model_name(), node=platform.node(),
               torch_interop_threads=torch.get_num_interop_threads(),
               process_count=1, intraop_threads=torch.get_num_threads())
    assert_unchanged(provenance)
    evidence = {"experiment": "F007", "phase": "A2_cpu_benchmark",
                "status": "completed", "partition": "validation",
                "evidence_level": "diagnostic_only" if diagnostic_requests is not None else "official",
                "test_inference_authorized": False,
                "protocol_sha256": sha256_file(ROOT / "configs/f007_protocol.json"),
                "request_ids": inputs.ids, "warmup_ids": inputs.ids[:warmup],
                "timed_requests": count, "warmup_requests": warmup, "repeats": repeats,
                "encoder_chunk_max": 4,
                "point_ids": [point.name for point in points],
                "points": [point.__dict__ for point in points],
                "checkpoint_hashes": {name: value.get("sha256", value.get("checkpoint_sha256"))
                                      for name, value in protocol["checkpoints"].items()},
                "checkpoints": protocol["checkpoints"], "frozen_inputs": verification,
                "provenance": provenance,
                "direct_chunk4_max_abs_logit_difference": agreement,
                "model_only": model_only, "switching": switching,
                "practical_jpeg_pipeline": practical, "policy_clocks": policy_clocks,
                "environment": env, "created_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": "CPU latency/throughput measured separately from full and cached FLOPs"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.json").write_text(json.dumps(evidence, indent=2, sort_keys=True)
                                            + "\n", encoding="utf-8")
    return {"output": str(output), "evidence_level": evidence["evidence_level"],
            "points": len(points), "requests": count, "repeats": repeats,
            "maximum_logit_difference": max(agreement.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--official", action="store_true")
    choice.add_argument("--diagnostic-requests", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts/F007" / f"cpu_validation_{stamp}"
    print(json.dumps(run(diagnostic_requests=None if args.official else
                         args.diagnostic_requests, output=output), indent=2), flush=True)


if __name__ == "__main__":
    main()
