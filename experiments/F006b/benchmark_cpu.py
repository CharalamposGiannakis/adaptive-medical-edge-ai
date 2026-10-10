"""T012b model-only CPU benchmark of the selected frozen F005b operating points.

All points are timed in one process on one CPU.  Input images are decoded and
normalized before the clock starts; frame selection, width setting, encoding,
pooling and the binary head are inside it.  No training or test rows are used.
"""
from __future__ import annotations

import argparse
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from experiments.F003.adaptive import F003Sequences, positions_in_k16
from experiments.F004.run_f004 import timed_pass
from experiments.F005.prune import set_width
from experiments.F005.run_f005b import CANONICAL_NESTED, load_deployable
from src import f002_pipeline as pipe
from src.f002_protocol import benchmark_request_ids, key

DEFAULT_OUTPUT = ROOT / "artifacts" / "F006b" / "cpu_benchmark_local.json"
AGREEMENT_TOLERANCE = 1e-4  # the same absolute logit check as F004's chunked CPU study


@dataclass(frozen=True)
class Inputs:
    ids: tuple[str, ...]
    frames16: tuple[torch.Tensor, ...]
    positions: tuple[dict[int, torch.Tensor], ...]
    source: str

    def frames(self, index: int, k: int) -> torch.Tensor:
        return self.frames16[index].index_select(0, self.positions[index][k])


def cpu_model_name() -> str:
    """Return a real CPU identity on Windows where platform.processor is empty."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as cpu:
                name, _ = winreg.QueryValueEx(cpu, "ProcessorNameString")
                if name.strip():
                    return name.strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or "unknown"


def validation_inputs(ks: tuple[int, ...], count: int) -> Inputs:
    """First frozen timing IDs, each loaded through the train/validation-only F003 loader."""
    if not 1 <= count <= 200:
        raise ValueError("Validation benchmark request count must be within 1..200")
    validation = pipe.load_inputs()["validation"]
    expected = benchmark_request_ids(validation)["timed"][:count]
    by_key = {key(row): row for row in validation}
    rows = [by_key[request_id] for request_id in expected]
    sequences = F003Sequences(rows, 224)
    frames16 = tuple(sequences.frames(i, 16) for i in range(count))
    positions = tuple({k: torch.tensor(positions_in_k16(int(row["frame_count"]), k))
                       for k in ks} for row in rows)
    return Inputs(tuple(f"{exam}/{zone}" for exam, zone in expected), frames16,
                  positions, "T006 validation; first frozen F002 timing IDs")


def synthetic_inputs(ks: tuple[int, ...], count: int) -> Inputs:
    """Explicit smoke mode; never claim synthetic timing represents DAS-5 evidence."""
    if not 1 <= count <= 50:
        raise ValueError("Synthetic request count must be within 1..50")
    generator = torch.Generator().manual_seed(2026)
    frames16 = tuple(torch.randn(16, 3, 224, 224, generator=generator) for _ in range(count))
    positions = tuple({k: torch.tensor(positions_in_k16(16, k)) for k in ks}
                      for _ in range(count))
    return Inputs(tuple(f"synthetic-{i:03d}" for i in range(count)), frames16,
                  positions, "seed-2026 synthetic normalized-shape tensors; smoke timing only")


def infer_logit(model, width: float, frames: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """One shared-weight nested model, with either a direct or chunked encoder pass."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    set_width(model.encoder, width)
    with torch.inference_mode():
        features = torch.cat([model.encoder(frames[start:start + chunk_size])
                              for start in range(0, len(frames), chunk_size)], dim=0)
        return model.head(features.mean(0, keepdim=True)).squeeze(-1)[0]


def verify_chunk_agreement(model, points, inputs: Inputs, chunk_size: int) -> dict[str, float]:
    """Fail closed if chunking changes any selected point's validation-request logit."""
    differences = {}
    for point in points:
        largest = 0.0
        for i in range(len(inputs.ids)):
            frames = inputs.frames(i, point.k)
            direct = infer_logit(model, point.width, frames, point.k)
            chunked = infer_logit(model, point.width, frames, chunk_size)
            difference = float((direct - chunked).abs())
            largest = max(largest, difference)
            if difference > AGREEMENT_TOLERANCE or bool((direct >= 0) != (chunked >= 0)):
                raise RuntimeError(f"Direct/chunked inference differs for {point.name}, "
                                   f"request {inputs.ids[i]}: |Δlogit|={difference}")
        differences[point.name] = largest
    return differences


def timed_configs(model, points, inputs: Inputs, warmup: int, repeats: int,
                  chunk_size: int) -> dict:
    """Round-robin direct/chunked timing avoids confounding point with job drift."""
    if not 1 <= warmup <= len(inputs.ids) or repeats < 1:
        raise ValueError("Warm-up must use available inputs; repeats must be positive")
    configs = {}
    for point in points:
        for mode, size in (("direct", point.k), (f"chunk{chunk_size}", chunk_size)):
            def once(i, point=point, size=size):
                infer_logit(model, point.width, inputs.frames(i, point.k), size)
            configs[f"{point.name}_{mode}"] = (once, point.k)
    names = list(configs)
    measurements = {name: [] for name in names}
    order = list(range(len(inputs.ids)))
    for repeat in range(repeats):
        for name in names[repeat % len(names):] + names[:repeat % len(names)]:
            measurements[name].append(timed_pass(configs[name][0], order,
                                                 order[:warmup], lambda: None))
        print(f"CPU point-timing repeat {repeat + 1}/{repeats}", flush=True)
    return {name: pipe.summarize_timing(measurements[name], k) | {"raw": measurements[name]}
            for name, (_, k) in configs.items()}


def switching_requests(points, request_count: int) -> tuple[list[tuple[int, str]],
                                                              list[tuple[int, str]]]:
    """Same point×input multiset, once interleaved and once grouped by point."""
    requests = [(i, point.name) for i in range(request_count) for point in points]
    random.Random(2026).shuffle(requests)
    grouped = sorted(requests, key=lambda request: (request[1], request[0]))
    return requests, grouped


def timed_switching(model, points, inputs: Inputs, warmup: int,
                    repeats: int, chunk_size: int) -> dict:
    """F005-style grouped/interleaved comparison, with no cross-clip feature cache."""
    by_name = {point.name: point for point in points}
    interleaved, grouped = switching_requests(points, len(inputs.ids))
    warm = [(i, point.name) for point in points for i in range(warmup)]

    def serve(request):
        i, name = request
        point = by_name[name]
        infer_logit(model, point.width, inputs.frames(i, point.k), chunk_size)

    def measure(order):
        for request in warm:
            serve(request)
        latencies = []
        began_pass = time.perf_counter()
        for request in order:
            began = time.perf_counter()
            serve(request)
            latencies.append((time.perf_counter() - began) * 1000)
        return {"latencies_ms": latencies, "wall_seconds": time.perf_counter() - began_pass}

    passes = []
    for repeat in range(repeats):
        first = "interleaved" if repeat % 2 == 0 else "grouped"
        result = {}
        for name in (first, "grouped" if first == "interleaved" else "interleaved"):
            result[name] = measure(interleaved if name == "interleaved" else grouped)
        passes.append(result)
        print(f"CPU switching repeat {repeat + 1}/{repeats}", flush=True)

    by_point = {}
    for point in points:
        differences = []
        medians = []
        for passed in passes:
            inter = statistics.median(latency for (_, name), latency in
                                      zip(interleaved, passed["interleaved"]["latencies_ms"])
                                      if name == point.name)
            group = statistics.median(latency for (_, name), latency in
                                      zip(grouped, passed["grouped"]["latencies_ms"])
                                      if name == point.name)
            differences.append(inter - group)
            medians.append({"interleaved_ms": inter, "grouped_ms": group})
        by_point[point.name] = {
            "median_interleaved_minus_grouped_ms": statistics.median(differences),
            "min_difference_ms": min(differences), "max_difference_ms": max(differences),
            "per_repeat": medians,
        }
    return {
        "interleaved_requests": interleaved, "grouped_requests": grouped,
        "interleaved_configuration_changes": sum(a[1] != b[1] for a, b in
                                                  zip(interleaved, interleaved[1:])),
        "grouped_configuration_changes": sum(a[1] != b[1] for a, b in
                                              zip(grouped, grouped[1:])),
        "passes": passes, "by_point": by_point,
        "method": "same point × input multiset; width set on every request; no feature cache",
    }


def run(args) -> dict:
    from experiments.F006b.controller import load_policy_points, selected_configurations

    if args.threads < 1 or args.chunk_size < 1:
        raise ValueError("threads and chunk size must be positive")
    if torch.cuda.is_available() and args.device != "cpu":
        raise ValueError("This benchmark is CPU-only")
    torch.set_num_threads(args.threads)
    points = tuple(selected_configurations(load_policy_points()))
    if not points or len({point.name for point in points}) != len(points):
        raise ValueError("Controller returned no unique selected points")
    ks = tuple(sorted({point.k for point in points}))
    inputs = (validation_inputs(ks, args.requests) if args.input_mode == "validation"
              else synthetic_inputs(ks, args.requests))
    model = load_deployable(device="cpu")
    agreement = verify_chunk_agreement(model, points, inputs, args.chunk_size)
    timing = timed_configs(model, points, inputs, args.warmup, args.repeats, args.chunk_size)
    switching = timed_switching(model, points, inputs, args.warmup,
                                args.repeats, args.chunk_size)
    env = pipe.environment_record()
    env["cpu_model"] = cpu_model_name()
    env["torch_interop_threads"] = torch.get_num_interop_threads()
    return {
        "experiment": "F006b", "evidence_type": "supporting local CPU model-only timing",
        "primary_resource": "integer FLOPs; CPU latency is supporting evidence, not a budget",
        "device": "cpu", "input_source": inputs.source, "request_ids": inputs.ids,
        "selected_points": [{"name": p.name, "width": p.width, "k": p.k, "flops": p.flops}
                            for p in points],
        "checkpoint_sha256": pipe.sha256_file(CANONICAL_NESTED / "checkpoint.pt"),
        "run_sha256": pipe.sha256_file(CANONICAL_NESTED / "run.json"),
        "protocol": {"warmup": args.warmup, "timed_requests": args.requests,
                     "repeats": args.repeats, "chunk_size": args.chunk_size,
                     "agreement_absolute_logit_tolerance": AGREEMENT_TOLERANCE,
                     "timing": "frame selection, width setting, encoder, mean pooling, binary head; "
                               "preloaded and normalized inputs; no JPEG decode or data transfer"},
        "direct_chunked_max_abs_logit_difference": agreement,
        "point_timings": timing, "switching": switching, "environment": env,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input-mode", choices=("validation", "synthetic"), default="validation")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite timing evidence: {args.output}")
    record = run(args)
    pipe.write_json(args.output, record)
    for name, point in record["point_timings"].items():
        print(f"{name}: median {point['median_latency_ms']:.2f} ms; "
              f"{point['sequence_throughput_s']:.2f} sequences/s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
