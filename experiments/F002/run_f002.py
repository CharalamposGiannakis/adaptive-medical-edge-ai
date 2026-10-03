"""T008 command line for the D024 F002 protocol (see README for the run order).

Every command writes under ignored artifacts/F002/. No command runs model
inference on the T006 test partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init (§5)

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import f002_pipeline as pipe  # noqa: E402
from src.f002_flops import build_sequence_model, count_sequence_flops  # noqa: E402
from src.f002_protocol import (benchmark_request_ids, reduce_k16_candidates,  # noqa: E402
                               seed43_confirmation_required)

ART = ROOT / "artifacts" / "F002"
EXPECTED_UNITS = {"train": 1208, "validation": 254}


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(stage: str, candidate: str, seed: int) -> Path:
    return ART / stage / f"{candidate}_seed{seed}"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_preflight(args) -> None:
    """§11 steps 1-3: T006 hashes, units, sampling, strict FLOPs (no training)."""
    inputs = pipe.load_inputs()
    units = {name: len({inputs["unit_of"][pipe.key(r)] for r in inputs[name]})
             for name in ("train", "validation")}
    if units != EXPECTED_UNITS:
        raise SystemExit(f"Evaluation units changed: {units}")
    for name in ("train", "validation"):
        for k in (4, 8, 16):
            pipe.Sequences(inputs[name], k)  # T006 path hash + nested sampling for every row
    flops = {}
    for candidate in pipe.CANDIDATES:
        model = build_sequence_model(candidate, pretrained=False)
        for k in (4, 8, 16):
            flops[f"{candidate}_k{k}"] = count_sequence_flops(model, k) | {
                "candidate": candidate,
                "parameters": sum(p.numel() for p in model.parameters())}
            print(candidate, k, f"{flops[f'{candidate}_k{k}']['flops'] / 1e9:.3f} GFLOPs")
    requests = benchmark_request_ids(inputs["validation"])
    pipe.write_json(ART / "preflight" / "flops.json", flops)
    pipe.write_json(ART / "preflight" / "inputs.json", {
        "t006_verified": True, "evaluation_units": units,
        "rows": {name: len(inputs[name]) for name in ("train", "validation")},
        "train_weight_sum": sum(inputs["train_weights"].values()),
        "benchmark_timed_ids": [f"{e}/{z}" for e, z in requests["timed"]],
        "benchmark_warmup_ids": [f"{e}/{z}" for e, z in requests["warmup"]],
        "environment": pipe.environment_record()})
    print("preflight passed")


def cmd_references(args) -> None:
    """§11 step 4: all-normal and train-fitted metadata-only validation references."""
    inputs = pipe.load_inputs()
    all_normal = pipe.all_normal_reference(inputs)
    metadata = pipe.metadata_reference(inputs)
    out = ART / "references"
    pipe.write_json(out / "all_normal.json", {k: v for k, v in all_normal.items() if k != "rows"} |
                    {"units": len(all_normal["units"])})
    pipe.write_predictions(out / "metadata_validation.csv", metadata)
    pipe.write_json(out / "metadata.json", {k: v for k, v in metadata.items()
                                            if k not in ("rows", "units")})
    print("all-normal", json.dumps(all_normal["duplicate_aware"]))
    print("metadata  ", json.dumps(metadata["duplicate_aware"]))


def all_normal_f1() -> float:
    return read(ART / "references" / "all_normal.json")["duplicate_aware"]["macro_f1"]


def cmd_train(args) -> None:
    """§11 steps 5/8/10: one K16 run with its single allowed rescue."""
    if args.stage == "f002b" and (args.candidate != "mobilenet_v2" or
                                   args.seed != 2026 or args.decision != "D025"):
        raise SystemExit("F002b requires MobileNetV2, seed 2026 and decision D025")
    if args.stage == "f002a" and args.seed not in (42, 43):
        raise SystemExit("F002a uses seed 42 (and 43 only for the confirmation trigger)")
    out = run_dir(args.stage, args.candidate, args.seed)
    if (out / "run.json").exists():
        raise SystemExit(f"{out} already has a completed run; refusing to overwrite")
    run = pipe.train_candidate(args.candidate, args.seed, out, device(), all_normal_f1())
    if args.decision:
        run["backbone_decision"] = args.decision
        pipe.write_json(out / "run.json", run)
    print(json.dumps({k: run[k] for k in ("candidate", "seed", "status", "attempts", "selected")}))
    if args.stage == "f002b" and run["status"] != "valid":
        raise SystemExit("F002b failed its validity gate; frozen baseline unavailable")


def cmd_evaluate(args) -> None:
    """§11 step 7: same K16 checkpoint at K4/8/16 on validation, no retraining."""
    out = run_dir(args.stage, args.candidate, args.seed)
    inputs = pipe.load_inputs()
    run, model = pipe.load_checkpoint(out, device())
    flops = read(ART / "preflight" / "flops.json")
    response = {}
    for k in (4, 8, 16):
        evaluation = pipe.aggregate_validation(
            inputs["validation"], pipe.predict(model, pipe.Sequences(inputs["validation"], k),
                                               inputs["unit_of"], device()), inputs["unit_of"])
        pipe.write_predictions(out / "k_response" / f"validation_k{k}.csv", evaluation)
        response[k] = {"duplicate_aware": evaluation["duplicate_aware"],
                       "row_level": evaluation["row_level"],
                       "flops": flops[f"{args.candidate}_k{k}"]["flops"]}
        print(k, json.dumps(evaluation["duplicate_aware"]))
    if response[16]["duplicate_aware"]["macro_f1"] != run["selected"]["macro_f1"]:
        raise SystemExit("K16 re-evaluation does not reproduce the selected checkpoint metrics")
    pipe.write_json(out / "k_response.json", {"candidate": args.candidate, "seed": args.seed,
                                              "checkpoint_sha256": run["checkpoint_sha256"],
                                              "by_k": response,
                                              "environment": pipe.environment_record()})


def cmd_benchmark(args) -> None:
    """§9 runtime on the fixed 200 validation requests for each requested K."""
    out = run_dir(args.stage, args.candidate, args.seed)
    for k in args.k:
        result = pipe.benchmark_checkpoint(out, k, device())
        pipe.write_json(out / f"benchmark_k{k}.json", result)
        primary = result["paths"][result["primary_path"]]
        print(k, result["primary_path"], f"{primary['median_latency_ms']:.2f} ms",
              f"{primary['sequence_throughput_s']:.1f} seq/s")


def cmd_gate_k16(args) -> None:
    """§6 validity + deterministic reduction to at most two K16 finalists."""
    flops = read(ART / "preflight" / "flops.json")
    rows, runs = [], {}
    requests = benchmark_request_ids(pipe.load_inputs()["validation"])
    hardware = None
    for candidate in pipe.CANDIDATES:
        out = run_dir("f002a", candidate, args.seed)
        run = read(out / "run.json")
        runs[candidate] = run
        if run["status"] == "optimization_failed":
            continue
        bench = read(out / "benchmark_k16.json")
        hardware = pipe.validate_benchmark_evidence(
            bench, run, stage="f002a", candidate=candidate, seed=args.seed,
            k=16, request_ids=requests, hardware=hardware)
        rows.append(pipe.k16_gate_row(run, bench, flops[f"{candidate}_k16"]))
    gate = reduce_k16_candidates(rows, all_normal_f1())
    table = [row | {"parameters": runs[row["id"]]["parameters"],
                    "checkpoint_bytes": runs[row["id"]]["checkpoint_bytes"],
                    "status": runs[row["id"]]["status"], "rescued": runs[row["id"]]["rescued"],
                    "selected_epoch": runs[row["id"]]["selected"]["epoch"]} for row in rows]
    failed = [c for c, r in runs.items() if r["status"] == "optimization_failed"]
    result = gate | {"optimization_failed": failed, "table": table,
                     "all_normal_macro_f1": all_normal_f1(), "seed": args.seed}
    pipe.write_json(ART / "gates" / f"k16_seed{args.seed}.json", result)
    print(json.dumps({k: result[k] for k in ("retained", "invalid", "dominated", "optimization_failed")}))


def cmd_gate_final(args) -> None:
    """§6 two-finalist K4/8/16 dominance; reports the seed-43 trigger."""
    gate = read(ART / "gates" / "k16_seed42.json")  # seed 43 confirms the same finalists
    evidence, status = {}, {}
    requests = benchmark_request_ids(pipe.load_inputs()["validation"])
    hardware = None
    for candidate in gate["retained"]:
        out = run_dir("f002a", candidate, args.seed)
        run = read(out / "run.json")
        status[candidate] = run["status"]
        if status[candidate] != "valid":
            continue  # §6: a failed seed-43 run goes to the human gate, never substituted
        response_record = read(out / "k_response.json")
        if (response_record.get("candidate") != candidate or
                response_record.get("seed") != args.seed or
                response_record.get("checkpoint_sha256") != run["checkpoint_sha256"]):
            raise ValueError("K-response differs from the selected checkpoint")
        response = response_record["by_k"]
        evidence[candidate] = {}
        for k in (4, 8, 16):
            bench = read(out / f"benchmark_k{k}.json")
            hardware = pipe.validate_benchmark_evidence(
                bench, run, stage="f002a", candidate=candidate, seed=args.seed,
                k=k, request_ids=requests, hardware=hardware)
            primary = bench["paths"][bench["primary_path"]]
            evidence[candidate][k] = {
                **response[str(k)]["duplicate_aware"], "flops": response[str(k)]["flops"],
                "model_only_median_latency_ms": primary["median_latency_ms"],
                "model_only_p95_latency_ms": primary["p95_latency_ms"],
                "model_only_sequence_throughput_s": primary["sequence_throughput_s"],
                "model_only_frame_throughput_s": primary["frame_throughput_s"]}
    winner = (pipe.final_dominance(evidence)
              if evidence and len(evidence) == len(gate["retained"]) else None)
    finalists = [{"id": c, "macro_f1": evidence[c][16]["macro_f1"]} for c in evidence]
    result = {"seed": args.seed, "finalists": gate["retained"], "status": status,
              "automatic_backbone": winner,
              "seed43_required": args.seed == 42 and len(finalists) == 2 and
              seed43_confirmation_required(finalists, winner),
              "human_gate_required": winner is None, "evidence": evidence}
    pipe.write_json(ART / "gates" / f"final_seed{args.seed}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight").set_defaults(func=cmd_preflight)
    sub.add_parser("references").set_defaults(func=cmd_references)
    for name, func in (("train", cmd_train), ("evaluate", cmd_evaluate),
                       ("benchmark", cmd_benchmark)):
        command = sub.add_parser(name)
        command.add_argument("--candidate", required=True, choices=pipe.CANDIDATES)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--stage", choices=("f002a", "f002b"), default="f002a")
        if name == "train":
            command.add_argument("--decision", help="decision-log ID recording the backbone (F002b)")
        if name == "benchmark":
            command.add_argument("--k", type=int, nargs="+", default=[16], choices=(4, 8, 16))
        command.set_defaults(func=func)
    for name, func in (("gate-k16", cmd_gate_k16), ("gate-final", cmd_gate_final)):
        command = sub.add_parser(name)
        command.add_argument("--seed", type=int, default=42)
        command.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
