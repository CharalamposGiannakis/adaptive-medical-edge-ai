"""Fail-closed checks for the accepted F002b run and D024 timing evidence."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src import f002_pipeline as pipe
from src.f002_protocol import PRIMARY_BENCHMARK_FLAGS

SPEC = importlib.util.spec_from_file_location(
    "run_f002", Path(__file__).resolve().parents[1] / "experiments/F002/run_f002.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def fixed_requests():
    timed = [(f"Pt{i}", "z1") for i in range(200)]
    return {"timed": timed, "warmup": timed[:50]}


def synthetic_run(candidate="mobilenet_v2", seed=42, stage="f002a"):
    return {"candidate": candidate, "seed": seed, "status": "valid",
            "checkpoint": f"artifacts/F002/{stage}/{candidate}_seed{seed}/lr1e-4/checkpoint.pt",
            "checkpoint_sha256": "abc123", "backbone_decision": "D025" if stage == "f002b" else None}


def synthetic_bench(candidate="mobilenet_v2", seed=42, k=16, stage="f002a"):
    ids = fixed_requests()
    primary = "model_only_fused" if candidate == "deit_tiny" else "model_only"
    return {"candidate": candidate, "seed": seed, "stage": stage, "k": k,
            "checkpoint_sha256": "abc123", "benchmark_platform": "DAS-5",
            "execution_device": "cuda:0", "benchmark_flags": dict(PRIMARY_BENCHMARK_FLAGS),
            "request_ids": [f"{e}/{z}" for e, z in ids["timed"]],
            "warmup_ids": [f"{e}/{z}" for e, z in ids["warmup"]],
            "timed_requests": 200, "warmup_per_repeat": 50, "repeats": 5,
            "environment": {"gpu_model": "GTX 1080 Ti", "node_features": "gpunode,GTX1080Ti",
                            "hostname": "node201", "slurm_node": "node201",
                            "slurm_job_id": "123"},
            "primary_path": primary,
            "paths": {primary: {"median_latency_ms": 8, "p95_latency_ms": 9,
                                "sequence_throughput_s": 100, "frame_throughput_s": 1600,
                                "raw": [{"latencies_ms": [8.0] * 200, "wall_seconds": 2.0}
                                        for _ in range(5)]}},
            "fused_runtime_valid": True if candidate == "deit_tiny" else None,
            "deit_equivalence": {"equivalent": True, "requests": 200, "k": k,
                                 "tolerance": 1e-5, "max_abs_logit_difference": 1e-6}
            if candidate == "deit_tiny" else None}


class FrozenF002bTests(unittest.TestCase):
    def test_train_rejects_every_wrong_frozen_choice_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ART", Path(tmp)):
            for candidate, seed, decision in (
                    ("resnet18", 2026, "D025"), ("deit_tiny", 2026, "D025"),
                    ("mobilenet_v2", 42, "D025"), ("mobilenet_v2", 2026, "D024"),
                    ("mobilenet_v2", 2026, None)):
                args = SimpleNamespace(stage="f002b", candidate=candidate,
                                       seed=seed, decision=decision)
                with self.subTest(candidate=candidate, seed=seed, decision=decision):
                    with patch.object(cli.pipe, "train_candidate") as train, self.assertRaises(SystemExit):
                        cli.cmd_train(args)
                    train.assert_not_called()

    def test_invalid_f002b_training_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ART", Path(tmp)), \
                patch.object(cli, "device", return_value=SimpleNamespace(type="cpu")), \
                patch.object(cli, "all_normal_f1", return_value=0.4), \
                patch.object(cli.pipe, "train_candidate", return_value={
                    "candidate": "mobilenet_v2", "seed": 2026, "status": "invalid_validity_gate",
                    "attempts": ["lr1e-4"], "selected": {}}):
            args = SimpleNamespace(stage="f002b", candidate="mobilenet_v2",
                                   seed=2026, decision="D025")
            with self.assertRaises(SystemExit):
                cli.cmd_train(args)

    def test_f002b_run_metadata_and_checkpoint_are_required(self):
        run_dir = Path("artifacts/F002/f002b/mobilenet_v2_seed2026")
        run = synthetic_run("mobilenet_v2", 2026, "f002b")
        for field, value in (("status", "invalid_validity_gate"),
                             ("candidate", "resnet18"), ("seed", 43),
                             ("backbone_decision", "D024"),
                             ("checkpoint_sha256", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipe.require_frozen_f002b(run | {field: value}, run_dir)
        with patch.object(pipe, "sha256_file", return_value="different"), self.assertRaises(ValueError):
            pipe.require_frozen_f002b(run, run_dir)

    def test_invalid_f002b_cannot_be_evaluated_or_benchmarked(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ART", Path(tmp)):
            out = cli.run_dir("f002b", "mobilenet_v2", 2026)
            pipe.write_json(out / "run.json", synthetic_run("mobilenet_v2", 2026, "f002b") |
                            {"status": "invalid_validity_gate"})
            args = SimpleNamespace(stage="f002b", candidate="mobilenet_v2", seed=2026, k=[16])
            with patch.object(cli, "device", return_value=torch.device("cpu")), \
                    patch.object(cli.pipe, "load_inputs", return_value={"validation": []}), \
                    self.assertRaises(ValueError):
                cli.cmd_evaluate(args)
            with patch.object(cli, "device", return_value=torch.device("cuda")), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(cli.pipe, "set_benchmark_flags"), \
                    patch.object(cli.pipe, "environment_record", return_value={
                        "slurm_job_id": "1", "slurm_node": "node201",
                        "node_features": "gpunode,GTX1080Ti",
                        "gpu_model": "GTX 1080 Ti"}), \
                    self.assertRaises(ValueError):
                cli.cmd_benchmark(args)
            self.assertFalse((out / "benchmark_k16.json").exists())

    def test_cpu_cannot_create_primary_benchmark(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            pipe.benchmark_checkpoint(Path("unused"), 16, SimpleNamespace(type="cpu"))


class BenchmarkEvidenceTests(unittest.TestCase):
    def validate(self, bench, run=None, hardware=None, k=16):
        return pipe.validate_benchmark_evidence(
            bench, run or synthetic_run(), stage="f002a", candidate="mobilenet_v2",
            seed=42, k=k, request_ids=fixed_requests(), hardware=hardware)

    def test_matched_evidence_and_legacy_stage(self):
        bench = synthetic_bench()
        self.assertEqual(self.validate(bench), ("GTX 1080 Ti", "gpunode,GTX1080Ti"))
        del bench["stage"]  # completed DAS-5 artifacts predate the explicit field
        self.assertEqual(self.validate(bench), ("GTX 1080 Ti", "gpunode,GTX1080Ti"))

    def test_rejects_provenance_comparability_and_primary_path_mismatches(self):
        mutations = {
            "cpu": lambda b: b.update(execution_device="cpu"),
            "platform": lambda b: b.update(benchmark_platform="workstation"),
            "gpu": lambda b: b["environment"].update(gpu_model="A100"),
            "node": lambda b: b["environment"].update(node_features="other"),
            "no job": lambda b: b["environment"].update(slurm_job_id=None),
            "flags": lambda b: b["benchmark_flags"].update({"torch.backends.cudnn.benchmark": False}),
            "timed ids": lambda b: b["request_ids"].reverse(),
            "warmup ids": lambda b: b["warmup_ids"].reverse(),
            "checkpoint": lambda b: b.update(checkpoint_sha256="wrong"),
            "candidate": lambda b: b.update(candidate="resnet18"),
            "seed": lambda b: b.update(seed=43),
            "stage": lambda b: b.update(stage="f002b"),
            "K": lambda b: b.update(k=8),
            "missing path": lambda b: b.update(paths={}),
            "invalid path": lambda b: b["paths"]["model_only"].update(median_latency_ms=0),
            "short timing": lambda b: b["paths"]["model_only"]["raw"][0].update(
                latencies_ms=[8.0] * 199),
        }
        for name, mutate in mutations.items():
            bench = synthetic_bench()
            mutate(bench)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(bench, hardware=("GTX 1080 Ti", "gpunode,GTX1080Ti"))

    def test_deit_requires_fused_equivalent_primary(self):
        run = synthetic_run("deit_tiny")
        bench = synthetic_bench("deit_tiny")
        args = dict(stage="f002a", candidate="deit_tiny", seed=42, k=16,
                    request_ids=fixed_requests())
        self.assertTrue(pipe.validate_benchmark_evidence(bench, run, **args))
        bench["deit_equivalence"]["max_abs_logit_difference"] = 1e-4
        with self.assertRaises(ValueError):
            pipe.validate_benchmark_evidence(bench, run, **args)

    def test_deit_equivalence_failure_stops_before_any_timing(self):
        import torch
        run = synthetic_run("deit_tiny")
        failures = (RuntimeError("difference"),
                    {"equivalent": False, "requests": 200, "k": 16,
                     "tolerance": 1e-5, "max_abs_logit_difference": 0},
                    {"equivalent": True, "requests": 200, "k": 16,
                     "tolerance": 1e-5, "max_abs_logit_difference": 1e-4})
        for failure in failures:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(pipe, "set_benchmark_flags"), \
                    patch.object(pipe, "environment_record", return_value={
                        "slurm_job_id": "1", "slurm_node": "node201",
                        "node_features": "gpunode,GTX1080Ti",
                        "gpu_model": "GTX 1080 Ti"}), \
                    patch.object(pipe, "load_checkpoint", return_value=(run, object())), \
                    patch.object(pipe, "load_inputs", return_value={"validation": [
                        {"exam_id": e, "zone_id": z, "partition": "validation"}
                        for e, z in fixed_requests()["timed"]]}), \
                    patch.object(pipe, "Sequences") as sequences, \
                    patch("src.f002_flops.verify_deit_fused_equivalence",
                          side_effect=failure if isinstance(failure, Exception) else None,
                          return_value=None if isinstance(failure, Exception) else failure), \
                    patch("src.f002_flops.set_deit_fused_attention"), \
                    patch.object(pipe, "_timed_requests") as timing:
                sequences.return_value.__len__.return_value = 200
                sequences.return_value.frames.return_value = MagicMock()
                with self.assertRaisesRegex(ValueError, "stopped"):
                    pipe.benchmark_checkpoint(Path(tmp), 16, torch.device("cuda"))
                timing.assert_not_called()
                self.assertTrue((Path(tmp) / "benchmark_equivalence_failure_k16.json").exists())

    def test_final_gate_checks_both_finalists_and_every_k_before_dominance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ART", Path(tmp)), \
                patch.object(cli.pipe, "load_inputs", return_value={"validation": []}), \
                patch.object(cli, "benchmark_request_ids", return_value=fixed_requests()):
            pipe.write_json(Path(tmp) / "gates/k16_seed42.json",
                            {"retained": ["mobilenet_v2", "resnet18"]})
            for candidate in ("mobilenet_v2", "resnet18"):
                out = cli.run_dir("f002a", candidate, 42)
                pipe.write_json(out / "run.json", synthetic_run(candidate))
                pipe.write_json(out / "k_response.json", {
                    "candidate": candidate, "seed": 42, "checkpoint_sha256": "abc123",
                    "by_k": {str(k): {"duplicate_aware": {"macro_f1": 0.8 if candidate ==
                        "mobilenet_v2" else 0.7}, "flops": k * (100 if candidate ==
                        "mobilenet_v2" else 200)} for k in (4, 8, 16)}})
                for k in (4, 8, 16):
                    pipe.write_json(out / f"benchmark_k{k}.json", synthetic_bench(candidate, k=k))
            altered = cli.run_dir("f002a", "resnet18", 42) / "benchmark_k8.json"
            bad = json.loads(altered.read_text())
            bad["environment"]["gpu_model"] = "A100"
            pipe.write_json(altered, bad)
            with self.assertRaises(ValueError):
                cli.cmd_gate_final(SimpleNamespace(seed=42))
            self.assertFalse((Path(tmp) / "gates/final_seed42.json").exists())
            pipe.write_json(altered, synthetic_bench("resnet18", k=8))
            cli.cmd_gate_final(SimpleNamespace(seed=42))
            self.assertTrue((Path(tmp) / "gates/final_seed42.json").exists())

    def test_k16_gate_checks_saved_evidence_before_reduction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ART", Path(tmp)), \
                patch.object(cli.pipe, "load_inputs", return_value={"validation": []}), \
                patch.object(cli, "benchmark_request_ids", return_value=fixed_requests()), \
                patch.object(cli, "all_normal_f1", return_value=0.4):
            pipe.write_json(Path(tmp) / "preflight/flops.json", {
                "mobilenet_v2_k16": {"flops": 5e9}})
            for candidate in pipe.CANDIDATES:
                out = cli.run_dir("f002a", candidate, 42)
                if candidate == "mobilenet_v2":
                    run = synthetic_run() | {"parameters": 10, "checkpoint_bytes": 100,
                        "rescued": False, "selected": {"epoch": 1, "macro_f1": 0.8,
                        "accuracy": 0.8, "normal_recall": 0.8, "abnormal_recall": 0.8}}
                    pipe.write_json(out / "benchmark_k16.json", synthetic_bench())
                else:
                    run = {"status": "optimization_failed"}
                pipe.write_json(out / "run.json", run)
            altered = cli.run_dir("f002a", "mobilenet_v2", 42) / "benchmark_k16.json"
            bad = json.loads(altered.read_text())
            bad["request_ids"].reverse()
            pipe.write_json(altered, bad)
            with self.assertRaises(ValueError):
                cli.cmd_gate_k16(SimpleNamespace(seed=42))
            self.assertFalse((Path(tmp) / "gates/k16_seed42.json").exists())
            pipe.write_json(altered, synthetic_bench())
            cli.cmd_gate_k16(SimpleNamespace(seed=42))
            self.assertTrue((Path(tmp) / "gates/k16_seed42.json").exists())


if __name__ == "__main__":
    unittest.main()
