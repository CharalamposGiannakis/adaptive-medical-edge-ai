"""Common F007 execution provenance and clean-source gate."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.f007_protocol import ROOT, sha256_file


SOURCE_MODULES = (
    "src/f007_protocol.py",
    "src/f002_protocol.py",
    "src/f002_pipeline.py",
    "src/f002_flops.py",
    "experiments/F002c/model.py",
    "experiments/F003/adaptive.py",
    "experiments/F004/family.py",
    "experiments/F004/run_f004.py",
    "experiments/F005/run_f005b.py",
    "experiments/F006a_2/run_analysis.py",
    "experiments/F006b/controller.py",
    "experiments/F006b/benchmark_cpu.py",
    "experiments/F006b/runtime.py",
    "experiments/F007/phase_a1.py",
    "experiments/F007/validation_run.py",
    "experiments/F007/analysis.py",
    "experiments/F007/cache_diagnostic.py",
    "experiments/F007/benchmark_cpu.py",
    "experiments/F007/provenance.py",
    "experiments/F005/prune.py",
)
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn",
            "matplotlib", "Pillow", "timm", "fvcore")


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(["git", *arguments], cwd=root,
                                       stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify F007 Git provenance: {arguments[0]}") from exc


def git_state(root: Path = ROOT) -> dict:
    """Inspect relevant repository changes without entering local agent folders."""
    commit = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("F007 provenance needs a full Git commit SHA")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal", "--",
                  ".", ":(exclude)CLAUDE", ":(exclude)CODEX").decode("utf-8")
    return {"commit_sha": commit, "working_tree_dirty": bool(status.strip())}


def source_hashes(root: Path = ROOT) -> dict[str, dict[str, str | None]]:
    """Both executed checkout bytes and canonical committed source bytes."""
    values = {}
    for name in SOURCE_MODULES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"F007 source module is missing: {name}")
        local = sha256_file(path)
        try:
            committed = _git(root, "show", f"HEAD:{name}")
        except RuntimeError:
            committed_sha = None  # a new, uncommitted diagnostic module
        else:
            committed_sha = hashlib.sha256(committed).hexdigest()
        values[name] = {"working_tree_sha256": local,
                        "committed_blob_sha256": committed_sha}
    return values


def _package_versions() -> dict[str, str | None]:
    values = {}
    for name in PACKAGES:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def capture(protocol: dict, frozen_inputs: dict, *, official: bool,
            execution_device: str, parameters: dict, root: Path = ROOT) -> dict:
    """Record provenance; reject dirty or uncommitted official execution."""
    state = git_state(root)
    if official and state["working_tree_dirty"]:
        raise RuntimeError("Official F007 execution requires a clean committed working tree")
    sources = source_hashes(root)
    if official and any(value["committed_blob_sha256"] is None
                        for value in sources.values()):
        raise RuntimeError("Official F007 execution requires committed source modules")
    import torch
    from experiments.F006b.benchmark_cpu import cpu_model_name
    gpu_model = (torch.cuda.get_device_name(execution_device)
                 if execution_device.startswith("cuda") and torch.cuda.is_available()
                 else None)
    return {
        **state, "official": official, "source_sha256": sources,
        "protocol_sha256": sha256_file(root / "configs/f007_protocol.json"),
        "frozen_checkpoints": protocol["checkpoints"],
        "frozen_inputs": frozen_inputs,
        "python": {"version": platform.python_version(), "implementation":
                   platform.python_implementation(), "executable": sys.executable},
        "packages": _package_versions(),
        "system": {"os": platform.platform(), "node": platform.node(),
                   "machine": platform.machine(), "cpu_model": cpu_model_name(),
                   "gpu_model": gpu_model,
                   "logical_cpu_count": os.cpu_count()},
        "execution_device": execution_device,
        "threads": {"torch_intraop": torch.get_num_threads(),
                    "torch_interop": torch.get_num_interop_threads(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": {"cwd": str(Path.cwd()), "argv": [sys.executable, *sys.argv],
                    "original_argv": list(getattr(sys, "orig_argv", sys.argv))},
        "parameters": parameters,
    }


def assert_unchanged(start: dict, *, root: Path = ROOT) -> None:
    """Fail if source/protocol or Git state changed during evidence generation."""
    now = git_state(root)
    if now != {field: start[field] for field in ("commit_sha", "working_tree_dirty")} or \
            source_hashes(root) != start["source_sha256"] or \
            sha256_file(root / "configs/f007_protocol.json") != start["protocol_sha256"]:
        raise RuntimeError("F007 source or protocol changed during execution")
