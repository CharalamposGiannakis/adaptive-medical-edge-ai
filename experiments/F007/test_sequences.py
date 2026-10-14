"""Authorized F007 test-row TinyViT loader using the frozen F003 image path."""
from __future__ import annotations

from pathlib import Path

from experiments.F003.adaptive import F003Sequences
from src import f002_pipeline as pipe
from src.f002_protocol import key, verify_t006
from src.f007_protocol import ROOT, require_test_authorization


class AuthorizedTestTinyViTSequences(F003Sequences):
    """F003 methods on verified test rows, without weakening F002c's guard."""

    def __init__(self, rows: list[dict], protocol: dict, *, raw: Path | None = None):
        require_test_authorization(protocol)
        if not rows or any(row.get("partition") != "test" for row in rows):
            raise ValueError("F007 TinyViT loader requires test rows")
        allowed = {key(row): row for row in verify_t006(
            ROOT / "configs/f002_protocol.json", ROOT / "data/openpocus/t006")["test"]}
        if len({key(row) for row in rows}) != len(rows) or any(
                allowed.get(key(row)) != row for row in rows):
            raise ValueError("F007 TinyViT rows must exactly match verified T006 test rows")
        # Reuse the canonical K16 retained-frame inventory. Inherited F003
        # positions, frames and frames_at retain bicubic RGB224/ImageNet behavior.
        pipe.Sequences.__init__(self, rows, 16, pipe.RAW if raw is None else raw)
        self.resolution = 224
