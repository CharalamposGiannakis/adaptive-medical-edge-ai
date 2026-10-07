"""F005 (T011): compression of the shared F004 family weights.

F005a is INT8 dynamic post-training quantization: every ``nn.Linear`` (61% of
the family's FLOPs) stores INT8 weights and quantizes its input activations per
call; convolutions, attention matmuls, norms and softmax stay FP32. No
calibration data and no training. One quantized weight set serves every K, so
the family stays one artifact.
"""
from __future__ import annotations

import io

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from experiments.F004.family import AdaptiveTinyViT, load_family

SCHEMES = ("fp32", "int8_dynamic")


def set_quantized_engine() -> str:
    """x86 (fbgemm successor) on the DAS-5 Xeons; qnnpack where it is the only engine."""
    supported = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No quantized engine available: {supported}")


def build(scheme: str) -> AdaptiveTinyViT:
    """The F004 family on CPU, FP32 or with INT8 dynamic Linear layers, eval mode."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme}")
    model = load_family(device="cpu")
    if scheme == "fp32":
        return model
    set_quantized_engine()
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval()


def serialized_bytes(model: nn.Module) -> int:
    """Size of the model's state dict as torch.save writes it."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def quantized_linear_count(model: nn.Module) -> tuple[int, int]:
    """(INT8 dynamic Linear modules, remaining FP32 nn.Linear modules)."""
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    quantized = sum(isinstance(m, DynamicLinear) for m in model.modules())
    plain = sum(type(m) is nn.Linear for m in model.modules())
    return quantized, plain
