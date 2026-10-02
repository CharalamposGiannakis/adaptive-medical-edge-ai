"""F002 no-training sequence model and strict fvcore operation coverage.

One counted operation is one multiply-add for conv/linear/matmul, following
fvcore. Elementwise arithmetic is one operation per output element; softmax
uses five operations per output element; max pool counts comparisons.
"""
from __future__ import annotations

import math
from collections import Counter


MODEL_IDS = {
    "mobilenet_v2": "torchvision.models.mobilenet_v2",
    "resnet18": "torchvision.models.resnet18",
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
}
WEIGHT_IDS = {
    "mobilenet_v2": "MobileNet_V2_Weights.IMAGENET1K_V2",
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
}


def build_sequence_model(candidate: str, pretrained: bool = False,
                         fused_attn: bool = False):
    """Construct the common K-frame architecture; false never downloads weights.

    T008 may use ``pretrained=True`` for training initialization. FLOPs tests
    always pass ``False``. Canonical training/validation/FLOPs use unfused DeiT
    attention. Only the equivalent optimized runtime probe enables fused
    attention on the identical checkpoint weights.
    """
    import torch
    import timm
    from torchvision import models

    if candidate == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        encoder = models.mobilenet_v2(weights=weights)
        width = encoder.classifier[1].in_features
        encoder.classifier = torch.nn.Identity()
    elif candidate == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        width = encoder.fc.in_features
        encoder.fc = torch.nn.Identity()
    elif candidate == "deit_tiny":
        encoder = timm.create_model(MODEL_IDS[candidate], pretrained=pretrained,
                                    num_classes=0)
        width = encoder.num_features
        for module in encoder.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = fused_attn
    else:
        raise ValueError(f"Unknown F002 candidate: {candidate}")
    if candidate != "deit_tiny" and fused_attn:
        raise ValueError("Fused attention is only defined for DeiT-Tiny")

    class SequenceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(width, 1)

        def forward(self, frames):
            batch, k, channels, height, width = frames.shape
            features = self.encoder(frames.reshape(batch * k, channels, height, width))
            pooled = features.reshape(batch, k, -1).mean(dim=1)
            return self.head(pooled).squeeze(-1)

    return SequenceModel()


def set_deit_fused_attention(model, enabled: bool) -> None:
    """Toggle only timm's supported attention kernel path, never weights."""
    modules = [module for module in model.modules() if hasattr(module, "fused_attn")]
    if not modules:
        raise ValueError("Model has no DeiT fused-attention modules")
    for module in modules:
        module.fused_attn = enabled


def verify_deit_fused_equivalence(model, requests, tolerance: float = 1e-5,
                                  expected_requests: int = 200) -> dict:
    """Compare both execution paths on identical tensors, restoring unfused.

    T008 passes all 200 fixed benchmark requests for each K. This helper only
    checks numerical equivalence; it does not time, train or load test rows.
    """
    import torch

    if tolerance != 1e-5 or model.training or expected_requests <= 0:
        raise ValueError("F002 equivalence requires eval mode and tolerance 1e-5")
    modules = [module for module in model.modules() if hasattr(module, "fused_attn")]
    if not modules or any(module.fused_attn for module in modules):
        raise ValueError("Expected canonical unfused DeiT checkpoint execution")
    maximum = 0.0
    count = 0
    k_seen = None
    try:
        with torch.inference_mode():
            for frames in requests:
                if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] not in (4, 8, 16):
                    raise ValueError("Expected one fixed-K sequence request per equivalence check")
                if k_seen is None:
                    k_seen = frames.shape[1]
                elif k_seen != frames.shape[1]:
                    raise ValueError("Check DeiT attention equivalence separately for each K")
                set_deit_fused_attention(model, False)
                reference = model(frames)
                set_deit_fused_attention(model, True)
                fused = model(frames)
                if not torch.isfinite(reference).all() or not torch.isfinite(fused).all():
                    raise RuntimeError("Nonfinite DeiT fused/unfused logits")
                maximum = max(maximum, (reference - fused).abs().max().item())
                count += 1
    finally:
        set_deit_fused_attention(model, False)
    if count != expected_requests:
        raise ValueError(f"Expected {expected_requests} DeiT equivalence requests, got {count}")
    if maximum > tolerance:
        raise RuntimeError(f"DeiT fused/unfused max logit difference {maximum} > {tolerance}")
    return {"requests": count, "k": k_seen, "max_abs_logit_difference": maximum,
            "tolerance": tolerance, "equivalent": True}


def _numel(value) -> int:
    shape = value.type().sizes()
    if shape is None or any(size is None for size in shape):
        raise ValueError("F002 FLOPs trace has unknown tensor shape")
    return math.prod(shape)


def _elementwise(inputs, outputs) -> Counter:
    return Counter({"elementwise": _numel(outputs[0])})


def _relu(inputs, outputs) -> Counter:
    return Counter({"relu": _numel(outputs[0])})


def _clamp(inputs, outputs) -> Counter:
    return Counter({"clamp": _numel(outputs[0])})


def _softmax(inputs, outputs) -> Counter:
    return Counter({"softmax": 5 * _numel(outputs[0])})


def _mean(inputs, outputs) -> Counter:
    # Each input element contributes once to the reduction/division budget.
    return Counter({"mean_pool": _numel(inputs[0])})


def _max_pool(inputs, outputs) -> Counter:
    kernel = inputs[1].toIValue()
    if kernel is None:
        kernel = tuple(value.toIValue() for value in inputs[1].node().inputs())
    if not kernel or len(kernel) != 2:
        raise ValueError("Unresolved max-pool kernel in FLOPs trace")
    return Counter({"max_pool": _numel(outputs[0]) * (math.prod(kernel) - 1)})


CUSTOM_HANDLERS = {
    "aten::mul": _elementwise,
    "aten::add": _elementwise,
    "aten::add_": _elementwise,
    "aten::hardtanh_": _elementwise,
    "aten::relu": _relu,
    "aten::relu_": _relu,
    "aten::clamp": _clamp,
    "aten::clamp_": _clamp,
    "aten::gelu": _elementwise,
    "aten::softmax": _softmax,
    "aten::mean": _mean,
    "aten::max_pool2d": _max_pool,
}


# Exact names observed in all three pinned K4/K8/K16 traces. These operators
# carry tensor shape/index metadata or move data without arithmetic; dropout
# is identity in eval mode. Anything newly ignored by fvcore fails closed.
BOOKKEEPING_ALLOWLIST = frozenset({
    "aten::Int", "aten::cat", "aten::dropout", "aten::expand",
    "aten::flatten", "aten::permute", "aten::reshape", "aten::select",
    "aten::size", "aten::slice", "aten::squeeze", "aten::transpose",
    "aten::unbind", "prim::Constant", "prim::ListConstruct",
    "prim::ListUnpack", "prim::NumToTensor",
})


def count_sequence_flops(model, k: int) -> dict:
    """Count a complete batch-one K-frame forward; reject any omitted op."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    class StrictFlopCountAnalysis(FlopCountAnalysis):
        def __init__(self, model, inputs):
            super().__init__(model, inputs)
            self.bookkeeping_seen = Counter()

        def _should_ignore_node(self, node):
            kind = node.kind()
            if kind in BOOKKEEPING_ALLOWLIST:
                self.bookkeeping_seen[kind] += 1
                return True
            if super()._should_ignore_node(node):
                raise RuntimeError(f"Unapproved fvcore-ignored F002 operator: {kind}")
            return False

    if k not in (4, 8, 16):
        raise ValueError("Only preregistered K values are countable")
    model.eval()
    input_device = next(model.parameters()).device
    dummy = torch.zeros(1, k, 3, 224, 224, device=input_device)
    analysis = StrictFlopCountAnalysis(model, dummy).set_op_handle(**CUSTOM_HANDLERS)
    analysis.unsupported_ops_warnings(False)
    total = int(analysis.total())
    unsupported = dict(analysis.unsupported_ops())
    if unsupported:
        raise RuntimeError(f"Uncounted F002 operators: {unsupported}")
    by_operator = dict(analysis.by_operator())
    head_linear = int(analysis.by_module_and_operator().get("head", {}).get("linear", 0))
    if total <= 0 or head_linear != model.head.in_features * model.head.out_features or by_operator.get("mean_pool", 0) <= 0:
        raise RuntimeError("Incomplete sequence encoder/pool/head FLOPs trace")
    if any(hasattr(module, "fused_attn") for module in model.modules()):
        if by_operator.get("matmul", 0) <= 0 or by_operator.get("softmax", 0) <= 0:
            raise RuntimeError("Transformer attention FLOPs omitted")
    return {"flops": total, "by_operator": by_operator, "unsupported": unsupported,
            "unapproved_ignored": {}, "allowed_bookkeeping": dict(analysis.bookkeeping_seen),
            "binary_head_flops": head_linear,
            "k": k, "convention": "fvcore_one_fma_one_flop"}
