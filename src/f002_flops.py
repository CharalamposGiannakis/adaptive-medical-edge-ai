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


def build_sequence_model(candidate: str, pretrained: bool = False):
    """Construct the common K-frame architecture; false never downloads weights.

    T008 may use ``pretrained=True`` for training initialization. FLOPs tests
    always pass ``False``. DeiT fused attention is disabled uniformly in every
    run so its QK and attention-value matrix products are visible to fvcore.
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
                module.fused_attn = False
    else:
        raise ValueError(f"Unknown F002 candidate: {candidate}")

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


def _numel(value) -> int:
    shape = value.type().sizes()
    if shape is None or any(size is None for size in shape):
        raise ValueError("F002 FLOPs trace has unknown tensor shape")
    return math.prod(shape)


def _elementwise(inputs, outputs) -> Counter:
    return Counter({"elementwise": _numel(outputs[0])})


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
    "aten::gelu": _elementwise,
    "aten::softmax": _softmax,
    "aten::mean": _mean,
    "aten::max_pool2d": _max_pool,
}


def count_sequence_flops(model, k: int) -> dict:
    """Count a complete batch-one K-frame forward; reject any omitted op."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    if k not in (4, 8, 16):
        raise ValueError("Only preregistered K values are countable")
    model.eval()
    input_device = next(model.parameters()).device
    dummy = torch.zeros(1, k, 3, 224, 224, device=input_device)
    analysis = FlopCountAnalysis(model, dummy).set_op_handle(**CUSTOM_HANDLERS)
    analysis.unsupported_ops_warnings(False)
    total = int(analysis.total())
    unsupported = dict(analysis.unsupported_ops())
    if unsupported:
        raise RuntimeError(f"Uncounted F002 operators: {unsupported}")
    by_operator = dict(analysis.by_operator())
    if total <= 0 or by_operator.get("linear", 0) <= 0 or by_operator.get("mean_pool", 0) <= 0:
        raise RuntimeError("Incomplete sequence encoder/pool/head FLOPs trace")
    if any(hasattr(module, "fused_attn") for module in model.modules()):
        if by_operator.get("matmul", 0) <= 0 or by_operator.get("softmax", 0) <= 0:
            raise RuntimeError("Transformer attention FLOPs omitted")
    return {"flops": total, "by_operator": by_operator, "unsupported": unsupported,
            "k": k, "convention": "fvcore_one_fma_one_flop"}
