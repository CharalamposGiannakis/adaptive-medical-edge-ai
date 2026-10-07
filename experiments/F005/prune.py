"""F005b: nested structured pruning of TinyViT hidden channels.

Twelve hidden-channel groups are prunable without touching the embedding
widths: the expansion channels of the two stage-0 MBConv blocks (conv1 out,
depthwise conv2, conv3 in; 18% of a frame's FLOPs) and the MLP hidden units of
the ten transformer blocks (fc1 out, fc2 in; 41.5%). Channels are reordered by
importance, which leaves the network's function unchanged, and width w keeps
the first round(w × H) channels of every group. Smaller widths therefore use a
strict subset of the larger widths' weights: one weight file serves every
width.

Batch norm. conv3's BN normalizes a sum over the hidden channels, so its input
statistics change with width. Each width keeps its own conv3 BN statistics,
chosen so that on calibration data the BN output has the mean and standard
deviation it has at full width (width 1.0 keeps D026's statistics exactly). The
other BNs act per hidden channel and are sliced with their channels.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.tiny_vit import MBConv, NormMlp

WIDTHS = (1.0, 0.75, 0.5, 0.25)


def groups(encoder: nn.Module) -> list[tuple[str, nn.Module]]:
    """(name, module) of the twelve prunable groups, in network order."""
    found = [(n, m) for n, m in encoder.named_modules() if isinstance(m, (MBConv, NormMlp))]
    if len(found) != 12:
        raise ValueError(f"Expected 12 prunable TinyViT-5M groups, found {len(found)}")
    return found


def hidden_size(module: nn.Module) -> int:
    return module.fc1.out_features if isinstance(module, NormMlp) else module.conv1.conv.out_channels


def channels(module: nn.Module, width: float) -> int:
    return max(1, round(width * hidden_size(module)))


def _bn_scale(bn: nn.BatchNorm2d) -> torch.Tensor:
    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


# ---------------------------------------------------------------- importance

def l1_scores(module: nn.Module) -> torch.Tensor:
    """Data-free score per hidden channel: L1 norm of its in- and out-going weights.

    MBConv weights are folded with their BN scales first, so the score reflects
    what the channel actually passes on.
    """
    with torch.no_grad():
        if isinstance(module, NormMlp):
            return module.fc1.weight.abs().sum(1) + module.fc2.weight.abs().sum(0)
        w1 = module.conv1.conv.weight.flatten(1) * _bn_scale(module.conv1.bn)[:, None]
        w2 = module.conv2.conv.weight.flatten(1) * _bn_scale(module.conv2.bn)[:, None]
        return w1.abs().sum(1) + w2.abs().sum(1) + module.conv3.conv.weight.flatten(1).abs().sum(0)


class TaylorRecorder:
    """First-order Taylor score per hidden channel, |Σ a · ∂L/∂a| per sample, summed.

    Hooks the hidden activation that the pruned channels produce: the input of
    fc2 (MLP) or of conv3 (MBConv).
    """

    def __init__(self, encoder: nn.Module):
        self.scores, self.handles = {}, []
        for name, module in groups(encoder):
            target = module.fc2 if isinstance(module, NormMlp) else module.conv3.conv
            self.scores[name] = torch.zeros(hidden_size(module))
            self.handles.append(target.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, output):
            activation = inputs[0]
            if not activation.requires_grad:
                return
            saved = activation.detach()  # not the hooked tensor itself: no reference cycle

            def grad_hook(grad):
                product = saved * grad
                dims = tuple(range(1, product.ndim - 1)) if product.ndim == 3 else (2, 3)
                per_sample = product.sum(dim=dims) if product.ndim == 4 else product.sum(dim=1)
                self.scores[name] += per_sample.abs().sum(0).cpu()
            activation.register_hook(grad_hook)
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


# ---------------------------------------------------------------- reorder

def _take(tensor: torch.Tensor, order: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor.index_select(dim, order.to(tensor.device))


def reorder(module: nn.Module, order: torch.Tensor) -> None:
    """Permute a group's hidden channels in place (most important first)."""
    with torch.no_grad():
        if isinstance(module, NormMlp):
            module.fc1.weight.copy_(_take(module.fc1.weight, order, 0))
            module.fc1.bias.copy_(_take(module.fc1.bias, order, 0))
            module.fc2.weight.copy_(_take(module.fc2.weight, order, 1))
            return
        module.conv1.conv.weight.copy_(_take(module.conv1.conv.weight, order, 0))
        module.conv2.conv.weight.copy_(_take(module.conv2.conv.weight, order, 0))
        module.conv3.conv.weight.copy_(_take(module.conv3.conv.weight, order, 1))
        for bn in (module.conv1.bn, module.conv2.bn):
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(bn, name).copy_(_take(getattr(bn, name), order, 0))


# ---------------------------------------------------------------- width-aware modules

def _key(width: float) -> str:
    return f"{int(round(width * 100)):03d}"


class WidthMlp(NormMlp):
    """NormMlp that runs on its first ``active`` hidden units."""

    def forward(self, x):
        k = self.active
        x = self.norm(x)
        x = self.drop1(self.act(F.linear(x, self.fc1.weight[:k], self.fc1.bias[:k])))
        return self.drop2(F.linear(x, self.fc2.weight[:, :k], self.fc2.bias))


class WidthMBConv(MBConv):
    """MBConv on its first ``active`` expansion channels, with conv3 BN statistics per width."""

    def forward(self, x):
        k, shortcut = self.active, x
        bn1, bn2, bn3 = self.conv1.bn, self.conv2.bn, self.conv3.bn
        x = F.conv2d(x, self.conv1.conv.weight[:k])
        x = self.act1(F.batch_norm(x, bn1.running_mean[:k], bn1.running_var[:k], bn1.weight[:k],
                                   bn1.bias[:k], False, 0.0, bn1.eps))
        conv2 = self.conv2.conv
        x = F.conv2d(x, conv2.weight[:k], None, conv2.stride, conv2.padding, conv2.dilation, k)
        x = self.act2(F.batch_norm(x, bn2.running_mean[:k], bn2.running_var[:k], bn2.weight[:k],
                                   bn2.bias[:k], False, 0.0, bn2.eps))
        x = F.conv2d(x, self.conv3.conv.weight[:, :k])
        if getattr(self, "moments", None) is not None:  # recalibration: per-channel sums
            flat = x.detach().double().transpose(0, 1).flatten(1)
            self.moments[0] += flat.sum(1); self.moments[1] += (flat ** 2).sum(1); self.moments[2] += flat.shape[1]
        mean, var = getattr(self, f"bn3_mean_{self.key}"), getattr(self, f"bn3_var_{self.key}")
        x = F.batch_norm(x, mean, var, bn3.weight, bn3.bias, False, 0.0, bn3.eps)
        x = self.drop_path(x) + shortcut
        return self.act3(x)


def make_nested(encoder: nn.Module, orders: dict[str, torch.Tensor] | None = None) -> None:
    """Reorder (if given) and convert the twelve groups to width-aware modules, in place.

    Every width starts with conv3 BN statistics equal to the full-width ones;
    ``recalibrate`` then fits them. Use ``set_width`` before a forward.
    """
    for name, module in groups(encoder):
        if orders is not None:
            reorder(module, orders[name])
        if isinstance(module, NormMlp):
            module.__class__ = WidthMlp
        else:
            module.__class__ = WidthMBConv
            for width in WIDTHS:
                module.register_buffer(f"bn3_mean_{_key(width)}", module.conv3.bn.running_mean.clone())
                module.register_buffer(f"bn3_var_{_key(width)}", module.conv3.bn.running_var.clone())
    set_width(encoder, 1.0)


def set_width(encoder: nn.Module, width: float) -> None:
    if width not in WIDTHS:
        raise ValueError(f"Width must be one of {WIDTHS}")
    for _, module in groups(encoder):
        module.active, module.key = channels(module, width), _key(width)
    for module in getattr(encoder, "_width_bns", ()):
        module.key = _key(width)


@torch.no_grad()
def recalibrate(encoder: nn.Module, frames: list[torch.Tensor]) -> dict:
    """Fit each width's conv3 BN statistics so its output matches full width on ``frames``.

    For every MBConv, with pre-BN conv3 output mean/std m₁, s₁ at width 1 and
    m_w, s_w at width w (per channel, over ``frames``): σ_w = σ₁ s_w / s₁ and
    μ_w = m_w − σ_w (m₁ − μ₁) / σ₁, so (x_w − μ_w) / σ_w has the same mean and
    std as at width 1. Width 1.0 keeps its statistics. The two MBConv blocks
    are calibrated in order, the second with the first already calibrated.
    """
    stem = lambda x: encoder.stages[0](encoder.patch_embed(x))  # noqa: E731 (MBConvs live here)
    report = {}
    for name, module in [(n, m) for n, m in groups(encoder) if isinstance(m, WidthMBConv)]:
        moments = {}
        for width in WIDTHS:
            set_width(encoder, width)
            channels_out = module.conv3.bn.num_features
            module.moments = [torch.zeros(channels_out, dtype=torch.float64, device=frames[0].device)
                              for _ in range(2)] + [0]
            for batch in frames:
                stem(batch)
            total, squares, count = module.moments
            module.moments = None
            mean = total / count
            moments[width] = (mean.float(), torch.sqrt(squares / count - mean ** 2).float())
        bn = module.conv3.bn
        mu1, sigma1 = bn.running_mean, torch.sqrt(bn.running_var + bn.eps)
        m1, s1 = moments[1.0]
        for width in WIDTHS[1:]:
            mw, sw = moments[width]
            sigma_w = sigma1 * sw / s1
            getattr(module, f"bn3_mean_{_key(width)}").copy_(mw - sigma_w * (m1 - mu1) / sigma1)
            getattr(module, f"bn3_var_{_key(width)}").copy_(sigma_w ** 2 - bn.eps)
        report[name] = {str(w): {"mean_shift": float((moments[w][0] - m1).abs().mean()),
                                 "std_ratio": float((moments[w][1] / s1).mean())} for w in WIDTHS}
    set_width(encoder, 1.0)
    return report


# ---------------------------------------------------------------- every BN per width

class WidthBN(nn.BatchNorm2d):
    """Eval-mode BatchNorm2d with running statistics per width (``key``)."""

    def forward(self, x):
        if getattr(self, "moments", None) is not None:  # recalibration: per-channel sums
            flat = x.detach().double().transpose(0, 1).flatten(1)
            self.moments[0] += flat.sum(1); self.moments[1] += (flat ** 2).sum(1); self.moments[2] += flat.shape[1]
        mean, var = getattr(self, f"wmean_{self.key}"), getattr(self, f"wvar_{self.key}")
        return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0.0, self.eps)


def downstream_bns(encoder: nn.Module) -> list[tuple[str, nn.Module]]:
    """BatchNorm2d layers whose input changes with width, other than the MBConv internals.

    That is every BN after the first pruned group: the PatchMerging BNs and the
    local-convolution BNs of the transformer blocks. MBConv conv1/conv2 BNs act
    per hidden channel and are sliced; conv3 has its own per-width statistics.
    """
    inside = {id(m.conv1.bn) for _, m in groups(encoder) if isinstance(m, MBConv)} | \
             {id(m.conv2.bn) for _, m in groups(encoder) if isinstance(m, MBConv)} | \
             {id(m.conv3.bn) for _, m in groups(encoder) if isinstance(m, MBConv)}
    return [(n, m) for n, m in encoder.named_modules()
            if isinstance(m, nn.BatchNorm2d) and not n.startswith("patch_embed") and id(m) not in inside]


def make_all_bn_width_aware(encoder: nn.Module) -> None:
    """Give every downstream BN its own statistics per width (all start equal to D026's)."""
    bns = downstream_bns(encoder)
    for _, module in bns:
        module.__class__ = WidthBN
        for width in WIDTHS:
            module.register_buffer(f"wmean_{_key(width)}", module.running_mean.clone())
            module.register_buffer(f"wvar_{_key(width)}", module.running_var.clone())
    encoder._width_bns = [m for _, m in bns]
    set_width(encoder, 1.0)


@torch.no_grad()
def recalibrate_all(encoder: nn.Module, frames: list[torch.Tensor]) -> dict:
    """Moment-match every downstream BN per width, in forward order (as ``recalibrate``).

    Each BN is fitted with all earlier BNs already fitted; width 1.0 keeps D026's
    statistics, so the full-width model is unchanged.
    """
    bns = encoder._width_bns
    order = []
    hooks = [m.register_forward_pre_hook(lambda mod, inp: order.append(mod) if mod not in order else None)
             for m in bns]
    set_width(encoder, 1.0)
    encoder(frames[0][:1])
    for h in hooks:
        h.remove()
    report = {}
    names = {id(m): n for n, m in encoder.named_modules()}
    for module in order:
        moments = {}
        for width in WIDTHS:
            set_width(encoder, width)
            module.moments = [torch.zeros(module.num_features, dtype=torch.float64, device=frames[0].device)
                              for _ in range(2)] + [0]
            for batch in frames:
                encoder(batch)
            total, squares, count = module.moments
            module.moments = None
            mean = total / count
            moments[width] = (mean.float(), torch.sqrt(squares / count - mean ** 2).float())
        mu1, sigma1 = module.running_mean, torch.sqrt(module.running_var + module.eps)
        m1, s1 = moments[1.0]
        for width in WIDTHS[1:]:
            mw, sw = moments[width]
            sigma_w = sigma1 * sw / s1
            getattr(module, f"wmean_{_key(width)}").copy_(mw - sigma_w * (m1 - mu1) / sigma1)
            getattr(module, f"wvar_{_key(width)}").copy_(sigma_w ** 2 - module.eps)
        report[names[id(module)]] = {str(w): {"mean_shift": float((moments[w][0] - m1).abs().mean()),
                                              "std_ratio": float((moments[w][1] / s1).mean())} for w in WIDTHS}
    set_width(encoder, 1.0)
    return report


class NestedAnytimeClip:
    """Anytime serving of one clip by the nested model, cache keyed by (width, frame).

    A frame's features depend on the width that encoded it, so a request at
    width w reuses only features encoded at w: moving up in K at the same width
    encodes only the added frames, and a width change encodes the K frames anew
    at the new width (features at the old width stay cached for a move back).
    ``model`` has ``encoder`` (made nested) and ``head``; ``load(frame_numbers)``
    returns those frames as n×3×224×224.
    """

    def __init__(self, model: nn.Module, frame_count: int, load):
        self.model, self.frame_count, self.load = model, frame_count, load
        self.features: dict[tuple[str, int], torch.Tensor] = {}

    def predict(self, width: float, k: int) -> tuple[torch.Tensor, int]:
        from experiments.F003.adaptive import frame_indices
        chosen = frame_indices(self.frame_count, k)
        key = _key(width)
        missing = [i for i in chosen if (key, i) not in self.features]
        if missing:
            set_width(self.model.encoder, width)
            self.features.update(zip(((key, i) for i in missing), self.model.encoder(self.load(missing))))
        stacked = torch.stack([self.features[(key, i)] for i in chosen]).mean(0, keepdim=True)
        return self.model.head(stacked).squeeze(-1)[0], len(missing)


# ---------------------------------------------------------------- physical copy

def physical(encoder: nn.Module, width: float) -> nn.Module:
    """A standard timm encoder whose hidden layers are physically ``width`` wide.

    The copy uses plain NormMlp/MBConv modules with smaller tensors, so fvcore,
    ONNX export and any runtime see the reduced computation directly.
    """
    set_width(encoder, width)
    out = copy.deepcopy(encoder)
    set_width(encoder, 1.0)
    for (_, source), (_, target) in zip(groups(encoder), groups(out)):
        k = channels(source, width)
        with torch.no_grad():
            if isinstance(source, WidthMlp):
                fc1, fc2 = source.fc1, source.fc2
                target.fc1 = nn.Linear(fc1.in_features, k).to(fc1.weight)
                target.fc1.weight.copy_(fc1.weight[:k]); target.fc1.bias.copy_(fc1.bias[:k])
                target.fc2 = nn.Linear(k, fc2.out_features).to(fc2.weight)
                target.fc2.weight.copy_(fc2.weight[:, :k]); target.fc2.bias.copy_(fc2.bias)
                target.__class__ = NormMlp
                continue
            c1, c2, c3 = source.conv1, source.conv2, source.conv3
            target.conv1.conv = nn.Conv2d(c1.conv.in_channels, k, 1, bias=False).to(c1.conv.weight)
            target.conv1.conv.weight.copy_(c1.conv.weight[:k])
            target.conv2.conv = nn.Conv2d(k, k, c2.conv.kernel_size, c2.conv.stride, c2.conv.padding,
                                          groups=k, bias=False).to(c2.conv.weight)
            target.conv2.conv.weight.copy_(c2.conv.weight[:k])
            target.conv3.conv = nn.Conv2d(k, c3.conv.out_channels, 1, bias=False).to(c3.conv.weight)
            target.conv3.conv.weight.copy_(c3.conv.weight[:, :k])
            for name, bn in (("conv1", c1.bn), ("conv2", c2.bn)):
                small = nn.BatchNorm2d(k, eps=bn.eps).to(bn.weight)
                for attr in ("weight", "bias", "running_mean", "running_var"):
                    getattr(small, attr).copy_(getattr(bn, attr)[:k])
                getattr(target, name).bn = small
            target.conv3.bn.running_mean.copy_(getattr(source, f"bn3_mean_{_key(width)}"))
            target.conv3.bn.running_var.copy_(getattr(source, f"bn3_var_{_key(width)}"))
            for w in WIDTHS:
                for stat in ("mean", "var"):
                    delattr(target, f"bn3_{stat}_{_key(w)}")
            target.__class__ = MBConv
        for attr in ("active", "key"):
            target.__dict__.pop(attr, None)
    for module in list(out.modules()):
        if isinstance(module, WidthBN):
            with torch.no_grad():
                module.running_mean.copy_(getattr(module, f"wmean_{_key(width)}"))
                module.running_var.copy_(getattr(module, f"wvar_{_key(width)}"))
            for w in WIDTHS:
                delattr(module, f"wmean_{_key(w)}")
                delattr(module, f"wvar_{_key(w)}")
            module.__dict__.pop("key", None)
            module.__class__ = nn.BatchNorm2d
    out.__dict__.pop("_width_bns", None)
    return out.eval()
