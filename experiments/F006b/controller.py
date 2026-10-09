"""T012b provisional quality-first controller over the canonical F005b grid.

Selection is a deterministic lookup over all 24 validation-measured width × K
points. It never inspects an input image, trains a model, or changes the frozen
threshold 0.5. A budget is an exact rational fraction of Cmax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from experiments.F006a import run_analysis as f006a
from experiments.F006a_2 import run_analysis as f006a_2

C_MAX = f006a_2.EXPECTED_CMAX
POLICIES = ("recall_priority", "minimum_quality", "f006a_rule_a", "fixed_full_k16")


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    width: float
    k: int
    flops: int
    quality: f006a_2.Quality
    recorded_macro_f1: float


@dataclass(frozen=True)
class Selection:
    policy: str
    resource: Fraction
    permitted_flops: Fraction
    point: OperatingPoint | None
    reason: str

    @property
    def deferred(self) -> bool:
        return self.point is None


def resource_fraction(value: Fraction | Decimal | str | float | int) -> Fraction:
    """Interpret user-facing decimals as decimals; retain exact rational inputs."""
    if isinstance(value, bool):
        raise ValueError("Resource b must be numeric in [0, 1]")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Resource b must be finite")
        value = str(value)
    try:
        b = Fraction(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError("Resource b must be numeric in [0, 1]") from exc
    if not 0 <= b <= 1:
        raise ValueError("Resource b must lie in [0, 1]")
    return b


def load_policy_points() -> tuple[OperatingPoint, ...]:
    """Reconstruct all 24 cells through F006a_2's hash-checked evidence path."""
    f006a_2.read_baseline()  # verify the frozen F002b source and checkpoint
    profiles = f006a_2.reconstruct_profiles()
    if len(profiles) != 24:
        raise ValueError("The canonical F005b grid must contain 24 points")
    points = tuple(OperatingPoint(p.point.name, p.point.width, p.point.k,
                                  p.point.flops, p.quality, p.point.macro_f1)
                   for p in profiles)
    if len({p.name for p in points}) != 24 or full_width_k16(points).flops != C_MAX:
        raise ValueError("Canonical operating-point identities or Cmax changed")
    return points


def full_width_k16(points: tuple[OperatingPoint, ...]) -> OperatingPoint:
    matches = [p for p in points if p.width == 1.0 and p.k == 16]
    if len(matches) != 1:
        raise ValueError("Expected one full-width K16 point")
    return matches[0]


def quality_qualified(point: OperatingPoint, baseline: f006a_2.Quality) -> bool:
    q = point.quality
    return (q.accuracy >= baseline.accuracy and q.macro_f1 >= baseline.macro_f1
            and q.abnormal_recall >= baseline.abnormal_recall)


def _defer(policy: str, b: Fraction, reason: str) -> Selection:
    return Selection(policy, b, b * C_MAX, None, reason)


def select(points: tuple[OperatingPoint, ...], baseline: f006a_2.Quality,
           resource: Fraction | Decimal | str | float | int,
           policy: str = "recall_priority") -> Selection:
    """Choose one measured point or DEFER, with no quality or FLOP tolerance."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    b = resource_fraction(resource)
    if b == 0:
        return _defer(policy, b, "resource_unavailable")
    limit = b * C_MAX
    feasible = [p for p in points if p.flops <= limit]
    by_name = {p.name: p for p in points}
    if len(by_name) != len(points):
        raise ValueError("Duplicate operating-point name")

    if policy == "fixed_full_k16":
        fixed = full_width_k16(points)
        if fixed.flops > limit:
            return _defer(policy, b, "fixed_point_over_budget")
        chosen = fixed
    elif policy == "minimum_quality":
        qualified = [p for p in points if quality_qualified(p, baseline)]
        if not qualified:
            return _defer(policy, b, "no_quality_qualified_point")
        fixed = min(qualified, key=lambda p: (p.flops, p.name))
        if fixed.flops > limit:
            return _defer(policy, b, "minimum_quality_point_over_budget")
        chosen = fixed
    elif policy == "f006a_rule_a":
        original = [f006a.Point(p.name, p.width, p.k, p.flops, p.recorded_macro_f1)
                    for p in points]
        rule_a = f006a.select(original, limit, "max_f1")
        if rule_a is None:
            return _defer(policy, b, "no_budget_feasible_point")
        chosen = by_name[rule_a.name]
    else:
        qualified = [p for p in feasible if quality_qualified(p, baseline)]
        if not qualified:
            return _defer(policy, b, "no_quality_qualified_point_within_budget")
        chosen = min(qualified, key=lambda p: (-p.quality.abnormal_recall,
                                               p.flops, p.name))
    if chosen.flops > limit:
        raise AssertionError("Selected point exceeds the exact resource budget")
    return Selection(policy, b, limit, chosen, "selected")


def selected_configurations(points: tuple[OperatingPoint, ...],
                            baseline: f006a_2.Quality | None = None
                            ) -> tuple[OperatingPoint, ...]:
    """Derive every recall-priority point reached anywhere on continuous b."""
    baseline = baseline or f006a_2.read_baseline()
    qualified = sorted((p for p in points if quality_qualified(p, baseline)),
                       key=lambda p: (p.flops, p.name))
    reached = {}
    for point in qualified:
        chosen = select(points, baseline, Fraction(point.flops, C_MAX)).point
        if chosen is not None:
            reached[chosen.name] = chosen
    return tuple(sorted(reached.values(), key=lambda p: (p.flops, p.name)))


def three_dimensional_dominators(point: OperatingPoint,
                                 points: tuple[OperatingPoint, ...]
                                 ) -> tuple[str, ...]:
    """Check FLOPs/F1/abnormal recall, not F006a's two-dimensional front."""
    return tuple(sorted(other.name for other in points if other.name != point.name
                        and other.flops <= point.flops
                        and other.quality.macro_f1 >= point.quality.macro_f1
                        and other.quality.abnormal_recall >= point.quality.abnormal_recall
                        and (other.flops < point.flops
                             or other.quality.macro_f1 > point.quality.macro_f1
                             or other.quality.abnormal_recall > point.quality.abnormal_recall)))


def cost_components(points: tuple[OperatingPoint, ...], width: float) -> tuple[int, int]:
    """Exact encoder-per-frame and head FLOPs; verify all six measured K."""
    by_k = {p.k: p.flops for p in points if p.width == width}
    if set(by_k) != set(f006a.KS):
        raise ValueError(f"Incomplete K costs for width {width}")
    encoder = by_k[2] - by_k[1]
    head = by_k[1] - encoder
    if encoder <= 0 or head < 0 or any(by_k[k] != k * encoder + head for k in by_k):
        raise ValueError(f"FLOP counts are not linear in K for width {width}")
    return encoder, head


def incremental_flops(points: tuple[OperatingPoint, ...], width: float,
                      new_frames: int) -> int:
    if not isinstance(new_frames, int) or new_frames < 0:
        raise ValueError("new_frames must be a nonnegative integer")
    encoder, head = cost_components(points, width)
    return new_frames * encoder + head
