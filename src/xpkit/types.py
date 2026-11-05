"""Shared result types.

Every estimator in xpkit returns an :class:`Estimate`, so the simulation harness,
report generator and app can treat them uniformly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from scipy import stats


@dataclass(frozen=True)
class Estimate:
    """A point estimate with uncertainty.

    Attributes:
        estimate: point estimate of the target quantity (e.g. ATE, ATT, lift).
        se: standard error of ``estimate``.
        ci_low, ci_high: two-sided ``1 - alpha`` confidence interval.
        p_value: two-sided p-value for H0: target == 0.
        method: short human-readable name of the estimator.
        alpha: significance level the CI was built for.
        n: number of units used, if meaningful.
        extra: estimator-specific diagnostics (always JSON-serialisable).
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    alpha: float = 0.05
    n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def covers(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normal_estimate(
    estimate: float,
    se: float,
    method: str,
    alpha: float = 0.05,
    n: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Estimate:
    """Build an :class:`Estimate` from a point estimate and SE using a normal approximation."""
    if not math.isfinite(se) or se < 0:
        raise ValueError(f"standard error must be finite and non-negative, got {se!r}")
    z = stats.norm.ppf(1 - alpha / 2)
    p = float(2 * stats.norm.sf(abs(estimate) / se)) if se > 0 else float(estimate == 0)
    return Estimate(
        estimate=float(estimate),
        se=float(se),
        ci_low=float(estimate - z * se),
        ci_high=float(estimate + z * se),
        p_value=p,
        method=method,
        alpha=alpha,
        n=n,
        extra=extra or {},
    )
