"""Sensitivity to unmeasured confounding: the E-value (VanderWeele & Ding, 2017).

The E-value is the minimum strength of association, on the risk-ratio scale, that an
unmeasured confounder would need with *both* the treatment and the outcome, beyond
the measured covariates, to fully explain away the observed association.

    E = RR + sqrt(RR * (RR - 1)),   with RR replaced by 1/RR if RR < 1.

The E-value for the CI uses the CI limit closest to the null. If the CI already
includes 1, that E-value is 1: no confounding is needed to make the result
compatible with no effect.

An E-value is a *communication* tool, not a test. A large E-value means the result is
robust only if no plausible confounder is that strong. Whether one is plausible is a
domain judgement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _e(rr: float) -> float:
    if rr <= 0 or not math.isfinite(rr):
        raise ValueError(f"risk ratio must be positive and finite, got {rr}")
    r = 1 / rr if rr < 1 else rr
    return r + math.sqrt(r * (r - 1))


@dataclass(frozen=True)
class EValue:
    rr: float
    rr_ci_low: float | None
    rr_ci_high: float | None
    e_value_point: float
    e_value_ci: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def e_value(rr: float, ci_low: float | None = None, ci_high: float | None = None) -> EValue:
    """E-value for a risk ratio and, optionally, its confidence interval.

    Example:
        >>> round(e_value(3.9, 1.8, 8.7).e_value_point, 2)
        7.26
    """
    point = _e(rr)
    ci_e: float | None = None
    if ci_low is not None and ci_high is not None:
        if ci_low > ci_high:
            raise ValueError("ci_low must be <= ci_high")
        ci_e = 1.0 if ci_low <= 1 <= ci_high else _e(ci_low if rr > 1 else ci_high)
    return EValue(rr, ci_low, ci_high, point, ci_e)


def rr_from_smd(d: float) -> float:
    """Approximate risk ratio for a standardized mean difference d: RR ≈ exp(0.91 d)."""
    return math.exp(0.91 * d)


def rr_from_risks(p_treated: float, p_control: float) -> float:
    if not (0 < p_treated < 1 and 0 < p_control < 1):
        raise ValueError("risks must lie strictly in (0, 1)")
    return p_treated / p_control
