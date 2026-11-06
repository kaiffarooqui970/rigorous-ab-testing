"""Sample size, power and minimum detectable effect (MDE) for two-arm tests.

All formulas use the normal approximation for a two-sided test at level ``alpha``,
with a treatment:control allocation ratio ``k = n_t / n_c``.

Proportions use the *unpooled* alternative variance and *pooled* null variance
(Fleiss, Levin & Paik 2003, eq. 4.14):

    n_c = [ z_{1-a/2} * sqrt(pbar*qbar*(1 + 1/k)) + z_{1-b} * sqrt(p0*q0 + p1*q1/k) ]^2 / delta^2

Means:

    n_c = (z_{1-a/2} + z_{1-b})^2 * sigma^2 * (1 - rho^2) * (1 + 1/k) / delta^2

where ``rho`` is the correlation between the outcome and a pre-experiment covariate
used for CUPED. CUPED cuts the outcome variance by a factor of (1 - rho^2), and the
required sample size falls by the same factor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from scipy import optimize, stats

from xpkit._validation import check_positive, check_probability


@dataclass(frozen=True)
class SampleSize:
    """Required units per arm for a two-arm test, plus the design inputs that produced them."""

    n_control: int
    n_treatment: int
    alpha: float
    power: float
    mde: float
    detail: dict[str, Any]

    @property
    def n_total(self) -> int:
        return self.n_control + self.n_treatment

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_total"] = self.n_total
        return d


def _z(alpha: float, power: float) -> tuple[float, float]:
    check_probability("alpha", alpha)
    check_probability("power", power)
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def _n_control_proportions(p0: float, p1: float, alpha: float, power: float, ratio: float) -> float:
    """Unrounded control-arm size from the Fleiss formula (continuous in the effect)."""
    za, zb = _z(alpha, power)
    k = ratio
    pbar = (p0 + k * p1) / (1 + k)
    num = za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k)) + zb * math.sqrt(
        p0 * (1 - p0) + p1 * (1 - p1) / k
    )
    return (num / (p1 - p0)) ** 2


def sample_size_proportions(
    baseline: float,
    mde: float,
    *,
    relative: bool = True,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> SampleSize:
    """Units per arm needed to detect a change in a conversion rate.

    Args:
        baseline: control conversion rate p0.
        mde: minimum detectable effect, relative (0.1 = +10%) or absolute (0.002 = +0.2pp).
        relative: interpret ``mde`` as relative (default) or absolute.
        alpha: two-sided significance level.
        power: probability of detecting an effect of exactly ``mde``.
        ratio: allocation ratio n_treatment / n_control.

    Example:
        >>> sample_size_proportions(0.05, 0.10).n_control
        31234
    """
    check_probability("baseline", baseline)
    check_positive("ratio", ratio)
    if mde == 0:
        raise ValueError("mde must be non-zero")
    p1 = baseline * (1 + mde) if relative else baseline + mde
    check_probability("baseline + mde (target rate)", p1)
    p0, delta = baseline, p1 - baseline
    n_c = math.ceil(_n_control_proportions(p0, p1, alpha, power, ratio))
    return SampleSize(
        n_control=n_c,
        n_treatment=math.ceil(n_c * ratio),
        alpha=alpha,
        power=power,
        mde=delta,
        detail={"baseline": p0, "target": p1, "relative_mde": delta / p0},
    )


def sample_size_means(
    sd: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    cuped_rho: float = 0.0,
) -> SampleSize:
    """Units per arm needed to detect an absolute difference ``mde`` in a mean.

    ``cuped_rho`` is the expected correlation between the outcome and the pre-period
    covariate. Pass 0 for an unadjusted analysis.
    """
    check_positive("sd", sd)
    check_positive("ratio", ratio)
    if mde == 0:
        raise ValueError("mde must be non-zero")
    if not 0 <= abs(cuped_rho) < 1:
        raise ValueError("cuped_rho must be in (-1, 1)")
    za, zb = _z(alpha, power)
    var = sd**2 * (1 - cuped_rho**2)
    n_c = math.ceil((za + zb) ** 2 * var * (1 + 1 / ratio) / mde**2)
    return SampleSize(
        n_control=n_c,
        n_treatment=math.ceil(n_c * ratio),
        alpha=alpha,
        power=power,
        mde=abs(mde),
        detail={"sd": sd, "cuped_rho": cuped_rho, "variance_factor": 1 - cuped_rho**2},
    )


def mde_means(
    sd: float,
    n_control: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    cuped_rho: float = 0.0,
) -> float:
    """Smallest absolute mean difference detectable with the given sample size."""
    check_positive("sd", sd)
    check_positive("n_control", n_control)
    za, zb = _z(alpha, power)
    return (za + zb) * sd * math.sqrt((1 - cuped_rho**2) * (1 + 1 / ratio) / n_control)


def mde_proportions(
    baseline: float,
    n_control: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    relative: bool = True,
) -> float:
    """Smallest detectable *increase* in a conversion rate with the given sample size.

    Solves :func:`sample_size_proportions` for the effect size numerically. Returns a
    relative lift by default.
    """
    check_probability("baseline", baseline)
    if n_control < 2:
        raise ValueError("n_control must be >= 2")

    def gap(d: float) -> float:
        return _n_control_proportions(baseline, baseline + d, alpha, power, ratio) - n_control

    hi = (1 - baseline) * 0.999
    if gap(hi) > 0:
        raise ValueError("sample too small to detect any effect below a rate of 1")
    abs_mde = optimize.brentq(gap, 1e-12, hi, xtol=1e-12)
    return abs_mde / baseline if relative else abs_mde


def power_means(
    sd: float,
    mde: float,
    n_control: int,
    *,
    alpha: float = 0.05,
    ratio: float = 1.0,
    cuped_rho: float = 0.0,
) -> float:
    """Power of a two-sided z-test to detect an absolute mean difference ``mde``."""
    check_positive("sd", sd)
    za = stats.norm.ppf(1 - alpha / 2)
    se = sd * math.sqrt((1 - cuped_rho**2) * (1 + 1 / ratio) / n_control)
    shift = abs(mde) / se
    return float(stats.norm.sf(za - shift) + stats.norm.cdf(-za - shift))
