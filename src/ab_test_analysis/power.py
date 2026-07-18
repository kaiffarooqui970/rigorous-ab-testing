"""A priori power analysis for a future, properly randomized redesign test.

This is deliberately forward-looking: it does not size the historical data
we already have (sizing a test after the sample is collected is circular).
Instead it takes a pilot baseline conversion rate -- observed from historical
data -- and a relative minimum detectable effect (MDE), and answers "how
many sessions per arm would a future randomized test need, at a given
alpha/power, to reliably detect a lift of that size?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class SampleSizeResult:
    baseline_rate: float
    target_rate: float
    relative_mde: float
    alpha: float
    power: float
    effect_size_h: float
    n_per_arm: int


def required_sample_size(
    baseline_rate: float,
    relative_mde: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> SampleSizeResult:
    """Sessions needed per arm to detect a relative_mde lift over baseline_rate.

    Uses Cohen's h (the standard effect size for two proportions) and a
    two-sided z-test power calculation via statsmodels' NormalIndPower --
    the correct family for a two-proportion test, since the underlying
    significance test itself will be a proportions z-test, not a t-test.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if relative_mde <= 0:
        raise ValueError("relative_mde must be positive")

    target_rate = baseline_rate * (1 + relative_mde)
    if not 0 < target_rate < 1:
        raise ValueError(
            f"target_rate={target_rate:.4f} implied by relative_mde is out of (0, 1)"
        )

    effect_size_h = proportion_effectsize(baseline_rate, target_rate)
    analysis = NormalIndPower()
    n_per_arm = analysis.solve_power(
        effect_size=abs(effect_size_h),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )

    return SampleSizeResult(
        baseline_rate=baseline_rate,
        target_rate=target_rate,
        relative_mde=relative_mde,
        alpha=alpha,
        power=power,
        effect_size_h=effect_size_h,
        n_per_arm=math.ceil(n_per_arm),
    )
