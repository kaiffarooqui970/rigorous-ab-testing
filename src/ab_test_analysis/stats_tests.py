"""Test selection for the two metrics used in this analysis.

Primary metric (conversion rate, binary): a two-proportion z-test is the
correct test by construction -- no distributional assumptions to check
beyond the normal approximation to the binomial, which holds comfortably at
these sample sizes (np and n(1-p) both far above the usual >=5 rule).

Secondary metric (revenue per session, continuous): assumptions are checked
explicitly before a test is picked. Revenue per session is zero-inflated
(most sessions don't convert) and right-skewed (a few large orders), so a
t-test's normality assumption is expected to fail -- this is verified with a
formal test rather than assumed, and Mann-Whitney U is used instead when it
does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep


@dataclass
class ProportionTestResult:
    n_a: int
    n_b: int
    successes_a: int
    successes_b: int
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    alpha: float


def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> ProportionTestResult:
    """Two-sided two-proportion z-test with a Newcombe/score CI on the difference.

    Labels: "a" is the baseline/reference group, "b" is the comparison group.
    The reported diff and CI are rate_b - rate_a.
    """
    rate_a = successes_a / n_a
    rate_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (rate_b - rate_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    try:
        ci_low, ci_high = confint_proportions_2indep(
            count1=successes_b,
            nobs1=n_b,
            count2=successes_a,
            nobs2=n_a,
            method="score",
            alpha=alpha,
        )
    except ValueError:
        # Score-CI inversion can fail to bracket a root on near-degenerate
        # cells (e.g. very few conversions in a low-volume segment). Falls
        # back to the simple Wald CI on the difference, which is always
        # defined -- less accurate at small counts, but only reached there.
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se_diff = np.sqrt(
            rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b
        )
        diff = rate_b - rate_a
        ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff

    return ProportionTestResult(
        n_a=n_a,
        n_b=n_b,
        successes_a=successes_a,
        successes_b=successes_b,
        rate_a=rate_a,
        rate_b=rate_b,
        diff=rate_b - rate_a,
        ci_low=ci_low,
        ci_high=ci_high,
        z_stat=z_stat,
        p_value=p_value,
        alpha=alpha,
    )


@dataclass
class AssumptionCheck:
    shapiro_p_a: float
    shapiro_p_b: float
    levene_p: float
    normal_alpha: float
    both_normal: bool
    equal_variance: bool


@dataclass
class ContinuousTestResult:
    test_name: str  # "welch_t" or "mann_whitney_u"
    statistic: float
    p_value: float
    median_a: float
    median_b: float
    mean_a: float
    mean_b: float
    assumptions: AssumptionCheck


def compare_continuous_metric(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    normal_alpha: float = 0.05,
    shapiro_subsample_n: int = 5000,
) -> ContinuousTestResult:
    """Checks normality/variance assumptions, then picks Welch's t-test or
    Mann-Whitney U accordingly, rather than assuming one upfront.

    Shapiro-Wilk is unreliable/overpowered above a few thousand points (it
    will reject near-perfect normality on trivial deviations at n in the
    hundreds of thousands), so it's run on a random subsample -- this is a
    diagnostic check, not the hypothesis test itself, so subsampling for it
    is appropriate.
    """
    rng = np.random.default_rng(42)

    def _subsample(x: np.ndarray) -> np.ndarray:
        if len(x) <= shapiro_subsample_n:
            return x
        idx = rng.choice(len(x), size=shapiro_subsample_n, replace=False)
        return x[idx]

    _, shapiro_p_a = stats.shapiro(_subsample(sample_a))
    _, shapiro_p_b = stats.shapiro(_subsample(sample_b))
    _, levene_p = stats.levene(sample_a, sample_b)

    both_normal = shapiro_p_a > normal_alpha and shapiro_p_b > normal_alpha
    equal_variance = levene_p > normal_alpha

    assumptions = AssumptionCheck(
        shapiro_p_a=shapiro_p_a,
        shapiro_p_b=shapiro_p_b,
        levene_p=levene_p,
        normal_alpha=normal_alpha,
        both_normal=both_normal,
        equal_variance=equal_variance,
    )

    if both_normal:
        stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_variance)
        test_name = "welch_t" if not equal_variance else "students_t"
    else:
        stat, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        test_name = "mann_whitney_u"

    return ContinuousTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        median_a=float(np.median(sample_a)),
        median_b=float(np.median(sample_b)),
        mean_a=float(np.mean(sample_a)),
        mean_b=float(np.mean(sample_b)),
        assumptions=assumptions,
    )
