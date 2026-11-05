"""Two-arm effect estimates with confidence intervals.

* :func:`diff_in_proportions`: absolute difference in conversion rates. The CI is
  Newcombe's hybrid score interval, which keeps good coverage at small counts where
  Wald undercovers. The p-value comes from the pooled two-proportion z-test.
* :func:`diff_in_means`: Welch difference in means (unequal variances). Pass
  ``cluster=`` when units are not independent (e.g. sessions nested in users), to get
  a cluster-robust (CR1) standard error.
* :func:`relative_lift`: (mean_t / mean_c - 1) with a delta-method CI.

All return :class:`xpkit.types.Estimate`. The effect is always *treatment minus control*.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from xpkit._validation import as_1d_float, as_treatment, check_probability, check_same_length
from xpkit.types import Estimate, normal_estimate


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def diff_in_proportions(
    successes_t: int,
    n_t: int,
    successes_c: int,
    n_c: int,
    alpha: float = 0.05,
) -> Estimate:
    """Difference in conversion rates (treatment - control).

    The CI is Newcombe (1998) method 10: the hybrid of the two Wilson score intervals.
    The p-value comes from the pooled-variance z-test. Both are score-based, so they
    rarely disagree. ``se`` is the unpooled standard error.

    Example:
        >>> e = diff_in_proportions(550, 10_000, 500, 10_000)
        >>> round(e.estimate, 4)
        0.005
    """
    check_probability("alpha", alpha)
    for name, s, n in (("treatment", successes_t, n_t), ("control", successes_c, n_c)):
        if n <= 0:
            raise ValueError(f"{name} sample size must be positive, got {n}")
        if not 0 <= s <= n:
            raise ValueError(f"{name} successes must be in [0, n], got {s} of {n}")
    p_t, p_c = successes_t / n_t, successes_c / n_c
    diff = p_t - p_c
    z = float(stats.norm.ppf(1 - alpha / 2))

    l_t, u_t = _wilson(successes_t, n_t, z)
    l_c, u_c = _wilson(successes_c, n_c, z)
    ci_low = diff - math.sqrt((p_t - l_t) ** 2 + (u_c - p_c) ** 2)
    ci_high = diff + math.sqrt((u_t - p_t) ** 2 + (p_c - l_c) ** 2)

    p_pool = (successes_t + successes_c) / (n_t + n_c)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    p_value = 1.0 if se_pool == 0 else float(2 * stats.norm.sf(abs(diff) / se_pool))
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)

    return Estimate(
        estimate=diff,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        method="diff_in_proportions (Newcombe CI, pooled z p-value)",
        alpha=alpha,
        n=n_t + n_c,
        extra={
            "rate_treatment": p_t,
            "rate_control": p_c,
            "relative_lift": diff / p_c if p_c else None,
        },
    )


def _cluster_se_diff(y: np.ndarray, t: np.ndarray, cluster: np.ndarray) -> float:
    """CR1 cluster-robust SE of the OLS slope in y ~ 1 + t (equals the difference in means)."""
    X = np.column_stack([np.ones_like(y), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    _, inv = np.unique(cluster, return_inverse=True)
    g = int(np.max(inv)) + 1
    scores = np.zeros((g, 2))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores
    n, k = X.shape
    if g < 2:
        raise ValueError("need at least two clusters for a cluster-robust SE")
    adj = (g / (g - 1)) * ((n - 1) / (n - k))
    vcov = adj * xtx_inv @ meat @ xtx_inv
    return float(math.sqrt(vcov[1, 1]))


def diff_in_means(
    y: ArrayLike,
    treatment: ArrayLike,
    alpha: float = 0.05,
    cluster: ArrayLike | None = None,
) -> Estimate:
    """Welch difference in means (treatment - control).

    The CI uses the Welch-Satterthwaite t distribution. With ``cluster``, the SE is
    cluster-robust (CR1) and the reference distribution is t with G-1 degrees of
    freedom, which matters when there are few clusters.
    """
    yy = as_1d_float("y", y)
    t = as_treatment("treatment", treatment)
    check_same_length(y=yy, treatment=t)
    y1, y0 = yy[t == 1], yy[t == 0]
    n1, n0 = len(y1), len(y0)
    if min(n1, n0) < 2:
        raise ValueError(f"need at least 2 units per arm, got treatment={n1}, control={n0}")
    diff = float(y1.mean() - y0.mean())
    v1, v0 = y1.var(ddof=1) / n1, y0.var(ddof=1) / n0

    if cluster is None:
        se = math.sqrt(v1 + v0)
        df = (v1 + v0) ** 2 / (v1**2 / (n1 - 1) + v0**2 / (n0 - 1)) if se > 0 else n1 + n0 - 2
        method = "diff_in_means (Welch)"
    else:
        cl = np.asarray(cluster)
        check_same_length(y=yy, cluster=cl)
        se = _cluster_se_diff(yy, t.astype(float), cl)
        df = len(np.unique(cl)) - 1
        method = "diff_in_means (cluster-robust CR1)"

    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    p = float(2 * stats.t.sf(abs(diff) / se, df)) if se > 0 else float(diff == 0)
    return Estimate(
        estimate=diff,
        se=se,
        ci_low=diff - tcrit * se,
        ci_high=diff + tcrit * se,
        p_value=p,
        method=method,
        alpha=alpha,
        n=n1 + n0,
        extra={
            "mean_treatment": float(y1.mean()),
            "mean_control": float(y0.mean()),
            "df": float(df),
        },
    )


def relative_lift(y: ArrayLike, treatment: ArrayLike, alpha: float = 0.05) -> Estimate:
    """Relative lift mean_t / mean_c - 1, with a delta-method standard error.

    Var(m1/m0) ≈ Var(m1)/m0^2 + m1^2 Var(m0)/m0^4 (the arms are independent).
    """
    yy = as_1d_float("y", y)
    t = as_treatment("treatment", treatment)
    check_same_length(y=yy, treatment=t)
    y1, y0 = yy[t == 1], yy[t == 0]
    m1, m0 = y1.mean(), y0.mean()
    if m0 == 0:
        raise ValueError("control mean is zero; relative lift is undefined")
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = math.sqrt(v1 / m0**2 + m1**2 * v0 / m0**4)
    return normal_estimate(
        float(m1 / m0 - 1), se, "relative_lift (delta method)", alpha=alpha, n=len(yy)
    )
