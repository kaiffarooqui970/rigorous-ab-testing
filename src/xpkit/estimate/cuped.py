"""CUPED: Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).

Replace the outcome Y with

    Y_cuped = Y - theta * (X - mean(X)),   theta = Cov(Y, X) / Var(X)

where X is measured *before* assignment, so it can't be affected by treatment. The
treatment-control difference in Y_cuped stays unbiased, because E[X] is the same in
both arms under randomization. Its variance falls by a factor of (1 - rho^2), where
rho = Corr(Y, X).

Here theta is estimated on the pooled sample, ignoring treatment. With equal
allocation this matches the original paper. The SE treats theta as fixed. That error
is O(1/n) and asymptotically negligible; the simulation study in
``validation/stage2_experiments.py`` checks coverage empirically.

:func:`lin_regression_adjustment` is the multi-covariate generalisation (Lin 2013):
OLS of Y on T, centred covariates and their interactions with T, with HC2 robust SEs.
It is never asymptotically less efficient than the unadjusted estimator, even if the
regression model is wrong.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from xpkit._validation import as_1d_float, as_treatment, check_same_length
from xpkit.estimate.effects import diff_in_means
from xpkit.types import Estimate


def cuped_theta(y: ArrayLike, x: ArrayLike) -> float:
    """OLS slope of y on x: Cov(y, x) / Var(x)."""
    yy, xx = as_1d_float("y", y), as_1d_float("x", x)
    check_same_length(y=yy, x=xx)
    vx = xx.var(ddof=1)
    if vx == 0:
        raise ValueError("pre-experiment covariate has zero variance; CUPED is undefined")
    return float(np.cov(yy, xx, ddof=1)[0, 1] / vx)


def cuped(y: ArrayLike, treatment: ArrayLike, x_pre: ArrayLike, alpha: float = 0.05) -> Estimate:
    """CUPED-adjusted difference in means.

    Args:
        y: outcome.
        treatment: 0/1 assignment.
        x_pre: pre-experiment covariate (e.g. the same metric in the weeks before launch).
            It **must not** be affected by treatment.
        alpha: CI level.

    Returns:
        :class:`Estimate`. ``extra`` holds ``theta``, ``rho`` and
        ``variance_reduction`` = 1 - Var(Y_cuped) / Var(Y).
    """
    yy, xx = as_1d_float("y", y), as_1d_float("x_pre", x_pre)
    t = as_treatment("treatment", treatment)
    check_same_length(y=yy, x_pre=xx, treatment=t)
    theta = cuped_theta(yy, xx)
    y_adj = yy - theta * (xx - xx.mean())
    est = diff_in_means(y_adj, t, alpha=alpha)
    rho = float(np.corrcoef(yy, xx)[0, 1])
    return Estimate(
        estimate=est.estimate,
        se=est.se,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        p_value=est.p_value,
        method="CUPED",
        alpha=alpha,
        n=est.n,
        extra={
            "theta": theta,
            "rho": rho,
            "variance_reduction": float(1 - y_adj.var(ddof=1) / yy.var(ddof=1)),
        },
    )


def lin_regression_adjustment(
    y: ArrayLike, treatment: ArrayLike, X: ArrayLike, alpha: float = 0.05
) -> Estimate:
    """Lin (2013) interacted regression adjustment with HC2 standard errors.

    Fits y ~ 1 + T + Xc + T:Xc, where Xc = X - mean(X). The coefficient on T is the ATE.
    """
    yy = as_1d_float("y", y)
    t = as_treatment("treatment", treatment).astype(float)
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm[:, None]
    if Xm.shape[0] != len(yy):
        raise ValueError(f"X has {Xm.shape[0]} rows but y has {len(yy)}")
    Xc = Xm - Xm.mean(axis=0)
    D = np.column_stack([np.ones_like(yy), t, Xc, Xc * t[:, None]])
    n, k = D.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    xtx_inv = np.linalg.pinv(D.T @ D)
    beta = xtx_inv @ D.T @ yy
    resid = yy - D @ beta
    h = np.einsum("ij,jk,ik->i", D, xtx_inv, D)
    w = resid**2 / np.clip(1 - h, 1e-12, None)
    vcov = xtx_inv @ (D.T * w) @ D @ xtx_inv
    se = math.sqrt(vcov[1, 1])
    est = float(beta[1])
    df = n - k
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    return Estimate(
        estimate=est,
        se=se,
        ci_low=est - tcrit * se,
        ci_high=est + tcrit * se,
        p_value=float(2 * stats.t.sf(abs(est) / se, df)),
        method="Lin regression adjustment (HC2)",
        alpha=alpha,
        n=n,
        extra={"n_covariates": Xm.shape[1]},
    )
