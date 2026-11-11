"""Inverse probability weighting (IPW) and augmented IPW (AIPW, doubly robust).

Identification needs **no unmeasured confounding** (conditional exchangeability given
the covariates), **positivity** (0 < e(x) < 1) and **consistency/SUTVA**. None of
these can be tested from the data. The diagnostics here only check the
model-dependent parts: overlap and covariate balance.

* :func:`ipw`: normalised (Hajek) IPW for the ATE or ATT. Uncertainty comes from
  - ``se="sandwich"``: a stacked M-estimation sandwich that accounts for estimating the
    logistic propensity model (Lunceford & Davidian 2004). Logistic PS only.
  - ``se="bootstrap"``: nonparametric bootstrap that re-fits the propensity model in
    every replicate. Works with any PS model.
* :func:`aipw`: augmented IPW for the ATE. It is consistent if *either* the propensity
  model *or* the outcome model is correct. The SE is the empirical SD of the
  efficient influence function. Flexible (GBM) nuisance models are cross-fitted
  (double machine learning).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from xpkit._validation import as_1d_float, as_treatment, require_columns
from xpkit.causal.propensity import (
    Model,
    ate_weights,
    att_weights,
    effective_sample_size,
    fit_propensity,
    overlap_summary,
    unpenalized_logistic,
)
from xpkit.types import Estimate, normal_estimate

Estimand = Literal["ate", "att"]


def _hajek(
    y: NDArray[np.float64], t: NDArray[np.int_], ps: NDArray[np.float64], estimand: str
) -> float:
    w = ate_weights(t, ps) if estimand == "ate" else att_weights(t, ps)
    m1 = np.sum(w * t * y) / np.sum(w * t)
    m0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    return float(m1 - m0)


def _logistic_design(X: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return Z, np.column_stack([np.ones(len(Z)), Z])


def _cluster_sum(
    rows: NDArray[np.float64], cluster: NDArray[np.generic] | None
) -> tuple[NDArray[np.float64], float]:
    """Sum score rows within clusters. Returns (summed rows, small-sample factor G/(G-1))."""
    if cluster is None:
        return rows, 1.0
    _, inv = np.unique(cluster, return_inverse=True)
    g = int(np.max(inv)) + 1
    if g < 2:
        raise ValueError("need at least two clusters")
    out = np.zeros((g, rows.shape[1]))
    np.add.at(out, inv, rows)
    return out, g / (g - 1)


def _sandwich_se(
    y: NDArray[np.float64],
    t: NDArray[np.int_],
    X: NDArray[np.float64],
    estimand: str,
    cluster: NDArray[np.generic] | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Hajek IPW estimate and its stacked-sandwich SE, with a logistic PS fitted on X."""
    Z, Xt = _logistic_design(X)
    clf = unpenalized_logistic().fit(Z, t)
    e = expit(Xt @ np.r_[clf.intercept_, clf.coef_.ravel()])
    n, p = Xt.shape
    tau = _hajek(y, t, e, estimand)
    if estimand == "ate":
        w1, w0 = t / e, (1 - t) / (1 - e)
    else:
        w1, w0 = t.astype(float), (1 - t) * e / (1 - e)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)

    # Estimating functions, stacked: [beta (p), mu1, mu0]
    psi = np.column_stack([Xt * (t - e)[:, None], w1 * (y - mu1), w0 * (y - mu0)])
    J = np.zeros((p + 2, p + 2))
    J[:p, :p] = -(Xt * (e * (1 - e))[:, None]).T @ Xt / n
    if estimand == "ate":
        J[p, :p] = -np.mean((t * (y - mu1) * (1 - e) / e)[:, None] * Xt, axis=0)
        J[p + 1, :p] = np.mean(((1 - t) * (y - mu0) * e / (1 - e))[:, None] * Xt, axis=0)
    else:
        J[p + 1, :p] = np.mean(((1 - t) * (y - mu0) * e / (1 - e))[:, None] * Xt, axis=0)
    J[p, p] = -np.mean(w1)
    J[p + 1, p + 1] = -np.mean(w0)
    S, adj = _cluster_sum(psi, cluster)
    B = adj * S.T @ S / n
    J_inv = np.linalg.inv(J)
    V = J_inv @ B @ J_inv.T / n
    c = np.zeros(p + 2)
    c[p], c[p + 1] = 1.0, -1.0
    return tau, float(np.sqrt(c @ V @ c)), e


def ipw(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    estimand: Estimand = "ate",
    ps_model: Model = "logistic",
    se: Literal["sandwich", "bootstrap"] = "sandwich",
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    cluster: str | None = None,
) -> Estimate:
    """Hajek IPW estimate of the ATE or ATT.

    ``cluster`` names a column (e.g. a user id). The sandwich then sums the estimating
    functions within clusters (rows are not independent), and the bootstrap resamples
    whole clusters.

    ``extra`` holds overlap diagnostics and effective sample sizes. Read them before
    the estimate.
    """
    require_columns(df, [outcome, treatment, *covariates])
    if estimand not in ("ate", "att"):
        raise ValueError("estimand must be 'ate' or 'att'")
    y = as_1d_float(outcome, df[outcome].to_numpy())
    t = as_treatment(treatment, df[treatment].to_numpy())
    X = df[list(covariates)].to_numpy(dtype=float)
    cl = df[cluster].to_numpy() if cluster is not None else None

    if se == "sandwich":
        if ps_model != "logistic":
            raise ValueError(
                "the sandwich SE assumes a logistic propensity model; use se='bootstrap'"
            )
        tau, s, ps = _sandwich_se(y, t, X, estimand, cl)
        boot_info: dict[str, float] = {}
    elif se == "bootstrap":
        ps = fit_propensity(df, treatment, covariates, ps_model, seed)
        tau = _hajek(y, t, ps, estimand)
        rng = np.random.default_rng(seed)
        reps = []
        if cl is not None:
            uniq, inv = np.unique(cl, return_inverse=True)
            members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
        for _ in range(n_boot):
            if cl is None:
                idx = rng.integers(0, len(df), len(df))
            else:
                pick = rng.integers(0, len(uniq), len(uniq))
                idx = np.concatenate([members[k] for k in pick])
            sub = df.iloc[idx]
            if sub[treatment].nunique() < 2:
                continue
            ps_b = np.clip(
                fit_propensity(sub, treatment, covariates, ps_model, seed), 1e-6, 1 - 1e-6
            )
            reps.append(_hajek(y[idx], t[idx], ps_b, estimand))
        if len(reps) < 0.9 * n_boot:
            raise RuntimeError(f"only {len(reps)}/{n_boot} bootstrap replicates were usable")
        s = float(np.std(reps, ddof=1))
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        boot_info = {
            "percentile_ci_low": float(lo),
            "percentile_ci_high": float(hi),
            "n_boot": len(reps),
        }
    else:
        raise ValueError("se must be 'sandwich' or 'bootstrap'")

    ps_c = np.clip(ps, 1e-12, 1 - 1e-12)
    w = ate_weights(t, ps_c) if estimand == "ate" else att_weights(t, ps_c)
    extra = {
        "estimand": estimand,
        "ps_model": ps_model,
        "overlap": overlap_summary(t, ps_c).to_dict(),
        "ess_treated": effective_sample_size(w[t == 1]),
        "ess_control": effective_sample_size(w[t == 0]),
        "cluster": cluster,
        **boot_info,
    }
    return normal_estimate(tau, s, f"IPW-{estimand} ({se} SE)", alpha=alpha, n=len(y), extra=extra)


def _outcome_preds(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    fit_mask: NDArray[np.bool_],
    pred_mask: NDArray[np.bool_],
    model: str,
    seed: int,
) -> NDArray[np.float64]:
    if model == "linear":
        m = LinearRegression().fit(X[fit_mask], y[fit_mask])
    elif model == "gbm":
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
        m.fit(X[fit_mask], y[fit_mask])
    else:
        raise ValueError(f"unknown outcome model {model!r}; use 'linear' or 'gbm'")
    return m.predict(X[pred_mask])


def aipw(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    ps_covariates: Sequence[str],
    outcome_covariates: Sequence[str] | None = None,
    ps_model: Model = "logistic",
    outcome_model: Literal["linear", "gbm"] = "linear",
    n_folds: int = 5,
    clip: float = 1e-3,
    alpha: float = 0.05,
    seed: int = 0,
    cluster: str | None = None,
) -> Estimate:
    """Augmented IPW (doubly robust) estimate of the ATE.

    tau = mean[ m1(X) - m0(X) + T (Y - m1(X)) / e(X) - (1 - T)(Y - m0(X)) / (1 - e(X)) ].

    If either nuisance model is flexible (GBM), both are cross-fitted over ``n_folds`` so
    in-sample overfitting doesn't bias the estimate. ``clip`` bounds e(X) away from 0/1
    in the augmentation term. Clipping is reported, never hidden.

    ``cluster`` names a column whose groups share errors (e.g. a user id). The influence
    function is then summed within clusters for the SE.

    ``extra`` also reports the estimated potential-outcome means and a risk ratio with a
    delta-method CI, which is useful for E-values on binary outcomes.
    """
    out_cov = list(outcome_covariates) if outcome_covariates is not None else list(ps_covariates)
    require_columns(df, [outcome, treatment, *ps_covariates, *out_cov])
    y = as_1d_float(outcome, df[outcome].to_numpy())
    t = as_treatment(treatment, df[treatment].to_numpy())
    Xo = df[out_cov].to_numpy(dtype=float)
    n = len(y)
    crossfit = ps_model == "gbm" or outcome_model == "gbm"

    m1: NDArray[np.float64] = np.empty(n)
    m0: NDArray[np.float64] = np.empty(n)
    ps: NDArray[np.float64]
    if crossfit:
        folds = np.random.default_rng(seed).integers(0, n_folds, n)
        ps = np.empty(n)
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            ps[te] = _fold_ps(df, treatment, ps_covariates, ps_model, tr, te, seed)
            m1[te] = _outcome_preds(Xo, y, tr & (t == 1), te, outcome_model, seed)
            m0[te] = _outcome_preds(Xo, y, tr & (t == 0), te, outcome_model, seed)
    else:
        ps = fit_propensity(df, treatment, ps_covariates, ps_model, seed)
        all_rows = np.ones(n, dtype=bool)
        m1 = _outcome_preds(Xo, y, t == 1, all_rows, outcome_model, seed)
        m0 = _outcome_preds(Xo, y, t == 0, all_rows, outcome_model, seed)

    n_clipped = int(np.sum((ps < clip) | (ps > 1 - clip)))
    e = np.clip(ps, clip, 1 - clip)
    phi1 = m1 + t * (y - m1) / e
    phi0 = m0 + (1 - t) * (y - m0) / (1 - e)
    phi = phi1 - phi0
    tau = float(phi.mean())
    mu1, mu0 = float(phi1.mean()), float(phi0.mean())
    cl = df[cluster].to_numpy() if cluster is not None else None
    S, adj = _cluster_sum(np.column_stack([phi - tau, phi1 - mu1, phi0 - mu0]), cl)
    V = adj * S.T @ S / n**2
    se = float(np.sqrt(V[0, 0]))
    rr_extra: dict[str, float] = {}
    if mu1 > 0 and mu0 > 0:
        g = np.array([0.0, 1 / mu1, -1 / mu0])  # gradient of log(mu1/mu0)
        se_log = float(np.sqrt(g @ V @ g))
        z = float(stats.norm.ppf(1 - alpha / 2))
        rr = mu1 / mu0
        rr_extra = {
            "risk_ratio": rr,
            "rr_ci_low": rr * np.exp(-z * se_log),
            "rr_ci_high": rr * np.exp(z * se_log),
        }
    extra = {
        "mean_treated_po": mu1,
        "mean_control_po": mu0,
        **rr_extra,
        "cluster": cluster,
        "ps_model": ps_model,
        "outcome_model": outcome_model,
        "crossfit": crossfit,
        "n_ps_clipped": n_clipped,
        "overlap": overlap_summary(t, np.clip(ps, 1e-12, 1 - 1e-12)).to_dict(),
    }
    return normal_estimate(tau, se, "AIPW (doubly robust)", alpha=alpha, n=n, extra=extra)


def _fold_ps(
    df: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    model: Model,
    tr: NDArray[np.bool_],
    te: NDArray[np.bool_],
    seed: int,
) -> NDArray[np.float64]:
    X = df[list(covariates)].to_numpy(dtype=float)
    t = df[treatment].to_numpy().astype(int)
    if model == "logistic":
        Z, _ = _logistic_design(X)
        clf = unpenalized_logistic().fit(Z[tr], t[tr])
        return clf.predict_proba(Z[te])[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier

    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    gbm.fit(X[tr], t[tr])
    return gbm.predict_proba(X[te])[:, 1]
