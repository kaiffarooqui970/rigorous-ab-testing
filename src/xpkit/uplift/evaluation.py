"""Held-out evaluation of uplift scores on **randomized** data.

Individual effects are never observed, so an uplift model can't be scored like a
classifier. Instead, rank the test set by predicted uplift, and for each targeting
depth (top k% of the ranking) compare treated and control outcomes *within* the
targeted group. Randomization makes that within-group comparison unbiased.

* **Qini curve** (Radcliffe 2007): Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k), the
  incremental outcomes in the top-k, scaled to the treated count.
* **Uplift curve**: U(k) = (Y_t(k)/N_t(k) - Y_c(k)/N_c(k)) * (N_t(k) + N_c(k)).
* **Random-targeting baseline**: the straight line from (0, 0) to (1, Q(1)). Targeting
  a random k% captures k% of the total incremental effect.
* **Qini coefficient**: the area between the model curve and the random line, divided
  by the number of units, so it is comparable across test sets. 0 means no better than
  random. A bootstrap CI over the test set says whether the gain is distinguishable
  from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from xpkit._validation import as_1d_float, as_treatment, check_same_length
from xpkit.estimate.effects import diff_in_means


@dataclass(frozen=True)
class UpliftCurve:
    """Qini and uplift curves at increasing targeting depths, with the random baseline."""

    fraction: NDArray[np.float64]  # share of population targeted, 0..1
    qini: NDArray[np.float64]
    uplift: NDArray[np.float64]
    random_qini: NDArray[np.float64]
    n: int

    @property
    def qini_coefficient(self) -> float:
        gain = self.qini - self.random_qini
        area = np.sum(np.diff(self.fraction) * (gain[1:] + gain[:-1]) / 2)  # trapezoid rule
        return float(area / self.n)


def _inputs(
    y: ArrayLike, t: ArrayLike, score: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    ya, ta, sa = as_1d_float("y", y), as_treatment("treatment", t), as_1d_float("score", score)
    check_same_length(y=ya, treatment=ta, score=sa)
    return ya, ta, sa


def uplift_curve(y: ArrayLike, t: ArrayLike, score: ArrayLike, n_points: int = 100) -> UpliftCurve:
    """Qini and uplift curves evaluated at ``n_points`` targeting depths (plus 0).

    Ties in ``score`` are broken randomly with a fixed seed, so a constant score
    reproduces the random baseline instead of an artefact of row order.
    """
    ya, ta, sa = _inputs(y, t, score)
    n = len(ya)
    tie = np.random.default_rng(0).random(n)
    order = np.lexsort((tie, -sa))
    yt, tt = ya[order], ta[order]
    cum_nt, cum_nc = np.cumsum(tt), np.cumsum(1 - tt)
    cum_yt, cum_yc = np.cumsum(yt * tt), np.cumsum(yt * (1 - tt))
    idx = np.unique(np.linspace(1, n, n_points).round().astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt[idx] - np.where(cum_nc[idx] > 0, cum_yc[idx] * cum_nt[idx] / cum_nc[idx], 0.0)
        rate_t = np.where(cum_nt[idx] > 0, cum_yt[idx] / cum_nt[idx], 0.0)
        rate_c = np.where(cum_nc[idx] > 0, cum_yc[idx] / cum_nc[idx], 0.0)
    u = (rate_t - rate_c) * (idx + 1)
    frac = (idx + 1) / n
    frac, q, u = np.r_[0.0, frac], np.r_[0.0, q], np.r_[0.0, u]
    return UpliftCurve(fraction=frac, qini=q, uplift=u, random_qini=frac * q[-1], n=n)


def qini_coefficient_ci(
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Qini coefficient with a nonparametric bootstrap percentile CI over the test set."""
    ya, ta, sa = _inputs(y, t, score)
    point = uplift_curve(ya, ta, sa).qini_coefficient
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ya), len(ya))
        if np.unique(ta[i]).size < 2:
            continue
        reps.append(uplift_curve(ya[i], ta[i], sa[i], n_points=50).qini_coefficient)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def uplift_by_bin(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, n_bins: int = 10, alpha: float = 0.05
) -> pd.DataFrame:
    """Observed treated-minus-control difference within each predicted-uplift bin.

    Bin 1 holds the highest predicted uplift. A useful model shows observed uplift
    falling across bins. A flat profile means the model can't tell people apart.
    """
    ya, ta, sa = _inputs(y, t, score)
    tie = np.random.default_rng(0).random(len(sa))
    ranks = np.lexsort((tie, -sa)).argsort()
    bins = (ranks * n_bins // len(sa)) + 1
    rows = []
    for b in range(1, n_bins + 1):
        m = bins == b
        try:
            e = diff_in_means(ya[m], ta[m], alpha=alpha)
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "predicted_uplift": float(sa[m].mean()),
                    "observed_uplift": e.estimate,
                    "ci_low": e.ci_low,
                    "ci_high": e.ci_high,
                }
            )
        except ValueError:
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "predicted_uplift": float(sa[m].mean()),
                    "observed_uplift": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
    return pd.DataFrame(rows)


def pehe(tau_true: ArrayLike, tau_hat: ArrayLike) -> float:
    """PEHE = sqrt(mean((tau_hat - tau)^2)). Needs the true tau, so simulation only."""
    a, b = as_1d_float("tau_true", tau_true), as_1d_float("tau_hat", tau_hat)
    check_same_length(tau_true=a, tau_hat=b)
    return float(np.sqrt(np.mean((a - b) ** 2)))
