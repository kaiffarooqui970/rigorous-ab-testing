"""Difference-in-differences with cluster-robust inference and an event-study pre-trend check.

Identification needs **parallel trends**: without treatment, the treated group's
average outcome would have moved in parallel with the control group's. It also
needs no anticipation and no spillovers. Parallel trends can't be tested for the
post-period. The event study tests whether trends were parallel *before*
treatment. That is necessary evidence, not sufficient, and a pre-trend test with low
power can "pass" a real violation.

Both estimators work on a **balanced panel with a single adoption date** (block
design). Staggered adoption with heterogeneous effects breaks two-way fixed effects
(Goodman-Bacon 2021) and is out of scope here (see docs/limitations).

Inference clusters at the unit level. Outcomes are serially correlated within a unit,
and treatment is assigned per unit, so iid OLS SEs are too small (Bertrand, Duflo &
Mullainathan 2004). The validation study shows this directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from xpkit._ols import ols
from xpkit._validation import require_columns
from xpkit.types import Estimate


def _check_panel(df: pd.DataFrame, unit: str, period: str, treated: str) -> None:
    counts = df.groupby(unit)[period].nunique()
    if counts.nunique() != 1 or len(df) != counts.iloc[0] * len(counts):
        raise ValueError("panel must be balanced (every unit observed once in every period)")
    if (df.groupby(unit)[treated].nunique() > 1).any():
        raise ValueError(f"'{treated}' must be a time-invariant group indicator (0/1 per unit)")


def did(
    df: pd.DataFrame,
    outcome: str = "y",
    unit: str = "unit",
    period: str = "period",
    treated: str = "treated",
    post: str = "post",
    se: Literal["cluster", "classical"] = "cluster",
    alpha: float = 0.05,
) -> Estimate:
    """Two-way fixed-effects DiD: y_it = a_i + g_t + tau * (treated_i * post_t) + e_it.

    On a balanced block-adoption panel this equals the 2x2 difference of group-by-period
    means. ``se="cluster"`` (default) clusters by unit, with a t(G-1) reference
    distribution. ``se="classical"`` is the iid OLS SE. It is offered only to show how
    wrong it can be.
    """
    require_columns(df, [outcome, unit, period, treated, post])
    _check_panel(df, unit, period, treated)
    d = df.sort_values([unit, period])
    y = d[outcome].to_numpy(float)
    x = (d[treated] * d[post]).to_numpy(float)
    yd = _two_way_demean(y, d[unit].to_numpy(), d[period].to_numpy())
    xd = _two_way_demean(x, d[unit].to_numpy(), d[period].to_numpy())
    n_units, n_periods = d[unit].nunique(), d[period].nunique()
    fe = n_units + n_periods - 1
    fit = ols(
        yd,
        xd[:, None],
        cov="cluster" if se == "cluster" else "classical",
        cluster=d[unit].to_numpy(),
        extra_dof=fe,
    )
    tau, s = float(fit.beta[0]), float(fit.se[0])
    dof = n_units - 1 if se == "cluster" else len(y) - 1 - fe
    tcrit = float(stats.t.ppf(1 - alpha / 2, dof))
    return Estimate(
        estimate=tau,
        se=s,
        ci_low=tau - tcrit * s,
        ci_high=tau + tcrit * s,
        p_value=float(2 * stats.t.sf(abs(tau) / s, dof)),
        method=f"DiD TWFE ({se} SE)",
        alpha=alpha,
        n=len(y),
        extra={"n_units": n_units, "n_periods": n_periods, "dof": dof},
    )


def _two_way_demean(v: np.ndarray, unit: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Within transformation for a balanced panel: v - mean_i - mean_t + grand mean."""
    s = pd.Series(v)
    mi = s.groupby(unit).transform("mean").to_numpy()
    mt = s.groupby(period).transform("mean").to_numpy()
    return v - mi - mt + v.mean()


@dataclass(frozen=True)
class EventStudy:
    """Event-study coefficients (relative to the reference period) and the joint pre-trend test."""

    coefficients: pd.DataFrame  # event_time, estimate, se, ci_low, ci_high
    reference_period: int
    pretrend_wald: float
    pretrend_dof: int
    pretrend_p_value: float
    alpha: float

    @property
    def pretrend_rejected(self) -> bool:
        return self.pretrend_p_value < self.alpha


def event_study(
    df: pd.DataFrame,
    outcome: str = "y",
    unit: str = "unit",
    period: str = "period",
    treated: str = "treated",
    event_time: str = "event_time",
    reference: int = -1,
    alpha: float = 0.05,
) -> EventStudy:
    """Dynamic TWFE: y_it = a_i + g_t + sum_{e != ref} b_e * treated_i * 1[event_time = e] + e_it.

    The coefficients are effects relative to event time ``reference`` (default: the
    last pre-period). The pre-trend test is a cluster-robust Wald test that all
    pre-period b_e (e < 0, e != ref) are 0, using an F(q, G-1) reference distribution.
    """
    require_columns(df, [outcome, unit, period, treated, event_time])
    _check_panel(df, unit, period, treated)
    d = df.sort_values([unit, period])
    ets = sorted(int(e) for e in d[event_time].unique() if e != reference)
    if reference not in set(d[event_time].unique()):
        raise ValueError(f"reference event time {reference} is not in the data")
    u, p = d[unit].to_numpy(), d[period].to_numpy()
    X = np.column_stack(
        [
            _two_way_demean(((d[treated] == 1) & (d[event_time] == e)).to_numpy(float), u, p)
            for e in ets
        ]
    )
    yd = _two_way_demean(d[outcome].to_numpy(float), u, p)
    n_fe = d[unit].nunique() + d[period].nunique() - 1
    fit = ols(yd, X, cov="cluster", cluster=u, extra_dof=n_fe)
    g = d[unit].nunique()
    tcrit = float(stats.t.ppf(1 - alpha / 2, g - 1))
    coefs = pd.DataFrame(
        {
            "event_time": ets,
            "estimate": fit.beta,
            "se": fit.se,
            "ci_low": fit.beta - tcrit * fit.se,
            "ci_high": fit.beta + tcrit * fit.se,
        }
    )
    pre = [i for i, e in enumerate(ets) if e < 0]
    if pre:
        b, V = fit.beta[pre], fit.vcov[np.ix_(pre, pre)]
        wald = float(b @ np.linalg.solve(V, b))
        q = len(pre)
        p_pre = float(stats.f.sf(wald / q, q, g - 1))
    else:
        wald, q, p_pre = float("nan"), 0, float("nan")
    return EventStudy(coefs, reference, wald, q, p_pre, alpha)
