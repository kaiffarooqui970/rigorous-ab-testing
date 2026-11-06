"""Turn uplift scores into a targeting decision with explicit economics.

For a targeting depth k (treat the top k% by predicted uplift), the expected
incremental profit per eligible user is

    profit(k) = k * ( uplift_in_top_k * value_per_conversion - cost_per_treatment )

``uplift_in_top_k`` is the *observed* treated-minus-control difference among the top k%
of an evaluation set that the model never saw. It is unbiased under randomization.
The random-targeting comparator at the same depth is k * (ATE * value - cost), and
"treat everyone" is the case k = 1.

Selecting k on the same data used to report its value overstates the result (a
winner's curse). :func:`choose_depth` therefore picks k on a *validation* split, and
the reported value comes from a separate *test* split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from xpkit._validation import as_1d_float, as_treatment, check_same_length
from xpkit.estimate.effects import diff_in_means


@dataclass(frozen=True)
class Economics:
    """Assumptions behind the targeting decision. Every one is configurable and stated
    in the report.

    Attributes:
        value_per_conversion: margin earned from one incremental outcome (e.g. 1 visit).
        cost_per_treatment: cost of treating one user (e.g. ad impression, discount).
        population: number of eligible users, used to scale per-user values to totals.
    """

    value_per_conversion: float
    cost_per_treatment: float
    population: int = 1_000_000

    def __post_init__(self) -> None:
        if self.value_per_conversion < 0 or self.cost_per_treatment < 0:
            raise ValueError("value_per_conversion and cost_per_treatment must be non-negative")
        if self.population <= 0:
            raise ValueError("population must be positive")

    @property
    def breakeven_uplift(self) -> float:
        """Minimum uplift per treated user for treatment to pay for itself."""
        if self.value_per_conversion == 0:
            return float("inf")
        return self.cost_per_treatment / self.value_per_conversion


DEFAULT_DEPTHS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def policy_table(
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    economics: Economics,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Profit per eligible user at each targeting depth, for the model and for random targeting.

    Columns: depth, n_targeted, uplift_top (with CI), profit_per_user (with CI),
    random_profit_per_user, total_profit, total_random_profit.
    """
    ya, ta, sa = as_1d_float("y", y), as_treatment("treatment", t), as_1d_float("score", score)
    check_same_length(y=ya, treatment=ta, score=sa)
    if any(not 0 < d <= 1 for d in depths):
        raise ValueError("depths must lie in (0, 1]")
    ate = diff_in_means(ya, ta, alpha=alpha)
    tie = np.random.default_rng(0).random(len(sa))
    order = np.lexsort((tie, -sa))
    v, c, pop = economics.value_per_conversion, economics.cost_per_treatment, economics.population
    rows = []
    for k in depths:
        top = order[: max(round(k * len(sa)), 4)]
        e = diff_in_means(ya[top], ta[top], alpha=alpha)
        prof = k * (e.estimate * v - c)
        rows.append(
            {
                "depth": k,
                "n_targeted": len(top),
                "uplift_top": e.estimate,
                "uplift_ci_low": e.ci_low,
                "uplift_ci_high": e.ci_high,
                "profit_per_user": prof,
                "profit_ci_low": k * (e.ci_low * v - c),
                "profit_ci_high": k * (e.ci_high * v - c),
                "random_profit_per_user": k * (ate.estimate * v - c),
                "total_profit": prof * pop,
                "total_random_profit": k * (ate.estimate * v - c) * pop,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PolicyDecision:
    """Targeting depth chosen on validation data and its value measured on test data."""

    chosen_depth: float
    chosen_on: str
    test_profit_per_user: float
    test_profit_ci: tuple[float, float]
    test_random_profit_per_user: float
    test_treat_all_profit_per_user: float
    test_uplift_top: float
    breakeven_uplift: float
    economics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_depth(
    val: tuple[ArrayLike, ArrayLike, ArrayLike],
    test: tuple[ArrayLike, ArrayLike, ArrayLike],
    economics: Economics,
    depths: Sequence[float] = DEFAULT_DEPTHS,
) -> PolicyDecision:
    """Pick the profit-maximising depth on ``val`` = (y, t, score) and report it on ``test``.

    If no depth is profitable on validation, the chosen depth is 0 (treat no one).
    """
    vt = policy_table(*val, economics=economics, depths=depths)
    best = vt.iloc[int(np.argmax(vt.profit_per_user.to_numpy()))]
    chosen = float(best.depth) if best.profit_per_user > 0 else 0.0
    tt = policy_table(*test, economics=economics, depths=sorted(set(depths) | {1.0}))
    all_row = tt.loc[tt.depth == 1.0].iloc[0]
    if chosen == 0.0:
        prof, ci, rnd, up = 0.0, (0.0, 0.0), 0.0, float("nan")
    else:
        r = tt.loc[np.isclose(tt.depth, chosen)].iloc[0]
        prof, ci = float(r.profit_per_user), (float(r.profit_ci_low), float(r.profit_ci_high))
        rnd, up = float(r.random_profit_per_user), float(r.uplift_top)
    return PolicyDecision(
        chosen_depth=chosen,
        chosen_on="validation",
        test_profit_per_user=prof,
        test_profit_ci=ci,
        test_random_profit_per_user=rnd,
        test_treat_all_profit_per_user=float(all_row.profit_per_user),
        test_uplift_top=up,
        breakeven_uplift=economics.breakeven_uplift,
        economics=asdict(economics),
    )
