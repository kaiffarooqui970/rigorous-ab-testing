"""Vectorised simulation of interim-look strategies for a two-arm conversion test.

At each of K equally spaced looks, a two-proportion z-statistic (pooled SE) is
computed on the cumulative data. The strategies compared are:

* ``single_look``: test once, at the final look only (the fixed-horizon baseline)
* ``naive_peeking``: stop at the first look with |z| > z_{1-a/2}
* ``bonferroni``: stop at the first look with |z| > z_{1-a/(2K)}
* ``obrien_fleming`` and ``pocock``: Lan-DeMets boundaries from :mod:`xpkit.estimate.sequential`

For each strategy the simulation reports the rejection rate (the false-positive rate
when ``lift == 0``, otherwise power) and the expected sample size as a share of the
maximum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from xpkit.estimate.sequential import group_sequential_design


@dataclass(frozen=True)
class StrategyResult:
    """Rejection rate and expected sample fraction of one interim-look strategy."""

    strategy: str
    rejection_rate: float
    rejection_mcse: float
    expected_sample_fraction: float
    boundaries: list[float]


def simulate_peeking_strategies(
    baseline: float = 0.05,
    relative_lift: float = 0.0,
    n_looks: int = 10,
    n_per_look_per_arm: int = 1000,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> dict[str, StrategyResult]:
    """Simulate ``n_sims`` experiments and apply every stopping strategy to the same data."""
    if n_looks < 1 or n_sims < 10:
        raise ValueError("need n_looks >= 1 and n_sims >= 10")
    rng = np.random.default_rng(seed)
    p_t = baseline * (1 + relative_lift)
    inc_c = rng.binomial(n_per_look_per_arm, baseline, size=(n_sims, n_looks))
    inc_t = rng.binomial(n_per_look_per_arm, p_t, size=(n_sims, n_looks))
    s_c, s_t = inc_c.cumsum(axis=1), inc_t.cumsum(axis=1)
    n = n_per_look_per_arm * np.arange(1, n_looks + 1)
    pool = (s_c + s_t) / (2 * n)
    se = np.sqrt(pool * (1 - pool) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (s_t / n - s_c / n) / se, 0.0)

    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    rules: dict[str, np.ndarray] = {
        "naive_peeking": np.full(n_looks, zcrit),
        "bonferroni": np.full(n_looks, float(stats.norm.ppf(1 - alpha / (2 * n_looks)))),
        "obrien_fleming": np.array(
            group_sequential_design(n_looks, alpha, "obrien_fleming").boundaries
        ),
        "pocock": np.array(group_sequential_design(n_looks, alpha, "pocock").boundaries),
    }
    single = np.full(n_looks, np.inf)
    single[-1] = zcrit
    rules = {"single_look": single, **rules}

    out: dict[str, StrategyResult] = {}
    for name, bounds in rules.items():
        cross = np.abs(z) >= bounds[None, :]
        rejected = cross.any(axis=1)
        first = np.where(rejected, cross.argmax(axis=1) + 1, n_looks)
        rate = float(rejected.mean())
        out[name] = StrategyResult(
            strategy=name,
            rejection_rate=rate,
            rejection_mcse=float(np.sqrt(rate * (1 - rate) / n_sims)),
            expected_sample_fraction=float(first.mean() / n_looks),
            boundaries=[float(b) for b in bounds],
        )
    return out


def strategies_to_records(results: dict[str, StrategyResult]) -> list[dict[str, object]]:
    """Convert strategy results to JSON-friendly dicts."""
    return [asdict(r) for r in results.values()]
