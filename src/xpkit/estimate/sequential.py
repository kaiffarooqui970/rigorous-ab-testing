"""Group-sequential testing with Lan-DeMets alpha-spending boundaries.

Problem: if you look at a running experiment K times and stop at the first look with
p < 0.05, the false-positive rate is far above 5% (about 19% for K = 10; see
``validation/stage2_experiments.py``).

Solution: fix the looks in advance and spend the alpha budget across them. A
spending function alpha(t) gives the cumulative type I error allowed by information
fraction t in (0, 1]. The boundary c_k for look k solves

    P_H0( |Z_1| < c_1, ..., |Z_{k-1}| < c_{k-1}, |Z_k| >= c_k ) = alpha(t_k) - alpha(t_{k-1}).

* O'Brien-Fleming-type: per tail, alpha_side(t) = 2 - 2*Phi(z_{1-a/4} / sqrt(t)). Very strict early
  and close to the fixed-sample critical value at the end. This is the usual choice
  in industry.
* Pocock-type: alpha(t) = a * ln(1 + (e - 1) * t). Spends evenly, so it is easier to
  stop early but pays more at the final look.

The boundaries are computed here by numerical integration (the Armitage-McPherson-Rowe
recursion), not taken from a library. Under H0 the score statistic S_k = Z_k*sqrt(t_k)
is Brownian motion at times t_k. The code propagates the density of S on the
continuation region, look by look, with Simpson's rule, then root-finds each c_k so
the exit probability matches the spending increment.

Check: for 5 equally spaced looks, two-sided alpha = 0.05 and OBF spending, this gives
4.877, 3.357, 2.680, 2.290, 2.031, which matches published Lan-DeMets tables
(e.g. R ``gsDesign``/``ldbounds``). See ``tests/test_estimate.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, stats

Spending = Literal["obrien_fleming", "pocock"]
_GRID = 801  # odd, for Simpson's rule


def spending_function(kind: Spending, alpha: float) -> Callable[[float], float]:
    """Cumulative two-sided alpha spent by information fraction t.

    Follows the convention of R ``gsDesign`` (test.type=2) and ``ldbounds``. The
    one-sided OBF-type function is applied to each tail at alpha/2, so
    alpha(t) = 2 * [2 - 2*Phi(z_{1 - alpha/4} / sqrt(t))].
    """
    z_side = stats.norm.ppf(1 - alpha / 4)
    if kind == "obrien_fleming":
        return lambda t: float(2 * (2 - 2 * stats.norm.cdf(z_side / np.sqrt(t)))) if t > 0 else 0.0
    if kind == "pocock":
        return lambda t: float(alpha * np.log(1 + (np.e - 1) * t))
    raise ValueError(f"unknown spending function {kind!r}; use 'obrien_fleming' or 'pocock'")


@dataclass(frozen=True)
class SequentialDesign:
    """Two-sided efficacy boundaries on the z-scale for pre-planned looks."""

    info_fractions: list[float]
    boundaries: list[float]
    alpha_spent: list[float]  # cumulative
    alpha: float
    spending: str

    @property
    def n_looks(self) -> int:
        return len(self.boundaries)

    def nominal_p_thresholds(self) -> list[float]:
        """Two-sided p-value thresholds equivalent to the z boundaries."""
        return [float(2 * stats.norm.sf(c)) for c in self.boundaries]


def _norm_pdf(x: NDArray[np.float64], sd: float) -> NDArray[np.float64]:
    return np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _exit_gap(
    c: float,
    t: float,
    sd: float,
    grid: NDArray[np.float64] | None,
    dens: NDArray[np.float64] | None,
    target: float,
) -> float:
    """P(first exit at this look | boundary c) - target. grid=None means the first look."""
    if grid is None or dens is None:
        return float(2 * stats.norm.sf(c)) - target
    b = c * np.sqrt(t)
    p_cross = stats.norm.cdf((-b - grid) / sd) + stats.norm.sf((b - grid) / sd)
    return float(integrate.simpson(dens * p_cross, x=grid)) - target


def group_sequential_design(
    n_looks: int | None = None,
    alpha: float = 0.05,
    spending: Spending = "obrien_fleming",
    info_fractions: Sequence[float] | None = None,
) -> SequentialDesign:
    """Compute two-sided Lan-DeMets efficacy boundaries.

    Args:
        n_looks: number of equally spaced looks (ignored if ``info_fractions`` is given).
        alpha: overall two-sided type I error.
        spending: ``"obrien_fleming"`` or ``"pocock"``.
        info_fractions: increasing fractions in (0, 1] ending at 1. For a fixed-horizon
            test these are cumulative sample size divided by planned total.
    """
    if info_fractions is None:
        if n_looks is None or n_looks < 1:
            raise ValueError("give n_looks >= 1 or explicit info_fractions")
        fracs = np.arange(1, n_looks + 1) / n_looks
    else:
        fracs = np.asarray(info_fractions, dtype=float)
        if fracs.ndim != 1 or fracs.size == 0:
            raise ValueError("info_fractions must be a non-empty list")
        if np.any(np.diff(fracs) <= 0) or fracs[0] <= 0 or not np.isclose(fracs[-1], 1.0):
            raise ValueError("info_fractions must be strictly increasing in (0, 1] and end at 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    spend = spending_function(spending, alpha)
    cum = [spend(float(t)) for t in fracs]

    boundaries: list[float] = []
    grid: NDArray[np.float64] | None = None
    dens: NDArray[np.float64] | None = None
    prev_t = 0.0
    for k, t in enumerate(fracs):
        target = cum[k] - (cum[k - 1] if k else 0.0)
        dt = t - prev_t
        sd = float(np.sqrt(dt))

        if target <= 1e-15:
            c_k = 40.0
        else:
            c_k = float(
                optimize.brentq(
                    _exit_gap, 0.1, 40.0, args=(float(t), sd, grid, dens, target), xtol=1e-10
                )
            )
        boundaries.append(c_k)

        # Density of S_k on the continuation region (-b_k, b_k).
        b_k = c_k * np.sqrt(t)
        new_grid = np.linspace(-b_k, b_k, _GRID)
        if k == 0:
            new_dens = _norm_pdf(new_grid, np.sqrt(t))
        else:
            assert grid is not None and dens is not None
            kern = _norm_pdf(new_grid[:, None] - grid[None, :], sd)
            new_dens = integrate.simpson(kern * dens[None, :], x=grid, axis=1)
        grid, dens, prev_t = new_grid, new_dens, float(t)

    return SequentialDesign(
        info_fractions=fracs.tolist(),
        boundaries=boundaries,
        alpha_spent=cum,
        alpha=alpha,
        spending=spending,
    )


@dataclass(frozen=True)
class SequentialDecision:
    """Outcome of applying a sequential design to the z-statistics observed so far."""

    stopped: bool
    look: int | None  # 1-based look at which the boundary was crossed
    z_at_stop: float | None
    boundary_at_stop: float | None


def sequential_decision(z_stats: Sequence[float], design: SequentialDesign) -> SequentialDecision:
    """Apply the design to the z-statistics seen so far (one per completed look)."""
    if len(z_stats) > design.n_looks:
        raise ValueError(f"got {len(z_stats)} looks but the design has {design.n_looks}")
    for i, (z, c) in enumerate(zip(z_stats, design.boundaries, strict=False)):
        if abs(z) >= c:
            return SequentialDecision(True, i + 1, float(z), float(c))
    return SequentialDecision(False, None, None, None)
