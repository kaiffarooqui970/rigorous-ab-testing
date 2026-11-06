"""Seeded panel simulator for difference-in-differences with a known ATT.

Model, for unit i in period t (treatment switches on for every treated unit at ``treat_start``):

    y_it = alpha_i + gamma_t + delta * G_i * (t - treat_start)      # differential trend
           + D_it * effect(t - treat_start) + e_it

* ``alpha_i ~ N(level_gap * G_i, 1)``: treated units may have different *levels*.
  DiD differences this away, so it does not bias the estimate.
* ``gamma_t ~ N(0, time_effect_sd)``: common shocks. These are also differenced away.
* ``delta`` (``trend_violation``): a differential linear trend in the treated group. It
  breaks parallel trends and biases DiD. It is active in the pre-period too, so an
  event-study pre-trend check can detect it.
* ``e_it`` is AR(1) within unit with autocorrelation ``ar_rho``. Naive OLS SEs that
  ignore it are too small, which is why clustered SEs are needed.
* ``effect(e)`` is either constant or a linear ramp whose post-period average equals
  ``effect``, so the true ATT over post periods is always ``effect``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from xpkit._validation import check_probability


@dataclass(frozen=True)
class PanelSpec:
    """Configuration for :func:`simulate_panel`."""

    n_units: int = 200
    n_periods: int = 10
    treat_start: int = 6
    treated_share: float = 0.5
    effect: float = 1.0
    dynamic: Literal["constant", "ramp"] = "constant"
    trend_violation: float = 0.0
    level_gap: float = 1.0
    ar_rho: float = 0.6
    noise_sd: float = 1.0
    time_effect_sd: float = 0.5

    def __post_init__(self) -> None:
        if self.n_units < 4:
            raise ValueError("n_units must be >= 4")
        if not 1 <= self.treat_start < self.n_periods:
            raise ValueError(
                f"treat_start must be in [1, n_periods), got {self.treat_start} with "
                f"n_periods={self.n_periods}: need at least one pre and one post period"
            )
        check_probability("treated_share", self.treated_share)
        if not -1 < self.ar_rho < 1:
            raise ValueError(f"ar_rho must be in (-1, 1), got {self.ar_rho}")
        if self.dynamic not in ("constant", "ramp"):
            raise ValueError(f"dynamic must be 'constant' or 'ramp', got {self.dynamic!r}")

    @property
    def n_post(self) -> int:
        return self.n_periods - self.treat_start

    def with_(self, **changes: object) -> PanelSpec:
        return replace(self, **changes)  # type: ignore[arg-type]

    def event_effects(self) -> dict[int, float]:
        """True treatment effect at each post-treatment event time e = t - treat_start."""
        m = self.n_post
        if self.dynamic == "constant":
            return {e: self.effect for e in range(m)}
        return {e: self.effect * 2 * (e + 1) / (m + 1) for e in range(m)}


@dataclass(frozen=True)
class SimulatedPanel:
    """Output of :func:`simulate_panel`: long-format panel and the true (event-time) ATT."""

    data: pd.DataFrame  # unit, period, treated, post, d, y, event_time
    true_att: float
    event_effects: dict[int, float]
    spec: PanelSpec


def simulate_panel(spec: PanelSpec, seed: int | np.random.Generator = 0) -> SimulatedPanel:
    """Simulate a balanced panel. ``true_att`` is the average effect over treated post periods."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n, periods = spec.n_units, spec.n_periods
    n_treated = max(1, min(n - 1, round(spec.treated_share * n)))
    g = np.zeros(n, dtype=int)
    g[rng.choice(n, size=n_treated, replace=False)] = 1

    alpha = rng.normal(spec.level_gap * g, 1.0)
    gamma = rng.normal(0.0, spec.time_effect_sd, periods)

    innov_sd = spec.noise_sd * np.sqrt(1 - spec.ar_rho**2)
    e = np.empty((n, periods))
    e[:, 0] = rng.normal(0.0, spec.noise_sd, n)
    for t in range(1, periods):
        e[:, t] = spec.ar_rho * e[:, t - 1] + rng.normal(0.0, innov_sd, n)

    t_idx = np.arange(periods)
    event_time = t_idx - spec.treat_start
    effects = spec.event_effects()
    eff_t = np.array([effects.get(int(et), 0.0) for et in event_time])
    post = (t_idx >= spec.treat_start).astype(int)

    y = (
        alpha[:, None]
        + gamma[None, :]
        + spec.trend_violation * g[:, None] * event_time[None, :]
        + g[:, None] * post[None, :] * eff_t[None, :]
        + e
    )
    data = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n), periods),
            "period": np.tile(t_idx, n),
            "treated": np.repeat(g, periods),
            "post": np.tile(post, n),
            "event_time": np.tile(event_time, n),
            "y": y.ravel(),
        }
    )
    data["d"] = data["treated"] * data["post"]
    return SimulatedPanel(
        data=data,
        true_att=float(np.mean(list(effects.values()))),
        event_effects=effects,
        spec=spec,
    )
