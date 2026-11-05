"""Seeded simulator for unit-level experiments and observational studies with a known truth.

Data-generating process (all covariates i.i.d. standard normal):

* ``x0`` -- *prognostic*: drives the outcome strongly. Its pre-period version is
  ``pre_metric``, which is what CUPED adjusts for.
* ``x1`` -- *confounder* candidate: drives the outcome moderately and, when
  ``confounding > 0``, also drives treatment uptake.
* ``x2`` -- *effect modifier*: carries the heterogeneity in the treatment effect
  (``hte``) and has no effect on the baseline outcome. That makes it a clean
  target for uplift models.
* ``x3..`` -- pure noise covariates.
* ``u`` -- an unobserved confounder. It is never returned in ``data``, only in ``truth``.

Potential outcomes:

* continuous: ``Y(0) = baseline + b0*x0 + 0.5*x1 + c_u*u + eps``,
  ``Y(1) = Y(0) + tau(x)``, ``tau(x) = effect + hte * h(x2)``
* binary: ``P(Y(0)=1) = expit(a + b0*x0 + 0.5*x1 + c_u*u)``, with ``a`` calibrated so
  that ``E[P(Y(0)=1)] = baseline``, and ``P(Y(1)=1) = clip(P(Y(0)=1) + tau(x), 0, 1)``

``h`` is either ``x2`` ("linear") or ``sign(x2)`` ("step"); both have mean zero, so
``effect`` is the average treatment effect before any clipping on the binary scale.

The *population* ATE and ATT are computed by Monte Carlo on a fixed 1,000,000-draw
sample. They are exact for the continuous outcome (up to MC error in the ATT) and
include the effect of probability clipping for the binary outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import expit, logit

from xpkit._validation import check_positive, check_probability

OutcomeType = Literal["continuous", "binary"]
_TRUTH_MC_DRAWS = 1_000_000
_TRUTH_SEED = 20240531


@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for :func:`simulate_experiment`.

    Args:
        n: number of units *before* any SRM loss.
        outcome: ``"continuous"`` or ``"binary"``.
        baseline: control mean (continuous) or average control conversion rate (binary).
        effect: average additive treatment effect on the outcome scale.
        noise_sd: SD of idiosyncratic outcome noise (continuous only).
        n_covariates: number of observed covariates (>= 3).
        covariate_effect: coefficient of ``x0`` on the outcome. Larger values give a more
            predictive pre-period metric, and so more CUPED variance reduction.
        pre_period_noise: noise SD of ``pre_metric`` around its structural mean.
        hte: size of effect heterogeneity along ``x2``.
        hte_form: ``"linear"`` (tau linear in x2) or ``"step"`` (tau shifts with sign(x2)).
        treat_prob: marginal assignment probability under randomization.
        confounding: log-odds coefficient of ``x0 + x1`` on treatment uptake (0 = randomized).
        unobserved_confounding: coefficient of the hidden ``u`` on both treatment log-odds
            and the outcome (0 = no hidden confounding).
        srm_drop: fraction of *treated* units lost after assignment (sample ratio mismatch).
        srm_mechanism: ``"random"`` loss (harmless to bias, but still a trust failure) or
            ``"selective"`` loss of low-``x0`` treated units (biases the naive estimate).
    """

    n: int = 10_000
    outcome: OutcomeType = "continuous"
    baseline: float = 10.0
    effect: float = 0.0
    noise_sd: float = 1.0
    n_covariates: int = 3
    covariate_effect: float = 1.0
    pre_period_noise: float = 0.5
    hte: float = 0.0
    hte_form: Literal["linear", "step"] = "linear"
    treat_prob: float = 0.5
    confounding: float = 0.0
    unobserved_confounding: float = 0.0
    srm_drop: float = 0.0
    srm_mechanism: Literal["random", "selective"] = "random"

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError(f"n must be at least 4, got {self.n}")
        if self.outcome not in ("continuous", "binary"):
            raise ValueError(f"outcome must be 'continuous' or 'binary', got {self.outcome!r}")
        if self.outcome == "binary":
            check_probability("baseline (binary outcome)", self.baseline)
        check_positive("noise_sd", self.noise_sd)
        check_positive("pre_period_noise", self.pre_period_noise)
        if self.n_covariates < 3:
            raise ValueError(
                "n_covariates must be >= 3 (x0 prognostic, x1 confounder, x2 modifier)"
            )
        check_probability("treat_prob", self.treat_prob)
        if not 0 <= self.srm_drop < 1:
            raise ValueError(f"srm_drop must be in [0, 1), got {self.srm_drop}")
        if self.srm_mechanism == "selective" and self.srm_drop > 0.5:
            raise ValueError(
                "selective srm_drop must be <= 0.5 (it drops 2*srm_drop of low-x0 treated)"
            )
        if self.hte_form not in ("linear", "step"):
            raise ValueError(f"hte_form must be 'linear' or 'step', got {self.hte_form!r}")

    def with_(self, **changes: object) -> ExperimentSpec:
        """Return a copy with some fields changed."""
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SimulatedExperiment:
    """Output of :func:`simulate_experiment`.

    ``data`` holds only what an analyst would observe. ``truth`` holds the hidden
    quantities (individual effects, potential-outcome means, the unobserved confounder),
    aligned row-by-row with ``data``.
    """

    data: pd.DataFrame
    truth: pd.DataFrame
    true_ate: float
    true_att: float
    spec: ExperimentSpec

    @property
    def sample_ate(self) -> float:
        return float(self.truth["tau"].mean())


def _h(x2: NDArray[np.float64], form: str) -> NDArray[np.float64]:
    return x2 if form == "linear" else np.sign(x2)


def _linear_predictor(
    spec: ExperimentSpec, x0: NDArray[np.float64], x1: NDArray[np.float64], u: NDArray[np.float64]
) -> NDArray[np.float64]:
    return spec.covariate_effect * x0 + 0.5 * x1 + spec.unobserved_confounding * u


def _treat_logit(
    spec: ExperimentSpec, x0: NDArray[np.float64], x1: NDArray[np.float64], u: NDArray[np.float64]
) -> NDArray[np.float64]:
    return logit(spec.treat_prob) + spec.confounding * (x0 + x1) + spec.unobserved_confounding * u


@lru_cache(maxsize=256)
def _binary_intercept(
    baseline: float, covariate_effect: float, unobserved_confounding: float
) -> float:
    """Intercept a such that E[expit(a + lin)] == baseline, solved on a fixed MC sample."""
    rng = np.random.default_rng(_TRUTH_SEED)
    x0, x1, u = rng.standard_normal((3, 200_000))
    lin = covariate_effect * x0 + 0.5 * x1 + unobserved_confounding * u
    return float(
        optimize.brentq(lambda a: expit(a + lin).mean() - baseline, -30.0, 30.0, xtol=1e-10)
    )


def _potential_means(
    spec: ExperimentSpec,
    x0: NDArray[np.float64],
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    u: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lin = _linear_predictor(spec, x0, x1, u)
    tau = spec.effect + spec.hte * _h(x2, spec.hte_form)
    if spec.outcome == "continuous":
        mu0 = spec.baseline + lin
        return mu0, mu0 + tau
    a = _binary_intercept(spec.baseline, spec.covariate_effect, spec.unobserved_confounding)
    p0 = expit(a + lin)
    return p0, np.clip(p0 + tau, 0.0, 1.0)


@lru_cache(maxsize=256)
def population_truth(spec: ExperimentSpec) -> tuple[float, float]:
    """(ATE, ATT) in the population defined by ``spec``, by Monte Carlo on 1e6 draws.

    SRM loss is a *measurement* failure, so it does not change the estimand. The truth
    is computed for the population as assigned, before any loss.
    """
    rng = np.random.default_rng(_TRUTH_SEED + 1)
    x0, x1, x2, u = rng.standard_normal((4, _TRUTH_MC_DRAWS))
    mu0, mu1 = _potential_means(spec, x0, x1, x2, u)
    diff = mu1 - mu0
    p_treat = expit(_treat_logit(spec, x0, x1, u))
    ate = float(diff.mean())
    att = float(np.sum(diff * p_treat) / np.sum(p_treat))
    if spec.outcome == "continuous" and spec.hte == 0:
        ate = att = float(spec.effect)  # exact, no MC error
    return ate, att


def simulate_experiment(
    spec: ExperimentSpec, seed: int | np.random.Generator = 0
) -> SimulatedExperiment:
    """Simulate one dataset from ``spec``.

    Returns observed columns ``unit_id, treatment, y, pre_metric, x0..x{k-1}``.

    Example:
        >>> sim = simulate_experiment(ExperimentSpec(n=1000, effect=0.2), seed=1)
        >>> sim.data.columns.tolist()[:4]
        ['unit_id', 'treatment', 'y', 'pre_metric']
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n, k = spec.n, spec.n_covariates
    x = rng.standard_normal((n, k))
    u = rng.standard_normal(n)
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

    t = (rng.random(n) < expit(_treat_logit(spec, x0, x1, u))).astype(int)
    mu0, mu1 = _potential_means(spec, x0, x1, x2, u)
    lin = _linear_predictor(spec, x0, x1, u)

    if spec.outcome == "continuous":
        y = np.where(t == 1, mu1, mu0) + rng.normal(0.0, spec.noise_sd, n)
        pre_metric = spec.baseline + lin + rng.normal(0.0, spec.pre_period_noise, n)
    else:
        y = (rng.random(n) < np.where(t == 1, mu1, mu0)).astype(float)
        pre_metric = lin + rng.normal(0.0, spec.pre_period_noise, n)

    keep = np.ones(n, dtype=bool)
    if spec.srm_drop > 0:
        if spec.srm_mechanism == "random":
            keep = ~((t == 1) & (rng.random(n) < spec.srm_drop))
        else:
            keep = ~((t == 1) & (x0 < 0) & (rng.random(n) < 2 * spec.srm_drop))

    data = pd.DataFrame({"unit_id": np.arange(n), "treatment": t, "y": y, "pre_metric": pre_metric})
    for j in range(k):
        data[f"x{j}"] = x[:, j]
    truth = pd.DataFrame(
        {"unit_id": np.arange(n), "tau": mu1 - mu0, "mu0": mu0, "mu1": mu1, "u": u}
    )

    true_ate, true_att = population_truth(spec)
    return SimulatedExperiment(
        data=data.loc[keep].reset_index(drop=True),
        truth=truth.loc[keep].reset_index(drop=True),
        true_ate=true_ate,
        true_att=true_att,
        spec=spec,
    )
