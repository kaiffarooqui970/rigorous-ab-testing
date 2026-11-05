"""Monte Carlo validation harness.

Runs an estimator over many seeded simulated datasets with a known truth. It reports:

* **bias** = mean(estimate - truth)
* **RMSE** = sqrt(mean((estimate - truth)^2))
* **coverage**: share of CIs containing the truth (target: 1 - alpha)
* **rejection rate**: share with p < alpha. When the truth is 0 this is the
  **false-positive rate**; otherwise it is **power**.
* **SE calibration**: mean reported SE divided by the empirical SD of the estimates
  (target: 1.0)

Each rate comes with its Monte Carlo standard error, so "close to nominal" can be
judged against simulation noise instead of by eye.

Several estimators can be run on the *same* datasets (paired comparison). That is how
CUPED is compared with difference-in-means.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

from xpkit.types import Estimate

D = TypeVar("D")


@dataclass(frozen=True)
class ValidationResult:
    """Summary of one estimator over ``n_reps`` simulated datasets."""

    estimator: str
    n_reps: int
    n_failed: int
    truth: float
    mean_estimate: float
    bias: float
    bias_mcse: float
    rmse: float
    empirical_sd: float
    mean_se: float
    se_ratio: float
    coverage: float
    coverage_mcse: float
    rejection_rate: float
    rejection_mcse: float
    alpha: float
    seconds: float

    @property
    def null_is_true(self) -> bool:
        return self.truth == 0

    def coverage_ok(self, tol_mcse: float = 3.0) -> bool:
        """True if coverage is within ``tol_mcse`` Monte Carlo SEs of ``1 - alpha``."""
        target = 1 - self.alpha
        mcse = math.sqrt(target * (1 - target) / max(self.n_reps - self.n_failed, 1))
        return abs(self.coverage - target) <= tol_mcse * mcse

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationDraw(Generic[D]):
    data: D
    truth: float


def _summarise(
    name: str,
    ests: list[Estimate],
    truths: list[float],
    n_failed: int,
    seconds: float,
    alpha: float,
) -> ValidationResult:
    if not ests:
        raise RuntimeError(f"estimator {name!r} failed on every replication")
    est = np.array([e.estimate for e in ests])
    se = np.array([e.se for e in ests])
    tr = np.array(truths)
    err = est - tr
    covered = np.array([e.ci_low <= t <= e.ci_high for e, t in zip(ests, truths, strict=True)])
    rejected = np.array([e.p_value < alpha for e in ests])
    r = len(ests)
    sd = float(est.std(ddof=1)) if r > 1 else float("nan")
    cov, rej = float(covered.mean()), float(rejected.mean())
    return ValidationResult(
        estimator=name,
        n_reps=r + n_failed,
        n_failed=n_failed,
        truth=float(tr.mean()),
        mean_estimate=float(est.mean()),
        bias=float(err.mean()),
        bias_mcse=float(err.std(ddof=1) / math.sqrt(r)) if r > 1 else float("nan"),
        rmse=float(np.sqrt(np.mean(err**2))),
        empirical_sd=sd,
        mean_se=float(se.mean()),
        se_ratio=float(se.mean() / sd) if sd and sd > 0 else float("nan"),
        coverage=cov,
        coverage_mcse=math.sqrt(cov * (1 - cov) / r),
        rejection_rate=rej,
        rejection_mcse=math.sqrt(rej * (1 - rej) / r),
        alpha=alpha,
        seconds=seconds,
    )


def run_validation(
    simulate: Callable[[int], SimulationDraw[D]],
    estimators: Mapping[str, Callable[[D], Estimate]],
    n_reps: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    max_failure_rate: float = 0.01,
) -> dict[str, ValidationResult]:
    """Run every estimator on the same ``n_reps`` simulated datasets.

    Args:
        simulate: ``seed -> SimulationDraw(data, truth)``. It must be deterministic in ``seed``.
        estimators: name -> function returning an :class:`Estimate`.
        n_reps: number of Monte Carlo replications.
        seed: master seed. Per-replication seeds come from ``numpy.random.SeedSequence``.
        alpha: level used for the coverage and rejection metrics.
        max_failure_rate: raise if an estimator errors on more than this share of datasets.
            Failures are never silently dropped; ``n_failed`` reports them.
    """
    if n_reps < 2:
        raise ValueError("n_reps must be >= 2")
    if not estimators:
        raise ValueError("pass at least one estimator")
    seeds = np.random.SeedSequence(seed).generate_state(n_reps, dtype=np.uint32)
    results: dict[str, list[Estimate]] = {k: [] for k in estimators}
    truths: dict[str, list[float]] = {k: [] for k in estimators}
    failures: dict[str, int] = dict.fromkeys(estimators, 0)
    timing: dict[str, float] = dict.fromkeys(estimators, 0.0)
    errors: dict[str, str] = {}

    for s in seeds:
        draw = simulate(int(s))
        for name, fn in estimators.items():
            t0 = time.perf_counter()
            try:
                est = fn(draw.data)
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError) as exc:
                failures[name] += 1
                errors.setdefault(name, repr(exc))
                continue
            finally:
                timing[name] += time.perf_counter() - t0
            results[name].append(est)
            truths[name].append(draw.truth)

    for name, n_fail in failures.items():
        if n_fail / n_reps > max_failure_rate:
            raise RuntimeError(
                f"estimator {name!r} failed on {n_fail}/{n_reps} replications "
                f"(first error: {errors.get(name)})"
            )
    return {
        name: _summarise(name, results[name], truths[name], failures[name], timing[name], alpha)
        for name in estimators
    }


def results_frame(results: Mapping[str, ValidationResult]) -> pd.DataFrame:
    """Tidy table of validation results, one row per estimator."""
    return pd.DataFrame([r.to_dict() for r in results.values()]).set_index("estimator")
