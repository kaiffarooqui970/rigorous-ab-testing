"""Sample ratio mismatch (SRM) check.

If units were meant to be split 50/50 and the observed split is 50.6/49.4 on a million
users, something in assignment, logging or triggering is broken. The effect estimate
can't be trusted until the cause is found. The test is a Pearson chi-square
goodness-of-fit test of the observed arm counts against the designed allocation.

The conventional SRM threshold is strict (p < 0.001). The check runs on every
experiment, and a false SRM alarm is expensive, but a real SRM is usually extreme
enough to clear any threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    observed: list[int]
    expected: list[float]
    expected_ratios: list[float]
    chi2: float
    dof: int
    p_value: float
    threshold: float

    @property
    def passed(self) -> bool:
        """True when there is *no* evidence of a sample ratio mismatch."""
        return self.p_value >= self.threshold

    @property
    def observed_ratios(self) -> list[float]:
        total = sum(self.observed)
        return [o / total for o in self.observed]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        d["observed_ratios"] = self.observed_ratios
        return d


def srm_test(
    counts: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test of arm sizes against the designed split.

    Args:
        counts: observed units per arm, e.g. ``[control_n, treatment_n]``.
        expected_ratios: designed allocation. It is normalised to sum to 1. Defaults to equal.
        threshold: p-value below which SRM is declared (default 0.001).

    Example:
        >>> srm_test([50_000, 50_600]).passed
        True
        >>> srm_test([50_000, 51_500]).passed
        False
    """
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("counts must list at least two arms")
    if np.any(obs < 0) or not np.all(np.isfinite(obs)):
        raise ValueError(f"counts must be non-negative and finite, got {list(counts)}")
    if obs.sum() == 0:
        raise ValueError("all arm counts are zero")
    if expected_ratios is None:
        ratios = np.full(obs.size, 1.0 / obs.size)
    else:
        ratios = np.asarray(expected_ratios, dtype=float)
        if ratios.shape != obs.shape:
            raise ValueError(f"expected_ratios has {ratios.size} entries but counts has {obs.size}")
        if np.any(ratios <= 0):
            raise ValueError("expected_ratios must all be positive")
        ratios = ratios / ratios.sum()
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")

    expected = obs.sum() * ratios
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    dof = obs.size - 1
    p = float(stats.chi2.sf(chi2, dof))
    return SRMResult(
        observed=[int(o) for o in obs],
        expected=expected.tolist(),
        expected_ratios=ratios.tolist(),
        chi2=chi2,
        dof=dof,
        p_value=p,
        threshold=threshold,
    )
