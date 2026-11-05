"""Multiple-testing corrections across metrics (or arms, or segments).

* Bonferroni and Holm control the **family-wise error rate** (FWER): the chance of
  *any* false positive. Holm is uniformly more powerful than Bonferroni and needs no
  extra assumptions, so use it as the default for a small set of decision metrics.
* Benjamini-Hochberg controls the **false discovery rate** (FDR): the expected share of
  false positives among the discoveries. It is valid under independence or positive
  dependence (PRDS). Use it for exploratory metric sets or segment scans.

Each function returns *adjusted* p-values. Compare them with the same alpha you would
have used for a single test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Method = Literal["bonferroni", "holm", "bh", "none"]


def _check(p: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("p-values must be a non-empty 1-D array")
    if np.any((arr < 0) | (arr > 1)) or not np.all(np.isfinite(arr)):
        raise ValueError("p-values must lie in [0, 1]")
    return arr


def bonferroni(p: ArrayLike) -> NDArray[np.float64]:
    arr = _check(p)
    return np.minimum(arr * arr.size, 1.0)


def holm(p: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down: multiply the i-th smallest p by (m - i), then enforce monotonicity."""
    arr = _check(p)
    m = arr.size
    order = np.argsort(arr)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * arr[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def benjamini_hochberg(p: ArrayLike) -> NDArray[np.float64]:
    """BH step-up: the i-th smallest is multiplied by m / i, with a running min from the top."""
    arr = _check(p)
    m = arr.size
    order = np.argsort(arr)
    ranked = arr[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


@dataclass(frozen=True)
class AdjustedPValues:
    method: str
    raw: list[float]
    adjusted: list[float]
    reject: list[bool]
    alpha: float


def adjust_pvalues(p: ArrayLike, method: Method = "holm", alpha: float = 0.05) -> AdjustedPValues:
    """Adjust p-values with ``method`` and flag which hypotheses are rejected at ``alpha``."""
    funcs = {
        "bonferroni": bonferroni,
        "holm": holm,
        "bh": benjamini_hochberg,
        "none": lambda x: _check(x).copy(),
    }
    if method not in funcs:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(funcs)}")
    raw = _check(p)
    adj = funcs[method](raw)
    return AdjustedPValues(
        method=method,
        raw=raw.tolist(),
        adjusted=adj.tolist(),
        reject=(adj < alpha).tolist(),
        alpha=alpha,
    )
