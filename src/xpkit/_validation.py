"""Small input-validation helpers with helpful error messages."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def check_probability(name: str, value: float, *, open_interval: bool = True) -> None:
    ok = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not ok:
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {bounds}, got {value!r}")


def check_positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def as_1d_float(name: str, x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} contains NaN or infinite values ({n_bad} rows)")
    return arr


def as_treatment(name: str, t: ArrayLike) -> NDArray[np.int_]:
    arr = np.asarray(t)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    uniq = set(np.unique(arr).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"{name} must be binary 0/1, found values {sorted(uniq)[:5]}")
    if len(uniq) < 2:
        raise ValueError(f"{name} contains only one group ({uniq}); need both treated and control")
    return arr.astype(int)


def check_same_length(**arrays: Iterable[object] | NDArray[np.generic]) -> None:
    lengths = {k: len(v) for k, v in arrays.items()}  # type: ignore[arg-type]
    if len(set(lengths.values())) > 1:
        raise ValueError(f"inputs must have the same length, got {lengths}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], what: str = "data") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{what} is missing required column(s) {missing}; available: {list(df.columns)}"
        )
