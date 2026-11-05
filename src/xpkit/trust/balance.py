"""Covariate balance diagnostics (standardized mean differences).

Used in two places:

* randomized experiments: pre-treatment covariates should be balanced. Large SMDs
  point to broken randomization, or to a "pre-treatment" variable that is really
  affected by treatment.
* observational studies: balance before and after propensity weighting shows
  whether the weighting did its job, *for the measured covariates only*.

The SMD uses the pooled *unweighted* SD in the denominator (Austin 2009 convention),
so weighted and unweighted SMDs share a scale. |SMD| < 0.1 is the usual
"negligible imbalance" rule of thumb.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from xpkit._validation import as_treatment, require_columns


def _wmean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = float(np.average(x, weights=w))
    v = float(np.average((x - m) ** 2, weights=w))
    return m, v


def balance_table(
    df: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    weights: ArrayLike | None = None,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Per-covariate means, standardized mean difference and variance ratio.

    Args:
        df: data frame containing the treatment and covariate columns.
        treatment: name of the 0/1 treatment column.
        covariates: numeric covariate columns to check.
        weights: optional unit weights (e.g. IPW weights). If omitted, weights are 1.
        threshold: |SMD| above which a covariate is flagged as imbalanced.

    Returns:
        DataFrame indexed by covariate with columns
        ``mean_treated, mean_control, smd, variance_ratio, imbalanced``.
    """
    require_columns(df, [treatment, *covariates])
    t = as_treatment(treatment, df[treatment].to_numpy())
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(df),):
        raise ValueError(f"weights must have length {len(df)}, got shape {w.shape}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")

    rows = []
    for c in covariates:
        x = df[c].to_numpy(dtype=float)
        m1, v1 = _wmean_var(x[t == 1], w[t == 1])
        m0, v0 = _wmean_var(x[t == 0], w[t == 0])
        pooled_sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
        smd = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "covariate": c,
                "mean_treated": m1,
                "mean_control": m0,
                "smd": smd,
                "variance_ratio": v1 / v0 if v0 > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("covariate")
    out["imbalanced"] = out["smd"].abs() > threshold
    return out
