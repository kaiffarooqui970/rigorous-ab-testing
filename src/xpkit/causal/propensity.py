"""Propensity-score estimation and overlap diagnostics.

The propensity score e(x) = P(T = 1 | X = x) is what IPW and AIPW reweight by. Two
diagnostics decide whether a weighted estimate is credible:

* **Overlap (positivity).** Every covariate profile must have some chance of either
  treatment. Scores piling up near 0 or 1 mean some units have no comparable
  counterparts. Their weights explode, and the estimate rests on extrapolation.
* **Balance after weighting.** The weighted covariate distributions should match
  (see :func:`xpkit.trust.balance_table`). This checks the propensity model, not the
  causal assumptions: balance on *measured* covariates says nothing about unmeasured ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from xpkit._validation import as_treatment, require_columns

Model = Literal["logistic", "gbm"]


def fit_propensity(
    df: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    model: Model = "logistic",
    seed: int = 0,
) -> NDArray[np.float64]:
    """Estimated propensity scores e(x) for every row.

    ``logistic`` is unpenalised logistic regression on standardized covariates (the
    standard choice, and what the M-estimation SEs assume). ``gbm`` is histogram
    gradient boosting, for non-linear assignment. Because it overfits in-sample, its
    scores are produced by 5-fold cross-fitting.

    Choose covariates that are *pre-treatment*. Adding a variable affected by
    treatment (a mediator or collider) biases every estimator downstream.
    """
    require_columns(df, [treatment, *covariates])
    if not covariates:
        raise ValueError("need at least one covariate for a propensity model")
    t = as_treatment(treatment, df[treatment].to_numpy())
    X = df[list(covariates)].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("covariates contain NaN/inf; impute or drop rows first")
    if model == "logistic":
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        clf = unpenalized_logistic()
        clf.fit((X - mu) / sd, t)
        return clf.predict_proba((X - mu) / sd)[:, 1]
    if model == "gbm":
        rng = np.random.default_rng(seed)
        folds = rng.integers(0, 5, len(t))
        ps = np.empty(len(t))
        for f in range(5):
            tr, te = folds != f, folds == f
            gbm = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, random_state=seed
            )
            gbm.fit(X[tr], t[tr])
            ps[te] = gbm.predict_proba(X[te])[:, 1]
        return ps
    raise ValueError(f"unknown propensity model {model!r}; use 'logistic' or 'gbm'")


@dataclass(frozen=True)
class OverlapSummary:
    ps_min_treated: float
    ps_max_control: float
    share_outside_05_95: float
    share_outside_01_99: float
    auc: float
    ess_treated: float
    ess_control: float
    max_weight_share: float
    n_treated: int
    n_control: int

    @property
    def adequate(self) -> bool:
        """Heuristic flag: fewer than 5% of units with scores outside [0.05, 0.95], and no
        single unit carrying more than 5% of the total weight."""
        return self.share_outside_05_95 < 0.05 and self.max_weight_share < 0.05

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["adequate"] = self.adequate
        return d


def ate_weights(t: NDArray[np.int_], ps: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(t == 1, 1 / ps, 1 / (1 - ps))


def att_weights(t: NDArray[np.int_], ps: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(t == 1, 1.0, ps / (1 - ps))


def effective_sample_size(w: NDArray[np.float64]) -> float:
    """Kish ESS = (sum w)^2 / sum w^2: how many equally weighted units the weights are worth."""
    return float(w.sum() ** 2 / np.sum(w**2))


def overlap_summary(t: NDArray[np.int_], ps: NDArray[np.float64]) -> OverlapSummary:
    """Overlap diagnostics for a set of propensity scores (ATE weights)."""
    t = as_treatment("treatment", t)
    ps = np.asarray(ps, dtype=float)
    if ps.shape != t.shape:
        raise ValueError("ps and treatment must have the same length")
    if np.any((ps <= 0) | (ps >= 1)):
        raise ValueError("propensity scores must be strictly inside (0, 1)")
    w = ate_weights(t, ps)
    return OverlapSummary(
        ps_min_treated=float(ps[t == 1].min()),
        ps_max_control=float(ps[t == 0].max()),
        share_outside_05_95=float(np.mean((ps < 0.05) | (ps > 0.95))),
        share_outside_01_99=float(np.mean((ps < 0.01) | (ps > 0.99))),
        auc=float(roc_auc_score(t, ps)),
        ess_treated=effective_sample_size(w[t == 1]),
        ess_control=effective_sample_size(w[t == 0]),
        max_weight_share=float(
            max(w[t == 1].max() / w[t == 1].sum(), w[t == 0].max() / w[t == 0].sum())
        ),
        n_treated=int((t == 1).sum()),
        n_control=int((t == 0).sum()),
    )


def trim_mask(ps: NDArray[np.float64], lower: float = 0.1, upper: float = 0.9) -> NDArray[np.bool_]:
    """Crump et al. (2009)-style trimming mask. Note that trimming **changes the estimand**:
    the result is the effect in the overlap population, not the full-population ATE."""
    if not 0 <= lower < upper <= 1:
        raise ValueError("need 0 <= lower < upper <= 1")
    return (ps >= lower) & (ps <= upper)


def unpenalized_logistic() -> LogisticRegression:
    """Plain (unpenalised) maximum-likelihood logistic regression.

    scikit-learn >= 1.8 deprecates ``penalty=None`` in favour of ``C=np.inf``. Older
    versions need ``penalty=None``.
    """
    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=np.inf, max_iter=1000)
    return LogisticRegression(penalty=None, max_iter=1000)
