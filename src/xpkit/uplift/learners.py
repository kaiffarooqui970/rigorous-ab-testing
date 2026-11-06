"""Meta-learners for conditional average treatment effects (CATE): tau(x) = E[Y(1) - Y(0) | X = x].

* :class:`TLearner`: fit separate outcome models on treated and control, then take
  tau(x) = m1(x) - m0(x). Simple, but each model sees only one arm, and errors in the
  two models don't cancel. It does badly when one arm is small.
* :class:`XLearner` (Künzel et al. 2019):
    1. fit m0 and m1 as in the T-learner;
    2. impute individual effects: D1 = Y - m0(X) for treated, D0 = m1(X) - Y for control;
    3. regress D1 on X (in treated) -> tau1, and D0 on X (in control) -> tau0;
    4. combine: tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x), with g = P(T=1|x).
  The arm with more data informs the imputation for the other. This helps when the
  allocation is unbalanced (e.g. 85/15).

Both accept any scikit-learn estimator factory. For a binary outcome, stage 1 uses a
classifier's ``predict_proba``, so tau is on the probability scale. The defaults are
histogram gradient boosting with conservative settings.

Fit the learners on *training* data only, and evaluate them on held-out data
(:mod:`xpkit.uplift.evaluation`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from xpkit._validation import as_1d_float, as_treatment, check_same_length

OutcomeKind = Literal["binary", "continuous"]


def default_regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    """Conservative gradient-boosting regressor used as the default base learner."""
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=seed
    )


def default_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    """Conservative gradient-boosting classifier used as the default base learner."""
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=seed
    )


def _prep(
    X: ArrayLike, t: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
    ta = as_treatment("treatment", t)
    ya = as_1d_float("y", y)
    check_same_length(X=Xa, treatment=ta, y=ya)
    return Xa, ta, ya


class _Base:
    def __init__(
        self,
        outcome: OutcomeKind = "binary",
        model: Callable[[], Any] | None = None,
        seed: int = 0,
    ) -> None:
        if outcome not in ("binary", "continuous"):
            raise ValueError("outcome must be 'binary' or 'continuous'")
        self.outcome = outcome
        self.seed = seed
        self._factory = model or (
            (lambda: default_classifier(seed))
            if outcome == "binary"
            else (lambda: default_regressor(seed))
        )
        self._fitted = False

    def _fit_outcome(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
        m = clone(self._factory())
        if self.outcome == "binary":
            if len(np.unique(y)) < 2:
                raise ValueError("an arm has no variation in the binary outcome; cannot fit")
            m.fit(X, y.astype(int))
        else:
            m.fit(X, y)
        return m

    def _mu(self, m: Any, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return m.predict_proba(X)[:, 1] if self.outcome == "binary" else m.predict(X)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, t, y) first")


class TLearner(_Base):
    """Two-model learner: tau(x) = m1(x) - m0(x)."""

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> TLearner:
        Xa, ta, ya = _prep(X, t, y)
        self.m0_ = self._fit_outcome(Xa[ta == 0], ya[ta == 0])
        self.m1_ = self._fit_outcome(Xa[ta == 1], ya[ta == 1])
        self._fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        Xa = np.asarray(X, dtype=float)
        return self._mu(self.m1_, Xa) - self._mu(self.m0_, Xa)


class XLearner(_Base):
    """X-learner. ``propensity`` defaults to the observed treated share (randomized data)."""

    def fit(
        self, X: ArrayLike, t: ArrayLike, y: ArrayLike, propensity: float | None = None
    ) -> XLearner:
        Xa, ta, ya = _prep(X, t, y)
        self.m0_ = self._fit_outcome(Xa[ta == 0], ya[ta == 0])
        self.m1_ = self._fit_outcome(Xa[ta == 1], ya[ta == 1])
        d1 = ya[ta == 1] - self._mu(self.m0_, Xa[ta == 1])
        d0 = self._mu(self.m1_, Xa[ta == 0]) - ya[ta == 0]
        self.tau1_ = clone(default_regressor(self.seed)).fit(Xa[ta == 1], d1)
        self.tau0_ = clone(default_regressor(self.seed)).fit(Xa[ta == 0], d0)
        g = float(ta.mean()) if propensity is None else float(propensity)
        if not 0 < g < 1:
            raise ValueError("propensity must be in (0, 1)")
        self.g_ = g
        self._fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        Xa = np.asarray(X, dtype=float)
        return self.g_ * self.tau0_.predict(Xa) + (1 - self.g_) * self.tau1_.predict(Xa)
