"""Minimal OLS with heteroskedasticity- and cluster-robust sandwich variances.

The sandwich is written out explicitly so it stays readable. Each variant is checked
against statsmodels in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Cov = Literal["classical", "HC1", "HC2", "cluster"]


@dataclass(frozen=True)
class OLSFit:
    beta: NDArray[np.float64]
    vcov: NDArray[np.float64]
    resid: NDArray[np.float64]
    n: int
    k: int
    n_clusters: int | None

    @property
    def se(self) -> NDArray[np.float64]:
        return np.sqrt(np.diag(self.vcov))


def ols(
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    cov: Cov = "HC1",
    cluster: NDArray[np.generic] | None = None,
    extra_dof: int = 0,
) -> OLSFit:
    """Fit y = X b + e and return b with the requested covariance.

    Args:
        cov: "classical" (iid errors), "HC1", "HC2" or "cluster" (CR1, which needs ``cluster``).
        extra_dof: parameters absorbed before the fit (e.g. fixed effects removed by
            demeaning). They are added to k in the small-sample correction.
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    k_eff = k + extra_dof
    g: int | None = None
    if cov == "classical":
        sigma2 = resid @ resid / (n - k_eff)
        vcov = sigma2 * xtx_inv
    elif cov == "HC1":
        meat = (X * resid[:, None] ** 2).T @ X
        vcov = n / (n - k_eff) * xtx_inv @ meat @ xtx_inv
    elif cov == "HC2":
        h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        meat = (X * (resid**2 / np.clip(1 - h, 1e-12, None))[:, None]).T @ X
        vcov = xtx_inv @ meat @ xtx_inv
    elif cov == "cluster":
        if cluster is None:
            raise ValueError("cov='cluster' requires a cluster array")
        _, inv = np.unique(cluster, return_inverse=True)
        g = int(np.max(inv)) + 1
        if g < 2:
            raise ValueError("need at least two clusters for a cluster-robust SE")
        scores = np.zeros((g, k))
        np.add.at(scores, inv, X * resid[:, None])
        adj = (g / (g - 1)) * ((n - 1) / (n - k_eff))
        vcov = adj * xtx_inv @ (scores.T @ scores) @ xtx_inv
    else:
        raise ValueError(f"unknown cov type {cov!r}")
    return OLSFit(beta=beta, vcov=vcov, resid=resid, n=n, k=k, n_clusters=g)
