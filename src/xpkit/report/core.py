"""Experiment analysis core shared by the CLI report and the Streamlit app.

The order is fixed: **trust checks run first**. If any check fails, the report still
shows the effect estimates, but under a prominent warning that they are not
interpretable until the failure is explained.

Config schema (YAML or dict)::

    experiment: Checkout button test
    unit_id: user_id                 # optional; duplicates are flagged
    treatment_column: variant
    control: control                 # value of treatment_column for control
    treatment: treatment             # value for the treatment arm
    expected_split: {control: 0.5, treatment: 0.5}
    alpha: 0.05
    srm_threshold: 0.001
    multiplicity: holm               # holm | bh | bonferroni | none
    pre_experiment_covariates: [pre_revenue, tenure_days]   # balance check
    metrics:
      - {name: converted, type: binary, primary: true}
      - {name: revenue, type: continuous, cuped_covariate: pre_revenue}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xpkit.design import mde_means, mde_proportions
from xpkit.estimate import adjust_pvalues, cuped, diff_in_means, diff_in_proportions
from xpkit.trust import balance_table, srm_test

VALID_TYPES = {"binary", "continuous"}


class ConfigError(ValueError):
    """Raised for an invalid report configuration, with a message saying how to fix it."""


@dataclass
class TrustCheck:
    """One trust check. A failed ``critical`` check makes the report non-interpretable."""

    name: str
    passed: bool
    detail: str
    severity: str = "critical"  # critical checks block interpretation; "warning" does not


@dataclass
class MetricResult:
    """Effect estimate for one metric (treatment minus control), with adjusted p-value."""

    name: str
    type: str
    primary: bool
    control_mean: float
    treatment_mean: float
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    adjusted_p_value: float
    relative_lift: float | None
    method: str
    mde_80: float | None
    cuped: dict[str, Any] | None = None


@dataclass
class ReportData:
    """Everything a report shows: trust checks first, then per-metric results."""

    experiment: str
    config: dict[str, Any]
    n_rows: int
    counts: dict[str, int]
    checks: list[TrustCheck]
    balance: pd.DataFrame | None
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "critical")

    @property
    def headline(self) -> str:
        if not self.trustworthy:
            failed = ", ".join(
                c.name for c in self.checks if not c.passed and c.severity == "critical"
            )
            return (
                f"DO NOT INTERPRET: trust check failed ({failed}). "
                "Investigate before reading effects."
            )
        prim = [m for m in self.metrics if m.primary]
        if not prim:
            return "No primary metric configured."
        m = prim[0]
        if m.adjusted_p_value < self.config["alpha"]:
            direction = "increased" if m.estimate > 0 else "decreased"
            return (
                f"Primary metric '{m.name}' {direction} by {m.estimate:+.4g} "
                f"(CI {m.ci_low:+.4g} to {m.ci_high:+.4g}); significant after "
                f"{self.config['multiplicity']} correction."
            )
        return (
            f"No significant change in primary metric '{m.name}' "
            f"(estimate {m.estimate:+.4g}, CI {m.ci_low:+.4g} to {m.ci_high:+.4g}). "
            f"Smallest effect detectable at 80% power: {m.mde_80:.4g}."
        )


def validate_config(cfg: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Fill defaults and check the config against the data. Raises ConfigError."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping (YAML dictionary)")
    c = {
        "experiment": "Experiment",
        "alpha": 0.05,
        "srm_threshold": 0.001,
        "multiplicity": "holm",
        "pre_experiment_covariates": [],
        "unit_id": None,
        **cfg,
    }
    for key in ("treatment_column", "control", "treatment", "metrics"):
        if key not in c:
            raise ConfigError(f"config is missing required key '{key}'")
    tc = c["treatment_column"]
    if tc not in df.columns:
        raise ConfigError(f"treatment_column '{tc}' not found; columns are {list(df.columns)}")
    arms = set(df[tc].dropna().astype(str).unique())
    for arm in ("control", "treatment"):
        if str(c[arm]) not in arms:
            raise ConfigError(
                f"{arm} value '{c[arm]}' not found in column '{tc}' (found {sorted(arms)})"
            )
    split = c.get("expected_split") or {"control": 0.5, "treatment": 0.5}
    if set(split) != {"control", "treatment"} or any(v <= 0 for v in split.values()):
        raise ConfigError("expected_split must give positive shares for 'control' and 'treatment'")
    c["expected_split"] = split
    if not c["metrics"]:
        raise ConfigError("configure at least one metric")
    n_primary = 0
    for m in c["metrics"]:
        if "name" not in m or "type" not in m:
            raise ConfigError(f"each metric needs 'name' and 'type', got {m}")
        if m["name"] not in df.columns:
            raise ConfigError(f"metric column '{m['name']}' not found in the data")
        if m["type"] not in VALID_TYPES:
            raise ConfigError(
                f"metric '{m['name']}' has type '{m['type']}'; use one of {sorted(VALID_TYPES)}"
            )
        if m["type"] == "binary" and not set(pd.unique(df[m["name"]].dropna())) <= {0, 1}:
            raise ConfigError(f"binary metric '{m['name']}' must contain only 0/1")
        cov = m.get("cuped_covariate")
        if cov and cov not in df.columns:
            raise ConfigError(f"cuped_covariate '{cov}' for metric '{m['name']}' not found")
        n_primary += bool(m.get("primary"))
    if n_primary > 1:
        raise ConfigError("mark at most one metric as primary")
    if n_primary == 0:
        c["metrics"][0]["primary"] = True
    for cov in c["pre_experiment_covariates"]:
        if cov not in df.columns:
            raise ConfigError(f"pre_experiment_covariate '{cov}' not found")
    if c["multiplicity"] not in ("holm", "bh", "bonferroni", "none"):
        raise ConfigError("multiplicity must be one of holm, bh, bonferroni, none")
    if c["unit_id"] and c["unit_id"] not in df.columns:
        raise ConfigError(f"unit_id column '{c['unit_id']}' not found")
    return c


def analyze(df: pd.DataFrame, cfg: dict[str, Any]) -> ReportData:
    """Run trust checks, then effect estimation, on an experiment table (one row per unit)."""
    c = validate_config(cfg, df)
    tc = c["treatment_column"]
    arm = df[tc].astype(str)
    is_t, is_c = arm == str(c["treatment"]), arm == str(c["control"])
    other = int((~(is_t | is_c)).sum())
    d = df.loc[is_t | is_c].copy()
    d["_t"] = is_t[is_t | is_c].astype(int).to_numpy()
    counts = {"control": int((d._t == 0).sum()), "treatment": int((d._t == 1).sum())}

    checks: list[TrustCheck] = []
    split = c["expected_split"]
    srm = srm_test(
        [counts["control"], counts["treatment"]],
        [split["control"], split["treatment"]],
        threshold=c["srm_threshold"],
    )
    obs_share = counts["treatment"] / max(sum(counts.values()), 1)
    exp_share = split["treatment"] / (split["control"] + split["treatment"])
    checks.append(
        TrustCheck(
            "Sample ratio mismatch",
            srm.passed,
            f"observed treatment share {obs_share:.4f} vs designed {exp_share:.4f}; "
            f"chi-square p = {srm.p_value:.3g} (fail if < {c['srm_threshold']})",
        )
    )
    if c["unit_id"]:
        dup = int(d[c["unit_id"]].duplicated().sum())
        multi = int((d.groupby(c["unit_id"])["_t"].nunique() > 1).sum())
        checks.append(
            TrustCheck(
                "Duplicate units",
                dup == 0,
                f"{dup} duplicated unit ids; {multi} units appear in both arms",
            )
        )
    metric_cols = [m["name"] for m in c["metrics"]]
    n_missing = int(d[metric_cols].isna().any(axis=1).sum())
    checks.append(
        TrustCheck(
            "Missing metric values",
            n_missing == 0,
            f"{n_missing} rows with a missing metric value (dropped per metric)",
            severity="warning",
        )
    )
    if other:
        checks.append(
            TrustCheck(
                "Unexpected arms",
                False,
                f"{other} rows with other values in '{tc}' were excluded",
                "warning",
            )
        )
    balance = None
    if c["pre_experiment_covariates"]:
        bd = d.dropna(subset=c["pre_experiment_covariates"])
        balance = balance_table(bd, "_t", c["pre_experiment_covariates"])
        worst = float(balance.smd.abs().max())
        checks.append(
            TrustCheck(
                "Pre-experiment covariate balance",
                worst < 0.1,
                f"max |SMD| = {worst:.3f} across {len(balance)} covariates (fail if >= 0.1)",
            )
        )

    results: list[MetricResult] = []
    alpha = c["alpha"]
    for m in c["metrics"]:
        sub = d.dropna(subset=[m["name"]])
        y, t = sub[m["name"]].to_numpy(float), sub._t.to_numpy()
        nc = int((t == 0).sum())
        if m["type"] == "binary":
            e = diff_in_proportions(
                int(y[t == 1].sum()), int(t.sum()), int(y[t == 0].sum()), nc, alpha
            )
            mde = (
                mde_proportions(
                    float(y[t == 0].mean()),
                    nc,
                    alpha=alpha,
                    relative=False,
                    ratio=max(t.sum(), 1) / max(nc, 1),
                )
                if 0 < y[t == 0].mean() < 1
                else None
            )
        else:
            e = diff_in_means(y, t, alpha)
            mde = mde_means(
                float(y[t == 0].std(ddof=1)), nc, alpha=alpha, ratio=max(t.sum(), 1) / max(nc, 1)
            )
        cu = None
        cov = m.get("cuped_covariate")
        if cov:
            s2 = sub.dropna(subset=[cov])
            ce = cuped(
                s2[m["name"]].to_numpy(float), s2._t.to_numpy(), s2[cov].to_numpy(float), alpha
            )
            cu = {
                "covariate": cov,
                "estimate": ce.estimate,
                "ci_low": ce.ci_low,
                "ci_high": ce.ci_high,
                "p_value": ce.p_value,
                "se": ce.se,
                "se_unadjusted": e.se,
                **ce.extra,
            }
        m0, m1 = float(y[t == 0].mean()), float(y[t == 1].mean())
        results.append(
            MetricResult(
                name=m["name"],
                type=m["type"],
                primary=bool(m.get("primary")),
                control_mean=m0,
                treatment_mean=m1,
                estimate=e.estimate,
                ci_low=e.ci_low,
                ci_high=e.ci_high,
                p_value=e.p_value,
                adjusted_p_value=np.nan,
                relative_lift=(m1 / m0 - 1) if m0 else None,
                method=e.method,
                mde_80=mde,
                cuped=cu,
            )
        )
    # Multiplicity: use the CUPED p-value where configured (it's the reported test).
    raw = [r.cuped["p_value"] if r.cuped else r.p_value for r in results]
    adj = adjust_pvalues(raw, c["multiplicity"], alpha).adjusted
    for r, a in zip(results, adj, strict=True):
        r.adjusted_p_value = float(a)
    return ReportData(c["experiment"], c, len(df), counts, checks, balance, results)
