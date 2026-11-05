"""Unit tests for Stage 3 causal methods (reference checks + behaviour on simulated data)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from xpkit._ols import ols
from xpkit.causal import (
    aipw,
    did,
    e_value,
    effective_sample_size,
    event_study,
    fit_propensity,
    ipw,
    overlap_summary,
    rr_from_risks,
    trim_mask,
)
from xpkit.sim import ExperimentSpec, PanelSpec, simulate_experiment, simulate_panel
from xpkit.trust import balance_table

COVS = ["x0", "x1", "x2"]


@pytest.fixture(scope="module")
def confounded():
    return simulate_experiment(ExperimentSpec(n=8000, effect=0.5, confounding=0.8), seed=3)


# ----------------------------------------------------------------------------- OLS helper


@pytest.mark.parametrize("cov", ["classical", "HC1", "HC2"])
def test_ols_matches_statsmodels(cov):
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    X = sm.add_constant(rng.normal(size=(300, 2)))
    y = X @ [1, 2, -1] + rng.normal(size=300) * (1 + np.abs(X[:, 1]))
    ours = ols(y, X, cov=cov)
    ref = sm.OLS(y, X).fit(cov_type="nonrobust" if cov == "classical" else cov)
    assert ours.se == pytest.approx(ref.bse, rel=1e-8)


# ----------------------------------------------------------------------------- propensity


def test_propensity_scores_in_unit_interval_and_predictive(confounded):
    ps = fit_propensity(confounded.data, "treatment", COVS)
    assert ((ps > 0) & (ps < 1)).all()
    ov = overlap_summary(confounded.data.treatment.to_numpy(), ps)
    assert ov.auc > 0.75


@pytest.mark.slow
def test_gbm_propensity_is_cross_fitted_and_reasonable(confounded):
    ps = fit_propensity(confounded.data, "treatment", COVS, model="gbm")
    ov = overlap_summary(confounded.data.treatment.to_numpy(), np.clip(ps, 1e-6, 1 - 1e-6))
    assert 0.7 < ov.auc < 0.95


def test_weighting_restores_balance(confounded):
    d = confounded.data
    ps = fit_propensity(d, "treatment", COVS)
    t = d.treatment.to_numpy()
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    before = balance_table(d, "treatment", COVS)
    after = balance_table(d, "treatment", COVS, weights=w)
    assert before.smd.abs().max() > 0.3
    assert after.smd.abs().max() < 0.05


def test_ess_and_trim():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)
    assert effective_sample_size(np.r_[np.ones(9), 100.0]) < 2
    assert trim_mask(np.array([0.05, 0.5, 0.95])).tolist() == [False, True, False]


def test_overlap_flags_extreme_assignment():
    d = simulate_experiment(ExperimentSpec(n=5000, confounding=3.0), seed=0).data
    ps = np.clip(fit_propensity(d, "treatment", COVS), 1e-9, 1 - 1e-9)
    assert not overlap_summary(d.treatment.to_numpy(), ps).adequate


# ----------------------------------------------------------------------------- IPW / AIPW


def test_ipw_recovers_effect_naive_does_not(confounded):
    d = confounded.data
    naive = d.loc[d.treatment == 1, "y"].mean() - d.loc[d.treatment == 0, "y"].mean()
    est = ipw(d, "y", "treatment", COVS)
    assert abs(naive - 0.5) > 0.5
    assert est.covers(0.5)


@pytest.mark.slow
def test_ipw_sandwich_is_close_to_bootstrap(confounded):
    d = confounded.data
    sw = ipw(d, "y", "treatment", COVS, se="sandwich")
    bs = ipw(d, "y", "treatment", COVS, se="bootstrap", n_boot=150, seed=1)
    assert sw.estimate == pytest.approx(bs.estimate, abs=1e-6)
    assert sw.se == pytest.approx(bs.se, rel=0.2)


def test_ipw_att_and_validation(confounded):
    d = confounded.data
    est = ipw(d, "y", "treatment", COVS, estimand="att")
    assert est.covers(confounded.true_att)
    with pytest.raises(ValueError, match="logistic"):
        ipw(d, "y", "treatment", COVS, ps_model="gbm", se="sandwich")
    with pytest.raises(KeyError):
        ipw(d, "y", "treatment", ["nope"])


def test_aipw_is_doubly_robust_to_one_misspecified_model(confounded):
    d = confounded.data
    # Outcome model omits confounder x1, PS correct -> still consistent.
    a = aipw(d, "y", "treatment", ps_covariates=COVS, outcome_covariates=["x0", "x2"])
    # PS omits x1, outcome model correct -> still consistent.
    b = aipw(d, "y", "treatment", ps_covariates=["x0", "x2"], outcome_covariates=COVS)
    # Both omit x1 -> biased.
    c = aipw(d, "y", "treatment", ps_covariates=["x0", "x2"], outcome_covariates=["x0", "x2"])
    assert a.covers(0.5) and b.covers(0.5)
    assert not c.covers(0.5)


@pytest.mark.slow
def test_aipw_crossfit_gbm_runs(confounded):
    est = aipw(confounded.data, "y", "treatment", COVS, ps_model="gbm", outcome_model="gbm")
    assert est.extra["crossfit"]
    assert abs(est.estimate - 0.5) < 0.15


def test_unmeasured_confounding_biases_ipw():
    sim = simulate_experiment(
        ExperimentSpec(n=20_000, effect=0.5, unobserved_confounding=1.0), seed=4
    )
    est = ipw(sim.data, "y", "treatment", COVS)
    assert est.estimate - 0.5 > 0.5 and not est.covers(0.5)


# ----------------------------------------------------------------------------- E-value


def test_e_value_matches_vanderweele_ding_example():
    ev = e_value(3.9, 1.8, 8.7)
    assert ev.e_value_point == pytest.approx(7.26, abs=0.01)
    assert ev.e_value_ci == pytest.approx(3.0, abs=0.01)


def test_e_value_protective_and_null_ci():
    assert e_value(0.5).e_value_point == pytest.approx(e_value(2.0).e_value_point)
    assert e_value(1.3, 0.9, 1.8).e_value_ci == 1.0
    assert rr_from_risks(0.02, 0.01) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        e_value(-1)


# ----------------------------------------------------------------------------- DiD


def test_did_matches_statsmodels_dummy_regression():
    p = simulate_panel(PanelSpec(n_units=60, n_periods=6, treat_start=3), seed=0).data
    ours = did(p)
    ref = smf.ols("y ~ d + C(unit) + C(period)", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p["unit"]}
    )
    assert ours.estimate == pytest.approx(ref.params["d"])
    assert ours.se == pytest.approx(ref.bse["d"], rel=1e-8)


def test_did_equals_2x2_means():
    p = simulate_panel(PanelSpec(n_units=100), seed=1).data
    m = p.groupby(["treated", "post"]).y.mean()
    assert did(p).estimate == pytest.approx((m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0]))


def test_classical_se_understates_with_serial_correlation():
    p = simulate_panel(PanelSpec(n_units=200, ar_rho=0.8), seed=2).data
    assert did(p, se="classical").se < 0.8 * did(p).se


def test_event_study_detects_pretrend_violation():
    ok = event_study(simulate_panel(PanelSpec(n_units=400), seed=3).data)
    bad = event_study(simulate_panel(PanelSpec(n_units=400, trend_violation=0.15), seed=3).data)
    assert not ok.pretrend_rejected
    assert bad.pretrend_rejected
    assert len(ok.coefficients) == 9  # 10 periods minus the reference


def test_event_study_matches_statsmodels():
    p = simulate_panel(PanelSpec(n_units=50, n_periods=5, treat_start=3), seed=4).data
    es = event_study(p)
    p = p.copy()
    for e in es.coefficients.event_time:
        p[f"e{e + 10}"] = ((p.treated == 1) & (p.event_time == e)).astype(float)
    terms = " + ".join(f"e{e + 10}" for e in es.coefficients.event_time)
    ref = smf.ols(f"y ~ {terms} + C(unit) + C(period)", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p["unit"]}
    )
    names = [f"e{e + 10}" for e in es.coefficients.event_time]
    assert es.coefficients.estimate.to_numpy() == pytest.approx(ref.params[names].to_numpy())
    assert es.coefficients.se.to_numpy() == pytest.approx(ref.bse[names].to_numpy(), rel=1e-8)


def test_did_rejects_unbalanced_panel():
    p = simulate_panel(PanelSpec(n_units=10), seed=0).data.iloc[1:]
    with pytest.raises(ValueError, match="balanced"):
        did(p)
    q = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],
            "period": [0, 1, 0, 1],
            "treated": [0, 1, 0, 0],
            "post": [0, 1, 0, 1],
            "y": [1.0, 2, 3, 4],
        }
    )
    with pytest.raises(ValueError, match="time-invariant"):
        did(q)


# ----------------------------------------------------------------------------- clustering


def _clustered_copy(seed=5):
    """Every unit appears 4 times with a shared unit-level shock, as repeat sessions would."""
    sim = simulate_experiment(ExperimentSpec(n=2000, effect=0.5, confounding=0.8), seed=seed)
    d = pd.concat([sim.data] * 4, ignore_index=True)
    d["user"] = np.tile(np.arange(2000), 4)
    d["y"] = d["y"] + np.random.default_rng(seed).normal(0, 0.3, len(d))
    return d


def test_cluster_se_larger_than_iid_with_duplicated_units():
    d = _clustered_copy()
    for f in (ipw, aipw):
        iid, cl = f(d, "y", "treatment", COVS), f(d, "y", "treatment", COVS, cluster="user")
        assert cl.se > 1.6 * iid.se  # ~sqrt(4) when rows are near-duplicates
        assert cl.estimate == pytest.approx(iid.estimate)


def test_singleton_clusters_match_iid_se(confounded):
    d = confounded.data.assign(uid=np.arange(len(confounded.data)))
    for f in (ipw, aipw):
        a, b = f(d, "y", "treatment", COVS), f(d, "y", "treatment", COVS, cluster="uid")
        assert b.se == pytest.approx(a.se, rel=1e-3)


@pytest.mark.slow
def test_cluster_bootstrap_runs():
    d = _clustered_copy()
    e = ipw(d, "y", "treatment", COVS, se="bootstrap", n_boot=60, cluster="user", seed=2)
    assert e.se > 0 and e.extra["cluster"] == "user"


def test_aipw_reports_risk_ratio_for_binary_outcome():
    sim = simulate_experiment(
        ExperimentSpec(n=20_000, outcome="binary", baseline=0.05, effect=0.02, confounding=0.5),
        seed=1,
    )
    e = aipw(sim.data, "y", "treatment", COVS)
    x = e.extra
    assert x["rr_ci_low"] < x["risk_ratio"] < x["rr_ci_high"]
    assert x["mean_treated_po"] - x["mean_control_po"] == pytest.approx(e.estimate)
