"""Tests for Stage 4 uplift modeling."""

import numpy as np
import pytest

from xpkit.sim import ExperimentSpec, simulate_experiment
from xpkit.uplift import (
    Economics,
    TLearner,
    XLearner,
    choose_depth,
    pehe,
    policy_table,
    qini_coefficient_ci,
    uplift_by_bin,
    uplift_curve,
)

FEATS = ["x0", "x1", "x2", "x3"]


def _split(spec, seed=0):
    sim = simulate_experiment(spec, seed)
    d, tau = sim.data, sim.truth.tau.to_numpy()
    n = len(d) // 2
    return d.iloc[:n], d.iloc[n:], tau[n:]


def test_uplift_curve_endpoints_and_random_baseline():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 2000)
    y = rng.random(2000) < 0.1 + 0.05 * t
    c = uplift_curve(y, t, rng.random(2000))
    assert c.fraction[0] == 0 and c.fraction[-1] == 1
    assert c.qini[-1] == pytest.approx(c.random_qini[-1])


def test_constant_score_has_zero_qini_coefficient_in_expectation():
    sim = simulate_experiment(ExperimentSpec(n=20_000, effect=0.2), seed=1).data
    q = uplift_curve(sim.y, sim.treatment, np.zeros(len(sim))).qini_coefficient
    assert abs(q) < 0.02


def test_oracle_score_beats_random():
    sim = simulate_experiment(ExperimentSpec(n=20_000, effect=0.2, hte=0.5, n_covariates=4), seed=2)
    q = uplift_curve(sim.data.y, sim.data.treatment, sim.truth.tau).qini_coefficient
    assert q > 0.05


@pytest.mark.parametrize("Learner", [TLearner, XLearner])
def test_learners_recover_heterogeneity_continuous(Learner):
    spec = ExperimentSpec(n=12_000, effect=0.3, hte=0.5, n_covariates=4)
    tr, te, tau = _split(spec)
    m = Learner(outcome="continuous").fit(tr[FEATS], tr.treatment, tr.y)
    pred = m.predict(te[FEATS])
    assert np.corrcoef(pred, tau)[0, 1] > 0.8
    assert pehe(tau, pred) < pehe(tau, np.full_like(tau, pred.mean()))


def test_learners_binary_outcome_returns_probability_differences():
    spec = ExperimentSpec(
        n=20_000,
        outcome="binary",
        baseline=0.1,
        effect=0.03,
        hte=0.03,
        hte_form="step",
        n_covariates=4,
    )
    tr, te, _ = _split(spec)
    pred = TLearner(outcome="binary").fit(tr[FEATS], tr.treatment, tr.y).predict(te[FEATS])
    assert np.all(np.abs(pred) <= 1)


def test_unfitted_learner_and_bad_inputs_raise():
    with pytest.raises(RuntimeError, match="not fitted"):
        TLearner().predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TLearner(outcome="ordinal")
    with pytest.raises(ValueError, match="2-D"):
        TLearner().fit(np.zeros(10), np.r_[np.zeros(5), np.ones(5)], np.zeros(10))


def test_uplift_by_bin_shape_and_ordering():
    sim = simulate_experiment(ExperimentSpec(n=20_000, effect=0.2, hte=0.5), seed=3)
    tab = uplift_by_bin(sim.data.y, sim.data.treatment, sim.truth.tau, n_bins=5)
    assert len(tab) == 5 and tab.n.sum() == len(sim.data)
    assert tab.observed_uplift.iloc[0] > tab.observed_uplift.iloc[-1]


def test_qini_ci_contains_point():
    sim = simulate_experiment(ExperimentSpec(n=4000, effect=0.2, hte=0.5), seed=4)
    p, lo, hi = qini_coefficient_ci(sim.data.y, sim.data.treatment, sim.truth.tau, n_boot=50)
    assert lo <= p <= hi


def test_policy_table_economics():
    sim = simulate_experiment(ExperimentSpec(n=20_000, effect=0.2, hte=0.5), seed=5)
    econ = Economics(value_per_conversion=1.0, cost_per_treatment=0.2)
    tab = policy_table(sim.data.y, sim.data.treatment, sim.truth.tau, econ)
    row = tab.loc[tab.depth == 1.0].iloc[0]
    assert row.profit_per_user == pytest.approx(row.random_profit_per_user)
    # Targeting by true tau should beat random at a partial depth.
    half = tab.loc[tab.depth == 0.5].iloc[0]
    assert half.profit_per_user > half.random_profit_per_user
    assert econ.breakeven_uplift == pytest.approx(0.2)


def test_choose_depth_treats_nobody_when_nothing_pays():
    sim = simulate_experiment(ExperimentSpec(n=8000, effect=0.0), seed=6)
    d = sim.data
    econ = Economics(value_per_conversion=1.0, cost_per_treatment=5.0)
    half = len(d) // 2
    s = np.zeros(len(d))
    dec = choose_depth(
        (d.y[:half], d.treatment[:half], s[:half]), (d.y[half:], d.treatment[half:], s[half:]), econ
    )
    assert dec.chosen_depth == 0.0 and dec.test_profit_per_user == 0.0


def test_economics_validation():
    with pytest.raises(ValueError):
        Economics(value_per_conversion=-1, cost_per_treatment=0)
