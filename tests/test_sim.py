"""Tests for the simulation engine: does it generate what it claims to?"""

import numpy as np
import pytest

from xpkit.sim import (
    ExperimentSpec,
    PanelSpec,
    SimulationDraw,
    population_truth,
    run_validation,
    simulate_experiment,
    simulate_panel,
)
from xpkit.types import Estimate, normal_estimate


def test_same_seed_same_data_different_seed_different_data():
    spec = ExperimentSpec(n=500, effect=0.3)
    a = simulate_experiment(spec, seed=11).data
    b = simulate_experiment(spec, seed=11).data
    c = simulate_experiment(spec, seed=12).data
    assert a.equals(b)
    assert not a.equals(c)


def test_observed_data_hides_truth_columns():
    sim = simulate_experiment(ExperimentSpec(n=200), seed=0)
    assert {"tau", "mu0", "mu1", "u"}.isdisjoint(sim.data.columns)
    assert list(sim.data.columns[:4]) == ["unit_id", "treatment", "y", "pre_metric"]
    assert len(sim.data) == len(sim.truth)


def test_continuous_truth_is_exact_without_hte():
    assert population_truth(ExperimentSpec(effect=0.25)) == (0.25, 0.25)


def test_continuous_randomized_difference_in_means_close_to_truth():
    sim = simulate_experiment(ExperimentSpec(n=200_000, effect=0.4), seed=3)
    d = sim.data
    diff = d.loc[d.treatment == 1, "y"].mean() - d.loc[d.treatment == 0, "y"].mean()
    assert diff == pytest.approx(0.4, abs=0.03)


def test_binary_baseline_is_calibrated_and_outcome_is_binary():
    spec = ExperimentSpec(
        n=300_000, outcome="binary", baseline=0.05, effect=0.01, covariate_effect=1.0
    )
    sim = simulate_experiment(spec, seed=4)
    d = sim.data
    assert set(np.unique(d.y)) <= {0.0, 1.0}
    assert d.loc[d.treatment == 0, "y"].mean() == pytest.approx(0.05, abs=0.002)
    # No clipping happens at these values, so the truth equals the nominal effect.
    assert sim.true_ate == pytest.approx(0.01, abs=1e-3)


def test_binary_truth_accounts_for_clipping():
    # A -0.05 effect on a 5% base rate can't apply in full to low-propensity units.
    ate, _ = population_truth(ExperimentSpec(outcome="binary", baseline=0.05, effect=-0.05))
    assert -0.05 < ate < -0.02


def test_pre_metric_correlation_increases_with_covariate_effect():
    weak = simulate_experiment(ExperimentSpec(n=20_000, covariate_effect=0.1), seed=0).data
    strong = simulate_experiment(ExperimentSpec(n=20_000, covariate_effect=2.0), seed=0).data
    # x1 (coef 0.5) also enters both, so the analytic weak-case corr is .26/sqrt(1.26*.51) ~= .32
    assert np.corrcoef(weak.y, weak.pre_metric)[0, 1] < 0.4
    assert np.corrcoef(strong.y, strong.pre_metric)[0, 1] > 0.8


def test_hte_lives_on_x2_and_averages_to_effect():
    sim = simulate_experiment(ExperimentSpec(n=50_000, effect=0.5, hte=1.0), seed=1)
    tau, x2 = sim.truth.tau.to_numpy(), sim.data.x2.to_numpy()
    assert np.corrcoef(tau, x2)[0, 1] > 0.99
    assert tau.mean() == pytest.approx(0.5, abs=0.02)
    assert sim.true_ate == pytest.approx(0.5, abs=0.005)


def test_step_hte_has_two_effect_levels():
    sim = simulate_experiment(ExperimentSpec(n=1000, effect=0.5, hte=0.5, hte_form="step"), seed=1)
    assert set(np.round(sim.truth.tau.unique(), 9)) == {0.0, 1.0}


def test_random_srm_drops_expected_share_of_treated():
    spec = ExperimentSpec(n=100_000, srm_drop=0.1)
    d = simulate_experiment(spec, seed=0).data
    share = d.treatment.mean()
    assert share == pytest.approx(0.45 / 0.95, abs=0.005)


def test_selective_srm_skews_treated_covariates():
    d = simulate_experiment(
        ExperimentSpec(n=50_000, srm_drop=0.2, srm_mechanism="selective"), seed=0
    ).data
    assert d.loc[d.treatment == 1, "x0"].mean() > d.loc[d.treatment == 0, "x0"].mean() + 0.1


def test_confounding_creates_covariate_imbalance_and_biases_naive_diff():
    sim = simulate_experiment(ExperimentSpec(n=100_000, effect=0.0, confounding=1.0), seed=0)
    d = sim.data
    assert d.loc[d.treatment == 1, "x0"].mean() - d.loc[d.treatment == 0, "x0"].mean() > 0.5
    naive = d.loc[d.treatment == 1, "y"].mean() - d.loc[d.treatment == 0, "y"].mean()
    assert naive > 0.5  # true effect is 0


def test_att_differs_from_ate_when_hte_correlates_with_uptake():
    # Treatment uptake depends on x0 + x1, HTE on x2: independent, so ATT ~= ATE.
    ate, att = population_truth(ExperimentSpec(effect=0.5, hte=1.0, confounding=1.0))
    assert ate == pytest.approx(att, abs=0.01)


@pytest.mark.parametrize(
    "bad",
    [
        {"n": 1},
        {"treat_prob": 1.0},
        {"srm_drop": 1.0},
        {"n_covariates": 2},
        {"outcome": "binary", "baseline": 2.0},
        {"hte_form": "quadratic"},
    ],
)
def test_spec_validation_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        ExperimentSpec(**bad)


# ----------------------------------------------------------------------------- panel


def test_panel_shape_and_columns():
    p = simulate_panel(PanelSpec(n_units=50, n_periods=8, treat_start=5), seed=0)
    assert len(p.data) == 50 * 8
    assert {"unit", "period", "treated", "post", "d", "y", "event_time"} <= set(p.data.columns)
    assert (p.data.d == p.data.treated * p.data.post).all()


def test_panel_ramp_effects_average_to_effect():
    spec = PanelSpec(effect=2.0, dynamic="ramp", n_periods=10, treat_start=6)
    p = simulate_panel(spec, seed=0)
    assert p.true_att == pytest.approx(2.0)
    effects = list(p.event_effects.values())
    assert effects == sorted(effects)


def test_panel_2x2_did_recovers_effect_with_parallel_trends():
    p = simulate_panel(PanelSpec(n_units=4000, effect=1.0, level_gap=3.0), seed=2).data
    m = p.groupby(["treated", "post"]).y.mean()
    did = (m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0])
    assert did == pytest.approx(1.0, abs=0.08)


def test_panel_trend_violation_biases_2x2_did_by_known_amount():
    spec = PanelSpec(n_units=4000, effect=1.0, trend_violation=0.2, n_periods=10, treat_start=6)
    p = simulate_panel(spec, seed=2).data
    m = p.groupby(["treated", "post"]).y.mean()
    did = (m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0])
    # Bias = delta * (mean post event time - mean pre event time) = 0.2 * (1.5 - (-3.5)) = 1.0
    assert did == pytest.approx(2.0, abs=0.1)


def test_panel_rejects_no_pre_period():
    with pytest.raises(ValueError, match="treat_start"):
        PanelSpec(treat_start=0)


# ----------------------------------------------------------------------------- harness


def _welch(data):
    y1, y0 = data.loc[data.treatment == 1, "y"], data.loc[data.treatment == 0, "y"]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return normal_estimate(y1.mean() - y0.mean(), se, "welch")


def _draw(spec):
    def f(seed):
        sim = simulate_experiment(spec, seed)
        return SimulationDraw(sim.data, sim.true_ate)

    return f


def test_harness_is_deterministic():
    spec = ExperimentSpec(n=200, effect=0.1)
    a = run_validation(_draw(spec), {"w": _welch}, n_reps=20, seed=5)["w"]
    b = run_validation(_draw(spec), {"w": _welch}, n_reps=20, seed=5)["w"]
    assert a.mean_estimate == b.mean_estimate


def test_harness_metrics_for_known_unbiased_estimator():
    res = run_validation(
        _draw(ExperimentSpec(n=400, effect=0.0)), {"w": _welch}, n_reps=400, seed=1
    )["w"]
    assert abs(res.bias) < 4 * res.bias_mcse
    assert res.coverage_ok()
    assert res.rejection_rate == pytest.approx(0.05, abs=0.035)
    assert res.se_ratio == pytest.approx(1.0, abs=0.12)


def test_harness_reports_power_when_effect_nonzero():
    res = run_validation(
        _draw(ExperimentSpec(n=400, effect=0.5)), {"w": _welch}, n_reps=100, seed=1
    )["w"]
    assert res.rejection_rate > 0.9


def test_harness_detects_biased_estimator():
    def biased(d):
        e = _welch(d)
        return normal_estimate(e.estimate + 0.3, e.se, "biased")

    res = run_validation(_draw(ExperimentSpec(n=400)), {"b": biased}, n_reps=100, seed=1)["b"]
    assert res.bias == pytest.approx(0.3, abs=0.05)
    assert not res.coverage_ok()


def test_harness_surfaces_failures_instead_of_hiding_them():
    def broken(_d) -> Estimate:
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="failed on 10/10"):
        run_validation(_draw(ExperimentSpec(n=100)), {"x": broken}, n_reps=10)
