"""Unit tests for Stage 2: trust checks, design, effects, CUPED, multiplicity, sequential.

Where a trusted reference implementation exists (scipy, statsmodels, published tables),
xpkit's hand-written estimator is checked against it.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from ab_test_analysis.stats_tests import two_proportion_ztest
from xpkit.design import (
    mde_means,
    mde_proportions,
    power_means,
    sample_size_means,
    sample_size_proportions,
)
from xpkit.estimate import (
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    cuped,
    cuped_theta,
    diff_in_means,
    diff_in_proportions,
    group_sequential_design,
    holm,
    lin_regression_adjustment,
    relative_lift,
    sequential_decision,
)
from xpkit.sim import ExperimentSpec, simulate_experiment
from xpkit.sim.peeking import simulate_peeking_strategies
from xpkit.trust import balance_table, srm_test

# ----------------------------------------------------------------------------- SRM


def test_srm_matches_scipy_chisquare():
    res = srm_test([50_000, 51_000])
    ref = stats.chisquare([50_000, 51_000])
    assert res.chi2 == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)


def test_srm_flags_real_mismatch_and_passes_balanced_split():
    assert srm_test([50_000, 50_100]).passed
    assert not srm_test([50_000, 51_500]).passed


def test_srm_respects_unequal_design():
    # 20/80 design observed as 20/80 is fine; tested against 50/50 it is a glaring SRM.
    assert srm_test([20_000, 80_000], [0.2, 0.8]).passed
    assert not srm_test([20_000, 80_000]).passed


def test_srm_three_arms():
    res = srm_test([21_300, 21_300, 21_400])
    assert res.dof == 2 and res.passed


@pytest.mark.parametrize("bad", [[10], [-1, 5], [0, 0]])
def test_srm_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        srm_test(bad)


# ----------------------------------------------------------------------------- balance


def test_balance_table_smd_matches_hand_calculation():
    df = pd.DataFrame({"t": [1, 1, 1, 0, 0, 0], "x": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0]})
    tab = balance_table(df, "t", ["x"])
    # means 4 vs 2, both sample SDs 1 -> SMD 2
    assert tab.loc["x", "smd"] == pytest.approx(2.0)
    assert bool(tab.loc["x", "imbalanced"])


def test_balance_table_randomized_data_is_balanced():
    d = simulate_experiment(ExperimentSpec(n=20_000), seed=0).data
    tab = balance_table(d, "treatment", ["x0", "x1", "x2", "pre_metric"])
    assert (tab.smd.abs() < 0.05).all()


# ----------------------------------------------------------------------------- design


def test_sample_size_proportions_close_to_statsmodels_cohens_h():
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    ours = sample_size_proportions(0.05, 0.10).n_control
    ref = NormalIndPower().solve_power(
        abs(proportion_effectsize(0.05, 0.055)), alpha=0.05, power=0.8
    )
    assert ours == pytest.approx(ref, rel=0.01)


def test_sample_size_proportions_absolute_and_relative_agree():
    a = sample_size_proportions(0.1, 0.01, relative=False)
    r = sample_size_proportions(0.1, 0.10, relative=True)
    assert a.n_control == r.n_control


def test_unequal_allocation_needs_more_total_units():
    eq = sample_size_proportions(0.05, 0.1)
    uneq = sample_size_proportions(0.05, 0.1, ratio=3.0)
    assert uneq.n_total > eq.n_total


def test_cuped_rho_shrinks_sample_size_by_one_minus_rho_squared():
    base = sample_size_means(sd=10, mde=0.5)
    adj = sample_size_means(sd=10, mde=0.5, cuped_rho=0.6)
    assert adj.n_control / base.n_control == pytest.approx(0.64, rel=0.01)


def test_mde_and_sample_size_are_inverses():
    n = sample_size_means(sd=2.0, mde=0.1).n_control
    assert mde_means(2.0, n) == pytest.approx(0.1, rel=0.01)
    assert power_means(2.0, 0.1, n) == pytest.approx(0.8, abs=0.005)
    n_p = sample_size_proportions(0.03, 0.05).n_control
    assert mde_proportions(0.03, n_p) == pytest.approx(0.05, rel=0.01)


def test_design_input_validation():
    with pytest.raises(ValueError):
        sample_size_proportions(0.0, 0.1)
    with pytest.raises(ValueError):
        sample_size_proportions(0.9, 0.5)  # target rate > 1
    with pytest.raises(ValueError):
        sample_size_means(sd=1, mde=0)


# ----------------------------------------------------------------------------- effects


def test_diff_in_proportions_ci_matches_statsmodels_newcombe():
    e = diff_in_proportions(130, 2000, 100, 2000)
    lo, hi = confint_proportions_2indep(130, 2000, 100, 2000, method="newcomb", compare="diff")
    assert (e.ci_low, e.ci_high) == pytest.approx((lo, hi), abs=1e-10)


def test_diff_in_proportions_pvalue_matches_legacy_ztest():
    e = diff_in_proportions(856, 208_725, 10_528, 664_479)
    legacy = two_proportion_ztest(10_528, 664_479, 856, 208_725)
    assert e.estimate == pytest.approx(legacy.diff)
    assert e.p_value == pytest.approx(legacy.p_value, rel=1e-6, abs=1e-300)
    # z = -41 here: the true p (~1e-368) is below the smallest double, so 0.0 is correct.


def test_p_value_no_longer_underflows_for_moderately_large_z():
    # z ~ 10: legacy 2*(1-cdf) returned exactly 0; norm.sf keeps ~1e-23.
    legacy = two_proportion_ztest(1000, 20_000, 1400, 20_000)
    ours = diff_in_proportions(1400, 20_000, 1000, 20_000)
    assert 0 < legacy.p_value < 1e-15
    assert ours.p_value == pytest.approx(legacy.p_value, rel=1e-6)


def test_diff_in_proportions_handles_zero_successes():
    e = diff_in_proportions(0, 500, 3, 500)
    assert np.isfinite([e.ci_low, e.ci_high]).all()
    assert e.ci_low < e.estimate < e.ci_high


def test_diff_in_means_matches_scipy_welch():
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(1.0, 2, 300), rng.normal(0.6, 1, 500)]
    t = np.r_[np.ones(300), np.zeros(500)]
    e = diff_in_means(y, t)
    ref = stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False)
    assert e.p_value == pytest.approx(ref.pvalue, rel=1e-8)


def test_cluster_se_matches_statsmodels():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 5)
    t = np.repeat(rng.integers(0, 2, 200), 5)
    y = 0.3 * t + np.repeat(rng.normal(0, 1, 200), 5) + rng.normal(0, 1, 1000)
    e = diff_in_means(y, t, cluster=g)
    ref = sm.OLS(y, sm.add_constant(t)).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert e.se == pytest.approx(ref.bse[1], rel=1e-8)
    assert e.se > diff_in_means(y, t).se * 1.5  # ignoring clustering understates the SE


def test_relative_lift_delta_method():
    rng = np.random.default_rng(2)
    y = np.r_[rng.normal(11, 1, 5000), rng.normal(10, 1, 5000)]
    t = np.r_[np.ones(5000), np.zeros(5000)]
    e = relative_lift(y, t)
    assert e.estimate == pytest.approx(0.1, abs=0.01)
    assert e.covers(0.1)


def test_effects_input_validation():
    with pytest.raises(ValueError, match="binary"):
        diff_in_means([1.0, 2.0, 3.0], [0, 1, 2])
    with pytest.raises(ValueError, match="one group"):
        diff_in_means([1.0, 2.0], [1, 1])
    with pytest.raises(ValueError, match="NaN"):
        diff_in_means([1.0, np.nan, 2.0, 3.0], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        diff_in_proportions(11, 10, 1, 10)


# ----------------------------------------------------------------------------- CUPED


def test_cuped_theta_is_ols_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(size=1000)
    y = 2.5 * x + rng.normal(size=1000)
    assert cuped_theta(y, x) == pytest.approx(np.polyfit(x, y, 1)[0])


def test_cuped_variance_reduction_close_to_rho_squared():
    d = simulate_experiment(ExperimentSpec(n=50_000, effect=0.1, covariate_effect=2.0), seed=0).data
    e = cuped(d.y, d.treatment, d.pre_metric)
    assert e.extra["variance_reduction"] == pytest.approx(e.extra["rho"] ** 2, abs=0.01)
    assert e.se < diff_in_means(d.y, d.treatment).se * 0.6


def test_cuped_with_useless_covariate_matches_unadjusted():
    rng = np.random.default_rng(4)
    d = simulate_experiment(ExperimentSpec(n=20_000, effect=0.1), seed=1).data
    noise = rng.normal(size=len(d))
    e, raw = cuped(d.y, d.treatment, noise), diff_in_means(d.y, d.treatment)
    assert e.se == pytest.approx(raw.se, rel=0.01)


def test_lin_adjustment_matches_statsmodels_hc2():
    d = simulate_experiment(ExperimentSpec(n=2000, effect=0.2, hte=0.5), seed=2).data
    X = d[["x0", "x1", "x2"]].to_numpy()
    e = lin_regression_adjustment(d.y, d.treatment, X)
    t = d.treatment.to_numpy(float)
    Xc = X - X.mean(0)
    D = sm.add_constant(np.column_stack([t, Xc, Xc * t[:, None]]))
    ref = sm.OLS(d.y.to_numpy(), D).fit(cov_type="HC2")
    assert e.estimate == pytest.approx(ref.params[1])
    assert e.se == pytest.approx(ref.bse[1], rel=1e-8)


# ----------------------------------------------------------------------------- multiplicity


@pytest.mark.parametrize(
    "ours,ref", [(bonferroni, "bonferroni"), (holm, "holm"), (benjamini_hochberg, "fdr_bh")]
)
def test_multiplicity_matches_statsmodels(ours, ref):
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert ours(p) == pytest.approx(multipletests(p, method=ref)[1])


def test_adjust_pvalues_reports_rejections():
    res = adjust_pvalues([0.001, 0.03, 0.5], method="holm")
    assert res.reject == [True, False, False]  # 0.03 * 2 = 0.06 > 0.05
    assert adjust_pvalues([0.001, 0.02, 0.5], method="holm").reject == [True, True, False]
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])


# ----------------------------------------------------------------------------- sequential


def test_obf_boundaries_match_published_lan_demets_table():
    # gsDesign::gsDesign(k=5, test.type=2, sfu=sfLDOF)$upper$bound / ldbounds, alpha=.05 two-sided
    d = group_sequential_design(5, alpha=0.05, spending="obrien_fleming")
    assert d.boundaries == pytest.approx([4.8769, 3.3569, 2.6803, 2.2898, 2.0310], abs=2e-4)
    assert d.alpha_spent[-1] == pytest.approx(0.05)


def test_pocock_boundaries_match_published_table():
    d = group_sequential_design(5, alpha=0.05, spending="pocock")
    assert d.boundaries == pytest.approx([2.438, 2.427, 2.410, 2.397, 2.386], abs=2e-3)


def test_single_look_is_fixed_sample_test():
    assert group_sequential_design(1).boundaries == pytest.approx([1.959964], abs=1e-5)


def test_unequal_info_fractions_and_validation():
    d = group_sequential_design(info_fractions=[0.3, 0.7, 1.0])
    assert len(d.boundaries) == 3 and d.boundaries[0] > d.boundaries[-1]
    with pytest.raises(ValueError):
        group_sequential_design(info_fractions=[0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        group_sequential_design(info_fractions=[0.5, 0.9])


def test_sequential_decision_stops_at_first_crossing():
    d = group_sequential_design(5)
    assert sequential_decision([1.0, 3.5], d).look == 2
    assert not sequential_decision([1.0, 2.0, 2.5], d).stopped
    with pytest.raises(ValueError):
        sequential_decision([0.0] * 6, d)


def test_peeking_simulation_fast_sanity():
    res = simulate_peeking_strategies(n_looks=5, n_sims=4000, seed=1)
    assert res["naive_peeking"].rejection_rate > 0.09
    assert res["obrien_fleming"].rejection_rate < 0.07


# ----------------------------------------------------------------------------- slow MC


@pytest.mark.slow
def test_sequential_controls_fpr_where_naive_peeking_fails():
    res = simulate_peeking_strategies(n_looks=10, n_sims=40_000, seed=7)
    for name in ("single_look", "obrien_fleming", "pocock"):
        r = res[name]
        assert abs(r.rejection_rate - 0.05) < 4 * r.rejection_mcse + 0.003, name
    assert res["naive_peeking"].rejection_rate > 0.15


@pytest.mark.slow
def test_obf_keeps_most_of_fixed_horizon_power():
    res = simulate_peeking_strategies(
        relative_lift=0.1, n_looks=5, n_per_look_per_arm=6000, n_sims=20_000, seed=3
    )
    assert res["obrien_fleming"].rejection_rate > res["single_look"].rejection_rate - 0.02
    assert res["obrien_fleming"].expected_sample_fraction < 0.9
