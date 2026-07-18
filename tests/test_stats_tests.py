import numpy as np
import pytest

from ab_test_analysis.stats_tests import compare_continuous_metric, two_proportion_ztest


def test_two_proportion_ztest_no_difference_is_not_significant():
    # Identical rates in both arms -> diff ~0, p-value should be large.
    result = two_proportion_ztest(successes_a=500, n_a=10000, successes_b=500, n_b=10000)
    assert result.diff == pytest.approx(0.0)
    assert result.p_value > 0.9
    assert result.ci_low < 0 < result.ci_high


def test_two_proportion_ztest_large_gap_is_significant_with_ci_excluding_zero():
    # Mirrors the real mobile-vs-desktop magnitude: large n, clear gap.
    result = two_proportion_ztest(
        successes_a=10528, n_a=664479, successes_b=856, n_b=208725
    )
    assert result.diff < 0
    assert result.p_value < 0.001
    assert result.ci_low < result.ci_high < 0


def test_two_proportion_ztest_diff_direction_is_b_minus_a():
    result = two_proportion_ztest(successes_a=100, n_a=1000, successes_b=200, n_b=1000)
    assert result.rate_b > result.rate_a
    assert result.diff == pytest.approx(result.rate_b - result.rate_a)


def test_two_proportion_ztest_handles_zero_count_segment_without_crashing():
    # Regression test: score-CI inversion can fail to bracket a root when one
    # arm has very few successes -- must fall back gracefully, not raise.
    result = two_proportion_ztest(successes_a=1, n_a=1363, successes_b=2, n_b=14832)
    assert np.isfinite(result.ci_low)
    assert np.isfinite(result.ci_high)


def test_compare_continuous_metric_picks_mann_whitney_for_skewed_zero_inflated_data():
    rng = np.random.default_rng(0)
    # Zero-inflated, right-skewed -- like real per-session revenue.
    zeros_a = np.zeros(950)
    spend_a = rng.exponential(scale=20, size=50)
    sample_a = np.concatenate([zeros_a, spend_a])

    zeros_b = np.zeros(980)
    spend_b = rng.exponential(scale=20, size=20)
    sample_b = np.concatenate([zeros_b, spend_b])

    result = compare_continuous_metric(sample_a, sample_b)
    assert result.test_name == "mann_whitney_u"
    assert not result.assumptions.both_normal


def test_compare_continuous_metric_picks_ttest_for_normal_data():
    # Seed/size chosen so Shapiro-Wilk reliably passes on both samples;
    # it has a real 5% false-positive rate, so an arbitrary seed can flake.
    rng = np.random.default_rng(3)
    sample_a = rng.normal(loc=10, scale=2, size=500)
    sample_b = rng.normal(loc=10.5, scale=2, size=500)

    result = compare_continuous_metric(sample_a, sample_b)
    assert result.test_name in ("welch_t", "students_t")
    assert result.assumptions.both_normal
