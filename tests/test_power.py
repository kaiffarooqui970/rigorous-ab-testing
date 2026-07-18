import pytest

from ab_test_analysis.power import required_sample_size


def test_larger_mde_requires_fewer_sessions():
    small_mde = required_sample_size(baseline_rate=0.02, relative_mde=0.05)
    large_mde = required_sample_size(baseline_rate=0.02, relative_mde=0.20)
    assert large_mde.n_per_arm < small_mde.n_per_arm


def test_lower_baseline_rate_requires_more_sessions_for_same_relative_mde():
    low_base = required_sample_size(baseline_rate=0.005, relative_mde=0.10)
    high_base = required_sample_size(baseline_rate=0.05, relative_mde=0.10)
    assert low_base.n_per_arm > high_base.n_per_arm


def test_higher_power_requires_more_sessions():
    lower_power = required_sample_size(baseline_rate=0.02, relative_mde=0.10, power=0.7)
    higher_power = required_sample_size(baseline_rate=0.02, relative_mde=0.10, power=0.95)
    assert higher_power.n_per_arm > lower_power.n_per_arm


def test_target_rate_matches_relative_mde():
    result = required_sample_size(baseline_rate=0.10, relative_mde=0.10)
    assert result.target_rate == pytest.approx(0.11)


def test_rejects_invalid_baseline_rate():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=0.0)
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=1.0)


def test_rejects_non_positive_mde():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=0.02, relative_mde=0.0)


def test_rejects_mde_pushing_target_rate_above_one():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=0.9, relative_mde=0.5)


def test_n_per_arm_is_a_positive_integer():
    result = required_sample_size(baseline_rate=0.02, relative_mde=0.10)
    assert isinstance(result.n_per_arm, int)
    assert result.n_per_arm > 0
