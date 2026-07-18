import numpy as np
import pandas as pd

from ab_test_analysis.traps import (
    novelty_trend,
    simpsons_paradox_check,
    simulate_peeking_inflation,
)


def _daily_device_frame(days=70, mobile_rate=0.01, desktop_rate=0.02, channel="Direct"):
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    rows = []
    for d in dates:
        rows.append(
            dict(
                date=d,
                device_category="mobile",
                channel_grouping=channel,
                sessions=1000,
                converting_sessions=int(1000 * mobile_rate),
            )
        )
        rows.append(
            dict(
                date=d,
                device_category="desktop",
                channel_grouping=channel,
                sessions=1000,
                converting_sessions=int(1000 * desktop_rate),
            )
        )
    return pd.DataFrame(rows)


def test_novelty_trend_flat_gap_is_not_significant():
    df = _daily_device_frame()
    result = novelty_trend(df)
    assert not result.trend_significant


def test_novelty_trend_detects_decaying_gap():
    dates = pd.date_range("2017-01-01", periods=84, freq="D")
    rows = []
    for i, d in enumerate(dates):
        # Mobile rate ramps up toward desktop's over time -> gap shrinks to 0.
        week = i // 7
        mobile_rate = 0.005 + week * 0.002
        rows.append(
            dict(
                date=d,
                device_category="mobile",
                channel_grouping="Direct",
                sessions=2000,
                converting_sessions=int(2000 * mobile_rate),
            )
        )
        rows.append(
            dict(
                date=d,
                device_category="desktop",
                channel_grouping="Direct",
                sessions=2000,
                converting_sessions=int(2000 * 0.02),
            )
        )
    df = pd.DataFrame(rows)
    result = novelty_trend(df)
    assert result.trend_significant
    assert result.slope > 0  # gap (mobile - desktop) is closing, i.e. rising


def test_simpsons_paradox_check_no_reversal_when_all_segments_agree():
    df = _daily_device_frame(mobile_rate=0.01, desktop_rate=0.02)
    result = simpsons_paradox_check(df, overall_diff=-0.01, min_sessions_per_arm=100)
    assert not result.any_reversal
    assert all(s.direction_matches_overall for s in result.segments)


def test_simpsons_paradox_check_detects_reversal_in_segment():
    agreeing = _daily_device_frame(
        days=90, mobile_rate=0.005, desktop_rate=0.03, channel="Direct"
    )
    reversing = _daily_device_frame(
        days=90, mobile_rate=0.03, desktop_rate=0.005, channel="Affiliates"
    )
    df = pd.concat([agreeing, reversing], ignore_index=True)
    result = simpsons_paradox_check(df, overall_diff=-0.01, min_sessions_per_arm=100)
    assert result.any_reversal
    reversing_segment = next(s for s in result.segments if s.channel == "Affiliates")
    assert not reversing_segment.direction_matches_overall
    assert reversing_segment.p_value < 0.05


def test_simulate_peeking_inflation_naive_exceeds_nominal_alpha():
    result = simulate_peeking_inflation(
        baseline_rate=0.05, n_looks=8, n_per_look_per_arm=200, n_simulations=1000, seed=1
    )
    assert result.naive_false_positive_rate > result.nominal_alpha
    # Single pre-registered look should track the nominal alpha reasonably closely.
    assert abs(result.single_look_false_positive_rate - result.nominal_alpha) < 0.03


def test_simulate_peeking_inflation_is_deterministic_given_seed():
    r1 = simulate_peeking_inflation(baseline_rate=0.05, n_simulations=200, seed=7)
    r2 = simulate_peeking_inflation(baseline_rate=0.05, n_simulations=200, seed=7)
    assert r1.naive_false_positive_rate == r2.naive_false_positive_rate
