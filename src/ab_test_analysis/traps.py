"""Three classic experimentation traps, checked with real analysis rather
than asserted in prose.

1. Novelty effect: does the mobile-vs-desktop gap hold steady over the year,
   or decay/grow -- checked by trending the weekly gap and testing the trend
   slope for significance.
2. Simpson's paradox: does the device effect reverse or vanish once you
   condition on channelGrouping -- checked by running the same
   two-proportion test within each channel that has enough volume.
3. Peeking / multiple-testing inflation: what happens to the true false
   positive rate if you test significance after every day's data instead of
   once at a pre-registered endpoint -- checked with a Monte Carlo simulation
   under the null (both arms given the same true rate), not just discussed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ab_test_analysis.stats_tests import two_proportion_ztest

# ---------------------------------------------------------------------------
# 1. Novelty effect
# ---------------------------------------------------------------------------


@dataclass
class NoveltyTrendResult:
    weekly: pd.DataFrame  # week, mobile_rate, desktop_rate, gap
    slope: float
    slope_p_value: float
    trend_significant: bool


def novelty_trend(
    daily_device: pd.DataFrame, alpha: float = 0.05
) -> NoveltyTrendResult:
    """Regresses the weekly (mobile_rate - desktop_rate) gap on week index.

    A significant nonzero slope means the gap is drifting over the year
    (consistent with a novelty effect decaying, or a genuine trend) rather
    than being a stable, one-off difference -- which matters for whether a
    single historical average is a fair estimate of the "true" gap.
    """
    df = daily_device.copy()
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    pivot = (
        df.groupby(["week", "device_category"])[["sessions", "converting_sessions"]]
        .sum()
        .reset_index()
    )
    wide = pivot.pivot(index="week", columns="device_category")
    wide.columns = [f"{metric}_{device}" for metric, device in wide.columns]
    wide = wide.reset_index().sort_values("week")

    wide["mobile_rate"] = wide["converting_sessions_mobile"] / wide["sessions_mobile"]
    wide["desktop_rate"] = wide["converting_sessions_desktop"] / wide["sessions_desktop"]
    wide["gap"] = wide["mobile_rate"] - wide["desktop_rate"]
    wide["week_index"] = np.arange(len(wide))

    reg = stats.linregress(wide["week_index"], wide["gap"])

    weekly_out = wide[["week", "mobile_rate", "desktop_rate", "gap"]]

    return NoveltyTrendResult(
        weekly=weekly_out,
        slope=reg.slope,
        slope_p_value=reg.pvalue,
        trend_significant=reg.pvalue < alpha,
    )


# ---------------------------------------------------------------------------
# 2. Simpson's paradox
# ---------------------------------------------------------------------------


@dataclass
class SegmentResult:
    channel: str
    n_mobile: int
    n_desktop: int
    mobile_rate: float
    desktop_rate: float
    diff: float
    p_value: float
    direction_matches_overall: bool


@dataclass
class SimpsonsCheckResult:
    overall_diff: float
    segments: list[SegmentResult]
    any_reversal: bool


def simpsons_paradox_check(
    daily_device_channel: pd.DataFrame,
    overall_diff: float,
    min_sessions_per_arm: int = 1000,
    alpha: float = 0.05,
) -> SimpsonsCheckResult:
    """Re-runs the mobile-vs-desktop proportions test within each channel.

    overall_diff is the pooled mobile-minus-desktop rate difference (should
    be negative here, i.e. mobile underperforms). A channel "reverses" the
    paradox if its own diff has the opposite sign and is itself significant
    -- a same-signed-but-insignificant segment is just noise, not a
    reversal, so both are checked.
    """
    grouped = (
        daily_device_channel.groupby(["channel_grouping", "device_category"])[
            ["sessions", "converting_sessions"]
        ]
        .sum()
        .reset_index()
    )

    segments: list[SegmentResult] = []
    for channel, sub in grouped.groupby("channel_grouping"):
        sub = sub.set_index("device_category")
        if "mobile" not in sub.index or "desktop" not in sub.index:
            continue
        n_mobile = int(sub.loc["mobile", "sessions"])
        n_desktop = int(sub.loc["desktop", "sessions"])
        if n_mobile < min_sessions_per_arm or n_desktop < min_sessions_per_arm:
            continue

        result = two_proportion_ztest(
            successes_a=int(sub.loc["desktop", "converting_sessions"]),
            n_a=n_desktop,
            successes_b=int(sub.loc["mobile", "converting_sessions"]),
            n_b=n_mobile,
        )
        direction_matches = (result.diff < 0) == (overall_diff < 0)
        segments.append(
            SegmentResult(
                channel=channel,
                n_mobile=n_mobile,
                n_desktop=n_desktop,
                mobile_rate=result.rate_b,
                desktop_rate=result.rate_a,
                diff=result.diff,
                p_value=result.p_value,
                direction_matches_overall=direction_matches,
            )
        )

    any_reversal = any(
        (not s.direction_matches_overall) and s.p_value < alpha for s in segments
    )

    return SimpsonsCheckResult(
        overall_diff=overall_diff, segments=segments, any_reversal=any_reversal
    )


# ---------------------------------------------------------------------------
# 3. Peeking / multiple-testing inflation
# ---------------------------------------------------------------------------


@dataclass
class PeekingSimulationResult:
    n_looks: int
    n_per_look_per_arm: int
    nominal_alpha: float
    n_simulations: int
    naive_false_positive_rate: float  # testing at every look, no correction
    single_look_false_positive_rate: float  # testing only at the final look
    bonferroni_alpha: float
    bonferroni_false_positive_rate: float  # every look, Bonferroni-corrected alpha


def simulate_peeking_inflation(
    baseline_rate: float,
    n_looks: int = 10,
    n_per_look_per_arm: int = 500,
    alpha: float = 0.05,
    n_simulations: int = 5000,
    seed: int = 42,
) -> PeekingSimulationResult:
    """Simulates repeated interim significance testing under the null
    (both arms truly equal, at baseline_rate) to measure how much the true
    false-positive rate inflates versus the nominal alpha.

    At each of n_looks cumulative checkpoints, a two-proportion z-test is
    run on the data accumulated so far. A simulated experiment is flagged
    "false positive" if ANY of those interim checks crosses alpha -- this is
    exactly the failure mode of "peek daily, ship when p < 0.05."
    Bonferroni correction (alpha / n_looks) is included as the simplest fix,
    to show it restores the nominal rate; a real deployment would prefer a
    proper sequential design (e.g. O'Brien-Fleming spending) since Bonferroni
    is conservative here.
    """
    rng = np.random.default_rng(seed)

    naive_rejections = np.zeros(n_simulations, dtype=bool)
    bonferroni_rejections = np.zeros(n_simulations, dtype=bool)
    final_look_rejections = np.zeros(n_simulations, dtype=bool)
    bonferroni_alpha = alpha / n_looks

    for sim in range(n_simulations):
        successes_a = rng.binomial(1, baseline_rate, size=n_per_look_per_arm * n_looks)
        successes_b = rng.binomial(1, baseline_rate, size=n_per_look_per_arm * n_looks)

        cum_a = np.cumsum(successes_a)
        cum_b = np.cumsum(successes_b)

        for look in range(1, n_looks + 1):
            n = n_per_look_per_arm * look
            s_a, s_b = int(cum_a[n - 1]), int(cum_b[n - 1])
            p_pool = (s_a + s_b) / (2 * n)
            if p_pool in (0, 1):
                continue
            se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
            z = (s_b / n - s_a / n) / se
            p_value = 2 * stats.norm.sf(abs(z))

            if p_value < alpha:
                naive_rejections[sim] = True
            if p_value < bonferroni_alpha:
                bonferroni_rejections[sim] = True
            if look == n_looks and p_value < alpha:
                final_look_rejections[sim] = True

    return PeekingSimulationResult(
        n_looks=n_looks,
        n_per_look_per_arm=n_per_look_per_arm,
        nominal_alpha=alpha,
        n_simulations=n_simulations,
        naive_false_positive_rate=float(naive_rejections.mean()),
        single_look_false_positive_rate=float(final_look_rejections.mean()),
        bonferroni_alpha=bonferroni_alpha,
        bonferroni_false_positive_rate=float(bonferroni_rejections.mean()),
    )
