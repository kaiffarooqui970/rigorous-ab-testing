"""Interactive explorer: filter the historical session data by date range and
channel, and re-run the mobile-vs-desktop significance test on whatever
subset is selected. Lets a reviewer check whether the headline finding holds
up under different slices, rather than just trusting the one number in the
memo.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ab_test_analysis.data import load_daily_device_channel
from ab_test_analysis.stats_tests import two_proportion_ztest

st.set_page_config(page_title="Mobile vs Desktop Conversion Explorer", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_daily_device_channel()


df = get_data()

st.title("Mobile vs Desktop Conversion Explorer")
st.caption(
    "Filter the historical Google Merchandise Store session data and re-run the "
    "significance test on the selected slice. Full findings and the ship/no-ship "
    "recommendation are in `memo/decision_memo.md`."
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

with st.sidebar:
    st.header("Filters")
    date_range = st.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    channels = sorted(df["channel_grouping"].unique())
    selected_channels = st.multiselect("Channels", channels, default=channels)
    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01)

if len(date_range) != 2:
    st.stop()

start_date, end_date = date_range
mask = (
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["channel_grouping"].isin(selected_channels))
)
filtered = df.loc[mask]

if filtered.empty:
    st.warning("No sessions match the current filters.")
    st.stop()

agg = filtered.groupby("device_category")[["sessions", "converting_sessions"]].sum()

if "desktop" not in agg.index or "mobile" not in agg.index:
    st.warning("Need both a desktop and a mobile session under the current filters.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Desktop sessions", f"{int(agg.loc['desktop', 'sessions']):,}")
col2.metric("Mobile sessions", f"{int(agg.loc['mobile', 'sessions']):,}")
col3.metric("Channels selected", f"{len(selected_channels)}/{len(channels)}")

result = two_proportion_ztest(
    successes_a=int(agg.loc["desktop", "converting_sessions"]),
    n_a=int(agg.loc["desktop", "sessions"]),
    successes_b=int(agg.loc["mobile", "converting_sessions"]),
    n_b=int(agg.loc["mobile", "sessions"]),
    alpha=alpha,
)

st.subheader("Two-proportion z-test result (selected slice)")
res_col1, res_col2 = st.columns(2)
with res_col1:
    st.metric("Desktop conversion rate", f"{result.rate_a:.3%}")
    st.metric("Mobile conversion rate", f"{result.rate_b:.3%}")
with res_col2:
    st.metric(
        "Difference (mobile - desktop)",
        f"{result.diff:.3%}",
        help=f"{(1 - alpha) * 100:.0f}% CI: [{result.ci_low:.3%}, {result.ci_high:.3%}]",
    )
    significance = "significant" if result.p_value < alpha else "not significant"
    st.metric("p-value", f"{result.p_value:.3g}", help=f"Result is {significance} at alpha={alpha}")

st.caption(
    f"{(1 - alpha) * 100:.0f}% confidence interval on the difference: "
    f"[{result.ci_low:.3%}, {result.ci_high:.3%}]"
)

st.subheader("Weekly conversion rate trend (selected slice)")
weekly = filtered.copy()
weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
weekly_agg = weekly.groupby(["week", "device_category"])[
    ["sessions", "converting_sessions"]
].sum()
weekly_agg["rate"] = weekly_agg["converting_sessions"] / weekly_agg["sessions"]
weekly_wide = weekly_agg["rate"].unstack("device_category")

fig, ax = plt.subplots(figsize=(10, 4))
for device in weekly_wide.columns:
    ax.plot(weekly_wide.index, weekly_wide[device], marker="o", ms=3, label=device)
ax.set_ylabel("Conversion rate")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("Per-channel breakdown (selected slice)")
channel_agg = (
    filtered.groupby(["channel_grouping", "device_category"])[
        ["sessions", "converting_sessions"]
    ]
    .sum()
    .reset_index()
)
channel_agg["conversion_rate"] = (
    channel_agg["converting_sessions"] / channel_agg["sessions"]
)
st.dataframe(
    channel_agg.pivot(index="channel_grouping", columns="device_category", values="conversion_rate"),
    use_container_width=True,
)
