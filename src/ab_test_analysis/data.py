"""Pulls and caches the two datasets used by the analysis from
bigquery-public-data.google_analytics_sample.

Two shapes are needed, not one, because they serve different tests:

- `load_daily_device_channel`: daily sessions/conversions by device_category x
  channelGrouping. This is small (device x channel x day), and is enough for
  the primary proportions test, the Simpson's paradox segment check, the
  novelty/trend check, and the peeking simulation -- all of those only need
  success/trial counts, not row-level data.
- `load_revenue_sample`: a random row-level sample of session revenue, used
  only for the secondary continuous-metric test (Mann-Whitney/t-test on
  revenue per session). Pulling this at row level for the full year would be
  wasteful; a random sample is what the hypothesis test needs anyway.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from google.cloud import bigquery

# Billing project for BigQuery queries. The original hardcoded project is kept as a
# fallback so the original notebook still runs unchanged; set GCP_PROJECT to override.
PROJECT_ID = os.environ.get("GCP_PROJECT", "gen-lang-client-0542921612")
DATASET = "bigquery-public-data.google_analytics_sample"
TABLE_WILDCARD = f"{DATASET}.ga_sessions_*"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_SUFFIX = "20160801"
END_SUFFIX = "20170801"

DAILY_DEVICE_CHANNEL_PATH = PROCESSED_DIR / "daily_device_channel.csv"
REVENUE_SAMPLE_PATH = PROCESSED_DIR / "revenue_sample.csv"


def _client() -> bigquery.Client:
    # Imported lazily so the package (and CI) does not need google-cloud-bigquery
    # unless a live pull is actually requested. Install with: pip install -e ".[bigquery]"
    from google.cloud import bigquery

    return bigquery.Client(project=PROJECT_ID)


def load_daily_device_channel(force_refresh: bool = False) -> pd.DataFrame:
    """Sessions and converting sessions per (date, device_category, channelGrouping)."""
    if DAILY_DEVICE_CHANNEL_PATH.exists() and not force_refresh:
        return pd.read_csv(DAILY_DEVICE_CHANNEL_PATH, parse_dates=["date"])

    query = f"""
    SELECT
      PARSE_DATE('%Y%m%d', date) AS date,
      device.deviceCategory AS device_category,
      channelGrouping AS channel_grouping,
      COUNT(*) AS sessions,
      SUM(IF(totals.transactions >= 1, 1, 0)) AS converting_sessions
    FROM `{TABLE_WILDCARD}`
    WHERE _TABLE_SUFFIX BETWEEN '{START_SUFFIX}' AND '{END_SUFFIX}'
      AND device.deviceCategory IN ('desktop', 'mobile')
    GROUP BY date, device_category, channel_grouping
    ORDER BY date
    """
    df = _client().query(query).result().to_dataframe()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DAILY_DEVICE_CHANNEL_PATH, index=False)
    return df


def load_revenue_sample(per_group_n: int = 5000, force_refresh: bool = False) -> pd.DataFrame:
    """Random per-session revenue sample for desktop and mobile.

    transactionRevenue is in micros in the raw schema; converted to currency
    units here. Non-converting sessions have revenue 0 (not null after the
    IFNULL below), which is the correct value for a per-session revenue
    metric -- most sessions don't buy anything.
    """
    if REVENUE_SAMPLE_PATH.exists() and not force_refresh:
        return pd.read_csv(REVENUE_SAMPLE_PATH, parse_dates=["date"])

    query = f"""
    WITH sessions AS (
      SELECT
        PARSE_DATE('%Y%m%d', date) AS date,
        device.deviceCategory AS device_category,
        IFNULL(totals.transactionRevenue, 0) / 1e6 AS revenue,
        RAND() AS r
      FROM `{TABLE_WILDCARD}`
      WHERE _TABLE_SUFFIX BETWEEN '{START_SUFFIX}' AND '{END_SUFFIX}'
        AND device.deviceCategory IN ('desktop', 'mobile')
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY device_category ORDER BY r) AS rn
      FROM sessions
    )
    SELECT date, device_category, revenue
    FROM ranked
    WHERE rn <= {per_group_n}
    """
    df = _client().query(query).result().to_dataframe()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REVENUE_SAMPLE_PATH, index=False)
    return df
