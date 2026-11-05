"""Loaders for the public case-study datasets. Data is downloaded at runtime and never committed.

Everything is cached under ``$XPKIT_DATA`` (default: ``<repo>/data/cache``), which is
gitignored. Every download is checked against a pinned SHA-256, so a changed upstream
file fails loudly instead of silently changing results.

* **Hillstrom**: minethatdata.com (Kevin Hillstrom, 2008). 4.0 MB, 64,000 rows. No licence
  stated; published as a public research challenge.
* **Criteo uplift v2.1**: huggingface.co/datasets/criteo/criteo-uplift. 297 MB gzip,
  13,979,592 rows. CC BY-NC-SA 4.0 (non-commercial, share-alike).
* **GA360 sample**: ``bigquery-public-data.google_analytics_sample``. The query scans ~89 MB.
  Google Cloud public dataset programme.

None of these are needed for tests or ``make demo``, which use bundled synthetic data.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
CRITEO_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
CRITEO_ROWS = 13_979_592


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset can't be fetched (offline, moved upstream, no credentials)."""


def data_dir() -> Path:
    root = os.environ.get("XPKIT_DATA")
    p = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xpkit-dataset-loader"})
            with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise DatasetUnavailable(f"could not download {url}: {exc}") from exc
        tmp.rename(dest)
    got = _sha256(dest)
    if got != sha256:
        raise DatasetUnavailable(
            f"{dest.name} checksum mismatch (expected {sha256[:12]}..., got {got[:12]}...). "
            "The upstream file changed; delete it and re-verify before trusting any results."
        )
    return dest


def load_hillstrom() -> pd.DataFrame:
    """Hillstrom MineThatData e-mail experiment: 64,000 customers randomized 1/3 each to a
    men's e-mail, a women's e-mail or no e-mail. Outcomes (visit, conversion, spend) are
    measured over the two weeks after the send.

    Pre-experiment covariates: ``recency`` (months since last purchase), ``history``
    (dollars spent in the past year), ``mens``/``womens`` (past purchase categories),
    ``zip_code``, ``newbie`` (new in the past 12 months), ``channel``.
    """
    path = _fetch(HILLSTROM_URL, data_dir() / "hillstrom" / "hillstrom.csv", HILLSTROM_SHA256)
    df = pd.read_csv(path)
    expected = {
        "recency",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "segment",
        "visit",
        "conversion",
        "spend",
    }
    missing = expected - set(df.columns)
    if missing:
        raise DatasetUnavailable(f"Hillstrom file is missing columns {sorted(missing)}")
    return df


def load_criteo_sample(n: int = 1_000_000, seed: int = 20240601) -> pd.DataFrame:
    """Seeded simple random sample of ~``n`` rows from Criteo Uplift v2.1 (13,979,592 rows).

    Each row is kept independently with probability n / 13,979,592, using a
    ``numpy.random.default_rng(seed)`` stream consumed in file order. The sample is
    therefore reproducible from the pinned file and the seed, and is cached as parquet.
    The full 297 MB gzip is downloaded once.
    """
    if not 0 < n <= CRITEO_ROWS:
        raise ValueError(f"n must be in (0, {CRITEO_ROWS}]")
    cache = data_dir() / "criteo" / f"sample_n{n}_seed{seed}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    gz = _fetch(
        CRITEO_URL, data_dir() / "criteo" / "criteo-research-uplift-v2.1.csv.gz", CRITEO_SHA256
    )
    rng = np.random.default_rng(seed)
    p = n / CRITEO_ROWS
    parts, total = [], 0
    with gzip.open(gz, "rt") as f:
        for chunk in pd.read_csv(f, chunksize=500_000):
            total += len(chunk)
            keep = rng.random(len(chunk)) < p
            parts.append(chunk.loc[keep])
    if total != CRITEO_ROWS:
        raise DatasetUnavailable(f"expected {CRITEO_ROWS} rows in Criteo file, read {total}")
    df = pd.concat(parts, ignore_index=True)
    for c in ("treatment", "conversion", "visit", "exposure"):
        df[c] = df[c].astype("int8")
    df.to_parquet(cache, index=False)
    return df


GA360_QUERY = """
SELECT
  PARSE_DATE('%Y%m%d', date) AS date,
  visitStartTime,
  fullVisitorId,
  device.deviceCategory AS device_category,
  geoNetwork.continent AS continent,
  geoNetwork.country AS country,
  channelGrouping AS channel_grouping,
  trafficSource.source AS traffic_source,
  visitNumber,
  IF(totals.transactions >= 1, 1, 0) AS converted,
  IFNULL(totals.transactionRevenue, 0) / 1e6 AS revenue
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
  AND device.deviceCategory IN ('desktop', 'mobile')
"""


def load_ga360_sessions(project: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Session-level GA360 sample (Google Merchandise Store), desktop and mobile only.

    One row per session, with only *session-start* attributes (geo, time, acquisition,
    prior visit count) and outcomes. In-session behaviour (pageviews, time on site,
    bounces) is deliberately not pulled: it is downstream of the device and must not be
    adjusted for. Needs ``google-cloud-bigquery`` and application-default credentials.
    The query scans ~89 MB.
    """
    cache = data_dir() / "ga360" / "sessions.parquet"
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise DatasetUnavailable('install the extra: pip install -e ".[bigquery]"') from exc
    proj = project or os.environ.get("GCP_PROJECT")
    if not proj:
        raise DatasetUnavailable("set GCP_PROJECT (a billing project for BigQuery queries)")
    try:
        df = bigquery.Client(project=proj).query(GA360_QUERY).result().to_dataframe()
    except Exception as exc:  # auth errors, network, quota: all mean "unavailable here"
        raise DatasetUnavailable(f"BigQuery query failed: {type(exc).__name__}: {exc}") from exc
    # Plain pandas dtypes, so the cache can be read without the db-dtypes extension.
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df["visitStartTime"] = df["visitStartTime"].astype("int64")
    df["visitNumber"] = df["visitNumber"].astype("int64")
    df["converted"] = df["converted"].astype("int8")
    df["revenue"] = df["revenue"].astype(float)
    for c in (
        "fullVisitorId",
        "device_category",
        "continent",
        "country",
        "channel_grouping",
        "traffic_source",
    ):
        df[c] = df[c].astype(str)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
