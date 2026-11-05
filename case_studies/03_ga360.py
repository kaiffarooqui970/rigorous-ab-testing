"""Case study 3: observational data. GA360 mobile vs desktop conversion, revisited.

The original analysis compared raw conversion rates with session-iid z-tests. This
script asks what can credibly be estimated when device is *chosen*, not assigned:

1. inference that respects repeat visitors (clustered by fullVisitorId);
2. propensity-score adjustment for session-start covariates only, with overlap and
   balance diagnostics;
3. a sensitivity adjustment set that adds channel, which may be a mediator;
4. exclusion of probable internal (Google-employee) traffic;
5. an E-value for how strong unmeasured confounding would need to be;
6. a decision: what evidence the business should collect next.

Run:  GCP_PROJECT=<billing project> python case_studies/03_ga360.py
Out:  results/case_studies/ga360.json, assets/case_studies/ga360_*.png, docs/memos/03_ga360.md
If BigQuery is unavailable and nothing is cached, the script exits with BLOCKED (code 2).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml
from _common import FIGURES, RESULTS, ROOT, git_commit, render_memo

from xpkit._io import read_json, write_json
from xpkit.causal import aipw, e_value, ipw
from xpkit.datasets import DatasetUnavailable, load_ga360_sessions
from xpkit.design import sample_size_proportions
from xpkit.estimate import diff_in_means, diff_in_proportions
from xpkit.trust import balance_table
from xpkit.viz import INK_2, MUTED, SERIES, apply_style, plt, save

CFG = yaml.safe_load((ROOT / "configs" / "ga360.yaml").read_text())
# Suspected internal traffic. Two patterns, neither verifiable because the sample
# obfuscates referrers: (1) channel "Referral" whose source is recorded as "(direct)",
# a desktop-only segment with an unusually high conversion rate; (2) Google-internal
# looking domains.
INTERNAL_DOMAINS = ("googleplex", "sites.google.com")


def features(g: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode session-start attributes. Returns the frame and the columns per group."""
    d = g.copy()
    d["treatment"] = (d.device_category == "mobile").astype(int)
    ts = pd.to_datetime(d.visitStartTime, unit="s", utc=True)
    d["month"] = d.date.dt.strftime("%Y-%m")
    d["weekday"] = ts.dt.day_name()
    d["hour_utc_block"] = (ts.dt.hour // 4 * 4).astype(str)
    top = d.country.value_counts().index[: CFG["n_top_countries"]]
    d["country_top"] = np.where(d.country.isin(top), d.country, "Other")
    d["returning"] = (d.visitNumber > 1).astype(int)
    d["log_prior_visits"] = np.log1p(d.visitNumber - 1)
    cols: dict[str, list[str]] = {
        "returning": ["returning"],
        "log_prior_visits": ["log_prior_visits"],
    }
    for c in ("month", "weekday", "hour_utc_block", "continent", "country_top", "channel_grouping"):
        levels = sorted(d[c].unique())
        names = []
        for lv in levels[1:]:  # drop first level as reference
            n = f"{c}={lv}"
            d[n] = (d[c] == lv).astype(np.int8)
            names.append(n)
        cols[c] = names
    return d, cols


def expand(groups: list[str], cols: dict[str, list[str]]) -> list[str]:
    return [c for g in groups for c in cols[g]]


def run(d: pd.DataFrame, cols: dict[str, list[str]], label: str) -> dict:
    y, t = d.converted.to_numpy(float), d.treatment.to_numpy()
    s1, n1 = int(y[t == 1].sum()), int(t.sum())
    s0, n0 = int(y[t == 0].sum()), int((1 - t).sum())
    out: dict = {
        "label": label,
        "n_sessions": len(d),
        "n_mobile": n1,
        "n_desktop": n0,
        "n_visitors": int(d.fullVisitorId.nunique()),
    }
    out["naive_iid"] = diff_in_proportions(s1, n1, s0, n0).to_dict()
    out["naive_clustered"] = diff_in_means(y, t, cluster=d.fullVisitorId.to_numpy()).to_dict()
    out["adjusted"] = {}
    for set_name, groups in CFG["adjustment_sets"].items():
        X = expand(groups, cols)
        e_ipw = ipw(d, "converted", "treatment", X, cluster=CFG["cluster"])
        e_aipw = aipw(d, "converted", "treatment", X, cluster=CFG["cluster"])
        out["adjusted"][set_name] = {
            "covariates": X,
            "ipw": e_ipw.to_dict(),
            "aipw": e_aipw.to_dict(),
        }
    return out


def main() -> int:
    if "--render-only" in sys.argv:
        render_memo("03_ga360.md.j2", "03_ga360.md", {"r": read_json(RESULTS / "ga360.json")})
        return 0
    try:
        g = load_ga360_sessions()
    except DatasetUnavailable as exc:
        print(f"BLOCKED: GA360 data unavailable ({exc}). Documented in docs/data_sources.md.")
        return 2
    d, cols = features(g)

    # Sanity link to the original analysis: same sessions and conversions.
    legacy = {
        dev: {
            "sessions": int((d.device_category == dev).sum()),
            "conversions": int(d.loc[d.device_category == dev, "converted"].sum()),
        }
        for dev in ("desktop", "mobile")
    }

    # Diagnostics on the primary adjustment set (before/after IPW balance).
    XA = expand(CFG["adjustment_sets"]["A_session_start"], cols)
    full = run(d, cols, "all sessions")
    from xpkit.causal import fit_propensity

    ps = fit_propensity(d, "treatment", XA)
    t = d.treatment.to_numpy()
    ps_w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    bal_before = balance_table(d, "treatment", XA)
    bal_after = balance_table(d, "treatment", XA, weights=ps_w)

    # Probable internal traffic (Google employees buying merchandise on desktop).
    referral_direct = (d.channel_grouping == "Referral") & (d.traffic_source == "(direct)")
    internal_domain = d.traffic_source.str.contains("|".join(INTERNAL_DOMAINS), regex=True)
    internal = referral_direct | internal_domain
    seg = d.loc[referral_direct]
    int_stats = {
        "n_sessions": int(internal.sum()),
        "referral_direct": {
            "n_desktop": int((seg.treatment == 0).sum()),
            "n_mobile": int((seg.treatment == 1).sum()),
            "conversions_desktop": int(seg.loc[seg.treatment == 0, "converted"].sum()),
            "conversion_rate_desktop": float(seg.loc[seg.treatment == 0, "converted"].mean()),
        },
        "internal_domain_sessions": int(internal_domain.sum()),
        "share_of_desktop_conversions": float(
            d.loc[internal & (t == 0), "converted"].sum() / d.loc[t == 0, "converted"].sum()
        ),
        "conversion_rate_desktop": float(d.loc[internal & (t == 0), "converted"].mean())
        if (internal & (t == 0)).any()
        else None,
        "sources_matched": sorted(d.loc[internal, "traffic_source"].unique().tolist())[:20],
    }
    no_internal = run(
        d.loc[~internal].reset_index(drop=True), cols, "excluding probable internal traffic"
    )

    # E-value on the primary adjusted estimate (AIPW risk ratio; protective direction).
    prim = full["adjusted"]["A_session_start"]["aipw"]["extra"]
    ev = e_value(prim["risk_ratio"], prim["rr_ci_low"], prim["rr_ci_high"])
    px = no_internal["adjusted"]["A_session_start"]["aipw"]["extra"]
    ev_x = e_value(px["risk_ratio"], px["rr_ci_low"], px["rr_ci_high"])

    # What a proper randomized test of a mobile redesign would need (sessions per arm).
    base = full["naive_iid"]["extra"]["rate_treatment"]
    months = d.date.dt.to_period("M").nunique()
    mobile_per_month = full["n_mobile"] / months
    pw = CFG["power_for_future_test"]
    power_rows = []
    for mde in pw["relative_mdes"]:
        ss = sample_size_proportions(base, mde, alpha=pw["alpha"], power=pw["power"])
        power_rows.append(
            {
                "relative_mde": mde,
                "n_per_arm": ss.n_control,
                "n_total": ss.n_total,
                "months_of_mobile_traffic": ss.n_total / mobile_per_month,
            }
        )

    se_inflation = full["naive_clustered"]["se"] / full["naive_iid"]["se"]
    a = full["adjusted"]["A_session_start"]["aipw"]
    results = {
        "dataset": (
            "bigquery-public-data.google_analytics_sample, 2016-08-01..2017-08-01, desktop+mobile"
        ),
        "git_commit_at_run": git_commit(),
        "config": CFG,
        "legacy_counts_check": legacy,
        "sessions_per_visitor": full["n_sessions"] / full["n_visitors"],
        "se_inflation_from_clustering": se_inflation,
        "all_sessions": full,
        "excluding_internal": no_internal,
        "internal_traffic": int_stats,
        "overlap_set_A": a["extra"]["overlap"],
        "balance_set_A": {
            "max_abs_smd_before": float(bal_before.smd.abs().max()),
            "max_abs_smd_after": float(bal_after.smd.abs().max()),
            "n_covariates": len(XA),
            "worst_before": bal_before.smd.abs().sort_values(ascending=False).head(5).to_dict(),
        },
        "e_value": ev.to_dict(),
        "e_value_excluding_internal": ev_x.to_dict(),
        "future_test_power": {
            "baseline_mobile_rate": base,
            "mobile_sessions_per_month": mobile_per_month,
            "rows": power_rows,
        },
        "decision": {
            "verdict": "collect better evidence",
            "reason": (
                "device is self-selected and the adjusted gap rests on an untestable "
                "no-unmeasured-confounding assumption; only a randomized test of a specific "
                "mobile change can say whether fixing mobile would pay"
            ),
        },
    }
    write_json(results, RESULTS / "ga360.json")
    charts(bal_before, bal_after, results)
    render_memo("03_ga360.md.j2", "03_ga360.md", {"r": results})
    print(
        f"GA360: naive {full['naive_iid']['estimate']:.5f}, AIPW(A) {a['estimate']:.5f}, "
        f"cluster SE x{se_inflation:.2f}, E-value {ev.e_value_point:.2f}"
    )
    return 0


def charts(before: pd.DataFrame, after: pd.DataFrame, r: dict) -> None:
    apply_style()
    top = before.smd.abs().sort_values(ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(7.5, 5))
    yy = np.arange(len(top))[::-1]
    ax.scatter(
        before.loc[top, "smd"].abs(), yy, color=SERIES[1], s=40, label="Before weighting", zorder=3
    )
    ax.scatter(after.loc[top, "smd"].abs(), yy, color=SERIES[0], s=40, label="After IPW", zorder=3)
    ax.axvline(0.1, color=INK_2, ls="--", lw=1)
    ax.set_yticks(yy, [c.replace("_", " ") for c in top], fontsize=8)
    ax.set_xlabel("|Standardized mean difference|, mobile vs desktop")
    ax.set_title("GA360: session-start covariates, before and after weighting (15 worst)")
    ax.legend(loc="lower right")
    save(fig, FIGURES / "ga360_balance.png")

    rows = [
        ("Naive, session-iid SE", r["all_sessions"]["naive_iid"]),
        ("Naive, visitor-clustered SE", r["all_sessions"]["naive_clustered"]),
    ]
    for k, lab in (
        ("A_session_start", "AIPW, session-start covariates"),
        ("B_plus_channel", "AIPW, + channel (possible mediator)"),
    ):
        rows.append((lab, r["all_sessions"]["adjusted"][k]["aipw"]))
    rows.append(
        (
            "AIPW (A), internal traffic excluded",
            r["excluding_internal"]["adjusted"]["A_session_start"]["aipw"],
        )
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (_lab, e) in enumerate(rows[::-1]):
        ax.errorbar(
            [100 * e["estimate"]],
            [i],
            xerr=[[100 * (e["estimate"] - e["ci_low"])], [100 * (e["ci_high"] - e["estimate"])]],
            fmt="o",
            ms=7,
            color=SERIES[0],
            capsize=3,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [x[0] for x in rows[::-1]], fontsize=9)
    ax.set_xlabel("Mobile − desktop conversion (pp), 95% CI")
    naive, prim = (
        r["all_sessions"]["naive_iid"]["estimate"],
        r["all_sessions"]["adjusted"]["A_session_start"]["aipw"]["estimate"],
    )
    excl = r["excluding_internal"]["adjusted"]["A_session_start"]["aipw"]["estimate"]
    ax.set_title(
        f"GA360: mobile − desktop gap is {100 * naive:.2f}pp raw, {100 * prim:.2f}pp adjusted, "
        f"{100 * excl:.2f}pp without suspected internal traffic",
        fontsize=10,
    )
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES / "ga360_estimates.png")


if __name__ == "__main__":
    sys.exit(main())
