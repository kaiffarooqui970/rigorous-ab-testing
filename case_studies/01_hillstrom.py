"""Case study 1: randomized experiment, Hillstrom MineThatData e-mail test.

Order of work (the report follows it too): trust checks -> effects -> variance
reduction -> multiplicity -> decision.

Run:  python case_studies/01_hillstrom.py
Out:  results/case_studies/hillstrom.json, assets/case_studies/hillstrom_*.png,
      docs/memos/01_hillstrom.md
"""

from __future__ import annotations

import sys

import pandas as pd
import yaml
from _common import FIGURES, RESULTS, ROOT, git_commit, render_memo

from xpkit._io import read_json, write_json
from xpkit.datasets import DatasetUnavailable, load_hillstrom
from xpkit.design import mde_means, mde_proportions
from xpkit.estimate import (
    adjust_pvalues,
    cuped,
    diff_in_means,
    diff_in_proportions,
    lin_regression_adjustment,
    relative_lift,
)
from xpkit.trust import balance_table, srm_test
from xpkit.viz import INK_2, SERIES, apply_style, plt, save

CFG = yaml.safe_load((ROOT / "configs" / "hillstrom.yaml").read_text())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for z in sorted(d.zip_code.unique()):
        d[f"zip={z.lower()}"] = (d.zip_code == z).astype(int)
    for c in sorted(d.channel.unique()):
        d[f"channel={c.lower()}"] = (d.channel == c).astype(int)
    return d


def arm_effects(d: pd.DataFrame, arm: str, alpha: float) -> dict:
    sub = d[d.segment.isin([arm, CFG["control_arm"]])]
    t = (sub.segment == arm).astype(int).to_numpy()
    out: dict = {"n_treatment": int(t.sum()), "n_control": int((1 - t).sum())}
    for m in ("visit", "conversion"):
        y = sub[m].to_numpy()
        e = diff_in_proportions(
            int(y[t == 1].sum()), int(t.sum()), int(y[t == 0].sum()), int((1 - t).sum()), alpha
        )
        out[m] = e.to_dict()
    y = sub.spend.to_numpy()
    out["spend"] = diff_in_means(y, t, alpha).to_dict()
    out["spend_relative_lift"] = relative_lift(y, t, alpha).to_dict()
    x = sub[CFG["cuped_covariate"]].to_numpy()
    out["spend_cuped"] = cuped(y, t, x, alpha).to_dict()
    zips = sorted(c for c in sub.columns if c.startswith("zip="))[1:]  # drop one reference level
    chans = sorted(c for c in sub.columns if c.startswith("channel="))[1:]
    X = sub[CFG["regression_covariates"] + zips + chans].to_numpy(float)
    out["spend_lin_adjusted"] = lin_regression_adjustment(y, t, X, alpha).to_dict()
    for m in ("visit", "conversion"):
        out[f"{m}_cuped"] = cuped(sub[m].to_numpy(float), t, x, alpha).to_dict()
    return out


def decide(primary: dict, adj_p: float, srm_ok: bool, balance_ok: bool, econ: dict) -> dict:
    """Decision rule, fixed before looking at estimates (see configs/hillstrom.yaml)."""
    m, c = econ["gross_margin"], econ["cost_per_email"]
    profit = primary["estimate"] * m - c
    lo, hi = primary["ci_low"] * m - c, primary["ci_high"] * m - c
    if not (srm_ok and balance_ok):
        verdict = "collect better evidence"
        why = "a trust check failed, so the effect estimate is not interpretable"
    elif adj_p < CFG["alpha"] and lo > 0:
        verdict = "launch"
        why = (
            "the primary metric is significant after Holm correction "
            "and the profit CI is above zero"
        )
    elif adj_p < CFG["alpha"] and hi > 0:
        verdict = "keep testing"
        why = "revenue lift is real but profit CI straddles zero under the stated economics"
    elif hi <= 0:
        verdict = "stop"
        why = "even the optimistic end of the CI does not cover the e-mail cost"
    else:
        verdict = "keep testing"
        why = "effect not distinguishable from zero after correction"
    return {
        "verdict": verdict,
        "reason": why,
        "profit_per_customer": profit,
        "profit_ci": [lo, hi],
        "total_profit": profit * econ["customers"],
        "total_profit_ci": [lo * econ["customers"], hi * econ["customers"]],
    }


def main() -> int:
    if "--render-only" in sys.argv:
        render_memo(
            "01_hillstrom.md.j2", "01_hillstrom.md", {"r": read_json(RESULTS / "hillstrom.json")}
        )
        return 0
    try:
        raw = load_hillstrom()
    except DatasetUnavailable as exc:
        print(f"BLOCKED: Hillstrom data unavailable: {exc}")
        return 2
    d = prepare(raw)
    alpha = CFG["alpha"]
    arms = [CFG["control_arm"], *CFG["treatment_arms"]]
    counts = [int((d.segment == a).sum()) for a in arms]

    # 1. Trust checks, before any effect estimate.
    srm = srm_test(counts)
    covs = ["recency", "history", "mens", "womens", "newbie"] + [
        c for c in d.columns if c.startswith(("zip=", "channel="))
    ]
    balance = {}
    for arm in CFG["treatment_arms"]:
        sub = d[d.segment.isin([arm, CFG["control_arm"]])].assign(
            t=lambda x, a=arm: (x.segment == a).astype(int)
        )
        balance[arm] = balance_table(sub, "t", covs)
    max_smd = max(float(b.smd.abs().max()) for b in balance.values())
    balance_ok = max_smd < 0.1

    # 2. Effects.
    effects = {arm: arm_effects(d, arm, alpha) for arm in CFG["treatment_arms"]}

    # 3. Multiplicity across arms x metrics.
    labels, pvals = [], []
    for arm in CFG["treatment_arms"]:
        for m in ["spend", "visit", "conversion"]:
            labels.append(f"{arm} / {m}")
            pvals.append(effects[arm][m]["p_value"])
    holm = adjust_pvalues(pvals, "holm", alpha)
    bh = adjust_pvalues(pvals, "bh", alpha)
    mult = [
        {"comparison": lab, "raw_p": r, "holm_p": h, "bh_p": b, "holm_reject": rj}
        for lab, r, h, b, rj in zip(
            labels, holm.raw, holm.adjusted, bh.adjusted, holm.reject, strict=True
        )
    ]

    # 4. Power actually achieved (MDE at 80% power for this sample size).
    ctrl = d[d.segment == CFG["control_arm"]]
    mde = {
        "spend_abs": mde_means(float(ctrl.spend.std()), len(ctrl)),
        "visit_rel": mde_proportions(float(ctrl.visit.mean()), len(ctrl)),
        "conversion_rel": mde_proportions(float(ctrl.conversion.mean()), len(ctrl)),
    }

    # 5. Decision per arm, on the primary metric.
    decisions = {}
    for arm in CFG["treatment_arms"]:
        adj = next(x["holm_p"] for x in mult if x["comparison"] == f"{arm} / spend")
        decisions[arm] = decide(
            effects[arm]["spend"], adj, srm.passed, balance_ok, CFG["economics"]
        )
    best = max(CFG["treatment_arms"], key=lambda a: decisions[a]["profit_per_customer"])

    results = {
        "dataset": "Hillstrom MineThatData E-Mail (2008), 64,000 customers",
        "git_commit_at_run": git_commit(),
        "config": CFG,
        "arm_counts": dict(zip(arms, counts, strict=True)),
        "control_rates": {m: float(ctrl[m].mean()) for m in ("visit", "conversion", "spend")},
        "srm": srm.to_dict(),
        "balance_max_abs_smd": max_smd,
        "balance_ok": balance_ok,
        "balance": {a: b.reset_index().to_dict(orient="records") for a, b in balance.items()},
        "effects": effects,
        "multiplicity": mult,
        "mde_80pct_power": mde,
        "decisions": decisions,
        "recommended_arm": best,
    }
    write_json(results, RESULTS / "hillstrom.json")
    charts(effects)
    # Render from the saved JSON, exactly as --render-only does, so both paths agree.
    render_memo(
        "01_hillstrom.md.j2", "01_hillstrom.md", {"r": read_json(RESULTS / "hillstrom.json")}
    )
    print(
        f"Hillstrom: SRM p={srm.p_value:.3f}, max|SMD|={max_smd:.3f}; "
        f"{best}: {decisions[best]['verdict']}"
    )
    return 0


def charts(effects: dict) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (m, lab, scale) in zip(
        axes,
        [
            ("visit", "Visit rate (pp)", 100),
            ("conversion", "Conversion rate (pp)", 100),
            ("spend", "Spend per customer ($)", 1),
        ],
        strict=True,
    ):
        for i, arm in enumerate(["Mens E-Mail", "Womens E-Mail"]):
            e = effects[arm][m]
            ax.errorbar(
                [e["estimate"] * scale],
                [i],
                xerr=[
                    [(e["estimate"] - e["ci_low"]) * scale],
                    [(e["ci_high"] - e["estimate"]) * scale],
                ],
                fmt="o",
                ms=7,
                color=SERIES[i],
                capsize=3,
            )
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_yticks([0, 1], ["Men's e-mail", "Women's e-mail"] if m == "visit" else ["", ""])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Hillstrom: effect vs no e-mail (95% CI, uncorrected)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, FIGURES / "hillstrom_effects.png")


if __name__ == "__main__":
    sys.exit(main())
