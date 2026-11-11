"""Case study 2: who should be targeted? Criteo Uplift v2.1 (seeded 1M-row sample).

Order: trust checks -> average effect -> uplift models (train) -> depth choice (validation)
-> value vs random targeting (test) -> sensitivity to economics -> decision.

Run:  python case_studies/02_criteo.py
Out:  results/case_studies/criteo.json, assets/case_studies/criteo_*.png,
      docs/memos/02_criteo.md, docs/memos/02_criteo_appendix.md
"""

from __future__ import annotations

import sys

import numpy as np
import yaml
from _common import FIGURES, RESULTS, ROOT, git_commit, render_memos

from xpkit._io import write_json
from xpkit.datasets import DatasetUnavailable, load_criteo_sample
from xpkit.estimate import diff_in_proportions
from xpkit.trust import balance_table, srm_test
from xpkit.uplift import (
    Economics,
    TLearner,
    XLearner,
    choose_depth,
    policy_table,
    qini_coefficient_ci,
    uplift_by_bin,
    uplift_curve,
)
from xpkit.viz import INK_2, SERIES, apply_style, plt, save

CFG = yaml.safe_load((ROOT / "configs" / "criteo.yaml").read_text())


def split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = CFG["split"]
    idx = np.random.default_rng(s["seed"]).permutation(n)
    a, b = int(s["train"] * n), int((s["train"] + s["validation"]) * n)
    return idx[:a], idx[a:b], idx[b:]


def model_vs_random_ci(
    y, t, score, depth: float, econ: Economics, n_boot: int = 200, seed: int = 0
):
    """Bootstrap CI (over the test set) for profit(model at depth) - profit(random at depth)."""
    rng = np.random.default_rng(seed)
    diffs = []
    y, t, score = np.asarray(y), np.asarray(t), np.asarray(score)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        tab = policy_table(y[i], t[i], score[i], econ, depths=[depth])
        diffs.append(float(tab.profit_per_user.iloc[0] - tab.random_profit_per_user.iloc[0]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    if "--render-only" in sys.argv:
        from xpkit._io import read_json

        render_memos("02_criteo", {"r": read_json(RESULTS / "criteo.json")})
        return 0
    try:
        d = load_criteo_sample(CFG["sample_rows"], CFG["sample_seed"])
    except DatasetUnavailable as exc:
        print(f"BLOCKED: Criteo data unavailable: {exc}")
        return 2
    feats, yname = CFG["features"], CFG["outcome"]
    share = CFG["designed_treatment_share"]

    # 1. Trust.
    counts = [int((d.treatment == 0).sum()), int((d.treatment == 1).sum())]
    srm = srm_test(counts, [1 - share, share])
    bal = balance_table(d, "treatment", feats)

    # 2. Average effects.
    ate = {}
    for m in (yname, CFG["secondary_outcome"]):
        y, t = d[m].to_numpy(), d.treatment.to_numpy()
        ate[m] = diff_in_proportions(
            int(y[t == 1].sum()), int(t.sum()), int(y[t == 0].sum()), int((1 - t).sum())
        ).to_dict()

    # 3-5. Uplift models: fit on train, pick depth on validation, report on test.
    tr, va, te = split(len(d))
    D = {
        k: d.iloc[ix].reset_index(drop=True) for k, ix in (("train", tr), ("val", va), ("test", te))
    }
    econ = Economics(**CFG["economics"])
    models, scores = {}, {}
    for name, L in (("t_learner", TLearner), ("x_learner", XLearner)):
        m = L(outcome="binary", seed=0).fit(
            D["train"][feats], D["train"].treatment, D["train"][yname]
        )
        models[name] = m
        scores[name] = {k: m.predict(D[k][feats]) for k in ("val", "test")}

    ev: dict = {}
    for name, s in scores.items():
        te_ = D["test"]
        q, lo, hi = qini_coefficient_ci(te_[yname], te_.treatment, s["test"], n_boot=200, seed=1)
        dec = choose_depth(
            (D["val"][yname], D["val"].treatment, s["val"]),
            (te_[yname], te_.treatment, s["test"]),
            econ,
        )
        vs_rand = (
            model_vs_random_ci(te_[yname], te_.treatment, s["test"], dec.chosen_depth, econ)
            if dec.chosen_depth > 0
            else (0.0, 0.0)
        )
        ev[name] = {
            "qini_coefficient": q,
            "qini_ci": [lo, hi],
            "uplift_by_decile": uplift_by_bin(te_[yname], te_.treatment, s["test"], 10).to_dict(
                orient="records"
            ),
            "policy_table_test": policy_table(te_[yname], te_.treatment, s["test"], econ).to_dict(
                orient="records"
            ),
            "decision": dec.to_dict(),
            "model_minus_random_profit_ci": list(vs_rand),
        }
    best = max(ev, key=lambda k: ev[k]["qini_coefficient"])

    # Secondary outcome: does the visit model also rank conversion uplift?
    te_ = D["test"]
    sec = qini_coefficient_ci(
        te_[CFG["secondary_outcome"]], te_.treatment, scores[best]["test"], n_boot=200, seed=2
    )

    # 6. Sensitivity to economics: decision at each break-even uplift (value fixed at $1).
    sens = []
    for be in CFG["sensitivity_breakeven_uplifts"]:
        e2 = Economics(value_per_conversion=1.0, cost_per_treatment=be, population=econ.population)
        dec = choose_depth(
            (D["val"][yname], D["val"].treatment, scores[best]["val"]),
            (te_[yname], te_.treatment, scores[best]["test"]),
            e2,
        )
        sens.append({"breakeven_uplift": be, **dec.to_dict()})

    # 7. Decision.
    b = ev[best]
    dec = b["decision"]
    lo_q = b["qini_ci"][0]
    if not srm.passed:
        verdict, why = (
            "collect better evidence",
            "sample ratio mismatch against the documented 85/15 split",
        )
    elif lo_q <= 0:
        verdict, why = (
            "keep testing",
            "the model's ranking is not distinguishable from random on held-out data",
        )
    elif dec["chosen_depth"] == 0:
        verdict, why = "stop", "no targeting depth is profitable under the stated economics"
    elif b["model_minus_random_profit_ci"][0] > 0 and dec["test_profit_ci"][0] > 0:
        verdict, why = (
            "launch",
            (
                "targeting beats random at the chosen depth and is profitable on "
                "the held-out test set"
            ),
        )
    else:
        verdict, why = (
            "keep testing",
            "targeting is not clearly better than random or not clearly profitable",
        )

    results = {
        "dataset": "Criteo Uplift v2.1 (CC BY-NC-SA 4.0), seeded sample",
        "git_commit_at_run": git_commit(),
        "config": CFG,
        "n_rows": len(d),
        "split_sizes": {k: len(v) for k, v in D.items()},
        "srm": srm.to_dict(),
        "balance_max_abs_smd": float(bal.smd.abs().max()),
        "balance": bal.reset_index().to_dict(orient="records"),
        "exposure_rate_treated": float(d.loc[d.treatment == 1, "exposure"].mean()),
        "ate": ate,
        "evaluation": ev,
        "best_model": best,
        "secondary_outcome_qini": {"coefficient": sec[0], "ci": [sec[1], sec[2]]},
        "sensitivity": sens,
        "decision": {"verdict": verdict, "reason": why},
    }
    write_json(results, RESULTS / "criteo.json")
    charts(D["test"], scores, yname, ev)
    render_memos("02_criteo", {"r": results})
    print(
        f"Criteo: SRM p={srm.p_value:.3f}; best={best} qini={b['qini_coefficient']:.5f} "
        f"CI={b['qini_ci']}; decision={verdict}"
    )
    return 0


def charts(te, scores, yname, ev) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (name, lab) in enumerate((("t_learner", "T-learner"), ("x_learner", "X-learner"))):
        c = uplift_curve(te[yname], te.treatment, scores[name]["test"])
        ax.plot(
            c.fraction * 100,
            c.qini,
            color=SERIES[j],
            label=f"{lab} (Qini coef {c.qini_coefficient:.4f})",
        )
    ax.plot(c.fraction * 100, c.random_qini, color=INK_2, ls="--", lw=1.5, label="Random targeting")
    ax.set_xlabel("Share of users targeted, ranked by predicted uplift (%)")
    ax.set_ylabel("Incremental visits (held-out test set)")
    ax.set_title(f"Criteo: Qini curve on {len(te):,} held-out users")
    ax.legend(loc="lower right")
    save(fig, FIGURES / "criteo_qini.png")

    best = max(ev, key=lambda k: ev[k]["qini_coefficient"])
    dec = ev[best]["uplift_by_decile"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = [r["bin"] for r in dec]
    yv = [100 * r["observed_uplift"] for r in dec]
    err = [
        [100 * (r["observed_uplift"] - r["ci_low"]) for r in dec],
        [100 * (r["ci_high"] - r["observed_uplift"]) for r in dec],
    ]
    ax.bar(x, yv, color=SERIES[0], width=0.6)
    ax.errorbar(x, yv, yerr=err, fmt="none", ecolor=INK_2, capsize=3, lw=1)
    ax.axhline(100 * ev[best]["decision"]["breakeven_uplift"], color=INK_2, ls="--", lw=1)
    be = 100 * ev[best]["decision"]["breakeven_uplift"]
    ax.text(10.45, be + 0.15, "break-even", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted-uplift decile (1 = highest)")
    ax.set_ylabel("Observed visit uplift (pp)")
    ax.set_title("Observed uplift by predicted decile, test set (95% CI)")
    ax.grid(axis="x", visible=False)
    save(fig, FIGURES / "criteo_deciles.png")


if __name__ == "__main__":
    sys.exit(main())
