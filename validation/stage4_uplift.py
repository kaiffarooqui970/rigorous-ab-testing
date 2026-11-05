"""Stage 4 validation: do T-/X-learners find real heterogeneity, and invent none when there is none?

Each replication draws a fresh randomized dataset, splits it 40/20/40 into
train/validation/test, fits both learners on train, picks the targeting depth on
validation and reports value on test. Run:  python validation/stage4_uplift.py [--quick]

Scenarios (binary outcomes unless noted; economics: value 1 per conversion, cost 0.02 per
treated user, so treating pays only where uplift > 2pp):
  U1 continuous outcome, step HTE on x2 (tau in {0.0, 0.6})
  U2 binary, step HTE (tau in {0, 4pp}), 50/50 allocation
  U3 binary, NO heterogeneity (tau = 2pp for everyone): the model should not beat random
  U4 binary, step HTE, 85/15 allocation (like Criteo): where the X-learner is meant to help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from xpkit._io import write_json
from xpkit.sim import ExperimentSpec, simulate_experiment
from xpkit.uplift import (
    Economics,
    TLearner,
    XLearner,
    choose_depth,
    pehe,
    qini_coefficient_ci,
    uplift_curve,
)
from xpkit.viz import INK_2, SERIES, apply_style, plt, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "stage4.json"
FIG = ROOT / "assets" / "validation"
FEATS = ["x0", "x1", "x2", "x3", "x4"]

SCENARIOS = {
    "U1_continuous_step_hte": ExperimentSpec(
        n=25_000, outcome="continuous", effect=0.3, hte=0.3, hte_form="step", n_covariates=5
    ),
    "U2_binary_step_hte": ExperimentSpec(
        n=25_000,
        outcome="binary",
        baseline=0.10,
        effect=0.02,
        hte=0.02,
        hte_form="step",
        n_covariates=5,
    ),
    "U3_binary_no_hte": ExperimentSpec(
        n=25_000, outcome="binary", baseline=0.10, effect=0.02, hte=0.0, n_covariates=5
    ),
    "U4_binary_step_hte_85_15": ExperimentSpec(
        n=25_000,
        outcome="binary",
        baseline=0.10,
        effect=0.02,
        hte=0.02,
        hte_form="step",
        n_covariates=5,
        treat_prob=0.85,
    ),
}
ECON = Economics(value_per_conversion=1.0, cost_per_treatment=0.02, population=1_000_000)
ECON_CONT = Economics(value_per_conversion=1.0, cost_per_treatment=0.3, population=1_000_000)


def one_rep(spec: ExperimentSpec, seed: int, n_boot: int) -> dict:
    sim = simulate_experiment(spec, seed)
    d, tau = sim.data, sim.truth.tau.to_numpy()
    n = len(d)
    i_tr, i_va = int(0.4 * n), int(0.6 * n)
    tr, va, te = d.iloc[:i_tr], d.iloc[i_tr:i_va], d.iloc[i_va:]
    tau_va, tau_te = tau[i_tr:i_va], tau[i_va:]
    econ = ECON if spec.outcome == "binary" else ECON_CONT
    out: dict = {}
    oracle = choose_depth((va.y, va.treatment, tau_va), (te.y, te.treatment, tau_te), econ)
    out["oracle"] = {
        "qini": uplift_curve(te.y, te.treatment, tau_te).qini_coefficient,
        "test_profit_per_user": oracle.test_profit_per_user,
        "chosen_depth": oracle.chosen_depth,
    }
    for name, Learner in (("t_learner", TLearner), ("x_learner", XLearner)):
        m = Learner(outcome=spec.outcome, seed=seed % 1000).fit(tr[FEATS], tr.treatment, tr.y)
        s_va, s_te = m.predict(va[FEATS]), m.predict(te[FEATS])
        q, lo, hi = qini_coefficient_ci(te.y, te.treatment, s_te, n_boot=n_boot, seed=seed)
        dec = choose_depth((va.y, va.treatment, s_va), (te.y, te.treatment, s_te), econ)
        sd_tau = float(np.std(tau_te))
        out[name] = {
            "pehe": pehe(tau_te, s_te),
            "corr_with_true_tau": float(np.corrcoef(s_te, tau_te)[0, 1]) if sd_tau > 0 else None,
            "qini": q,
            "qini_ci_low": lo,
            "qini_ci_high": hi,
            "qini_ci_excludes_zero": bool(lo > 0 or hi < 0),
            "chosen_depth": dec.chosen_depth,
            "test_profit_per_user": dec.test_profit_per_user,
            "random_profit_same_depth": dec.test_random_profit_per_user,
            "treat_all_profit_per_user": dec.test_treat_all_profit_per_user,
        }
    return out


def aggregate(reps: list[dict]) -> dict:
    agg: dict = {}
    for who in reps[0]:
        agg[who] = {}
        for k, v in reps[0][who].items():
            vals = [r[who][k] for r in reps if r[who][k] is not None]
            if not vals:
                agg[who][k] = None
            elif isinstance(v, bool):
                agg[who][k + "_rate"] = float(np.mean(vals))
            else:
                agg[who][k] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)),
                    "mcse": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                }
    return agg


def chart(results: dict, seed: int = 0) -> None:
    apply_style()
    spec = SCENARIOS["U2_binary_step_hte"]
    sim = simulate_experiment(spec, seed)
    d = sim.data
    n = len(d)
    tr, te = d.iloc[: int(0.4 * n)], d.iloc[int(0.6 * n) :]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (name, L) in enumerate((("T-learner", TLearner), ("X-learner", XLearner))):
        s = L(outcome="binary").fit(tr[FEATS], tr.treatment, tr.y).predict(te[FEATS])
        c = uplift_curve(te.y, te.treatment, s)
        ax.plot(
            c.fraction * 100,
            c.qini,
            color=SERIES[j],
            label=f"{name} (Qini coef {c.qini_coefficient:.4f})",
        )
    ax.plot(c.fraction * 100, c.random_qini, color=INK_2, ls="--", lw=1.5, label="Random targeting")
    ax.set_xlabel("Share of users targeted, ranked by predicted uplift (%)")
    ax.set_ylabel("Incremental conversions (test set)")
    s = results["scenarios"]["U2_binary_step_hte"]["summary"]
    ax.set_title("Qini curve, ONE simulated dataset (seed 0; true uplift 0 or 4pp)")
    ax.text(
        0.01,
        0.97,
        f"Mean Qini over {results['n_reps']} reps: T {s['t_learner']['qini']['mean']:.4f}, "
        f"X {s['x_learner']['qini']['mean']:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_2,
        va="top",
    )
    ax.legend(loc="lower right")
    save(fig, FIG / "qini_simulated.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    n_reps, n_boot = (3, 20) if args.quick else (40, 100)
    results: dict = {
        "quick": args.quick,
        "n_reps": n_reps,
        "n_boot": n_boot,
        "economics_binary": ECON.__dict__,
        "economics_continuous": ECON_CONT.__dict__,
        "scenarios": {},
    }
    for i, (name, spec) in enumerate(SCENARIOS.items()):
        seeds = np.random.SeedSequence(500 + i).generate_state(n_reps)
        reps = [one_rep(spec, int(s), n_boot) for s in seeds]
        results["scenarios"][name] = {"spec": spec.__dict__, "summary": aggregate(reps)}
        print(f"  {name}: done")
    out = OUT.with_name("stage4_quick.json") if args.quick else OUT
    write_json(results, out)
    if not args.quick:
        chart(results)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
