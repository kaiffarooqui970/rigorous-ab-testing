"""Stage 2 validation: do the experiment-analysis estimators behave as claimed?

Writes ``results/validation/stage2.json`` and charts in ``assets/validation/``.
Run:  python validation/stage2_experiments.py [--quick]

Studies (all seeded):
  A. CI coverage, false-positive rate and power for diff-in-means and diff-in-proportions,
     including a small-count case comparing Newcombe with Wald intervals.
  B. CUPED and Lin regression adjustment vs unadjusted, at three levels of covariate
     predictiveness: bias, SE, empirical variance reduction, coverage.
  C. Interim looks: naive peeking vs Bonferroni vs O'Brien-Fleming vs Pocock vs single look.
  D. Multiple metrics: FWER/FDR of unadjusted, Holm and BH testing across 10 correlated metrics.
  E. SRM: detection rate of the chi-square SRM check and the bias from selective loss.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from xpkit._io import write_json
from xpkit.estimate import adjust_pvalues, cuped, diff_in_means, diff_in_proportions
from xpkit.estimate.cuped import lin_regression_adjustment
from xpkit.sim import ExperimentSpec, SimulationDraw, run_validation, simulate_experiment
from xpkit.sim.peeking import simulate_peeking_strategies, strategies_to_records
from xpkit.trust import balance_table, srm_test
from xpkit.types import Estimate, normal_estimate
from xpkit.viz import CRITICAL, INK_2, MUTED, SERIES, apply_style, plt, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "stage2.json"
FIG = ROOT / "assets" / "validation"


def draws(spec: ExperimentSpec):
    def f(seed: int) -> SimulationDraw:
        sim = simulate_experiment(spec, seed)
        return SimulationDraw(sim.data, sim.true_ate)

    return f


def est_dim(d) -> Estimate:
    return diff_in_means(d.y.to_numpy(), d.treatment.to_numpy())


def est_prop(d) -> Estimate:
    t, y = d.treatment.to_numpy(), d.y.to_numpy()
    return diff_in_proportions(
        int(y[t == 1].sum()), int((t == 1).sum()), int(y[t == 0].sum()), int((t == 0).sum())
    )


def est_wald(d) -> Estimate:
    t, y = d.treatment.to_numpy(), d.y.to_numpy()
    p1, p0 = y[t == 1].mean(), y[t == 0].mean()
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    if se == 0:
        raise ValueError("Wald SE is zero (no conversions in either arm)")
    return normal_estimate(p1 - p0, se, "wald")


def est_cuped(d) -> Estimate:
    return cuped(d.y.to_numpy(), d.treatment.to_numpy(), d.pre_metric.to_numpy())


def est_lin(d) -> Estimate:
    return lin_regression_adjustment(
        d.y.to_numpy(), d.treatment.to_numpy(), d[["pre_metric", "x1", "x2"]].to_numpy()
    )


def study_a(reps: int) -> dict:
    out = {}
    cases = {
        "continuous_null": (ExperimentSpec(n=2000, effect=0.0), {"diff_in_means": est_dim}),
        "continuous_effect": (ExperimentSpec(n=2000, effect=0.15), {"diff_in_means": est_dim}),
        "binary_null": (
            ExperimentSpec(n=10_000, outcome="binary", baseline=0.05),
            {"newcombe": est_prop, "wald": est_wald},
        ),
        "binary_effect": (
            ExperimentSpec(n=10_000, outcome="binary", baseline=0.05, effect=0.01),
            {"newcombe": est_prop, "wald": est_wald},
        ),
        "binary_small_counts": (
            ExperimentSpec(n=600, outcome="binary", baseline=0.01, covariate_effect=0.0),
            {"newcombe": est_prop, "wald": est_wald},
        ),
    }
    for i, (name, (spec, ests)) in enumerate(cases.items()):
        res = run_validation(draws(spec), ests, n_reps=reps, seed=1000 + i, max_failure_rate=0.05)
        out[name] = {"spec": spec.__dict__, "results": {k: v.to_dict() for k, v in res.items()}}
    return out


def study_b(reps: int) -> dict:
    out = {}
    for label, beta in (("weak", 0.0), ("moderate", 0.5), ("strong", 2.0)):
        spec = ExperimentSpec(n=2000, effect=0.1, covariate_effect=beta, noise_sd=1.0)
        res = run_validation(
            draws(spec),
            {"unadjusted": est_dim, "cuped": est_cuped, "lin": est_lin},
            n_reps=reps,
            seed=100 + int(beta * 10),
        )
        sim = simulate_experiment(spec.with_(n=200_000), seed=0).data
        rho = float(np.corrcoef(sim.y, sim.pre_metric)[0, 1])
        var_red = 1 - (res["cuped"].empirical_sd / res["unadjusted"].empirical_sd) ** 2
        out[label] = {
            "covariate_effect": beta,
            "rho_y_pre": rho,
            "theoretical_variance_reduction": rho**2,
            "empirical_variance_reduction_cuped": var_red,
            "empirical_variance_reduction_lin": 1
            - (res["lin"].empirical_sd / res["unadjusted"].empirical_sd) ** 2,
            "results": {k: v.to_dict() for k, v in res.items()},
        }
    return out


def study_c(sims: int) -> dict:
    n_looks, baseline, lift = 10, 0.05, 0.10
    from xpkit.design import sample_size_proportions

    n_fixed = sample_size_proportions(baseline, lift).n_control
    per_look = math.ceil(n_fixed / n_looks)
    null = simulate_peeking_strategies(baseline, 0.0, n_looks, per_look, n_sims=sims, seed=11)
    alt = simulate_peeking_strategies(baseline, lift, n_looks, per_look, n_sims=sims, seed=12)
    return {
        "n_looks": n_looks,
        "baseline": baseline,
        "relative_lift": lift,
        "n_per_look_per_arm": per_look,
        "n_sims": sims,
        "null": strategies_to_records(null),
        "alternative": strategies_to_records(alt),
    }


def study_d(sims: int) -> dict:
    """10 metrics, correlation 0.3. Scenario 1: all null. Scenario 2: 3 real effects (z shift 3)."""
    rng = np.random.default_rng(21)
    m, rho = 10, 0.3
    cov = np.full((m, m), rho) + (1 - rho) * np.eye(m)
    out = {}
    for scen, shift in (
        ("all_null", np.zeros(m)),
        ("three_true", np.r_[np.full(3, 3.0), np.zeros(7)]),
    ):
        z = rng.multivariate_normal(shift, cov, size=sims)
        p = 2 * stats.norm.sf(np.abs(z))
        is_null = shift == 0
        res = {}
        for method in ("none", "holm", "bh"):
            rej = np.array([adjust_pvalues(row, method).reject for row in p])
            fp = (rej & is_null).sum(axis=1)
            disc = rej.sum(axis=1)
            res[method] = {
                "fwer": float((fp > 0).mean()),
                "fdr": float(np.mean(np.where(disc > 0, fp / np.maximum(disc, 1), 0))),
                "power_per_true_effect": float(rej[:, ~is_null].mean())
                if (~is_null).any()
                else None,
            }
        out[scen] = res
    out["config"] = {"n_metrics": m, "correlation": rho, "n_sims": sims, "true_effect_z": 3.0}
    return out


def study_e(reps: int) -> dict:
    rng = np.random.default_rng(31)
    n = 100_000
    detect = {}
    for drop in (0.0, 0.005, 0.01, 0.02):
        flags = []
        for _ in range(reps):
            t = rng.random(n) < 0.5
            kept_t = rng.binomial(t.sum(), 1 - drop)
            flags.append(not srm_test([int((~t).sum()), int(kept_t)]).passed)
        detect[str(drop)] = float(np.mean(flags))
    spec = ExperimentSpec(n=20_000, effect=0.0, srm_drop=0.05, srm_mechanism="selective")
    res = run_validation(draws(spec), {"diff_in_means": est_dim}, n_reps=min(reps, 500), seed=32)
    # How often does each trust check flag this (biased) experiment?
    n_check = min(reps, 500)
    srm_flag, bal_flag = [], []
    for s in np.random.SeedSequence(33).generate_state(n_check):
        d = simulate_experiment(spec, int(s)).data
        counts = d.treatment.value_counts().sort_index().tolist()
        srm_flag.append(not srm_test(counts).passed)
        smd = balance_table(d, "treatment", ["pre_metric"]).loc["pre_metric", "smd"]
        bal_flag.append(abs(smd) > 0.05)
    sim = simulate_experiment(spec, seed=0)
    counts = sim.data.treatment.value_counts().sort_index().tolist()
    return {
        "n_units": n,
        "threshold": 0.001,
        "detection_rate_by_drop": detect,
        "selective_srm_example": {
            "spec": spec.__dict__,
            "counts_control_treatment_seed0": counts,
            "srm_p_value_seed0": srm_test(counts).p_value,
            "naive_estimator": res["diff_in_means"].to_dict(),
            "n_check_reps": n_check,
            "srm_flag_rate": float(np.mean(srm_flag)),
            "pre_metric_smd_gt_0.05_flag_rate": float(np.mean(bal_flag)),
        },
    }


def charts(r: dict) -> None:
    apply_style()
    # Peeking: false-positive rate by strategy (single series, highlight the failure).
    rows = {x["strategy"]: x for x in r["C_sequential"]["null"]}
    order = ["single_look", "naive_peeking", "bonferroni", "pocock", "obrien_fleming"]
    labels = ["Single look", "Naive peeking", "Bonferroni", "Pocock", "O'Brien-Fleming"]
    vals = [rows[k]["rejection_rate"] for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [CRITICAL if k == "naive_peeking" else SERIES[0] for k in order]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.55)
    ax.axvline(0.05, color=INK_2, lw=1, ls="--")
    ax.text(0.052, -0.55, "nominal 5%", color=INK_2, fontsize=8, va="center")
    for b, v in zip(bars, vals[::-1], strict=True):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlabel("False-positive rate under H0 (10 looks)")
    ax.set_title(
        f"Peeking inflates false positives; sequential boundaries don't "
        f"({r['C_sequential']['n_sims']:,} sims)"
    )
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "sequential_fpr.png")

    # CUPED: CI half-width by predictiveness (3 series).
    b = r["B_cuped"]
    labs = ["weak", "moderate", "strong"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (est, name) in enumerate(
        (("unadjusted", "Unadjusted"), ("cuped", "CUPED"), ("lin", "Lin regression"))
    ):
        sd = [b[k]["results"][est]["empirical_sd"] for k in labs]
        ax.bar(x + (i - 1) * 0.26, sd, width=0.24, color=SERIES[i], label=name)
    ax.set_xticks(x, [f"{k}\n(ρ={b[k]['rho_y_pre']:.2f})" for k in labs])
    ax.set_ylabel("SD of effect estimate")
    ax.set_title("CUPED shrinks estimator noise as the pre-period covariate gets more predictive")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.115)
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "cuped_variance.png")

    # Coverage: newcombe vs wald small counts (2 series dot plot).
    a = r["A_effects"]
    cases = ["binary_null", "binary_effect", "binary_small_counts"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for i, (est, name) in enumerate((("newcombe", "Newcombe"), ("wald", "Wald"))):
        cov = [a[c]["results"][est]["coverage"] for c in cases]
        err = [2 * a[c]["results"][est]["coverage_mcse"] for c in cases]
        ax.errorbar(
            cov,
            np.arange(3) + (0.12 if i else -0.12),
            xerr=err,
            fmt="o",
            ms=7,
            color=SERIES[i],
            label=name,
            capsize=3,
        )
    ax.axvline(0.95, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(3), ["n=10k, p=5%, null", "n=10k, p=5%, +1pp", "n=600, p=1%, null"])
    ax.set_xlabel("95% CI coverage (±2 MC SE)")
    ax.set_title("Both near 95% at n=10k; at small counts Newcombe is conservative")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIG / "ci_coverage.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="fewer replications (smoke run)")
    args = ap.parse_args(argv)
    reps, sims = (100, 4000) if args.quick else (2000, 100_000)
    results = {
        "quick": args.quick,
        "A_effects": study_a(reps),
        "B_cuped": study_b(reps),
        "C_sequential": study_c(sims),
        "D_multiplicity": study_d(min(sims, 20_000)),
        "E_srm": study_e(min(reps, 1000)),
    }
    out = OUT.with_name("stage2_quick.json") if args.quick else OUT
    write_json(results, out)
    if not args.quick:
        charts(results)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
