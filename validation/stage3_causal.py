"""Stage 3 validation: when do observational estimators recover the truth, and when do they fail?

Writes ``results/validation/stage3.json`` and charts in ``assets/validation/``.
Run:  python validation/stage3_causal.py [--quick]

Observational (unit-level, true ATE = 0.5 unless noted):
  O1 measured confounding: naive, regression adjustment, IPW (sandwich), AIPW
  O2 same DGP: IPW with bootstrap SE (fewer reps; each fit is bootstrapped)
  O3 binary outcome with measured confounding (the GA360 case-study setting)
  O4 unmeasured confounding: every method that adjusts only for X is biased
  O5 PS/outcome model omits a confounder (x1): same failure, self-inflicted
  O6 poor overlap (strong confounding): IPW variance, AIPW, trimming
Panel DiD (true ATT = 1.0):
  D1 parallel trends, AR(1) errors: clustered vs classical SE coverage
  D2 small trend violation (0.03/period): biased, pre-trend test rarely flags it
  D3 large trend violation (0.15/period): biased, pre-trend test flags it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from xpkit._io import write_json
from xpkit.causal import (
    aipw,
    did,
    e_value,
    event_study,
    fit_propensity,
    ipw,
    rr_from_smd,
    trim_mask,
)
from xpkit.estimate import diff_in_means, lin_regression_adjustment
from xpkit.sim import (
    ExperimentSpec,
    PanelSpec,
    SimulationDraw,
    run_validation,
    simulate_experiment,
    simulate_panel,
)
from xpkit.types import Estimate
from xpkit.viz import INK_2, MUTED, SERIES, apply_style, plt, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "stage3.json"
FIG = ROOT / "assets" / "validation"
COVS = ["x0", "x1", "x2"]


def obs_draw(spec: ExperimentSpec):
    def f(seed: int) -> SimulationDraw:
        sim = simulate_experiment(spec, seed)
        return SimulationDraw(sim.data, sim.true_ate)

    return f


def naive(d) -> Estimate:
    return diff_in_means(d.y.to_numpy(), d.treatment.to_numpy())


def reg_adj(d) -> Estimate:
    return lin_regression_adjustment(d.y.to_numpy(), d.treatment.to_numpy(), d[COVS].to_numpy())


def ipw_sw(d) -> Estimate:
    return ipw(d, "y", "treatment", COVS)


def aipw_lin(d) -> Estimate:
    return aipw(d, "y", "treatment", COVS)


def ipw_boot(d) -> Estimate:
    return ipw(d, "y", "treatment", COVS, se="bootstrap", n_boot=100, seed=1)


def ipw_trimmed(d) -> Estimate:
    ps = fit_propensity(d, "treatment", COVS)
    keep = trim_mask(ps, 0.1, 0.9)
    return ipw(d.loc[keep].reset_index(drop=True), "y", "treatment", COVS)


def ipw_omit(d) -> Estimate:
    return ipw(d, "y", "treatment", ["x0", "x2"])


def aipw_omit(d) -> Estimate:
    return aipw(d, "y", "treatment", ["x0", "x2"])


def summarise(res) -> dict:
    return {k: v.to_dict() for k, v in res.items()}


def observational(reps: int) -> dict:
    base = ExperimentSpec(n=4000, effect=0.5, confounding=0.8)
    four = {"naive": naive, "regression_adjustment": reg_adj, "ipw": ipw_sw, "aipw": aipw_lin}
    out: dict = {}
    out["O1_measured_confounding"] = {
        "spec": base.__dict__,
        "results": summarise(run_validation(obs_draw(base), four, reps, seed=301)),
    }
    out["O2_ipw_bootstrap"] = {
        "spec": base.__dict__,
        "results": summarise(
            run_validation(
                obs_draw(base), {"ipw_bootstrap": ipw_boot}, max(reps // 5, 20), seed=302
            )
        ),
    }
    binary = ExperimentSpec(n=20_000, outcome="binary", baseline=0.05, effect=0.01, confounding=0.8)
    out["O3_binary_measured_confounding"] = {
        "spec": binary.__dict__,
        "results": summarise(
            run_validation(
                obs_draw(binary), {"naive": naive, "ipw": ipw_sw, "aipw": aipw_lin}, reps, seed=303
            )
        ),
    }
    hidden = base.with_(unobserved_confounding=0.8)
    res = run_validation(obs_draw(hidden), four, reps, seed=304)
    # E-value an analyst would report from a typical (seed 0) dataset, continuous-outcome
    # approximation RR ~= exp(0.91 * d), d = estimate / SD(y).
    d0 = simulate_experiment(hidden, 0).data
    e0 = aipw_lin(d0)
    sd = float(d0.y.std())
    ev = e_value(
        rr_from_smd(e0.estimate / sd), rr_from_smd(e0.ci_low / sd), rr_from_smd(e0.ci_high / sd)
    )
    out["O4_unmeasured_confounding"] = {
        "spec": hidden.__dict__,
        "results": summarise(res),
        "example_aipw_estimate_seed0": e0.estimate,
        "example_e_value": ev.to_dict(),
    }
    out["O5_omitted_confounder_in_models"] = {
        "spec": base.__dict__,
        "results": summarise(
            run_validation(
                obs_draw(base),
                {"ipw_without_x1": ipw_omit, "aipw_without_x1": aipw_omit},
                reps,
                seed=305,
            )
        ),
    }
    poor = base.with_(confounding=2.0)
    res = run_validation(
        obs_draw(poor),
        {"naive": naive, "ipw": ipw_sw, "aipw": aipw_lin, "ipw_trimmed_0.1_0.9": ipw_trimmed},
        reps,
        seed=306,
        max_failure_rate=0.05,
    )
    d0 = simulate_experiment(poor, 0).data
    ov = ipw_sw(d0).extra
    out["O6_poor_overlap"] = {
        "spec": poor.__dict__,
        "results": summarise(res),
        "example_overlap_seed0": ov["overlap"],
        "note": "Effect is homogeneous here, so trimming changes the target "
        "population but not the true value (0.5). With HTE it would.",
    }
    return out


def panel(reps: int) -> dict:
    out: dict = {}
    for key, spec in (
        ("D1_parallel_trends", PanelSpec(n_units=200, effect=1.0, ar_rho=0.6)),
        (
            "D2_small_trend_violation",
            PanelSpec(n_units=200, effect=1.0, ar_rho=0.6, trend_violation=0.03),
        ),
        (
            "D3_large_trend_violation",
            PanelSpec(n_units=200, effect=1.0, ar_rho=0.6, trend_violation=0.15),
        ),
    ):

        def draw(seed: int, spec=spec) -> SimulationDraw:
            p = simulate_panel(spec, seed)
            return SimulationDraw(p.data, p.true_att)

        res = run_validation(
            draw,
            {"twfe_cluster": lambda d: did(d), "twfe_classical": lambda d: did(d, se="classical")},
            reps,
            seed=400 + len(out),
        )
        rej = [
            event_study(simulate_panel(spec, int(s)).data).pretrend_rejected
            for s in np.random.SeedSequence(410 + len(out)).generate_state(reps)
        ]
        bias_theory = spec.trend_violation * (
            np.mean(range(spec.n_post)) - np.mean(range(-spec.treat_start, 0))
        )
        out[key] = {
            "spec": spec.__dict__,
            "results": summarise(res),
            "pretrend_test_rejection_rate": float(np.mean(rej)),
            "theoretical_bias": float(bias_theory),
        }
    return out


def charts(r: dict) -> None:
    apply_style()
    o = r["observational"]
    scen = [
        ("O1_measured_confounding", "Measured\nconfounding"),
        ("O4_unmeasured_confounding", "Unmeasured\nconfounding"),
        ("O5_omitted_confounder_in_models", "Confounder\nomitted"),
        ("O6_poor_overlap", "Poor\noverlap"),
    ]
    ests = [("naive", "Naive difference"), ("ipw", "IPW"), ("aipw", "AIPW")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, (key, name) in enumerate(ests):
        xs, ys, es = [], [], []
        for i, (s, _) in enumerate(scen):
            res = o[s]["results"]
            k = key if key in res else {"ipw": "ipw_without_x1", "aipw": "aipw_without_x1"}.get(key)
            if k not in res:
                continue
            xs.append(i + (j - 1) * 0.22)
            ys.append(res[k]["bias"])
            es.append(1.96 * res[k]["empirical_sd"])
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=7, color=SERIES[j], label=name, capsize=3)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(range(len(scen)), [s[1] for s in scen])
    ax.set_ylabel("Bias (estimate − truth)")
    ax.set_title("Adjustment removes measured confounding only; it can't fix what isn't measured")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "observational_bias.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, (trend, name) in enumerate(
        ((0.0, "Parallel trends hold"), (0.15, "Trend violation 0.15/period"))
    ):
        es = event_study(simulate_panel(PanelSpec(n_units=200, trend_violation=trend), 7).data)
        c = es.coefficients
        x = c.event_time + (j - 0.5) * 0.18
        ax.errorbar(
            x,
            c.estimate,
            yerr=[c.estimate - c.ci_low, c.ci_high - c.estimate],
            fmt="o",
            ms=6,
            color=SERIES[j],
            capsize=2,
            label=f"{name} (pre-trend p={es.pretrend_p_value:.2g})",
        )
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(-0.5, color=INK_2, lw=1, ls="--")
    ax.set_xlabel("Periods relative to treatment (reference = −1)")
    ax.set_ylabel("Effect vs reference period")
    ax.set_title("Event study: pre-period coefficients expose a parallel-trends violation (seed 7)")
    ax.legend(loc="upper left")
    save(fig, FIG / "event_study.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    reps = 40 if args.quick else 1000
    results = {"quick": args.quick, "observational": observational(reps), "panel": panel(reps)}
    out = OUT.with_name("stage3_quick.json") if args.quick else OUT
    write_json(results, out)
    if not args.quick:
        charts(results)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
