"""xpkit experimentation workbench.

Tabs: Planning, Analysis, Causal diagnostics, Uplift, and the original GA360 explorer.
Everything except the GA360 tab runs on bundled or simulated synthetic data, with no
cloud credentials.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from xpkit.causal import aipw, did, event_study, fit_propensity, ipw  # noqa: E402
from xpkit.design import mde_means, mde_proportions, sample_size_means, sample_size_proportions  # noqa: E402
from xpkit.estimate import diff_in_means, group_sequential_design  # noqa: E402
from xpkit.report import analyze  # noqa: E402
from xpkit.sim import ExperimentSpec, PanelSpec, simulate_experiment, simulate_panel  # noqa: E402
from xpkit.trust import balance_table  # noqa: E402
from xpkit.uplift import Economics, TLearner, XLearner, policy_table, uplift_curve  # noqa: E402
from xpkit.viz import INK_2, MUTED, SERIES, apply_style, plt  # noqa: E402

st.set_page_config(page_title="xpkit experimentation workbench", layout="wide")
apply_style()
EXAMPLES = ROOT / "examples"
COVS = ["x0", "x1", "x2"]

st.title("Experimentation workbench")
st.caption("Trust checks first, effects second. All tabs except the GA360 explorer use synthetic data.")
tab_plan, tab_an, tab_causal, tab_up, tab_ga = st.tabs(
    ["Planning", "Analysis", "Causal diagnostics", "Uplift", "GA360 explorer (observational)"]
)

# ----------------------------------------------------------------------------- planning
with tab_plan:
    st.subheader("Sample size and minimum detectable effect")
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Metric type", ["Conversion rate", "Continuous (e.g. revenue)"])
    alpha = c2.slider("α (two-sided)", 0.01, 0.10, 0.05, 0.01)
    power = c3.slider("Power", 0.5, 0.95, 0.8, 0.05)
    ratio = c3.number_input("Allocation ratio (treatment / control)", 0.1, 10.0, 1.0, 0.1)
    if kind == "Conversion rate":
        base = c1.number_input("Baseline rate", 0.0001, 0.99, 0.05, 0.001, format="%.4f")
        mde = c2.number_input("Relative MDE", 0.005, 2.0, 0.10, 0.005, format="%.3f")
        try:
            ss = sample_size_proportions(base, mde, alpha=alpha, power=power, ratio=ratio)
            st.metric("Units needed", f"{ss.n_total:,}", f"{ss.n_control:,} control / {ss.n_treatment:,} treatment")
        except ValueError as exc:
            st.error(str(exc))
        n_have = st.number_input("…or: control units available", 100, 100_000_000, 50_000, 1000)
        try:
            st.write(f"Detectable relative lift with {n_have:,} control units: "
                     f"**{mde_proportions(base, n_have, alpha=alpha, power=power, ratio=ratio):.2%}**")
        except ValueError as exc:
            st.warning(str(exc))
    else:
        sd = c1.number_input("Outcome SD", 0.01, 1e6, 10.0)
        mde = c2.number_input("Absolute MDE", 0.0001, 1e6, 0.5)
        rho = c2.slider("CUPED: corr(outcome, pre-period covariate)", 0.0, 0.95, 0.0, 0.05)
        ss = sample_size_means(sd, mde, alpha=alpha, power=power, ratio=ratio, cuped_rho=rho)
        base_ss = sample_size_means(sd, mde, alpha=alpha, power=power, ratio=ratio)
        st.metric("Units needed", f"{ss.n_total:,}",
                  f"{ss.n_total - base_ss.n_total:,} vs no CUPED" if rho else None, delta_color="inverse")
        n_have = st.number_input("…or: control units available", 100, 100_000_000, 10_000, 1000)
        st.write(f"Detectable difference: **{mde_means(sd, n_have, alpha=alpha, power=power, ratio=ratio, cuped_rho=rho):.4g}**")

    st.subheader("Group-sequential boundaries")
    k = st.slider("Planned looks", 1, 10, 5)
    spend = st.radio("Spending function", ["obrien_fleming", "pocock"], horizontal=True)
    des = group_sequential_design(k, alpha, spend)  # type: ignore[arg-type]
    st.dataframe(pd.DataFrame({"look": range(1, k + 1), "information": des.info_fractions,
                               "z boundary": np.round(des.boundaries, 4),
                               "nominal p threshold": des.nominal_p_thresholds(),
                               "cumulative α spent": des.alpha_spent}), hide_index=True)
    st.caption("Stop at the first look where |z| ≥ boundary. Peeking at every look with p < α "
               "instead inflates false positives (validation: 19.2% at 10 looks).")

# ----------------------------------------------------------------------------- analysis
with tab_an:
    st.subheader("Analyse an experiment")
    src = st.radio("Data", ["Bundled: healthy test", "Bundled: test with logging bug (SRM)", "Upload CSV"],
                   horizontal=True)
    if src.startswith("Bundled"):
        stem = "checkout_test" if "healthy" in src else "checkout_test_srm"
        df = pd.read_csv(EXAMPLES / f"{stem}.csv")
        cfg_text = (EXAMPLES / f"{stem}.yaml").read_text()
    else:
        up = st.file_uploader("CSV with one row per unit", type="csv")
        df = pd.read_csv(up) if up else None
        cfg_text = (EXAMPLES / "checkout_test.yaml").read_text()
    cfg_text = st.text_area("Config (YAML)", cfg_text, height=230)
    if df is not None:
        try:
            r = analyze(df, yaml.safe_load(cfg_text))
        except Exception as exc:  # show config/data problems to the user, don't crash the app
            st.error(f"Could not analyse: {exc}")
        else:
            st.markdown("#### 1. Trust checks")
            if not r.trustworthy:
                st.error("⚠ " + r.headline)
            for c in r.checks:
                (st.success if c.passed else st.error if c.severity == "critical" else st.warning)(
                    f"**{c.name}**: {c.detail}")
            st.markdown("#### 2. Effect estimates" + ("" if r.trustworthy else " (NOT interpretable)"))
            if r.trustworthy:
                st.info(r.headline)
            st.dataframe(pd.DataFrame([{
                "metric": m.name + (" (primary)" if m.primary else ""), "control": m.control_mean,
                "treatment": m.treatment_mean, "effect": m.estimate, "ci_low": m.ci_low,
                "ci_high": m.ci_high, "p": m.p_value, "adj. p": m.adjusted_p_value, "MDE 80%": m.mde_80,
                "CUPED effect": m.cuped["estimate"] if m.cuped else None,
                "CUPED var. reduction": m.cuped["variance_reduction"] if m.cuped else None,
            } for m in r.metrics]), hide_index=True)

# ----------------------------------------------------------------------------- causal
with tab_causal:
    st.subheader("Observational adjustment on simulated data (true ATE = 0.5)")
    c1, c2, c3 = st.columns(3)
    conf = c1.slider("Measured confounding", 0.0, 3.0, 0.8, 0.1)
    hidden = c2.slider("Unmeasured confounding", 0.0, 1.5, 0.0, 0.1)
    n = c3.select_slider("n", [1000, 2000, 4000, 8000, 16000], 4000)

    @st.cache_data
    def obs(conf: float, hidden: float, n: int) -> dict:
        d = simulate_experiment(ExperimentSpec(n=n, effect=0.5, confounding=conf,
                                               unobserved_confounding=hidden), 0).data
        ests = {"Naive difference": diff_in_means(d.y, d.treatment),
                "IPW": ipw(d, "y", "treatment", COVS), "AIPW": aipw(d, "y", "treatment", COVS)}
        ps = fit_propensity(d, "treatment", COVS)
        w = np.where(d.treatment == 1, 1 / ps, 1 / (1 - ps))
        bal = pd.DataFrame({"before": balance_table(d, "treatment", COVS).smd,
                            "after IPW": balance_table(d, "treatment", COVS, weights=w).smd})
        return {"ests": ests, "ps": ps, "t": d.treatment.to_numpy(), "bal": bal}

    o = obs(conf, hidden, n)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        for i, e in enumerate(o["ests"].values()):
            ax.errorbar([e.estimate], [i], xerr=[[e.estimate - e.ci_low], [e.ci_high - e.estimate]],
                        fmt="o", color=SERIES[0], capsize=3)
        ax.axvline(0.5, color=INK_2, ls="--", lw=1)
        ax.set_yticks(range(3), list(o["ests"]))
        ax.set_title("Estimates vs truth (dashed)")
        st.pyplot(fig)
        if hidden > 0:
            st.warning("With unmeasured confounding, no adjustment for x recovers the truth, and "
                       "the diagnostics on the right still look fine. That's the point.")
    with right:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        bins = np.linspace(0, 1, 41)
        ax.hist(o["ps"][o["t"] == 1], bins, alpha=0.7, color=SERIES[0], label="treated")
        ax.hist(o["ps"][o["t"] == 0], bins, alpha=0.7, color=SERIES[1], label="control")
        ax.set_title("Propensity overlap")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(o["bal"].round(3))

    st.subheader("Difference-in-differences (true ATT = 1.0)")
    c1, c2 = st.columns(2)
    trend = c1.slider("Parallel-trend violation (per period)", 0.0, 0.3, 0.0, 0.01)
    rho = c2.slider("Within-unit error autocorrelation", 0.0, 0.9, 0.6, 0.1)
    p = simulate_panel(PanelSpec(n_units=200, trend_violation=trend, ar_rho=rho), 1).data
    es, dd, dc = event_study(p), did(p), did(p, se="classical")
    c1.metric("DiD (clustered SE)", f"{dd.estimate:.3f}", f"CI {dd.ci_low:.2f} to {dd.ci_high:.2f}", delta_color="off")
    c1.caption(f"Classical-SE CI would be {dc.ci_low:.2f} to {dc.ci_high:.2f}: too narrow with serial correlation.")
    c2.metric("Pre-trend test p", f"{es.pretrend_p_value:.3g}",
              "violation detected" if es.pretrend_rejected else "not detected", delta_color="off")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    cf = es.coefficients
    ax.errorbar(cf.event_time, cf.estimate, yerr=[cf.estimate - cf.ci_low, cf.ci_high - cf.estimate],
                fmt="o", color=SERIES[0], capsize=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(-0.5, color=INK_2, ls="--", lw=1)
    ax.set_xlabel("Event time")
    ax.set_title("Event study (reference period −1)")
    st.pyplot(fig)

# ----------------------------------------------------------------------------- uplift
with tab_up:
    st.subheader("Who to target? Simulated randomized test with known heterogeneity")
    c1, c2, c3 = st.columns(3)
    hte = c1.slider("Uplift heterogeneity (step size on x2)", 0.0, 0.05, 0.02, 0.005, format="%.3f")
    share = c2.slider("Share treated", 0.5, 0.9, 0.5, 0.05)
    n = c3.select_slider("n (total)", [10_000, 25_000, 50_000, 100_000], 25_000)
    value = c1.number_input("Value per incremental conversion ($)", 0.0, 1000.0, 1.0)
    cost = c2.number_input("Cost per treated user ($)", 0.0, 100.0, 0.02, format="%.3f")

    @st.cache_data
    def up(hte: float, share: float, n: int) -> dict:
        spec = ExperimentSpec(n=n, outcome="binary", baseline=0.1, effect=0.02, hte=hte, hte_form="step",
                              n_covariates=5, treat_prob=share)
        d = simulate_experiment(spec, 3).data
        feats = [f"x{i}" for i in range(5)]
        half = n // 2
        tr, te = d.iloc[:half], d.iloc[half:]
        out = {"te": te}
        for name, L in (("T-learner", TLearner), ("X-learner", XLearner)):
            out[name] = L(outcome="binary").fit(tr[feats], tr.treatment, tr.y).predict(te[feats])
        return out

    u = up(hte, share, n)
    te = u["te"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for j, name in enumerate(("T-learner", "X-learner")):
        c = uplift_curve(te.y, te.treatment, u[name])
        ax.plot(c.fraction * 100, c.qini, color=SERIES[j], label=f"{name} (Qini {c.qini_coefficient:.4f})")
    ax.plot(c.fraction * 100, c.random_qini, color=INK_2, ls="--", label="Random")
    ax.set_xlabel("% targeted")
    ax.set_ylabel("Incremental conversions")
    ax.legend()
    st.pyplot(fig)
    st.caption("Single simulated test split (no separate validation split here), so treat the "
               "best depth below as optimistic. The case study picks depth on validation data.")
    tab = policy_table(te.y, te.treatment, u["X-learner"], Economics(value, cost, 1_000_000))
    st.dataframe(tab[["depth", "uplift_top", "profit_per_user", "random_profit_per_user"]].round(5),
                 hide_index=True)

# ----------------------------------------------------------------------------- GA360
with tab_ga:
    st.warning("Observational data: visitors chose their device. Differences are associations, "
               "not effects of the mobile experience. See docs/memos/03_ga360.md.")
    from ab_test_analysis.data import DAILY_DEVICE_CHANNEL_PATH

    if DAILY_DEVICE_CHANNEL_PATH.exists():
        import legacy_ga360_explorer

        legacy_ga360_explorer.render()
    else:
        st.info("This tab needs the cached BigQuery pull, which needs Google Cloud credentials. Run "
                "`pip install -e '.[bigquery]'`, authenticate with `gcloud auth application-default "
                "login`, then `python -c 'from ab_test_analysis.data import "
                "load_daily_device_channel as f; f()'`.")
