# Method-selection guide

Start from **how treatment was assigned**, not from the method you would like to use.

```
Was treatment randomized?
├─ yes ─ Do the trust checks pass (SRM, balance, no duplicate units)?
│        ├─ no  ─ STOP. Fix assignment/logging; no estimator rescues a broken experiment.
│        └─ yes ─ Difference in means/proportions (+ CUPED if a pre-period metric exists)
│                 ├─ several metrics or arms? ─ Holm (decision metrics) / BH (exploration)
│                 ├─ looking before the end?  ─ group-sequential boundaries
│                 └─ "who benefits?"          ─ uplift learners, evaluated on held-out data
└─ no ── Is there a before/after with an untreated comparison group?
         ├─ yes ─ Difference-in-differences + event study (if trends are plausibly parallel)
         └─ no ── Are all confounders plausibly measured, pre-treatment?
                  ├─ yes ─ IPW / AIPW + overlap and balance diagnostics + E-value
                  └─ no ── Don't claim a causal effect. Report associations, run an experiment.
```

| Method | Use when | Key assumptions | Don't use when | Check | xpkit |
|---|---|---|---|---|---|
| SRM χ² test | Always, first | Designed split is known | Never skip it | p < 0.001 → stop. Passing is necessary, not sufficient: subtle selective loss can pass | `trust.srm_test` |
| Balance (SMD) | Randomized: sanity check. Observational: before and after weighting | Covariates are pre-treatment | As proof that unmeasured confounders are balanced | \|SMD\| < 0.1 | `trust.balance_table` |
| Diff in proportions (Newcombe CI) | Binary metric, randomized | Independent units | Units are clustered (use cluster-robust means) | Coverage holds at small counts (slightly conservative) | `estimate.diff_in_proportions` |
| Welch / cluster-robust diff in means | Continuous metric; clustered when the analysis unit is below the randomization unit | Independent clusters | Few clusters (< ~30): CR1 undercovers | Cluster count, SE inflation | `estimate.diff_in_means(cluster=)` |
| CUPED | A pre-period version of the metric exists | Covariate is measured before assignment | Covariate could be affected by treatment | Reported variance reduction ≈ ρ² | `estimate.cuped` |
| Lin regression adjustment | Several pre-period covariates | Same as CUPED | Many covariates relative to n | HC2 SEs | `estimate.lin_regression_adjustment` |
| Holm / BH | Several metrics, arms or segments | BH: independence or positive dependence | Post-hoc slicing beyond the declared family | Declare the family in advance | `estimate.adjust_pvalues` |
| Group-sequential (OBF / Pocock) | You will look before the planned end | Looks fixed in advance; information fractions known | Unplanned looks, or changing the sample size mid-test | Stop only when \|z\| ≥ boundary | `estimate.group_sequential_design` |
| Propensity + IPW | Observational, all confounders measured | No unmeasured confounding; positivity | Poor overlap (use AIPW/trimming), post-treatment covariates | Overlap, ESS, weighted balance | `causal.ipw` |
| AIPW | Same, and you want robustness to one misspecified model | Either the PS or the outcome model is correct | Unmeasured confounding (still biased) | Same, plus clipping count | `causal.aipw` |
| E-value | After any observational estimate | Risk-ratio scale | As a test of "no confounding" | Compare with plausible confounder strength | `causal.e_value` |
| DiD (TWFE) | Untreated comparison group, single adoption date | Parallel trends, no anticipation, no spillovers | Staggered adoption with heterogeneous effects | Event-study pre-trends (weak test!) | `causal.did`, `causal.event_study` |
| T-learner | Randomized; both arms large | Overlap; enough data per arm | Strongly unbalanced arms | Held-out Qini vs random | `uplift.TLearner` |
| X-learner | Randomized; unbalanced arms | Same | Tiny data (four models) | Held-out Qini vs random | `uplift.XLearner` |
| Targeting policy | Treatment has a cost | Value and cost known | Choosing depth on the test split | Validation-chosen depth, test-reported value | `uplift.choose_depth` |

**Rules that held up in the validation studies** (`results/validation/`):

1. A significant pre-trend test means "stop". A passing pre-trend test does not mean "safe".
2. Under poor overlap, IPW SEs are too small. Prefer AIPW and report the effective sample size.
3. Adjustment fixes what you measure. With an unmeasured confounder, IPW, AIPW and regression were equally biased and their diagnostics looked fine.
4. Never adjust for variables measured after treatment starts (pageviews, time on site, in-session events). They can create bias even when there was none.
