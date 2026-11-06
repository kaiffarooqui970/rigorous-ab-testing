> **Erratum (November 2026).** This memo is kept as originally written. A later audit found problems with it; see `docs/memos/03_ga360.md` for the corrected analysis.
> 1. This is observational data, not an A/B test. Nothing here shows that device *causes* the gap.
> 2. "It doesn't shrink over time" is right, but the README's "stable across the year" was not: the weekly trend is significant (p = 3.5e-5) and the gap *widens*.
> 3. The CIs treat sessions as independent. Clustering by visitor widens them (see the new memo).
> 4. A desktop-only Referral segment, plausibly internal traffic, holds about half of desktop conversions. Excluding it roughly halves the adjusted gap.
> 5. The revenue claim relied on Mann-Whitney, which does not test a difference in *means*.
> 6. "Phone vs tablet was already split out" is incorrect: tablets were excluded.
> 7. The notebook summary says "~2 year test"; the computed figure was 3.8 years.

# Decision Memo: Mobile Checkout Experience

**To:** Head of E-commerce
**From:** Data Analytics
**Re:** Should we fund a mobile-first redesign of the Google Merchandise Store?
**Data:** 12 months of store traffic (Aug 2016–Aug 2017), ~900,000 sessions

## Recommendation: Ship a mobile redesign initiative — but test it properly before scaling it

Mobile visitors buy at a much lower rate than desktop visitors, and this gap is real,
large, and consistent — not a data artifact, not a phase we're growing out of, and not
explained away by where the traffic comes from. That said, we do **not** yet have a
randomized test of any *specific* redesign, so this memo recommends funding a redesign
effort and running one, not skipping straight to a full rollout.

## What we found

| | Desktop | Mobile |
|---|---|---|
| Sessions analyzed | 664,479 | 208,725 |
| Conversion rate | 1.58% | 0.41% |

Mobile converts at roughly **a quarter of the desktop rate** — a gap of 1.17 percentage
points. We're 95% confident the true gap is between **1.13 and 1.21 percentage points**.
With this much data, that gap is not a fluke: the odds of seeing this by chance if there
were truly no difference are astronomically small.

We checked this finding against three ways it could be misleading, and it held up:

- **Is this just a "mobile is new" effect that fades?** No. We tracked the gap week by
  week across the full year — it doesn't shrink over time. If anything, it holds steady
  or widens slightly.
- **Is this actually about where traffic comes from, not the device?** Mostly no. We
  re-checked the gap separately within every traffic channel (search, social, referral,
  paid, direct). Mobile underperforms desktop in every major channel. One small channel
  (Affiliates, under 2% of sessions) showed the opposite pattern, but it's not
  statistically reliable given its size — a footnote, not a real exception.
- **Do people spend less on mobile too, not just convert less?** Yes, consistent with
  the conversion story: average revenue per mobile session is about a third of desktop's,
  and this also held up under a statistical test suited to the very skewed nature of
  revenue data (most sessions spend $0, a few spend a lot).

## Why we're not saying "ship it" outright

The data above tells us mobile *currently* underperforms — it does not tell us that a
*specific redesign* will fix it, because we never ran a randomized test of one. Before
building anything, we sized what that follow-up test would actually require:

- To detect even a modest 10% relative improvement in mobile conversion (0.41% → about
  0.45%) with standard reliability (80% power, 5% false-positive tolerance), a
  randomized test would need about **400,000 sessions in each arm — 800,000 total.**
- At today's mobile traffic (~209,000 sessions/year), that's roughly **3.8 years** to
  collect if the test simply split existing mobile traffic in half.

That's not a reason to wait — it's a reason to be realistic about what a clean test
costs, and to plan around it rather than be surprised by it later, e.g. by testing a
bolder redesign (bigger expected effect = smaller required sample) or growing mobile
traffic first.

## Caveats

- **This is observational, not randomized.** Visitors weren't randomly assigned to
  mobile or desktop — they chose their own device. Some of the gap could reflect who
  tends to shop on mobile (e.g. browsing vs. buying intent), not the mobile experience
  itself. The channel-by-channel check reduces this concern but doesn't eliminate it.
- **If this becomes a live, monitored test later, don't check results daily and stop
  early.** We simulated what happens if you peek at results every day and stop as soon
  as you see significance: the real false-positive rate jumps from the intended 5% to
  about **19%** — nearly 4x higher than it looks. Decide on one analysis date in
  advance and hold to it.
- Revenue-per-session findings are supporting evidence, not the primary decision metric
  — conversion rate is the cleaner, less noisy signal here.

## What I'd want to test next

1. A randomized test of an actual mobile redesign (checkout flow, page speed, or
   navigation), sized using the power calculation above, with the analysis date fixed
   before launch.
2. If 3.8 years is too long to wait: test a bolder change first (a bigger expected
   effect needs a smaller sample), or prioritize growing mobile traffic before testing
   subtle refinements.
3. Segment the mobile experience by sub-device (phone vs. tablet was already split out;
   iOS vs. Android, or browser, was not) in case the underperformance is concentrated in
   one segment rather than spread evenly.
