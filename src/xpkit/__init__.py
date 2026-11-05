"""xpkit: experimentation and causal-inference toolkit.

Subpackages:
    sim       -- seeded simulators with known truth, plus a Monte Carlo validation harness
    design    -- sample size and minimum detectable effect
    trust     -- sample ratio mismatch and covariate-balance checks
    estimate  -- effect estimates, CUPED, multiple testing, group-sequential boundaries
    causal    -- propensity scores, IPW/AIPW, difference-in-differences, E-values
    uplift    -- T-/X-learners, Qini/uplift curves, targeting policy
    report    -- CSV + config -> Markdown/HTML experiment report (CLI: ``xpkit-report``)
"""

from xpkit.types import Estimate, normal_estimate

__version__ = "0.2.0"
__all__ = ["Estimate", "__version__", "normal_estimate"]
