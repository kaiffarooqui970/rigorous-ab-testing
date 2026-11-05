"""Effect estimation: two-arm effects, CUPED, multiple testing, group-sequential boundaries."""

from xpkit.estimate.cuped import cuped, cuped_theta, lin_regression_adjustment
from xpkit.estimate.effects import diff_in_means, diff_in_proportions, relative_lift
from xpkit.estimate.multiplicity import (
    AdjustedPValues,
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    holm,
)
from xpkit.estimate.sequential import (
    SequentialDecision,
    SequentialDesign,
    group_sequential_design,
    sequential_decision,
    spending_function,
)

__all__ = [
    "AdjustedPValues",
    "SequentialDecision",
    "SequentialDesign",
    "adjust_pvalues",
    "benjamini_hochberg",
    "bonferroni",
    "cuped",
    "cuped_theta",
    "diff_in_means",
    "diff_in_proportions",
    "group_sequential_design",
    "holm",
    "lin_regression_adjustment",
    "relative_lift",
    "sequential_decision",
    "spending_function",
]
