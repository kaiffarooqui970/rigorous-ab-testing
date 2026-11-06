"""Shared matplotlib styling for static charts (README, reports, case studies).

The palette is the validated categorical order: blue, orange, aqua. Charts keep to at
most three series. A single-series chart has no legend; with 2-3 series there is a
legend plus selective direct labels. Grid and axes are recessive. Text uses ink
colours, never series colours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#a3a29c"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
CRITICAL = "#d03b3b"


def apply_style() -> None:
    """Apply the shared, recessive matplotlib style used by every static chart."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: Any, path: str | Path) -> Path:
    """Save a figure at 150 dpi (creating parent dirs) and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p
