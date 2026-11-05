"""Shared helpers for case-study scripts: paths, memo rendering, number formatting."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "case_studies"
FIGURES = ROOT / "assets" / "case_studies"
MEMOS = ROOT / "docs" / "memos"
TEMPLATES = Path(__file__).resolve().parent / "templates"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    """Percentage points."""
    return f"{100 * x:+.{digits}f} pp"


def money(x: float, digits: int = 2) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{digits}f}"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}g}" if abs(x) < 1e-3 and x != 0 else f"{x:,.{digits}f}"


def pval(p: float) -> str:
    if p < 1e-300:
        return "< 1e-300"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_memo(template: str, out_name: str, context: dict[str, Any]) -> Path:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(
        pct=pct, pp=pp, money=money, num=num, pval=pval, sentence=lambda s: s[:1].upper() + s[1:]
    )
    text = env.get_template(template).render(**context)
    MEMOS.mkdir(parents=True, exist_ok=True)
    out = MEMOS / out_name
    out.write_text(text)
    return out
