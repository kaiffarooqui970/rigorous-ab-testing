"""Render :class:`ReportData` as Markdown or a self-contained HTML page."""

from __future__ import annotations

import html
import math
from importlib import resources
from typing import Any

from jinja2 import Environment, StrictUndefined

from xpkit import __version__
from xpkit.report.core import ReportData


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):
        if x != 0 and abs(x) < 1e-3:
            return f"{x:.2e}"
        return f"{x:.{digits}g}"
    return str(x)


def _p(x: float) -> str:
    return "< 1e-300" if x < 1e-300 else (f"{x:.2e}" if x < 1e-3 else f"{x:.4f}")


def _context(r: ReportData, source: str, data_sha256: str) -> dict[str, Any]:
    # No timestamp: the same data + config + version must give a byte-identical report.
    return {"r": r, "source": source, "version": __version__, "data_sha256": data_sha256}


def _env() -> Environment:
    env = Environment(
        undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, autoescape=False
    )
    env.filters.update(f=_fmt, p=_p, e=html.escape)
    return env


def render_markdown(r: ReportData, source: str = "", data_sha256: str = "") -> str:
    tpl = resources.files("xpkit.report").joinpath("templates/report.md.j2").read_text()
    return _env().from_string(tpl).render(**_context(r, source, data_sha256))


def render_html(r: ReportData, source: str = "", data_sha256: str = "") -> str:
    tpl = resources.files("xpkit.report").joinpath("templates/report.html.j2").read_text()
    env = _env()
    env.autoescape = True
    return env.from_string(tpl).render(**_context(r, source, data_sha256))
