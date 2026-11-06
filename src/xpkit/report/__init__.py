"""Trust-checked experiment reports (CLI: ``xpkit-report``)."""

from xpkit.report.core import ConfigError, MetricResult, ReportData, TrustCheck, analyze
from xpkit.report.render import render_html, render_markdown

__all__ = [
    "ConfigError",
    "MetricResult",
    "ReportData",
    "TrustCheck",
    "analyze",
    "render_html",
    "render_markdown",
]
