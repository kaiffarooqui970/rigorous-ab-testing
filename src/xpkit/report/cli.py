"""Command-line experiment report.

    xpkit-report DATA.csv CONFIG.yaml [-o report.html|report.md]

Exit codes: 0 = report written, all trust checks passed; 3 = report written but a
critical trust check FAILED (so CI pipelines can block on it); 1 = bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from xpkit.report.core import ConfigError, analyze
from xpkit.report.render import render_html, render_markdown


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``xpkit-report``. Returns the process exit code (0, 1 or 3)."""
    ap = argparse.ArgumentParser(prog="xpkit-report", description="Trust-checked experiment report")
    ap.add_argument("data", type=Path, help="CSV with one row per experimental unit")
    ap.add_argument("config", type=Path, help="YAML config (see xpkit.report.core docstring)")
    ap.add_argument("-o", "--out", type=Path, help="output .md or .html (default: print Markdown)")
    a = ap.parse_args(argv)
    try:
        raw = a.data.read_bytes()
        df = pd.read_csv(a.data)
        cfg = yaml.safe_load(a.config.read_text())
        report = analyze(df, cfg)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ConfigError, KeyError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sha = hashlib.sha256(raw).hexdigest()
    if a.out is None or a.out.suffix == ".md":
        text = render_markdown(report, source=a.data.name, data_sha256=sha)
    elif a.out.suffix in (".html", ".htm"):
        text = render_html(report, source=a.data.name, data_sha256=sha)
    else:
        print("error: --out must end in .md or .html", file=sys.stderr)
        return 1
    if a.out is None:
        print(text)
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print(f"wrote {a.out}")
    if not report.trustworthy:
        print("WARNING: a critical trust check failed; see the report banner.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
