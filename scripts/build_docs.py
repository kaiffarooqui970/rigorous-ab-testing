"""Render README.md, docs/limitations.md and docs/cv_bullets.md from generated results.

Every number in those files is pulled from results/**/*.json at render time, never
typed by hand. ``tests/test_docs.py`` fails if a committed doc differs from what this
script renders, so the docs can't drift from the results.

Run:  python scripts/build_docs.py          (write)
      python scripts/build_docs.py --check  (exit 1 if any doc is stale)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case_studies"))
from _common import money, num, pct, pp, pval  # noqa: E402

TARGETS = {
    "README.md.j2": "README.md",
    "limitations.md.j2": "docs/limitations.md",
    "cv_bullets.md.j2": "docs/cv_bullets.md",
}


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def context() -> dict:
    s2 = load("results/validation/stage2.json")
    seq = {s: {x["strategy"]: x for x in s2["C_sequential"][s]} for s in ("null", "alternative")}
    return {
        "s2": s2,
        "seq": seq,
        "s3": load("results/validation/stage3.json"),
        "s4": load("results/validation/stage4.json"),
        "hill": load("results/case_studies/hillstrom.json"),
        "crit": load("results/case_studies/criteo.json"),
        "ga": load("results/case_studies/ga360.json"),
    }


def render() -> dict[str, str]:
    env = Environment(
        loader=FileSystemLoader(ROOT / "docs" / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(
        pct=pct, pp=pp, money=money, num=num, pval=pval, sentence=lambda s: s[:1].upper() + s[1:]
    )
    ctx = context()
    return {out: env.get_template(tpl).render(**ctx) for tpl, out in TARGETS.items()}


def main() -> int:
    docs = render()
    if "--check" in sys.argv:
        stale = [p for p, text in docs.items() if (ROOT / p).read_text() != text]
        if stale:
            print(f"stale docs (run python scripts/build_docs.py): {stale}")
            return 1
        print("docs are up to date with results/")
        return 0
    for p, text in docs.items():
        (ROOT / p).write_text(text)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
