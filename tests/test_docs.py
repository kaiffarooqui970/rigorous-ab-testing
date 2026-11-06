"""Docs must match what the generators render from results/ (no hand-edited numbers)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_limitations_cv_are_rendered_from_results():
    out = subprocess.run(
        [sys.executable, "scripts/build_docs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_memos_are_rendered_from_results():
    for script, memo in (
        ("01_hillstrom.py", "01_hillstrom.md"),
        ("02_criteo.py", "02_criteo.md"),
        ("03_ga360.py", "03_ga360.md"),
    ):
        before = (ROOT / "docs" / "memos" / memo).read_text()
        subprocess.run(
            [sys.executable, script, "--render-only"], cwd=ROOT / "case_studies", check=True
        )
        assert (ROOT / "docs" / "memos" / memo).read_text() == before, f"{memo} is stale"


def test_no_raw_data_committed():
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    bad = [f for f in files if f.startswith("data/") or f.endswith((".parquet", ".gz"))]
    assert not bad, bad
