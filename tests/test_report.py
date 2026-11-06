"""Tests for the trust-checked report (Stage 6)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from xpkit.report import ConfigError, analyze, render_html, render_markdown
from xpkit.report.cli import main

EX = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def healthy():
    return pd.read_csv(EX / "checkout_test.csv"), yaml.safe_load(
        (EX / "checkout_test.yaml").read_text()
    )


@pytest.fixture(scope="module")
def broken():
    return (
        pd.read_csv(EX / "checkout_test_srm.csv"),
        yaml.safe_load((EX / "checkout_test_srm.yaml").read_text()),
    )


def test_healthy_example_passes_trust_and_covers_truth(healthy):
    r = analyze(*healthy)
    assert r.trustworthy
    rev = next(m for m in r.metrics if m.name == "revenue")
    assert rev.ci_low < 0.6 < rev.ci_high  # true effect from scripts/make_examples.py
    assert rev.cuped["variance_reduction"] > 0.1


def test_srm_example_fails_and_report_warns(broken):
    r = analyze(*broken)
    assert not r.trustworthy
    assert r.headline.startswith("DO NOT INTERPRET")
    md, html = render_markdown(r), render_html(r)
    assert "TRUST CHECK FAILED" in md and "TRUST CHECK FAILED" in html
    # The trust section must come before the effects section.
    assert md.index("Trust checks") < md.index("Effect estimates")


def test_report_is_deterministic(healthy):
    a = render_markdown(analyze(*healthy), "x.csv", "abc")
    b = render_markdown(analyze(*healthy), "x.csv", "abc")
    assert a == b


def test_cli_exit_codes(tmp_path):
    assert (
        main(
            [
                str(EX / "checkout_test.csv"),
                str(EX / "checkout_test.yaml"),
                "-o",
                str(tmp_path / "r.html"),
            ]
        )
        == 0
    )
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
    assert (
        main(
            [
                str(EX / "checkout_test_srm.csv"),
                str(EX / "checkout_test_srm.yaml"),
                "-o",
                str(tmp_path / "r.md"),
            ]
        )
        == 3
    )
    assert main([str(EX / "missing.csv"), str(EX / "checkout_test.yaml")]) == 1


@pytest.mark.parametrize(
    "patch,msg",
    [
        ({"treatment_column": "nope"}, "treatment_column 'nope' not found"),
        ({"treatment": "B"}, "treatment value 'B' not found"),
        ({"metrics": [{"name": "revenue", "type": "ratio"}]}, "use one of"),
        ({"metrics": [{"name": "revenue", "type": "binary"}]}, "only 0/1"),
        ({"metrics": [{"name": "zzz", "type": "binary"}]}, "'zzz' not found"),
        ({"multiplicity": "fdr"}, "multiplicity must be"),
    ],
)
def test_config_errors_are_helpful(healthy, patch, msg):
    df, cfg = healthy
    with pytest.raises(ConfigError, match=msg):
        analyze(df, {**cfg, **patch})


def test_duplicate_units_flagged(healthy):
    df, cfg = healthy
    dup = pd.concat([df, df.head(5)], ignore_index=True)
    r = analyze(dup, cfg)
    assert not next(c for c in r.checks if c.name == "Duplicate units").passed
    assert not r.trustworthy
