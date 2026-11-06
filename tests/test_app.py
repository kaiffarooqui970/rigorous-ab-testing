"""Smoke test: the Streamlit app runs end to end on bundled synthetic data, without credentials."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_app_runs_without_exceptions():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert labels[:4] == ["Planning", "Analysis", "Causal diagnostics", "Uplift"]


@pytest.mark.slow
def test_app_flags_srm_example():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.radio[2].set_value("Bundled: test with logging bug (SRM)").run()
    assert not at.exception
    assert any("DO NOT INTERPRET" in e.value for e in at.error)
