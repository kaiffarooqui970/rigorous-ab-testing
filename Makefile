# One-command entry points. Everything except `case-studies` runs on synthetic data,
# without credentials or downloads.
PY ?= python

.PHONY: install lint test test-all demo validate case-studies app examples clean

install:            ## editable install with pinned dependencies (pip)
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e ".[dev,app]" --no-deps

lint:               ## ruff lint + format check + mypy
	$(PY) -m ruff check src tests app validation case_studies scripts
	$(PY) -m ruff format --check .
	$(PY) -m mypy

test:               ## fast tests (what CI runs)
	$(PY) -m pytest -m "not slow" --doctest-modules src/xpkit tests

test-all:           ## everything, including slow Monte Carlo and app tests
	$(PY) -m pytest --doctest-modules src/xpkit tests

demo:               ## end-to-end demo on synthetic data (~1-2 min)
	$(PY) scripts/demo.py

validate:           ## full simulation studies -> results/validation/*.json + charts (~15 min)
	$(PY) validation/stage2_experiments.py
	$(PY) validation/stage3_causal.py
	$(PY) validation/stage4_uplift.py

case-studies:       ## real data: downloads Hillstrom (4 MB) + Criteo (297 MB); GA360 needs GCP_PROJECT
	cd case_studies && $(PY) 01_hillstrom.py
	cd case_studies && $(PY) 02_criteo.py
	cd case_studies && $(PY) 03_ga360.py

examples:           ## regenerate bundled synthetic example CSVs
	$(PY) scripts/make_examples.py

app:                ## launch the Streamlit workbench
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf reports results/**/*_quick.json .pytest_cache .mypy_cache .ruff_cache
