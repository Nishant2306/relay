# Relay — one-command workflows (SPEC §15: make up seed train loadtest drill harvest)

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
VENV_PY := python
else
PY := .venv/bin/python
VENV_PY := python3
endif

.PHONY: install up down seed train test test-all validate baseline loadtest drill harvest lint fmt

## Setup -----------------------------------------------------------------------
install:                ## create .venv and install the package + dev extras
	$(VENV_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

## Stack lifecycle -------------------------------------------------------------
up:                     ## build + start the full stack, blocking until healthy
	docker compose up -d --build --wait

down:                   ## stop the stack and drop volumes
	docker compose down -v

## Data + models ---------------------------------------------------------------
seed:                   ## create demo teams + limits in Postgres
	$(PY) scripts/seed_teams.py

train:                  ## train the complexity classifier on the frozen 600 (group-aware split)
	$(PY) scripts/train_classifier.py
	@echo "Reloading the gateway so it picks up the new model (it loads the"
	@echo "classifier once at startup; without this the stack keeps serving"
	@echo "with whatever it booted with)."
	-docker compose restart gateway verifier

validate:               ## dataset contract checks + split verification
	$(PY) scripts/validate_datasets.py
	$(PY) scripts/split_complexity_dataset.py

baseline:               ## the <50% length-only release gate
	$(PY) scripts/length_baseline.py

## Tests -----------------------------------------------------------------------
test:                   ## unit tests + dataset contracts (no Docker needed)
	$(PY) -m pytest -m "not integration and not slow" -q

test-all:               ## everything, incl. testcontainers integration tests (needs Docker)
	$(PY) -m pytest -q

## Load + chaos ----------------------------------------------------------------
loadtest:               ## build corpus + run the steady-state Locust scenario against the stack
	$(PY) scripts/build_loadtest_corpus.py
	$(PY) scripts/run_loadtest.py steady repeats storm budget

drill:                  ## 3-minute provider outage under load — target: zero client 5xx
	$(PY) scripts/run_loadtest.py outage

harvest:                ## print every README/resume number from Prometheus + Postgres
	$(PY) scripts/harvest_metrics.py

## Hygiene ---------------------------------------------------------------------
lint:
	$(PY) -m ruff check .
	$(PY) -m mypy gateway verifier mockprovider admin

fmt:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .
