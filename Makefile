# gpu-quantum-sim: one command per task. Run from Linux, WSL2 or Colab.
#   make all     install, test, run both experiments, redraw every plot
#   make quick   reduced run of both experiments (CPU, under 2 minutes each)
PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: all install test ghz h2 quick plots clean

all: install test ghz h2 plots

install: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	touch $@

test: install
	$(PY) -m pytest --cov=src --cov-report=term-missing

ghz: install
	$(PY) -m src.ghz_benchmark

h2: install
	$(PY) -m src.h2_vqe

quick: install
	$(PY) -m src.ghz_benchmark --quick --output-dir results/quick
	$(PY) -m src.h2_vqe --quick --output-dir results/quick

plots: install
	$(PY) -m src.ghz_benchmark --plot-only
	$(PY) -m src.h2_vqe --plot-only

clean:
	rm -rf $(VENV) .pytest_cache .coverage results/quick
