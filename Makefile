.PHONY: install sample features train predict test lint audit

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

sample:
	$(PYTHON) scripts/generate_synthetic_sample.py

features:
	$(PYTHON) -m microgrid_ml.cli prepare-features

train:
	$(PYTHON) -m microgrid_ml.cli train

predict:
	$(PYTHON) -m microgrid_ml.cli predict --site-id MG-SITE-001

audit:
	$(PYTHON) scripts/audit_confidentiality.py

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts
