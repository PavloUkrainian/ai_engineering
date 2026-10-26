.PHONY: install run step1 test clean

PYTHON ?= ./.venv/bin/python
PIP ?= ./.venv/bin/pip

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py

step1:
	$(PYTHON) main.py --step 1

test:
	$(PYTHON) -c "from src.schema import ExtractedDocument; from src.client import ProviderClient; print('All core modules import successfully!')"

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
