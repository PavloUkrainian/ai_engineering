.PHONY: run step1 test clean

PYTHON = ./.venv/bin/python
PIP = ./.venv/bin/pip

# Правило: папка .venv створюється автоматично, якщо її немає
.venv: requirements.txt
	@echo "==> Створення віртуального оточення .venv та встановлення залежностей..."
	python3 -m venv .venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	@touch .venv

# Ціль install викликає створення .venv
install: .venv

# run залежить від .venv: якщо .venv немає, make спочатку створить його!
run: .venv
	$(PYTHON) main.py

# step1 також автоматично підтягує .venv
step1: .venv
	$(PYTHON) main.py --step 1

test: .venv
	$(PYTHON) -m unittest discover tests

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
