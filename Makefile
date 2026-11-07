.PHONY: run step1 rag step1-rag step2-rag step3-rag step4-rag evals test clean

PYTHON = ./.venv/bin/python
PIP = ./.venv/bin/pip

# Правило: папка .venv створюється автоматично, якщо її немає
.venv:
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

# rag: Запуск повного наскрізного циклу Кейсу 2 (RAG & Evals: Кроки 1-4)
rag: .venv
	$(PYTHON) main.py --rag

# step1 також автоматично підтягує .venv (Кейс 1)
step1: .venv
	$(PYTHON) main.py --step 1

# step1-rag: Запуск Кроку 1 для RAG (Кейс 2: Ingestion & Chunking)
step1-rag: .venv
	$(PYTHON) src/run_step1.py

# step2-rag: Запуск Кроку 2 для RAG (Кейс 2: Index, Hybrid Search & Benchmark)
step2-rag: .venv
	$(PYTHON) src/eval_retrieval.py

# step3-rag: Запуск Кроку 3 для RAG (Кейс 2: RAG Pipeline з джерелами та контролем токенів)
step3-rag: .venv
	$(PYTHON) src/run_step3.py

# step4-rag: Запуск Кроку 4 для RAG (Кейс 2: Тестсет, Калібрування судді та Evals)
step4-rag: .venv
	$(PYTHON) src/run_evals.py

# evals: Запуск повної оцінки з перевіркою Quality Gates для CI
evals: .venv
	$(PYTHON) src/run_evals.py --ci-mode

test: .venv
	$(PYTHON) -m unittest discover tests

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
