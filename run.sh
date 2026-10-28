#!/usr/bin/env bash
set -e

# Якщо віртуальне середовище .venv ще не створено — створюємо його автоматично!
if [ ! -d ".venv" ]; then
    echo "==> [run.sh] Створення віртуального оточення .venv та встановлення залежностей..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip -q
    ./.venv/bin/pip install -r requirements.txt
fi

./.venv/bin/python main.py "$@"
