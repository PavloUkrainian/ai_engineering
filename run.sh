#!/usr/bin/env bash
set -e

# Використовуємо віртуальне середовище .venv, якщо воно існує
if [ -d ".venv" ]; then
    PYTHON="./.venv/bin/python"
else
    PYTHON="python3"
fi

$PYTHON main.py "$@"
