#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python3}"
fi

pytest_root="${FREEROUTER_TEST_TEMP:-data/pytest-temp}"
pytest_temp="${pytest_root}/freerouter-pytest-$$"
mkdir -p "$pytest_root"
"$python_bin" -m pytest -p no:cacheprovider --basetemp "$pytest_temp"
"$python_bin" -m ruff check .
npm run typecheck:web
npm run test:web
npm run build:web
npm run check:desktop
