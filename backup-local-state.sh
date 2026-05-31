#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python3}"
fi

"$python_bin" -m app.local_backup export "$@"
