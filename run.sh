#!/usr/bin/env bash
set -euo pipefail

host="${GATEWAY_HOST:-127.0.0.1}"
port="${GATEWAY_PORT:-8000}"
install_only=0
runtime_only=0
reload=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --install-only)
      install_only=1
      shift
      ;;
    --runtime-only)
      runtime_only=1
      shift
      ;;
    --no-reload)
      reload=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

python_bin="${PYTHON:-python3}"
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12+ is required. Set PYTHON=/path/to/python3.12 if needed." >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating local virtual environment at .venv..."
  "$python_bin" -m venv .venv
fi

echo "Installing/updating gateway dependencies..."
.venv/bin/python -m pip install --upgrade pip
if [[ "$runtime_only" -eq 1 ]]; then
  .venv/bin/python -m pip install -e .
else
  .venv/bin/python -m pip install -e ".[dev]"
fi

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add provider API keys before expecting live completions."
fi

if [[ "$install_only" -eq 1 ]]; then
  echo "Install complete. Start later with: ./run.sh"
  exit 0
fi

args=(app.main:app --host "$host" --port "$port")
if [[ "$reload" -eq 1 ]]; then
  args+=(--reload)
fi

echo "Starting FreeRouter at http://${host}:${port}/v1"
exec .venv/bin/python -m uvicorn "${args[@]}"
