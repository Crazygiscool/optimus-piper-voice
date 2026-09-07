#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPER_PYTHON="${PROJECT_ROOT}/piper-source/src/python"

export PYTHONPATH="${PIPER_PYTHON}:${PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

exec "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/test.py" "$@"
