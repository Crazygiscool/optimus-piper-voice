#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPER_PYTHON="${PROJECT_ROOT}/piper-source/src/python"

# 1. Clone piper-source if missing
if [ ! -d "${PROJECT_ROOT}/piper-source" ]; then
    echo "--- Cloning piper-source ---"
    git clone https://github.com/rhasspy/piper.git "${PROJECT_ROOT}/piper-source" --recurse-submodules --depth 1
fi

# 2. Install Python dependencies
echo "--- Installing Python dependencies ---"
uv pip install -r "${PROJECT_ROOT}/requirements.txt" --python "${VENV_PYTHON}"

# 3. Compile Cython monotonic_align extension
MONO_ALIGN_DIR="${PIPER_PYTHON}/piper_train/vits/monotonic_align"
if [ -d "${MONO_ALIGN_DIR}" ]; then
    echo "--- Compiling Cython monotonic_align extension ---"
    cd "${MONO_ALIGN_DIR}"
    "${PROJECT_ROOT}/.venv/bin/cythonize" -i core.pyx
else
    echo "--- Skipping Cython compile (monotonic_align dir not found) ---"
fi

echo "--- Setup complete ---"
