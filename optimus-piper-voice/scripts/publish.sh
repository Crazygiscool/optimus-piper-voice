#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPER_PYTHON="${PROJECT_ROOT}/piper-source/src/python"
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"
OUT_DIR="${PROJECT_ROOT}/published"
HF_REPO="crazygiscool/optimus-piper-voice"
GH_REPO="Crazygiscool/optimus-piper-voice"

export PYTHONPATH="${PIPER_PYTHON}:${PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "=== Publish Script ==="

# Resolve -c <path | name | glob> into a checkpoint path
resolve_ckpt() {
    local spec="$1"
    local name
    if [ -f "$spec" ]; then
        readlink -f "$spec"
        return 0
    fi
    if [ -f "${CHECKPOINTS_DIR}/$spec" ]; then
        readlink -f "${CHECKPOINTS_DIR}/$spec"
        return 0
    fi
    if [ -f "${PROJECT_ROOT}/$spec" ]; then
        readlink -f "${PROJECT_ROOT}/$spec"
        return 0
    fi
    name="$(basename "$spec")"
    find "${CHECKPOINTS_DIR}/lightning_logs" -type f -name "$name" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}'
}

# Parse args (-c <checkpoint>)
CKPT_ARG=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -c)
            CKPT_ARG="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1 (only -c <checkpoint> is supported)"
            exit 1
            ;;
    esac
done

# 1. Pick checkpoint: explicit -c, else prefer the lowest-loss checkpoint
if [ -n "$CKPT_ARG" ]; then
    LATEST_CKPT="$(resolve_ckpt "$CKPT_ARG")"
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: Could not resolve checkpoint: $CKPT_ARG"
        exit 1
    fi
    echo "Selected checkpoint: $(basename "$LATEST_CKPT")"
else
    LATEST_CKPT=$(find "${CHECKPOINTS_DIR}/lightning_logs" -name "best-gen-loss-*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$LATEST_CKPT" ]; then
        echo "Using best-loss checkpoint: $(basename "$LATEST_CKPT")"
    else
        LATEST_CKPT=$(find "${CHECKPOINTS_DIR}/lightning_logs" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
        if [ -n "$LATEST_CKPT" ]; then
            echo "No best-gen-loss checkpoint; using latest: $(basename "$LATEST_CKPT")"
        fi
    fi
fi

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in lightning_logs/"
    exit 1
fi

# 2. Copy model files to published/
mkdir -p "${OUT_DIR}"
cp "$LATEST_CKPT" "${OUT_DIR}/optimus-final.ckpt"
cp "${CHECKPOINTS_DIR}/config.json" "${OUT_DIR}/optimus-final.onnx.json"
cp "${PROJECT_ROOT}/published/README.md" "${OUT_DIR}/README.md" 2>/dev/null || true
rm -f "${OUT_DIR}/optimus.onnx" "${OUT_DIR}/optimus.onnx.ckpt" "${OUT_DIR}/optimus.onnx.json"

# 3. Export to ONNX
echo ""
echo "--- Exporting to ONNX ---"
"${VENV_PYTHON}" -c "
import sys
from functools import partial
import torch
sys.path.insert(0, '${PIPER_PYTHON}')
torch.onnx.export = partial(torch.onnx.export, dynamo=False)
from piper_train.export_onnx import main
sys.argv = ['export_onnx', '${OUT_DIR}/optimus-final.ckpt', '${OUT_DIR}/optimus-final.onnx']
main()
"

# 4. Simplify ONNX (optional, improves compatibility)
if command -v onnxsim &> /dev/null; then
    echo ""
    echo "--- Simplifying ONNX ---"
    onnxsim "${OUT_DIR}/optimus-final.onnx" "${OUT_DIR}/optimus-final.onnx" || true
fi

echo ""
echo "--- Published files ---"
ls -lh "${OUT_DIR}/"

# 5. Upload to Hugging Face
echo ""
echo "--- Uploading to Hugging Face ---"
"${VENV_PYTHON}" -c "
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = '${HF_REPO}'
out_dir = '${OUT_DIR}'

files = [
    ('optimus-final.onnx', 'ONNX model'),
    ('optimus-final.onnx.json', 'ONNX config'),
    ('optimus-final.ckpt', 'PyTorch checkpoint'),
    ('README.md', 'Model card'),
]

for filename, description in files:
    path = os.path.join(out_dir, filename)
    if os.path.exists(path):
        print(f'Uploading {filename} ({description})...')
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type='model',
        )
    else:
        print(f'WARNING: {filename} not found, skipping')

print('Upload complete!')
"

# 6. Publish model artifacts as GitHub release assets
echo ""
echo "--- Publishing to GitHub ---"
if ! command -v gh &> /dev/null; then
    echo "ERROR: GitHub CLI (gh) is required to publish release assets"
    exit 1
fi

RELEASE_TAG="optimus-$(basename "$LATEST_CKPT" .ckpt)"
RELEASE_TITLE="Optimus Prime Piper voice - $(basename "$LATEST_CKPT" .ckpt)"
RELEASE_ASSETS=(
    "${OUT_DIR}/optimus-final.ckpt"
    "${OUT_DIR}/optimus-final.onnx"
    "${OUT_DIR}/optimus-final.onnx.json"
)

if gh release view "${RELEASE_TAG}" --repo "${GH_REPO}" &> /dev/null; then
    gh release upload "${RELEASE_TAG}" "${RELEASE_ASSETS[@]}" --clobber --repo "${GH_REPO}"
else
    gh release create "${RELEASE_TAG}" "${RELEASE_ASSETS[@]}" \
        --title "${RELEASE_TITLE}" \
        --notes "Piper voice model export. Training dataset files are not included." \
        --repo "${GH_REPO}"
fi

echo ""
echo "Done."
echo "  Hugging Face: https://huggingface.co/${HF_REPO}"
echo "  GitHub: https://github.com/${GH_REPO}/releases/tag/${RELEASE_TAG}"
echo "  Local files: ${OUT_DIR}/"
