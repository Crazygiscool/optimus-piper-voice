#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPER_PYTHON="${PROJECT_ROOT}/piper-source/src/python"
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"
OUT_DIR="${PROJECT_ROOT}/published"
HF_REPO="crazygiscool/optimus-piper-voice"

export PYTHONPATH="${PIPER_PYTHON}:${PYTHONPATH}"

echo "=== Publish Script ==="

# 1. Find latest checkpoint
LATEST_CKPT=$(find "${CHECKPOINTS_DIR}/lightning_logs" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in lightning_logs/"
    exit 1
fi

echo "Latest checkpoint: $(basename "$LATEST_CKPT")"

# 2. Copy all files to published/
mkdir -p "${OUT_DIR}"
cp "$LATEST_CKPT" "${OUT_DIR}/optimus-final.ckpt"
cp "${CHECKPOINTS_DIR}/config.json" "${OUT_DIR}/optimus-final.onnx.json"
cp "${CHECKPOINTS_DIR}/dataset.jsonl" "${OUT_DIR}/dataset.jsonl"
cp "${CHECKPOINTS_DIR}/dataset.jsonl.gz" "${OUT_DIR}/dataset.jsonl.gz"
cp "${PROJECT_ROOT}/published/README.md" "${OUT_DIR}/README.md" 2>/dev/null || true
rm -f "${OUT_DIR}/optimus.onnx" "${OUT_DIR}/optimus.onnx.ckpt" "${OUT_DIR}/optimus.onnx.json"

# 3. Export to ONNX
echo ""
echo "--- Exporting to ONNX ---"
"${VENV_PYTHON}" -c "
import sys
sys.path.insert(0, '${PIPER_PYTHON}')
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
    ('dataset.jsonl', 'Training dataset'),
    ('dataset.jsonl.gz', 'Compressed dataset'),
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

# 6. Commit checkpoint to GitHub (large files tracked via LFS)
echo ""
echo "--- Committing to GitHub ---"
cd "${PROJECT_ROOT}"
git add published/ || true
git add checkpoints/epoch=*.ckpt || true
git commit -m "Publish: $(basename "$LATEST_CKPT")" || echo "Nothing to commit"
git push || echo "Push failed (not critical)"

echo ""
echo "Done."
echo "  Hugging Face: https://huggingface.co/${HF_REPO}"
echo "  Local files: ${OUT_DIR}/"
