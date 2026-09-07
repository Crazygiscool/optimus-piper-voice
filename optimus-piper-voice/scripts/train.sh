#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPER_PYTHON="${PROJECT_ROOT}/piper-source/src/python"

export PYTHONPATH="${PIPER_PYTHON}:${PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CKPT_DIR="${PROJECT_ROOT}/checkpoints"
BASE_CKPT="${CKPT_DIR}/lessac-medium.ckpt"

# Resume newest lightning-log checkpoint, else bootstrap from base lessac model
LATEST_CKPT=$(find "${CKPT_DIR}/lightning_logs" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')

RESUME=""
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME="--resume_from_checkpoint $LATEST_CKPT"
elif [ -f "$BASE_CKPT" ]; then
    echo "Bootstrap: fine-tuning from base checkpoint $BASE_CKPT"
    RESUME="--resume_from_checkpoint $BASE_CKPT"
else
    echo "ERROR: No checkpoint found. Download en_US-lessac-medium to ${BASE_CKPT} first."
    echo "  https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
    exit 1
fi

exec "${VENV_PYTHON}" -m piper_train \
    --dataset-dir "${CKPT_DIR}" \
    --default_root_dir "${CKPT_DIR}" \
    --accelerator cpu \
    --batch-size 4 \
    --validation-split 0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --checkpoint-epochs 1 \
    --precision 32 \
    --log_every_n_steps 1 \
    --learning-rate 1e-5 \
    --lr-decay 0.999875 \
    $RESUME \
    "$@"