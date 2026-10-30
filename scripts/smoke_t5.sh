#!/bin/bash
# One-step GPU smoke test for the WATcloud training stack.
#
# Run interactively from the repository root:
#   srun --partition=compute --gres=shard:rtx_3090:4096,tmpdisk:2048 \
#        --cpus-per-task=4 --mem=8G --time=00:20:00 bash scripts/smoke_t5.sh
#
# It writes a throwaway run to runs/smoke_t5 and checks CUDA + bf16 before
# starting.  It does not decode full tracks (--metrics-every 0).

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
    echo "error: Python not found at $PYTHON" >&2
    exit 1
fi

export PYTHONPATH="$REPO_DIR/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-runs/smoke_t5}"
if [[ "$OUTPUT_DIR" == "runs/smoke_t5" ]]; then
    rm -rf "$OUTPUT_DIR"
fi

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

exec "$PYTHON" -m model.train \
    --data-dir data/processed_guitarset \
    --vocab-path vocab/guitarset.json \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 2 \
    --batch-size 2 \
    --grad-accum 1 \
    --bf16 \
    --eval-every 1 \
    --eval-loss-batches 1 \
    --eval-batches 1 \
    --metrics-every 0 \
    --save-every 100 \
    --max-len 128