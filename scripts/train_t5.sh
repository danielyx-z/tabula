#!/bin/bash
#SBATCH --job-name=tabula-t5
#SBATCH --partition=compute_dense
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:rtx_3090:8192,tmpdisk:20480
#SBATCH --output=logs/%j-%x.out
#SBATCH --error=logs/%j-%x.err

# Fretting-Transformer / T5 training on WATcloud.
#
# Submit from the repository root (the logs/ directory must already exist):
#     mkdir -p logs
#     sbatch scripts/train_t5.sh
#
# Override any of these at submit time, e.g.:
#     BATCH_SIZE=16 GRAD_ACCUM=4 MAX_STEPS=75000 sbatch scripts/train_t5.sh
#
# Resource notes:
#   * shard:rtx_3090:8192 requests 8 GiB of a shared RTX 3090. The model is
#     tiny; this is ample for batch 8. The explicit GPU type avoids the
#     Pascal GTX 1080 node, which does not support bf16. Use
#     --gres=gpu:rtx_3090:1 for a dedicated card.
#   * compute_dense allows long jobs (7-day max). Use compute + time<=1 day if
#     you need a shorter slot.
#   * Checkpoint under runs/ so a requeued job can resume manually with
#     --resume-from runs/pretrain_t5_guitarset/best.
#
# Paper-scale: MAX_STEPS=75000 (approx 75k optimizer updates, ~5.5h on RTX 3090).
# Default:     MAX_STEPS=30000 (approx 2.2h on RTX 3070, ~1.1h on RTX 3090).
#
# The corpus is tokenised once into a chunk store cached under
# data/processed_guitarset/.chunk_cache (a few MB). Training epochs then only
# reshuffle indices, so the data path is ~1% of step time. Do not set
# NUM_WORKERS>0 for the GuitarSet corpus: worker IPC costs more than it saves.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"

mkdir -p logs

export PYTHONPATH="$REPO_DIR/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
    echo "error: Python not found at $PYTHON" >&2
    echo "Create the cluster venv first (see README.md / AGENTS.md)." >&2
    exit 1
fi

DATA_DIR="${DATA_DIR:-data/processed_guitarset}"
VOCAB_PATH="${VOCAB_PATH:-vocab/guitarset.json}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "error: $DATA_DIR is missing in $REPO_DIR" >&2
    exit 1
fi
if [[ ! -f "$VOCAB_PATH" ]]; then
    echo "error: $VOCAB_PATH is missing in $REPO_DIR" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_t5_guitarset}"

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON" --version

exec "$PYTHON" -m model.train \
    --data-dir "$DATA_DIR" \
    --vocab-path "$VOCAB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "${BATCH_SIZE:-8}" \
    --grad-accum "${GRAD_ACCUM:-8}" \
    --bf16 \
    --max-steps "${MAX_STEPS:-30000}" \
    --eval-every "${EVAL_EVERY:-500}" \
    --eval-loss-batches "${EVAL_LOSS_BATCHES:-0}" \
    --eval-batches "${EVAL_BATCHES:-10}" \
    --metrics-every "${METRICS_EVERY:-2500}" \
    --save-every "${SAVE_EVERY:-2500}" \
    --postproc full \
    --patience "${PATIENCE:-10}" \
    --dropout "${DROPOUT:-0.1}" \
    --weight-decay "${WEIGHT_DECAY:-0.0}" \
    ${LR:+--lr "$LR"} \
    --resume-from "${RESUME_FROM:-}"
