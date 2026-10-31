#!/bin/bash
#SBATCH --job-name=tabula-t5
#SBATCH --partition=compute_dense
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_4090:1
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
#   * gpu:rtx_4090:1 requests the whole 24 GiB RTX 4090 on trpro-slurm2, which
#     is what lets batch 512 fit. The 3090 nodes are shared 8 GiB shards.
#   * compute_dense allows long jobs (7-day max); this job asks for 1 day.
#   * Checkpoint under runs/ so a requeued job can resume manually with
#     --resume-from runs/pretrain_t5_guitarset/best.
#
# Paper-scale: MAX_STEPS=75000 (approx 75k optimizer updates).
# Default:     MAX_STEPS=30000.
#
# Fine-tuning uses weights-only initialisation (fresh optimizer and schedule):
#   INIT_FROM=runs/pretrain_t5_dadagp/best \
#   DATA_DIR=data/processed_guitarset OUTPUT_DIR=runs/finetune_t5_guitarset \
#   BATCH_SIZE=32 MAX_STEPS=3000 LR=1e-4 WARMUP_RATIO=0.1 \
#   EVAL_EVERY=100 METRICS_EVERY=500 EVAL_BEAM=4 \
#   sbatch scripts/train_t5.sh
#
# Inference-time decoding defaults to a pitch-valid mask plus a low-position
# prior (MAX_FRET=12).  Set PITCH_CONSTRAINT=0 for the paper's unconstrained
# baseline, or MAX_FRET=-1 to disable only the low-position rule.
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

PC_FLAG="--pitch-constraint"
if [[ "${PITCH_CONSTRAINT:-1}" == "0" ]]; then
    PC_FLAG="--no-pitch-constraint"
fi

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
    --eval-beam "${EVAL_BEAM:-1}" \
    --save-every "${SAVE_EVERY:-2500}" \
    --postproc "${POSTPROC:-full}" \
    "$PC_FLAG" \
    --max-fret "${MAX_FRET:-12}" \
    --fret-penalty "${FRET_PENALTY:-0.0}" \
    --warmup-ratio "${WARMUP_RATIO:-0.02}" \
    --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
    --patience "${PATIENCE:-10}" \
    --dropout "${DROPOUT:-0.1}" \
    --weight-decay "${WEIGHT_DECAY:-0.0}" \
    ${LR:+--lr "$LR"} \
    --init-from "${INIT_FROM:-}" \
    --resume-from "${RESUME_FROM:-}"
