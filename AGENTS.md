# Fretwire / Tabula — working memory

## Project

Reimplementation of **Fretting-Transformer: Encoder-Decoder Model for MIDI to
Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

- Input: symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`).
- Output: ASCII tab events (`TAB` + `TIME_SHIFT`).
- Model: paper-config T5, 2.81M parameters (`d_model=128`, `d_ff=1024`,
  `d_kv=64`, 3 encoder/3 decoder layers, 4 heads, vocab 423).
- The paper does **not** define a training-time alignment loss, note-count head,
  or pitch-constrained training loss. Do not reintroduce them. Its alignment
  step is post-processing: overlap correction + ±5-note neighbor search.
- Training chunks: 512 encoder tokens. Inference chunks: 20 notes with the
  previous note as context in encoder and decoder.

## Training is user-driven

Do **not** launch additional long training runs without asking. The user
explicitly requested the current WATcloud run; the local 8 GB GPU is still
not used for long training.

OOM is not a design concern. The cluster job requests a GPU shard; use a batch
size that is efficient (defaults are batch 8 × grad-accum 8 = effective 64).

## WATcloud / Slurm

- SSH login: `ssh watcloud`
- Slurm docs: <https://cloud.watonomous.ca/docs/compute-cluster/slurm>
- The Slurm CLI is not on the default login `PATH`; use
  `export PATH=/opt/slurm/bin:$PATH` or `/opt/slurm/bin/sbatch`.
- Submit from the repository root: `mkdir -p logs && sbatch slurm/train_t5.sh`
- Job script: `slurm/train_t5.sh` (edit paths/resources if needed).

Resource choices in the script:

| Resource | Value | Why |
| --- | --- | --- |
| Partition | `compute_dense` | long jobs; 7-day max |
| Time | `2-00:00:00` | training + periodic evaluation |
| GPUs | `--gres=shard:rtx_3090:8192` | 8 GB RTX 3090 shard; plenty for batch 8 |
| CPUs | `8` | tokenisation / dataloading |
| RAM | `32G` | sharded dataset + Python |
| Temp disk | `tmpdisk:20480` | 20 GB `/tmp` |

`shard` is the recommended shared-GPU request. The explicit `rtx_3090` type
avoids the Pascal GTX 1080 node, which does not support bf16. For a dedicated
GPU, replace it with `--gres=gpu:rtx_3090:1`. Do not request `gpu:1` on the
`compute` partition unless the job can actually use most of the card. The
tested node was `thor-slurm1`: RTX 3090, driver 580.178.04, CUDA 13.0.

`--resume-from` restores weights, vocabulary, optimizer/scheduler state and
early-stopping counters, and continues from the saved epoch. A checkpoint
without `trainer_state.pt` falls back to weights/vocabulary only.

Monitor:

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.10T %.10M %.9l %R"
tail -f logs/<jobid>-fretwire-t5.out
scancel <jobid>
```

WATcloud notifications (instead of Slurm mail):

```bash
./watcloud subscription <jobid> you@example.com
./watcloud subscription <jobid> --discord
```

## Current WATcloud run

- Job `724155` finished with early stopping at step 27,750 (epoch 109) on
  `thor-slurm1` (RTX 3090, 24 GiB).
- Best checkpoint: `runs/pretrain_t5_v5/best`, step 25,250 (epoch 99),
  validation loss `0.03631`.
- Final checkpoint: `runs/pretrain_t5_v5/final`, step 27,750.  Log:
  `logs/724155-fretwire-t5.out`.
- Full 10-track validation metrics from the run:
  - step 24,750: raw tab `94.72%`, post tab `94.72%`, post pitch `100%`,
    post difficulty `3.5183`.
  - step 26,000: raw tab `94.50%`, post tab `94.50%`, post difficulty `3.4722`.
  - step 27,250: raw tab `94.39%`, post tab `94.46%`, post difficulty `3.4851`.
- These numbers use the run's fixed shuffled 10-track validation sample
  (`eval_batches=10`, seed 43), not the full validation split.  Offline
  evaluation of `best` on the first 10 validation tracks (seed 42, no
  shuffle) gives post tab `88.07%`, post pitch `100%`, difficulty `3.0978`;
  on the shuffled seed-43 sample it gives post tab `93.13%`.  Treat the
  differences as track-sampling variance (and bf16-vs-fp32 decoding), not as
  a single benchmark.
- Early stopping fired after 10 evaluation intervals without improvement
  (`patience=10`, `min_delta=0.0001`).  The previous job (`722895`) had reached
  step 14,750 / epoch 58 before being stopped to fix post-processing.
- Post-processing fix: neighbor search matches by sequence position first, and
  full post-processing runs per 20-note chunk.  Offline evaluation of
  `checkpoint-12500` (10 valid tracks) gives raw tab `89.70%`, post tab
  `89.71%`, post pitch `100%`, post difficulty `3.07`; before the fix the
  global pitch-first pass reduced post tab accuracy to about `62%`.
- Re-evaluate without training:
  `PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py --checkpoint runs/pretrain_t5_v5/best --tracks 10 --postproc full --shuffle --seed 43`
- If training is extended, resume with:
  `RESUME_FROM=runs/pretrain_t5_v5/best sbatch slurm/train_t5.sh`
  This restores optimizer/schedule and early-stopping counters.

## Cluster setup

The cluster needs its own environment and data:

```bash
ssh watcloud
cd ~/tabula                     # this workspace's cluster copy
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The tested cluster driver is 580.178.04 (CUDA 13.0); the cu121 wheel above
works. If `torch.cuda.is_available()` is false, reinstall the CUDA wheel
matching the compute-node driver (see
<https://pytorch.org/get-started/locally/>) before submitting. The processed
corpus must exist at `data/processed` (train / valid / test shards). Check
available quota before copying checkpoints; the dataset is ~800 MB and
checkpoints are small (~11 MB each).

## Reference training command

```bash
PYTHONPATH=src .venv/bin/python3 -m fretwire.train \
  --data-dir data/processed \
  --vocab-path vocab_v1_full.json \
  --output-dir runs/pretrain_t5_v5 \
  --batch-size 8 \
  --grad-accum 8 \
  --bf16 \
  --steps-per-epoch 2000 \
  --eval-every 250 \
  --eval-loss-batches 50 \
  --eval-batches 10 \
  --metrics-every 1250 \
  --save-every 2500 \
  --max-epochs 300
```

`steps_per_epoch` counts **micro-batches**. With batch 8 and grad-accum 8, one
epoch is 250 optimizer steps (16k sequences, close to the paper's 16,451
training sequences). `eval_every`, `metrics_every`, and `save_every` count
optimizer steps.

Evaluation is two-tier:

- every `eval_every`: fast teacher-forced validation loss; used for early
  stopping and `best/` checkpoint;
- every `metrics_every`: full-track greedy decoding for raw and post-processed
  tab/pitch metrics. This is expensive, so it is less frequent.

Use `--eval-beam 1` for the paper-style greedy default. `--grad-checkpointing`
is off by default; enable it only if memory becomes tight. Do not use
`--no-grad-checkpointing` (the flag was replaced by `--grad-checkpointing`).

## Files that matter

- `src/fretwire/model.py` — paper-config T5 factory.
- `src/fretwire/tokenizer.py` — v3 event/target encoding.
- `src/fretwire/chunk.py` — 512-token train windows / 20-note inference chunks.
- `src/fretwire/postproc.py` — overlap correction + neighbor search.
- `src/fretwire/metrics.py` — tab/pitch accuracy and difficulty.
- `src/fretwire/train.py` — training, validation loss, full metrics.
- `src/fretwire/decode.py` — T5 generation helpers.
- `vocab_v1_full.json` — frozen 423-token vocab. Do not use the unpruned
  `DEFAULT_VOCAB` (531) for training.
