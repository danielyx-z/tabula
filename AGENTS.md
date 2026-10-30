# Tabula — working memory

## Project

Reimplementation of **Fretting-Transformer: Encoder-Decoder Model for MIDI to
Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

- Input: symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`).
- Output: ASCII tab events (`TAB` + `TIME_SHIFT`).
- Model: paper-config T5, 2.81M parameters (`d_model=128`, `d_ff=1024`,
  `d_kv=64`, 3 encoder/3 decoder layers, 4 heads). The project vocabulary
  is 246 tokens for GuitarSet.
- The paper does **not** define a training-time alignment loss, note-count head,
  or pitch-constrained training loss. Do not reintroduce them. Its alignment
  step is post-processing: overlap correction + ±5-note neighbor search.
- Training chunks: 512 encoder tokens. Inference chunks: 20 notes with the
  previous note as context in encoder and decoder.

## Training is user-driven

Do **not** launch long training runs without asking. The local 8 GB GPU is
not used for long training.

OOM is not a design concern. The cluster job requests a GPU shard; use a batch
size that is efficient (defaults are batch 8 × grad-accum 8 = effective 64).

## WATcloud / Slurm

- SSH login: `ssh watcloud`
- Slurm docs: <https://cloud.watonomous.ca/docs/compute-cluster/slurm>
- The Slurm CLI is not on the default login `PATH`; use
  `export PATH=/opt/scripts/bin:$PATH` or `/opt/scripts/bin/sbatch`.
- Submit from the repository root: `mkdir -p logs && sbatch scripts/train_t5.sh`
- Job script: `scripts/train_t5.sh` (edit paths/resources if needed).

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
tail -f logs/<jobid>-tabula-t5.out
scancel <jobid>
```

WATcloud notifications (instead of Slurm mail):

```bash
./watcloud subscription <jobid> you@example.com
./watcloud subscription <jobid> --discord
```

## Training status

The old `runs/pretrain_t5_v5` checkpoint (kept only on the workstation) and
its metrics were trained on the discarded synthetic corpus. They are **not comparable** to GuitarSet and the
vocabulary has since changed (246 tokens), so the old checkpoint cannot be
loaded with the current data. Treat it as historical.

A real GuitarSet training run has not been completed yet. The active defaults
are `data/processed_guitarset`, `vocab/guitarset.json` and
`runs/pretrain_t5_guitarset`.

Evaluate a checkpoint without training:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
  --checkpoint runs/pretrain_t5_guitarset/best --tracks 10 --postproc full
```

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
corpus must exist at `data/processed_guitarset` (train / valid / test shards).
The GuitarSet shard is small (<2 MB); checkpoints are ~11 MB each.

## Reference training command

```bash
PYTHONPATH=src .venv/bin/python3 -m model.train \
  --data-dir data/processed_guitarset \
  --vocab-path vocab/guitarset.json \
  --output-dir runs/pretrain_t5_guitarset \
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

- `src/model/model.py` — paper-config T5 factory.
- `src/tablature/tokenizer.py` — v3 event/target encoding.
- `src/model/chunk.py` — 512-token train windows / 20-note inference chunks.
- `src/tablature/postproc.py` — overlap correction + neighbor search.
- `src/tablature/metrics.py` — tab/pitch accuracy and difficulty.
- `src/model/train.py` — training, validation loss, full metrics.
- `src/model/decode.py` — T5 generation helpers.
- `vocab/guitarset.json` — frozen 246-token vocab. Do not use the unpruned
  `DEFAULT_VOCAB` (286) for training.
