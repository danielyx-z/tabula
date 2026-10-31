# Tabula — working memory

## Project

Reimplementation of **Fretting-Transformer: Encoder-Decoder Model for MIDI to
Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

- Input: symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`).
- Output: ASCII tab events (`TAB` + `TIME_SHIFT`).
- Model: paper-config T5, 2.79M parameters (`d_model=128`, `d_ff=1024`,
  `d_kv=64`, 3 encoder/3 decoder layers, 4 heads). The project vocabulary
  is 248 tokens for GuitarSet.
- The paper does **not** define a training-time alignment loss, note-count head,
  or pitch-constrained training loss. Do not reintroduce them. Its alignment
  step is post-processing: overlap correction + ±5-note neighbor search.
- Training chunks: 512 encoder tokens. Inference chunks: 20 notes with the
  previous note as context in encoder and decoder.
- Time is quantised to a 20-tick grid (PPQ 960). `TIME_SHIFT` tokens are
  compositional (11 values: 20..3840). Onsets/durations are snapped to this grid.
- Rhythm metrics (onset hit rate, IOI/duration MAE) are computed on raw
  model output; post-processing assigns GT timing and is not rhythm-meaningful.

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

See `docs/WATCLOUD.md` for the full cluster reference (node/GPU table, GRES
semantics, notifications, troubleshooting).

Verified throughput on `trpro-slurm1` (RTX 3090 shard, batch 8 × grad-accum 8,
warm chunk cache): ~4.2 optimizer steps/s, so 30,000 steps ≈ 2 h including
validation loss and periodic full-track metrics.

WATcloud notifications (instead of Slurm mail):

```bash
./watcloud subscription <jobid> you@example.com
./watcloud subscription <jobid> --discord
```

## Training status

The old `runs/pretrain_t5_v5` checkpoint (kept only on the workstation) and
its metrics were trained on the discarded synthetic corpus. They are **not comparable** to GuitarSet and the
vocabulary has since changed (248 tokens), so the old checkpoint cannot be
loaded with the current data. Treat it as historical.

A real GuitarSet training run has not been completed yet. The active defaults
are `data/processed_guitarset`, `vocab/guitarset.json` and
`runs/pretrain_t5_guitarset`.

Evaluate a checkpoint without training:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
  --checkpoint runs/pretrain_t5_guitarset/best --tracks 10 --postproc full \
  --context-mode pred
```

## Cluster setup

The cluster needs its own environment and data:

```bash
ssh watcloud
cd ~/tabula_gs                    # new GuitarSet workspace
git clone https://github.com/danielyx-z/tabula.git .
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pretty_midi
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The tested cluster driver is 580.178.04 (CUDA 13.0); the cu121 wheel above
works. If `torch.cuda.is_available()` is false, reinstall the CUDA wheel
matching the compute-node driver (see
<https://pytorch.org/get-started/locally/>) before submitting. The processed
corpus must exist at `data/processed_guitarset` (train / valid / test shards).
Copy `data/processed_guitarset/` and `vocab/guitarset.json` from the fractal
workstation (2 MB total). Verify: vocab size 248, hash `6265d9bb91e68095`.
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
  --max-steps 30000 \
  --eval-every 500 \
  --eval-loss-batches 0 \
  --eval-batches 10 \
  --metrics-every 2500 \
  --save-every 2500 \
  --postproc full
```

`max_steps` counts **optimizer steps** (micro-batches / grad_accum). With
batch 8 and grad-accum 8, 30,000 steps ≈ 240,000 sequences. The dataset
cycles infinitely; `eval_every`, `metrics_every`, and `save_every` count
optimizer steps. No partial grad-accum step occurs.

Evaluation is two-tier:

- every `eval_every`: fast teacher-forced validation loss on all validation
  chunks; used for early stopping and `best/` checkpoint;
- every `metrics_every`: full-track greedy decoding for raw and post-processed
  tab/pitch/rhythm metrics. This is expensive, so it is less frequent.

Use `--eval-beam 1` for the paper-style greedy default. `--grad-checkpointing`
is off by default; enable it only if memory becomes tight.

## Files that matter

- `src/model/model.py` — paper-config T5 factory.
- `src/tablature/tokenizer.py` — v3 event/target encoding.
- `src/model/chunk.py` — 512-token train windows / 20-note inference chunks.
- `src/corpus/chunk_store.py` — tokenises training chunks **once** into flat
  arrays and caches them under `<data-dir>/.chunk_cache/`. Training epochs then
  only reshuffle indices (~0.4 ms/batch vs ~47 ms of re-chunking per batch).
  Do not reintroduce per-epoch chunking in the training loop.
- `src/tablature/postproc.py` — overlap correction + neighbor search.
- `src/tablature/metrics.py` — tab/pitch accuracy, difficulty, **rhythm**.
- `src/model/train.py` — training, validation loss, full metrics.
- `src/model/decode.py` — T5 generation helpers.
- `src/core/quantise.py` — 20-tick grid, shift decomposition, `quantise_notes`.
- `vocab/guitarset.json` — frozen 248-token vocab. Do not use the unpruned
  `DEFAULT_VOCAB` (286) for training.