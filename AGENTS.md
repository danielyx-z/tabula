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
- `src/tablature/postproc.py` — overlap correction + neighbor search.
- `src/tablature/metrics.py` — tab/pitch accuracy, difficulty, **rhythm**.
- `src/model/train.py` — training, validation loss, full metrics.
- `src/model/decode.py` — T5 generation helpers.
- `src/core/quantise.py` — 20-tick grid, shift decomposition, `quantise_notes`,
  **`ticks_to_seconds(tick, tempo_changes, ppq)`** converts tick → seconds
  following a tempo map.
- `src/audio/synth.py` — FluidSynth guitar synthesis. Writes a temporary MIDI
  then renders with `fluidsynth -ni -g 1.0 -R 0 -C 0`. **Critical:** FluidSynth
  writes **stereo** WAVs; `synthesize()` reads `nchan`/`sampwidth` and downmixes
  to mono before trimming, otherwise interleaved L/R frames are read as
  twice as many samples and audio plays 2× slow / half-length. Accepts
  `tempo_changes` and `source_midi_path` to preserve the input tempo map.
- `scripts/midi2tab.py` — `write_tab_midi(path, notes, bpm, ppq,
  source_midi_path, tempo_changes)`. If `source_midi_path` is given, loads that
  MIDI, sets `pm.instruments = []`, appends six string tracks and writes — this
  preserves the source's full tempo map, key signature and time signature.
  `ticks_to_seconds` converts note ticks to seconds using the tempo map.
- `vocab/guitarset.json` — frozen 248-token vocab. Do not use the unpruned
  `DEFAULT_VOCAB` (286) for training.

## Fixed bugs (recent)

- **Tempo map flattening** — output `input/raw/final.mid` previously flattened
  to a single constant BPM. Now they load the source MIDI and replace instruments,
  keeping all original tempo changes (e.g. lover_girl.mid: 37 tempo changes,
  74–83 BPM, first 80.9 BPM, duration 176.56 s).
- **2× slow WAV synthesis** — FluidSynth writes stereo; the old code treated the
  interleaved L/R stream as consecutive mono samples, halving the playback rate
  and trimming to half the music. Fixed by downmixing to mono before trim.
  Every WAV (`input/raw/final.wav`) is now derived from its own MIDI at the
  correct speed.
- **Tab measure wrapping** — `notes_to_tab` now wraps at whole-measure
  boundaries so bar lines align across all six strings within each system.
- **Chunk-boundary duplication** — the T5 decoder could generate past its
  20-note inference window, and `scripts/midi2tab.py` only stripped one
  re-emitted context note. Output was therefore not time-sorted, contained
  exact duplicates, and had notes at onsets the input never had. Each chunk's
  output is now trimmed to `base_t < t <= last_input_t`, giving 0 duplicates
  and 0 backwards jumps (lover_girl: 1199 raw notes -> 965, 14 breaks -> 0).
- **Pitch provenance in `final.mid`** — `final.mid` now takes onset, duration
  and pitch from the input MIDI (ground truth) and only the string from the
  model (`fret = pitch - tuning[string]`), so final pitch accuracy is 1.0 by
  construction. `raw.mid` stays uncorrected so the model's own pitch errors
  are still visible.
- **Low-position prior** — `assign_onsets()` assigns each onset's chord
  jointly to unique strings, minimising `(notes above --max-fret, max fret,
  sum of frets, deviation from the model)`. `fix_chord_spans`, `limit_jumps`
  and `repair_overlaps` also filter candidates by `--max-fret` (default 9).
  On lover_girl, final frets >= 10 dropped from 12.9% to 0.0%, mean fret from
  5.66 to ~3.25, with pitch unchanged.
- **Unplayable input chords** — a piano-derived MIDI can contain six-note
  chords wider than any hand (e.g. Valentine's `[51,58,62,63,67,70]`). The old
  `fix_chord_spans` re-fingered greedily and could move two notes onto one
  string, after which `cap_simultaneous` dropped 42 notes. It is now a *joint*
  per-onset assignment, so it cannot create a string collision; when no
  voicing fits the target span it falls back to the most compact unique-string
  voicing (minimising the maximum fret). `playability_pass` also re-assigns
  any residual collision instead of dropping the note. Valentine now keeps all
  750 input pitches; its 21 six-note chords use the compact
  `11/13/12/8/8/6` voicing instead of a single fret 22.
- **One `playability_pass`** — `scripts/midi2tab.py` carried a duplicate copy
  of the pass; it now imports `tablature.playability.playability_pass`, so the
  CLI and library cannot drift apart.
