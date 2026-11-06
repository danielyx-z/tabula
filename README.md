# Tabula — data layer

MVP target: symbolic MIDI in, 6-string standard-tuning ASCII tab out. See
[PLAN.md](PLAN.md) for the model plan; this README covers the data layer that
exists today.

## Status

The pipeline is wired end to end: **audio/MIDI → canonical notes → shards →
T5 → decoder stream → MIDI/WAV/ASCII tab**. The current model was trained on a
now-discarded synthetic corpus and is not musically useful. The data layer now
targets GuitarSet; a real GuitarSet training run is the next step.

New to the project? Read:

* [`docs/FORMATS.md`](docs/FORMATS.md) — every input/output format.
* [`docs/STITCHING.md`](docs/STITCHING.md) — audio dead notes and articulations on top of MIDI-to-tab.
* [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) — the training data contract.
* [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what was tested and what
  still fails.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Convert GuitarSet annotations into training shards
PYTHONPATH=src .venv/bin/python3 scripts/preprocess_guitarset.py \
    --jams-dir /path/to/guitarSet_annotation --out data/processed_guitarset \
    --chord-window 40
PYTHONPATH=src .venv/bin/python3 scripts/verify.py --dataset data/processed_guitarset

# Build a vocabulary for that corpus (vocab/guitarset.json is committed;
# rebuild only when the corpus changes)
PYTHONPATH=src .venv/bin/python3 scripts/build_vocab.py \
    --data-dir data/processed_guitarset --out vocab/guitarset.json

# Train (max-steps counts optimizer steps; dataset cycles infinitely)
PYTHONPATH=src .venv/bin/python3 -m model.train \
    --data-dir data/processed_guitarset --vocab-path vocab/guitarset.json \
    --output-dir runs/pretrain_t5_guitarset \
    --max-steps 30000

# MIDI -> input/raw/final MIDI + WAV + tab + report
PYTHONPATH=src:. .venv/bin/python3 scripts/midi2tab.py song.mid \
    --checkpoint runs/pretrain_t5_guitarset/best --vocab vocab/guitarset.json \
    --out-dir out/song

# Fine-tuned checkpoint (canonical 262-token vocab) + automatic capo
PYTHONPATH=src:. .venv/bin/python3 scripts/midi2tab.py song.mid \
    --checkpoint runs/finetune_t5_guitarset/best --vocab vocab/canonical.json \
    --out-dir out/song --beam 4 --max-fret 6 --capo auto
```

The exported MIDI files carry **one track per string**, because a normal MIDI
file only stores pitch and time and cannot represent a fingering.

All output MIDI and WAV files preserve the **source tempo map** (every tempo,
key and time signature change, not one constant BPM). `write_tab_midi` loads
the source MIDI, sets `pm.instruments = []`, appends six string tracks and
writes; `ticks_to_seconds` converts note ticks to seconds using the tempo map.

WAVs are rendered via FluidSynth + the Live HQ Natural GM V3 SoundFont (by
UnderxPipe1985, CC BY; FluidR3_GM is the fallback), using GM program 24
(Acoustic Guitar (nylon) / "Concert Nylon Guitar"). FluidSynth writes stereo;
`synthesize()` downmixes to mono before trimming so interleaved L/R frames are
not read as twice as many samples (otherwise audio plays 2× slow and is cut at
half length). Each WAV is derived only from its own MIDI.

Final fingering prefers the lowest capo-relative fret (open strings first) and
soft-caps frets above 6. `--capo auto` scans feasible capos 0–6 and picks the
most playable one; `final.json`/`final_tab.txt` frets are capo-relative and
the tab shows `Capo: N`. If an input chord is genuinely unplayable, one inner
tone may be dropped; `report.json` lists it (Valentine: 21 duplicated-Eb drops,
729/750 notes).

## Web app

The Flask workbench turns a MIDI upload into a real Guitar Pro 5 score and
plays it in the browser with alphaTab. The score keeps the source tempo map,
uses the nylon-guitar program (24), and highlights the current measure during
playback. It invokes the same `midi2tab.py` pipeline with `--no-audio` and
exports `score.gp5` from that job's own `final.json` and `final.mid`.

On the machine that has the checkpoint and vocabulary:

```bash
.venv/bin/pip install -r webapp/requirements-web.txt
.venv/bin/python -m webapp.server --port 8000
# Reach it from another machine on the LAN:
.venv/bin/python -m webapp.server --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000>, upload a `.mid`/`.midi`, and wait for the score.
A conversion takes roughly 20–30 s on the 3070. Job files live under
`webapp/jobs/<id>/` (`upload.mid`, `score.gp5`, `final.mid`,
`final_tab.txt`, `report.json`); that directory is local data and is ignored
by git. Jobs older than 24 hours are removed on startup.

The page vendors alphaTab 1.9.0 (MPL-2.0) with its Bravura font and Sonivox
soundfont; keep the rendered attribution visible. See
`webapp/static/alphatab/README.md` and `webapp/static/fonts/README.md`.

## Layout

```
src/
  core/        # Note schema, fretboard, quantise, chord grouping
  corpus/      # shard dataset, filtering, MIDI/alphaTex parsers
  tablature/   # tokenizer, stream decoder, playability, metrics, render
  model/       # T5 config, chunking, decoding, training
  audio/       # guitar synth, MIDI writer, WAV/playback
  cli.py       # `python -m cli`
scripts/       # converters, training/eval, preview, Slurm jobs
tests/
docs/
vocab/
```

## Contract

| Thing | Value |
| --- | --- |
| PPQ | 960 |
| Time storage | integer ticks, 20-tick grid (1/48 note) |
| TIME_SHIFT | 11 compositional values; exact for every grid multiple |
| Strings | 1 = thickest (low E) … 6 = thinnest (high E) |
| Tuning | standard only, `(40, 45, 50, 55, 59, 64)` |
| Pitch | `pitch = tuning[s-1] + capo + fret`, frets 0–24 |
| Chords | notes at same onset sorted ascending by string (s=1..6) |
| Capo | auto 0–6; frets are capo-relative |
| Pitch names | Scientific notation: MIDI 60 = C4 (middle C) |

`src/core/fretboard.py` is the only place pitch ↔ (string, fret) happens;
`src/corpus/parse/alphatex.py` is the only place notation is interpreted.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On a GPU host, install the CUDA PyTorch build that matches the driver if the
plain wheel reports `False`.

## Tests

```bash
.venv/bin/python tests/test_core.py     # parser / fretboard unit checks
.venv/bin/python tests/test_metrics.py  # quantiser / post-processing / metrics checks
.venv/bin/python tests/test_chunk.py    # chord-safe chunking / string order
.venv/bin/python tests/test_synth.py    # FluidSynth wrapper
.venv/bin/python -m pytest tests/test_playability.py  # capo / open strings / chord drop
.venv/bin/python -m pytest -q           # full suite, including webapp GP5 export
.venv/bin/python scripts/verify.py      # invariants over the built dataset
.venv/bin/python scripts/audit_guitarset.py --jams-dir /path/to/guitarSet_annotation --data-dir data/processed_guitarset --vocab vocab/guitarset.json
```

## Preprocess

```bash
.venv/bin/python scripts/preprocess_guitarset.py --jams-dir /path/to/guitarSet_annotation --out data/processed_guitarset --chord-window 40
# ~30 s for 360 files, single core, ~2 MB output
```

Output is sharded and memory-mappable:

```
data/processed_guitarset/
  manifest.json          contract + counts + reject histogram + grid stats
  rejects.jsonl          every dropped track and why
  shard-00000/
    notes.npy            structured dtype, 11 B/note: (t, dur, pitch, s, f)
    offsets.npy          int64[n_tracks+1] -> slice per track
    tracks.jsonl         one metadata record per track
```

Load without copying:

```python
import numpy as np
notes = np.load("data/processed_guitarset/shard-00000/notes.npy", mmap_mode="r")
offsets = np.load("data/processed_guitarset/shard-00000/offsets.npy")
track = notes[offsets[i]:offsets[i + 1]]
```

or, across shards, `corpus.dataset.iter_tracks(root, split="train")`.

### GuitarSet corpus

| | |
| --- | --- |
| files | 360 |
| performances | 180 (each has comp + solo) |
| tracks kept | 360 |
| notes | 62,476 |
| pitch range | 40 (E2) – 81 (A5) |
| max fret used | 19 |
| train / valid / test | 272 / 50 / 38 (split by performance, not track) |

Splits key on `performance_key` (strips `_comp` / `_solo`) so comp/solo never
leak across splits.

### Things the measurement changed

- **Bare `r` means a whole-measure rest.** The exporter writes `r |`, not
  `:1 r |`. Snapping those bars to the track's modal bar length moved the
  4/4-bar share from 78 % to 85 % and stopped every following note from
  drifting.
- **Time signatures vary and are never declared.** 85 % of bars are 4/4, 7 %
  are 3/4, the rest are floaty/odd meters. Nothing in the MVP needs a meter,
  so meter is stored as `bar_ticks` / `bar_consistency` per track rather than
  filtered on.
- **Tuplets are 4.3 % of onsets.** A 60-tick (1/16) grid silently rewrites
  every triplet; a 20-tick (1/48) grid represents triplets, dotted values and
  32nds exactly, and leaves only 0.2 % of notes off-grid (quintuplets,
  septuplets). Storage keeps exact ticks. The model uses 11 compositional `TIME_SHIFT` values, so every multiple of
  20 ticks is encoded exactly instead of being snapped to a 60-tick step.
- **No capo in the corpus export**, no repeats, no ties, no dead notes, no
  time-signature directives — those parser branches are defensive only.
  (Inference can still apply a capo 0–6 to the fingering.)

## Training (WATcloud)

Training is intended for the WATcloud Slurm cluster; see [`AGENTS.md`](AGENTS.md)
for resource choices and monitoring. From the repository root:

```bash
export PATH=/opt/scripts/bin:$PATH   # the cluster login PATH omits Slurm
mkdir -p logs
sbatch scripts/train_t5.sh
```

Before a long run, the GPU smoke test checks CUDA, bf16 and two optimizer
steps without decoding full tracks:

```bash
srun --partition=compute --gres=shard:rtx_3090:4096,tmpdisk:2048 \
     --cpus-per-task=4 --mem=8G --time=00:20:00 \
     bash scripts/smoke_t5.sh
```

The defaults are paper-sized: batch 8 × grad-accum 8 (effective batch 64),
512-token training chunks, and 20-note inference chunks. `max_steps=30000`
gives ~30k optimizer steps (~5 h on RTX 3070, ~2.5 h on RTX 3090). For
paper-scale 75k steps, override with `MAX_STEPS=75000`. Validation loss is
checked every 500 optimizer steps; full-track tab metrics are decoded every
2,500 steps. `--resume-from runs/pretrain_t5_guitarset/best` reloads weights,
vocabulary, optimizer/schedule and early-stopping state. Post-processing runs
per 20-note inference chunk so one missing note cannot shift every later chunk.

To evaluate a saved checkpoint without training (same metrics as the job):

```bash
PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
  --checkpoint runs/pretrain_t5_guitarset/best --tracks 10 --postproc full \
  --context-mode pred
```

```bash
# local smoke test, no long training
PYTHONPATH=src .venv/bin/python3 -m model.train \
  --max-steps 2 \
  --batch-size 2 --grad-accum 1 --no-bf16 \
  --eval-every 1 --eval-loss-batches 1 --eval-batches 1 \
  --metrics-every 0 --output-dir /tmp/tabula-smoke
```

To inspect one real inference chunk, including the exact encoder/decoder
tokens, teacher-forcing answer, generated tokens and target/raw/post tabs:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/infer_sample.py \
  --checkpoint runs/pretrain_t5_guitarset/best --track-id 93 --chunk 1
```

## Preview (see it, hear it)

```bash
.venv/bin/python scripts/preview.py --index 8 --bars 8
.venv/bin/python scripts/preview.py --song nepali --bars 4
.venv/bin/python scripts/preview.py --tex path/to/song.tex --bars 4
.venv/bin/python scripts/preview.py --index 8 --no-play --mid out.mid
```

This is a development audition tool, not part of the training pipeline. It
prints the tab, synthesises a plucked-string rendition, plays it through
`aplay`, and walks a playhead through the tab in time with the audio. The tab is
wrapped into systems that fit the terminal width (override with `--width`);
while it plays it pages system by system, highlighting the sounding frets.
`--mid` also writes a Standard MIDI File for a DAW or MuseScore, and `--wav`
keeps the rendered audio.