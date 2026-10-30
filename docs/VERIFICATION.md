# Verification: data → train → infer → export

Run on 2026-10-30 on `fractal` (RTX 3070, 8 GB), PyTorch 2.5.1+cu121.

The goal was to confirm that **once a proper dataset exists, the rest of the
pipeline works end to end**, and to record the exact formats at each boundary.

## Summary

| Stage | Command | Result |
| --- | --- | --- |
| Convert new data | `scripts/preprocess_guitarset.py` | 360 JAMS files → 360 tracks, 62,476 notes |
| Dataset invariants | `scripts/verify.py` | **all invariants hold** |
| Build vocabulary | `scripts/build_vocab.py` | 316 tokens (41 NOTE_ON, 81 TIME_SHIFT) |
| Train (existing vocab) | `python -m fretwire.train` | 20 steps, val loss 6.75, checkpoints written |
| Train (new vocab) | same, `--vocab-path /tmp/gs_vocab.json` | 5 steps, val loss 6.40, checkpoints written |
| Infer with trained checkpoint | `scripts/midi2tab.py` | input/raw/final MIDI + WAV + tab + report |
| Infer with v5 checkpoint | `scripts/midi2tab.py` | 130 raw notes, 42.9 % pitch alignment |
| Unit tests | `tests/test_core.py`, `tests/test_metrics.py` | all checks passed |

## 1. New data conversion

GuitarSet ships six per-string `note_midi` annotations per performance, so the
converted records have real `(string, fret)` values.

```bash
python scripts/preprocess_guitarset.py \
    --jams-dir /tmp/guitarSet_annotation \
    --out /tmp/gs_processed
```

```
files: 360, tracks kept: 360, notes: 62,476, shards: 1
splits: train 279, valid 43, test 38
```

## 2. Invariants

```bash
python scripts/verify.py --dataset /tmp/gs_processed
```

```
shards      1
tracks      360
notes       62,476
songs       360

all invariants hold
```

This checks `pitch == tuning[s-1] + f`, string/fret ranges, one note per string
per onset, time ordering, offset bookkeeping, and split leakage.

## 3. Vocabulary

```bash
python scripts/build_vocab.py --data-dir /tmp/gs_processed --out /tmp/gs_vocab.json
```

```
vocab size : 316
NOTE_ON    : 41
TIME_SHIFT : 81
```

All 42 distinct pitches in GuitarSet (40–81) are already covered by
`vocab_v1_full.json`, so the existing vocabulary also works. Rebuilding is only
required when a new source exceeds the old pitch/timing range.

## 4. Training

Existing vocabulary (20 micro-batches, batch 4):

```
Model: T5, 2.81M params
Step 20 | Loss: 6.7198
Validation loss @ step 20: 6.7541
```

Dataset-derived vocabulary (5 micro-batches, batch 2):

```
Vocab size: 316
Model: T5, 2.80M params
Step 5 | Loss: 6.4272
Validation loss @ step 5: 6.4048
```

Both runs wrote `checkpoint-*`, `best/`, `final/` and `vocab.json` to their
output directory. These are **mechanics checks**, not quality runs: 5–20 steps
is far too few to learn anything.

## 5. Inference and export

With the 20-step checkpoint (proves a freshly trained checkpoint loads):

```bash
python scripts/midi2tab.py /tmp/guitarset_midi/00_BN1-129-Eb_comp.mid \
    --checkpoint /tmp/gs_run/final --vocab /tmp/gs_run/final/vocab.json \
    --out-dir /tmp/gs_infer --beam 4
```

It wrote all artefacts and a report. Quality was garbage, as expected after 20
steps; the important part is that every stage ran.

With the existing v5 checkpoint on the same GuitarSet MIDI:

```json
{
  "input_notes": 133,
  "raw_notes": 130,
  "aligned": 114,
  "pitch_correct": 57,
  "pitch_accuracy": 0.4286,
  "tab_exact": 0,
  "tab_accuracy": 0.0,
  "dropped_impossible": 7,
  "refingered_chord_span": 10,
  "refingered_jump": 31,
  "final_notes": 123
}
```

`input.mid`, `raw.mid` and `final.mid` all have different hashes. Each is
written one track per string so the fingering is actually stored.

## 6. Unit tests

```bash
python tests/test_core.py      # all checks passed
python tests/test_metrics.py   # all checks passed
```

## What this does and does not prove

Proved:

* GuitarSet JAMS converts into the canonical shard format.
* The dataset passes every structural invariant.
* Vocabulary construction from a dataset works.
* Training loads the new shards, runs forward/backward, validates and saves
  checkpoints.
* A saved checkpoint can be loaded by the inference CLI.
* Inference writes `input.mid`, `raw.mid`, `final.mid`, WAVs, ASCII tab and a
  JSON report, with no ground-truth timing or pitch leaking into `raw`.
* The format contract is compatible with the existing vocabulary.

Not proved:

* **Model quality.** The 20-step run is a smoke test. v5 was trained on the old
  GuitarPro corpus and scores 42.9 % pitch alignment on GuitarSet. A real
  training run on GuitarSet is still required.
* **GAPS ingestion.** GAPS ships MusicXML rather than JAMS and still needs a
  converter.
* **Audio front end on GuitarSet recordings.** The converter and MIDI path were
  tested; running basic-pitch over the GuitarSet audio and scoring the result
  is a separate experiment.

## Known failure mode

The playability pass drops notes that are physically impossible (two notes on
one string at the same tick). A badly decoded stream can therefore shrink a
lot: the 20-step checkpoint produced 520 raw notes and 4 playable ones. This is
intentional — the alternative is to emit a tab that cannot be played — but it
means a very poor model can look like it “lost” most of the input.
