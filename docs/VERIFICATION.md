# Verification: data → train → infer → export

Re-run after switching to the 20-tick grid, the 11-token compositional
TIME_SHIFT set and chord grouping. Host: `fractal` (RTX 3070, 8 GB),
PyTorch 2.5.1+cu121.

## Summary

| Stage | Command | Result |
| --- | --- | --- |
| Convert GuitarSet | `scripts/preprocess_guitarset.py` | 360 tracks, 62,476 notes, 38,052 onsets |
| Dataset invariants | `scripts/verify.py` | **all invariants hold** |
| Vocabulary | `scripts/build_vocab.py` | **246 tokens**, 11 TIME_SHIFT |
| Quantisation test | `tests/test_metrics.py` | every 20-tick multiple decomposes exactly |
| Training | `python -m fretwire.train` | 15 steps, val loss 6.27, checkpoints written |
| Inference | `scripts/midi2tab.py` | input/raw/final MIDI + WAV + tab + report |
| Unit tests | `test_core.py`, `test_metrics.py` | all checks passed |

## 1. Chord grouping

Performed GuitarSet chords are spread across 20–40 ticks. Grouping onsets
inside a 40-tick window changes the corpus from fast arpeggios into chords:

| | before | after |
| --- | --- | --- |
| distinct onsets | 51,666 | 38,052 |
| chord onsets | 8,867 (17.2 %) | 13,520 (35.5 %) |
| mean notes per onset | 1.21 | 1.64 |
| total notes | 62,476 | 62,476 |

## 2. TIME_SHIFT vocabulary

The old scheme allocated one token per 60-tick multiple. From 94,056 positive
timing deltas it emitted an average of **0.71 tokens per delta** — i.e. it
silently discarded about 29 % of gaps shorter than 30 ticks.

The new scheme uses 11 values:

```
20, 40, 60, 120, 180, 240, 480, 720, 960, 1920, 3840
```

| | old | new |
| --- | --- | --- |
| TIME_SHIFT tokens | 81 (172 unpruned) | **11** |
| vocabulary size | 316 | **246** |
| timing exactness | ~71 % of deltas | **100 %** |
| avg tokens per delta | 0.71 | 1.62 |

Every multiple of 20 up to 40,000 ticks was tested and decomposes exactly.

## 3. Dataset

```
files: 360, tracks kept: 360, notes: 62,476, shards: 1
splits: train 279, valid 43, test 38
```

`scripts/verify.py` confirms `pitch == tuning[s-1] + f`, one note per string
per onset, valid ranges, time ordering, offset bookkeeping and no split
leakage.

## 4. Vocabulary

```
vocab size : 246
NOTE_ON    : 41
NOTE_OFF   : 41
TIME_SHIFT : 11
TAB        : 150
```

Written to `vocab/guitarset.json` and used as the default by the training and
inference CLIs.

## 5. Training smoke test

```bash
python -m fretwire.train --data-dir /tmp/gs3 \
    --vocab-path vocab/guitarset.json --output-dir /tmp/gs3_run \
    --max-epochs 1 --steps-per-epoch 15 --batch-size 4 --grad-accum 1 \
    --eval-every 10 --eval-loss-batches 5 --metrics-every 0 \
    --save-every 10 --patience 2
```

```
Vocab size: 246
Model: T5, 2.79M params
Validation loss @ step 10: 6.2705
Epoch 0 | Avg Loss: 6.3544
```

This is a mechanics check, not a quality run.

## 6. Inference and export

```bash
python scripts/midi2tab.py /tmp/guitarset_midi/00_BN1-129-Eb_comp.mid \
    --checkpoint /tmp/gs3_run/final --vocab vocab/guitarset.json \
    --out-dir /tmp/gs3_infer --beam 4
```

The CLI wrote `input.mid`, `raw.mid`, `final.mid`, three WAVs, `final_tab.txt`
and `report.json`. After 15 steps the output is garbage (0/133 pitch
alignment), which is expected; the point is that the whole path runs with the
new vocabulary.

`input.mid`, `raw.mid` and `final.mid` have different hashes. Each is written
one track per string so the fingering is stored.

## 7. Tests

```
tests/test_core.py      all checks passed
tests/test_metrics.py   all checks passed
```

`test_metrics.py` now also asserts:

* every 20-tick multiple up to 40,000 decomposes exactly;
* `SHIFT_VOCAB` has exactly 11 entries;
* a strummed chord collapses to one onset and keeps its release tick.

## Not proved

* **Model quality.** 15 steps is a smoke test. A real GuitarSet training run is
  still required.
* **GAPS ingestion.** GAPS ships MusicXML rather than JAMS and still needs a
  converter.
* **Techniques.** Hammer-ons, slides, bends and vibrato are not representable
  in the current record or vocabulary.

## Known failure mode

The playability pass still drops physically impossible notes (two notes on one
string at the same tick). A model that has learned nothing can emit almost
every note at one tick, so the pass may keep only a handful. That is
intentional; the alternative is to emit an unplayable tab.
