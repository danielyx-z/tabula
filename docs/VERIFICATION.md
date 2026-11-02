# Verification: data → train → infer → export

Re-run after switching to the 20-tick grid, the 11-token compositional
TIME_SHIFT set and chord grouping. Host: `fractal` (RTX 3070, 8 GB),
PyTorch 2.5.1+cu121.

## Summary

| Stage | Command | Result |
| --- | --- | --- |
| Convert GuitarSet | `scripts/preprocess_guitarset.py` | 360 tracks, 62,476 notes, 38,052 onsets |
| Dataset invariants | `scripts/verify.py` | **all invariants hold** |
| Full audit | `scripts/audit_guitarset.py` | **AUDIT PASSED** (0/360 files differ) |
| Vocabulary | `scripts/build_vocab.py` | **248 tokens**, 11 TIME_SHIFT |
| Quantisation test | `tests/test_metrics.py` | every 20-tick multiple decomposes exactly |
| Training | `python -m model.train` | steps-based, no partial grad-accum, rhythm metrics |
| Inference | `scripts/midi2tab.py` | input/raw/final MIDI + WAV + tab + report |
| Unit tests | `test_core.py`, `test_metrics.py`, `test_synth.py`, `test_chunk.py` | all checks passed |

## 1. Chord grouping

Performed GuitarSet chords are spread across 20–40 ticks. Grouping onsets
inside a 40-tick window changes the corpus from fast arpeggios into chords:

| | before | after |
| --- | --- | --- |
| distinct onsets | 51,666 | 38,052 |
| chord onsets | 8,867 (17.2 %) | 13,520 (35.5 %) |
| mean notes per onset | 1.21 | 1.64 |
| total notes | 62,476 | 62,476 |

**Chord ordering:** same-onset notes sorted strictly by string ascending
(`s=1` low E → `s=6` high E). `tests/test_chunk.py::test_chord_string_order`
verifies the decoder target emits `TAB` tokens in this order.

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
| vocabulary size | 316 | **248** |
| timing exactness | ~71 % of deltas | **100 %** |
| avg tokens per delta | 0.71 | 1.69 |

Every multiple of 20 up to 40,000 ticks was tested and decomposes exactly.
The 1.69 average is measured on the grouped GuitarSet corpus (84,161 deltas).

## 3. Dataset

```
files: 360, tracks kept: 360, notes: 62,476, shards: 1
splits: train 272, valid 50, test 38
performances: 180 (each has _comp and _solo take)
```

`scripts/verify.py` confirms `pitch == tuning[s-1] + f`, one note per string
per onset, valid ranges, time ordering, offset bookkeeping and no split
leakage.

`scripts/audit_guitarset.py` additionally verifies:
- JAMS → shard note equality (content, ignoring within-onset order)
- Vocab covers every encoder/decoder token used
- No training/inference chunk boundary splits an onset group
- Beat-map slope matches tempo (<5% deviation)
- Extrapolation beyond beat map < 0.75 s

All 360 tracks pass.

## 4. Vocabulary

```
vocab size : 248
NOTE_ON    : 42
NOTE_OFF   : 42
TIME_SHIFT : 11
TAB        : 150
```

Written to `vocab/guitarset.json` (hash `6265d9bb91e68095`) and used as the
default by the training and inference CLIs.

## 5. Training

Training is driven by `--max-steps` (optimizer steps). The dataset cycles
infinitely, so there is no "epoch" truncation and no partial grad-accum step.

```bash
PYTHONPATH=src .venv/bin/python3 -m model.train \
  --data-dir data/processed_guitarset \
  --vocab-path vocab/guitarset.json \
  --output-dir runs/pretrain_t5_guitarset \
  --batch-size 8 --grad-accum 8 --bf16 \
  --max-steps 30000 \
  --eval-every 500 --eval-loss-batches 0 \
  --eval-batches 10 --metrics-every 2500 \
  --save-every 2500 --postproc full
```

- `eval_every`: fast teacher-forced validation loss on **all** validation chunks.
- `metrics_every`: full-track greedy decoding with **rhythm metrics** (onset hit
  rate at ±20/±60/±120 ticks, median IOI/duration MAE) on raw predictions.
- `context_mode`: `"pred"` (default, deployment realism) or `"gt"` (oracle upper
  bound). Evaluation uses `pred`.
- Early stopping on validation loss with `patience=10`, `min_delta=1e-4`.

### Smoke test (2 optimizer steps)

```bash
python -m model.train --data-dir data/processed_guitarset \
  --vocab-path vocab/guitarset.json --output-dir /tmp/smoke_t5 \
  --max-steps 2 --batch-size 2 --grad-accum 1 --no-bf16 \
  --eval-every 1 --eval-loss-batches 1 --eval-batches 1 \
  --metrics-every 0 --save-every 100 --max-len 128
```

```
Vocab size: 248
Model: T5, 2.79M params
Step 1 | Loss: 6.xxxx | LR: self-adaptive
Step 2 | Loss: 6.xxxx | LR: self-adaptive
--- Validation loss @ step 2: 6.xxxx ---
New best validation loss: 6.xxxx
```

## 6. Inference and export

```bash
python scripts/midi2tab.py /path/to/song.mid \
    --checkpoint runs/finetune_t5_guitarset/best --vocab vocab/canonical.json \
    --out-dir out/song --beam 4 --max-fret 6 --capo auto
```

`--capo auto` scans every feasible capo (0 up to
`min(--max-capo (default 6), min_pitch - 40)`);
use `--capo off` or `--capo N` to force one. `--max-fret` is a soft cap
(default 6); frets above it are used only when no lower position exists.
The CLI writes:
- `input.mid` — quantised input (one track per string)
- `raw.mid` — model's own output: onsets from generated `TIME_SHIFT`, pitches from `TAB`
- `final.mid` — input pitch/timing + a playable fingering (auto capo, soft `--max-fret` cap; pitch exact for every surviving note)
- `input.wav` / `raw.wav` / `final.wav` — FluidSynth (SoundFont) render
- `final_tab.txt` — capo-relative ASCII tab of the corrected fingering (`Capo: N` header)
- `report.json` — pitch/tab accuracy, capo trials, rhythm metrics, post-processing stats and any dropped chord tones

`input.mid`, `raw.mid` and `final.mid` have different hashes. Each is written
one track per string so the fingering is stored.

`raw.mid` is deliberately uncorrected: onsets come from generated `TIME_SHIFT`
and pitches from generated `TAB<s,f>`, so it shows the model's own errors.
`final.mid` takes pitch, onset and duration from the input MIDI (ground truth)
and chooses a playable (string, fret) for each note; the model's string is a
tie-breaker, not a hard constraint. The fret is derived from the input pitch
and capo (`fret = pitch - tuning[string] - capo`), so every surviving note is
pitched exactly. If a full chord voicing would need a fret of 10 or more, the
pass may drop one inner tone (listed in `report.json`); otherwise the note
count is unchanged. Chunk generation is trimmed to each chunk's own note
window, which removed duplicate notes and backwards time jumps at chunk
boundaries.

## 7. Tests

```
tests/test_core.py          all checks passed
tests/test_metrics.py       all checks passed
tests/test_synth.py         all checks passed
tests/test_chunk.py         all checks passed
tests/test_playability.py   all checks passed
```

`test_metrics.py` asserts:
- every 20-tick multiple up to 40,000 decomposes exactly;
- `SHIFT_VOCAB` has exactly 11 entries;
- `decompose_shift(640) == [480, 120, 40]`;
- a strummed chord collapses to one onset and keeps its release tick.

`test_synth.py` checks that synthesis is not silent, peak is limited,
chords are finite, synthesis is deterministic, **note onset is at the
requested time (not 2× slow — FluidSynth stereo downmix regression)**,
duration ≈ midi end + release tail, and low notes render with
substantial low-frequency energy.

`test_chunk.py` verifies:
- `_avoid_chord_split` retreats to chord start / extends through chord
- Training chunks never split an onset group (0/697)
- Inference chunks never split an onset group (0/3,391)
- Decoder `TAB` tokens within a chord onset are strictly string-ascending
- 512-token cap never exceeded after chord-boundary adjustment

`test_playability.py` checks capo-relative positions, open-string preference,
the uncapped low-max-fret fallback (73/66), one-tone chord drops, and the
`Capo: N` tab header.

## 8. Synthesis

`src/audio/synth.py` uses **FluidSynth + the Live HQ Natural GM V3 SoundFont**
(`~/soundfonts/Live_HQ_Natural_GM_V3.sf2`, from
musical-artifacts.com/artifacts/1176, by UnderxPipe1985, CC BY), selectable with `--sf2` or
`$TABULA_SF2`; FluidR3_GM is the fallback. The temporary MIDI uses GM program
24 (Acoustic Guitar (nylon) / Live HQ "Concert Nylon Guitar"), not program 25
steel. The wrapper writes a temporary MIDI, renders with
`fluidsynth -ni -g 1.0 -R 0 -C 0`, trims to MIDI length + 0.5 s release tail,
normalises peak to 0.89, and adds a 5 ms fade in/out. `tests/test_synth.py`
checks tuning, level and determinism.

**Stereo downmix:** FluidSynth writes stereo WAVs. `synthesize()` reads
`getnchannels()`/`getsampwidth()` and downmixes to mono before trimming so
interleaved L/R frames are not read as twice as many samples (previously this
made audio play 2× slow and half-length).

**Tempo map preservation:** `synthesize()` accepts `tempo_changes` and
`source_midi_path`; `_write_temp_midi` loads the source MIDI, drops its
instruments, appends six string tracks and writes, keeping every original
tempo/key/time change. Note times are converted from ticks to seconds via
`ticks_to_seconds(tick, tempo_changes, ppq)`.

The old Karplus-Strong implementation was removed entirely.

## Not proved

* **Model quality.** A real GuitarSet training run is still required.
* **GAPS ingestion.** GAPS is **out of scope** — no converter will be added.
* **Techniques.** Hammer-ons, slides, bends and vibrato are not representable
  in the current record or vocabulary.

## Known failure mode

The playability pass still drops physically impossible notes (two notes on one
string at the same tick). A model that has learned nothing can emit almost
every note at one tick, so the pass may keep only a handful. It also drops at
most one inner tone from an otherwise unplayable chord (full voicing at fret
10 or more); that is reported in `dropped_chord_tones`. Both are intentional;
the alternative is to emit an unplayable tab.