# Audio-to-tab stitching

This pipeline adds performance features that a MIDI-only tab cannot represent:

* dead notes / string mutes (`x`) detected from the source audio,
* inferred hammer-ons and pull-offs (`h` / `p`),
* manually supplied slides, bends, vibrato, and H/P overrides.

It does **not** retrain Tabula and does **not** replace the existing
`scripts/midi2tab.py` fingering stage.  Stage 2 runs that script unchanged.

## Inputs and assumptions

* Guitar-only audio.
* A quantised MIDI transcription (for example from MuScriptor), with a tempo
  map that follows the performance.  The default grid is `1/16`.
* The MIDI is the source of truth for pitch and rhythm.  Audio is used only for
  onsets that are absent from the MIDI.

## Run it

From the repository root:

```bash
.venv/bin/python scripts/stitch_tab.py \
  --audio song.wav \
  --midi transcribed.mid \
  --out-dir out/my_song
```

The default checkpoint and vocab are:

```text
runs/finetune_t5_guitarset/best
vocab/canonical.json
```

Override them with `--checkpoint` / `--vocab` if needed.

### Resumable stages

```bash
# 1. audio events + MIDI reconciliation
... --stage audio

# 2. the existing Tabula MIDI -> tab model
... --stage tab

# 3. dead notes, H/P, overlay, ASCII + GP5
... --stage stitch
```

`--stage all` is the default.  Stage 2 reuses an existing
`<out-dir>/tabula/final.json` unless `--force` is passed.

After editing `articulations.json`, rerun only the last stage:

```bash
.venv/bin/python scripts/stitch_tab.py \
  --audio song.wav --midi transcribed.mid --out-dir out/my_song \
  --stage stitch
```

## Outputs

| Path | Contents |
| --- | --- |
| `audio_events.json` | HPSS onset candidates with timbre/f0 features |
| `alignment_report.json` | audio/MIDI match rate, offset, residuals, warnings |
| `percussion_events.json` | onsets after reconciliation (matched/unmatched, snapped ticks) |
| `tabula/` | the job's own `final.json`, `final.mid`, `report.json`, logs |
| `dead_notes.json` | inserted `x` notes, with ticks and confidence |
| `hp_review.json` | inferred H/P pairs, gaps, intervals, confidence |
| `articulations.json` | editable manual overlay (created empty) |
| `tab_with_features.txt` | ASCII tab with `x`, `h`, `p`, slide/bend/vibrato markers |
| `song.gp5` | real GP5 from `final.json` + the job's `final.mid` |
| `pipeline_report.json` | counts, guard warnings, output paths |

## How it works

### Stage 1: audio events

1. Load mono audio, run HPSS (`librosa.effects.hpss`).
2. Detect percussive onsets on the percussive stem.
3. Measure spectral flatness, centroid, bandwidth, ZCR, RMS, HPSS harmonic
   ratio, and `pyin` voiced probability around each onset.
4. Convert onsets to PPQ-960 ticks via the MIDI tempo map.
5. Match each onset to a MIDI note onset within half a grid cell (default).
   A matched onset is a normal pluck attack and is discarded.  Unmatched
   onsets are dead-note candidates.

A coarse first pass estimates a global audio/MIDI offset for files with
leading silence.  The correction is reported as `applied_offset_sec`.

If the strict match rate is below `--min-match-rate` (default `0.5`), Stage 3
refuses to insert dead notes.  This prevents a mismatched audio/MIDI pair from
turning every pluck into an `x`.  Use `--allow-low-match-rate` only if you have
checked the alignment report.

### Stage 2: fingering

Runs the existing `scripts/midi2tab.py` with `--no-audio`.  It writes the
job's own `final.json`, `final.mid`, `final_tab.txt`, and `report.json` into
`<out-dir>/tabula/`.

### Stage 3: stitching

1. Snap candidate ticks to the chosen grid.
2. Insert dead notes.  Default string policy is `lowest`: try string 1
   (low E), then the next free string.  If all strings are occupied in that
   grid slot, the candidate is skipped and reported.
3. Shorten any sustained note on the same string at the dead-note tick, so the
   GP5 file never contains a tie and a dead note on the same string at once.
4. Infer H/P on adjacent same-string notes with conservative guards.
5. Apply `articulations.json` as the final override.
6. Render ASCII and export GP5.

`final.json` on disk is never mutated; dead notes are kept separately and
merged at render/export time.

## Dead-note string policy

```bash
--dead-strings lowest   # default: low E, then next free string
--dead-strings nearest  # start at the nearest pitched note's string
--dead-strings all      # dead strum: every free string
```

Occupancy is checked on the quantised grid slot, not only on exact ticks, so a
note at tick 260 cannot collide with a dead note at tick 240 in the ASCII/GP5
rendering.

## H/P guard rails

An inferred H/P requires all of the following:

* same string,
* adjacent notes in time with a positive gap,
* gap at most `--hp-max-gap-beats` quarter notes (default `0.5`),
* the first note still ringing (`gap <= first.dur + --ring-tolerance`),
* interval 1..`--hp-max-interval` semitones (default 7),
* both notes fretted (open strings skipped),
* no chord at either onset (unless `--allow-hp-in-chords`).

Every inferred pair is written to `hp_review.json`.  Disable inference with
`--no-auto-hp`.

## Manual overlay

`articulations.json` is tick-based:

```json
{
  "version": 1,
  "notes": [
    {"t": 480,  "s": 3, "f": 7, "hammer": true},
    {"t": 720,  "s": 3, "f": 7, "hammer": false},
    {"t": 960,  "s": 3, "f": 9, "slide": "legatoSlideTo"},
    {"t": 1440, "s": 3, "f": 9, "vibrato": true},
    {"t": 1920, "s": 3, "f": 9, "bend": [[0, 0], [12, 2]]},
    {"t": 2400, "s": 2, "f": 5, "dead": true}
  ]
}
```

* `t` is PPQ-960 ticks; use `hp_review.json` / `dead_notes.json` for values.
* `s` is `1 = low E` through `6 = high E`.
* `f` is the capo-relative fret from `final.json`; if supplied it must match.
* `hammer: false` or `"hp": "none"` clears an inferred H/P.
* `slide` accepts `shiftSlideTo`, `legatoSlideTo`, `outDownwards`,
  `outUpwards`, `intoFromBelow`, `intoFromAbove` (plus short aliases).
* `bend` points are `[position, value]` with `position` 0..12 and `value` 0..12
  semitones; the optional object form also accepts a `type` such as `bend`,
  `bendRelease`, `dip`, or `prebend`.
* `"dead": true` converts the note at that `(t, s)` to an `x`; if no pitched
  note exists there, it adds a dead note.

## Tuning knobs

* `--grid 1/16` — grid used for snapping audio events (also `1/8`, `1/32`,
  triplet forms, or a tick count).
* `--min-confidence 0.6` — dead-note candidate threshold.  Lower it to review
  more candidates; the full list is in `percussion_events.json`.
* `--min-match-rate 0.5` — safety guard against mismatched audio/MIDI.
* `--dead-strings lowest|nearest|all`.
* `--hp-max-gap-beats 0.5`, `--hp-max-interval 7`, `--ring-tolerance 60`.
* `--no-auto-hp`, `--no-gp5`, `--keep-intermediate`.
* `--onset-delta`, `--onset-wait`, `--window-before`, `--window-after` for the
  onset detector.

## GP5 contract

The exporter is a feature-aware copy of the web app exporter.  It writes:

* the job's own `final.json` notes,
* timing from the job's own `final.mid` (falling back to the supplied MIDI
  only if `final.mid` is absent),
* the original tempo/time-signature map,
* capo and standard tuning,
* nylon guitar program 24,
* `NoteType.dead` for `x`,
* `NoteEffect.hammer`, `slides`, `vibrato`, and `bend`.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_tabfeatures.py
.venv/bin/python -m pytest -q
```
