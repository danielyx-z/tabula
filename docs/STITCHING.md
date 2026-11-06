# Audio-to-tab stitching

This pipeline adds performance features that a MIDI-only tab cannot represent:

* dead notes / string mutes (`x`) detected from unmatched audio onsets,
* matched notes that are actually percussion or other non-note hits replaced
  with dead notes (`x`),
* slap-strum chords kept intact with one `x` placed on the lowest free string
  beneath the chord,
* inferred hammer-ons and pull-offs (`h` / `p`), gated by audio evidence,
* manually supplied slides, bends, vibrato, and H/P overrides.

It does **not** retrain Tabula and does **not** replace the existing
`scripts/midi2tab.py` fingering stage.  Stage 2 runs that script unchanged.

## Inputs and assumptions

* Guitar-only audio.
* A quantised MIDI transcription (for example from MuScriptor), with a tempo
  map that follows the performance.  The default grid is `1/16`.
* The MIDI is the source of truth for pitch and rhythm.  Audio is used for
  onsets absent from the MIDI, to validate matched notes against the expected
  pitch, and to gate H/P decisions.

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
| `non_note_review.json` | note decisions with presence, Basic Pitch, and event evidence |
| `basic_pitch_notes.json` | independent Basic Pitch note events used as a second opinion |
| `hp_review.json` | H/P candidates, accepted/rejected state, gaps, confidence |
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
4. Convert onsets to PPQ-960 ticks via the MIDI tempo map.  The pipeline's
   internal PPQ is 960; `final.mid` may keep another resolution (commonly 480).
   `tick_to_seconds` scales by `pm.resolution` and interpolates fractional
   MIDI ticks, so no song or file is assumed to be at 480 or 960 PPQ.
5. Match each onset to a MIDI note onset within half a grid cell (default).
   The pre-reconciliation audio class, confidence, and strength are preserved
   on the matched event (`audio_class`, `audio_confidence`,
   `audio_strength`).  A match does not prove the MIDI note is real: a body tap
   or muted knock can coincide with an invented pitch.  Unmatched non-pitched
   onsets remain dead-note candidates.

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
2. Validate matched notes against audio.  A high-confidence non-note on a
   single string is marked for removal and replaced by a dead note on its
   original string.
3. A high-confidence percussive event on a chord with real pitch support is a
   slap strum: the pitched chord stays and one dead note is placed on the
   lowest free string, so the `x` renders beneath the chord.
4. Insert dead notes from unmatched candidates.  Default string policy is
   `lowest`: try string 1 (low E), then the next free string.  If all strings
   are occupied in that grid slot, the candidate is skipped and reported.
5. Merge the dead-note sources, then shorten any sustained note on the same
   string at a dead-note tick, so the GP5 file never contains a tie and a dead
   note on the same string at once.
6. Infer H/P on adjacent same-string notes with conservative model and audio
   guards.
7. Apply `articulations.json` as the final override.
8. Render ASCII and export GP5.

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

## Non-note validation

Matched notes are not automatically trusted.  For each matched percussive
onset the validator measures:

* harmonicity and `pyin` voiced probability from the onset window,
* expected-pitch support: energy at the MIDI fundamental and its first
  harmonics compared with the surrounding analysis band,
* the pre-reconciliation audio class as a small prior.

The combined non-note score is deliberately not tied to a pitch, string, time,
or song.  A score at or above `--non-note-min-confidence` (default `0.60`)
converts the note to a dead note on its original string.

A percussive event on a chord is handled additively when at least one chord
tone has real pitch support (`--slap-min-pitch-support`, default `0.12`).  The
chord is kept and one `x` is placed on the lowest free string, which renders
beneath the chord.  This is the slap-strum case.

If a note has no matched onset, a nearby unmatched dead/percussive onset within
one grid cell can still validate it.  This catches dead notes where the audio
and MIDI clocks drift by a fraction of a grid cell.  The unmatched fallback
uses `--non-note-unmatched-min-confidence` (default `0.50`) and ignores
onsets above `--non-note-unmatched-max-harmonic-ratio` (default `0.70`) because
those are probably real notes that merely failed alignment.

Scores at or above `--non-note-review-confidence` (default `0.45`) but below
the conversion threshold are left as notes.  All candidates at or above the
review threshold are written to `non_note_review.json`, including the ones
that were converted, so every automatic decision can be inspected.  The
review entry's `action` is `replace`, `slap`, or `review`.

Ambiguous and medium-confidence events are never silently removed.  To keep a
note that the validator removed, add an overlay entry with `"keep": true` or
`"non_note": false` at that `(t, s)`.  Disable the validator with
`--no-non-note`.

A short repeated run on one string and pitch is handled as context.  If at
least four hits in the run are confident non-notes, the run spans at least two
quarter notes, and a majority of the run is non-note evidence, remaining
matched notes in that run are also converted.  This covers a percussion tap
that happens to excite a stable pitch on one hit.  Those entries use
`source: audio_non_note_run` and `reason: repeated_non_note_run` in
`dead_notes.json` / `non_note_review.json`.

Existing jobs created before raw audio classes were preserved are backfilled
from `audio_events.json` when Stage 3 is resumed.  Re-run `--stage audio`
first if that raw file is also missing.

## Basic Pitch single-note fallback

Basic Pitch is run once on the source audio.  Its note starts are matched
one-to-one against uploaded MIDI notes of the same pitch, allowing for the
quantisation/timing difference.

For a **single-note attack**:

- Basic Pitch has a matching note -> keep it pitched;
- Basic Pitch has no matching note -> replace it with `x` on its string;
- Basic Pitch was not run -> keep the event-heuristic decision.

Polyphonic chords keep the event/presence decision because Basic Pitch is less
reliable when several strings sound at once.  A fully pitched percussive chord
can still receive the slap-strum marker from the event heuristic.

Basic Pitch output is written to `basic_pitch_notes.json` for inspection.

## H/P guard rails

An inferred H/P requires all of the following:

* same string,
* adjacent notes in time with a positive gap,
* gap at most `--hp-max-gap-beats` quarter notes (default `0.5`),
* the first note still ringing (`gap <= first.dur + --ring-tolerance`),
* interval 1..`--hp-max-interval` semitones (default 7),
* both notes fretted (open strings skipped),
* no chord at either onset (unless `--allow-hp-in-chords`).

When audio evidence is available (the default), a candidate must also pass an
audio gate:

* the first note's expected pitch is still present just before the second
  onset,
* the second onset does not have a full separate pluck attack relative to the
  first note and nearby attacks,
* the second note has pitched support rather than being another non-note.

The gate is conservative: a strong second attack, a decayed first note, a weak
second pitch, or a low combined confidence rejects the candidate.  The default
threshold is `--hp-min-confidence 0.70`.  Use `--no-hp-audio` for the old
model-only rules.  Every candidate, accepted or rejected, is written to
`hp_review.json` with its metrics and rejection reason.  Disable inference
entirely with `--no-auto-hp`.

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
* `"keep": true` (or `"non_note": false`) keeps a note that the automatic
  non-note validator removed, and removes that automatic replacement.

## Tuning knobs

* `--grid 1/16` — grid used for snapping audio events (also `1/8`, `1/32`,
  triplet forms, or a tick count).
* `--min-confidence 0.6` — unmatched dead-note candidate threshold.  Lower it
  to review more candidates; the full list is in `percussion_events.json`.
* `--non-note-min-confidence 0.60` — matched non-note conversion threshold;
  candidates at `--non-note-review-confidence` and above are reported.
* `--non-note-unmatched-min-confidence 0.50` — conversion threshold for notes
  validated by a nearby unmatched percussive onset.
* `--non-note-unmatched-max-harmonic-ratio 0.70` — ignore unmatched onsets
  that look harmonic rather than percussive.
* `--slap-min-pitch-support 0.12` — chord pitch support needed to keep a
  percussive chord and add an `x` beneath it instead of replacing notes.
* `--basic-pitch` / `--no-basic-pitch` — enable or disable the independent
  Basic Pitch note-presence vote (default: on).
* `--basic-pitch-tolerance 0.12` — allowed Basic Pitch onset difference in
  seconds.
* `--no-non-note` — disable matched non-note conversion.
* `--min-match-rate 0.5` — safety guard against mismatched audio/MIDI.
* `--dead-strings lowest|nearest|all`.
* `--hp-max-gap-beats 0.5`, `--hp-max-interval 7`, `--ring-tolerance 60`.
* `--hp-min-confidence 0.70`, `--no-hp-audio` — tune or disable the audio gate.
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

A voice cannot sound two notes on one string at once.  If `final.json` contains
a sustained note and a later attack on the same string, the exporter ends the
earlier note at that attack (and drops exact same-tick duplicates) so the
written GP5 stays valid.  The count is reported as `gp5.overlap_trimmed`.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_tabfeatures.py
.venv/bin/python -m pytest -q
```
