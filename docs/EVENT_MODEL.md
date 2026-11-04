# Event model

The event model is an optional, tiny XGBoost layer for:

- dead / muted string notes,
- bass-string flicks,
- body percussion.

It does not replace Tabula and it does not change `final.json`.  It can add
dead-note records and can mark a matched note for removal when the model is
confident that the note is a dead-string hit.  Flicks and body percussion are
additive: the melody note stays and the `x` goes on the lowest free string.

## Why XGBoost

XGBoost is a classifier over tabular features.  It does not process the raw
waveform and does not replace the low-pass/HPSS/CQT feature pass.  The pipeline
is:

```text
audio -> onset candidates -> feature extraction -> XGBoost heads
```

The model is deliberately small: depth 3, a few hundred boosting rounds,
`binary:logistic` per head, and class weighting.  It is suitable for a few
hundred to a few thousand labelled onsets.

## Labels and heads

```text
pitched
dead_string
bass_flick
body_percussion
```

The heads are independent because a pitched note and a percussive hit may
coexist.  See [LABELING.md](LABELING.md) for the browser workflow.

## Features

`src/tabfeatures/events/features.py` extracts:

- low-band, mid-band, and high-band energy ratios,
- HPSS harmonic / percussive ratios, including a low-passed HPSS view,
- spectral flatness, centroid, bandwidth, rolloff, ZCR,
- f0 median, voiced probability, f0 spread,
- expected-pitch harmonic support when a MIDI pitch is available,
- attack/decay and spectral flux,
- optional pitch and string context.

Missing pitch/string values are `NaN`; XGBoost handles them natively.

## Training

Label sessions first, then:

```bash
.venv/bin/python scripts/train_event_model.py
```

The training defaults (thresholds, review margin, and feature window) come from
`configs/events.yaml` and are copied into the trained model metadata.  Change
the output directory with `--out-dir`; change the config with `--config`.
The output is written to `data/models/events/`:

```text
metadata.json
pitched.json
dead_string.json
bass_flick.json
body_percussion.json
```

Heads with fewer than one positive or one negative example are skipped and
reported in `metadata.json`.  A holdout session can be reserved:

```bash
.venv/bin/python scripts/train_event_model.py \
  --holdout-session <session_id>
```

Evaluate explicitly:

```bash
.venv/bin/python scripts/evaluate_event_model.py \
  --model data/models/events \
  --session <session_id>
```

Always split by recording/session.  Do not put windows from the same take in
both train and test.

## Pipeline inference

The model is opt-in:

```bash
.venv/bin/python scripts/stitch_tab.py \
  --audio take.wav \
  --midi take.mid \
  --out-dir out/take \
  --event-model data/models/events
```

If `--event-model` is omitted, the existing non-note heuristic is unchanged.
The web app automatically adds `--event-model` when
`data/models/events/metadata.json` exists; set `TABULA_EVENT_MODEL` to another
directory or an empty path to change that. When enabled, the pipeline writes:

```text
event_model_predictions.json
event_model_review.json
```

and reports `event_model` statistics in `pipeline_report.json`.

## Safety behavior

- `dead_string`: if a matched MIDI note has weak `pitched` confidence, the note
  is dropped and replaced with `x` on its original/nearby string.
- `bass_flick`: never drops the melody note; adds `x` on the lowest free string.
- `body_percussion`: adds `x` on the lowest free string with a staccato dot;
  never drops the melody note.
- If no string is free, the event is skipped and counted.
- Thresholds default to `0.60` for dead/flick/body and `0.50` for pitched.
  Values between `threshold - 0.15` and the threshold are written to review.

The model should be treated as high-precision assistance.  Manual overlays
remain the final override.
