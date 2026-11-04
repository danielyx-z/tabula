# Labeling guitar events

The browser labeller creates training data for the optional event model.  It
lives at `/label` in the local web app and stores sessions under:

```text
data/labels/<session_id>/
  audio.wav
  candidates.json
  labels.jsonl
  meta.json
```

`data/` is gitignored.  Nothing is written to `webapp/jobs`, because that
directory is cleaned automatically.

## Workflow

1. Open `http://127.0.0.1:8000/label`.
2. Choose an audio file.
3. The server detects onsets, merges detections closer than 60 ms, and creates
   one playback window per onset.
4. Listen to the window and press:

```text
1 = pitched
2 = dead_string
3 = bass_flick
4 = body_percussion
```

`space` replays, `s` skips, and `u` undoes the last label.  The browser saves
after every keypress.

A bass flick plus melody is labelled `3`; the melody note is kept by the note
validator, not replaced by this label.  Body percussion plus melody is `4`.

## Labels

- `pitched`: a real ringing note or chord.
- `dead_string`: a muted string attack without a sustained pitch.
- `bass_flick`: a percussive bass-string attack, with or without a simultaneous
  melody note.
- `body_percussion`: a non-string tap or knock.

The model expands these into binary heads:

```text
pitched          <- label 1
dead_string      <- labels 2 and 3
bass_flick       <- label 3
body_percussion  <- label 4
```

## Label format

`labels.jsonl` contains one record per keypress:

```json
{"session_id":"abc123","index":12,"onset_s":3.42,"label":"bass_flick","updated_at":"2026-11-04T04:10:34+00:00"}
```

Only the onset is stored.  The playback and feature window is derived from the
candidate record, so `ending_s` is not a human label.

## Export

`GET /api/label/sessions/<id>/labels.jsonl` downloads the session labels.  The
file can be used for training or copied elsewhere for backup.

## Body percussion in the tab

Body percussion becomes a dead note `x` on the lowest free string, with a
staccato dot:

```text
x.
```

The ASCII renderer shows `x.`; the GP5 writer sets `NoteEffect.staccato` on the
dead note.  The event model stores `kind = "body_percussion"` and
`staccato = true` so this is not inferred later.
