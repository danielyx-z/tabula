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
3. The server finds pitched note starts with basic-pitch, merges chord notes
   closer than 40 ms, and adds percussive onsets that do not overlap a detected
   note.  That gives one short window per attack: pitched notes stay aligned
   with the note, while dead strings, flicks, and body knocks still get
   candidates.  Playback uses a short lead-in, starts after the previous
   attack, and stops 30 ms before the next attack.  The guard is wider than one
   detection frame because a detected onset can lag the audible attack by
   ~10–20 ms; without it the next attack's transient leaks into the clip.
4. Listen to the target attack and press:

```text
1 = pitched
2 = dead_string
3 = bass_flick
4 = body_percussion
```

`space` replays, `s` skips, and `u` undoes the last label or pitched run.  The
browser saves after every keypress.

Label **one attack**, not a phrase.  If several strings are struck together,
that is still one event.  A second attack later in the clip is a separate
sample.  Do not merge a rapid run into one multi-note label: the model is an
onset classifier, and a multi-note label would be ambiguous.

`Shift+1` labels the current onset and the following contiguous attacks as
`pitched` until a gap over 250 ms, an already-labelled onset, or eight
candidates.  Use it only when the whole short run is clearly normal pitched
notes.  Undo removes the entire run.

A bass flick plus melody is labelled `3`; the melody note is kept by the note
validator, not replaced by this label.  Body percussion plus melody is `4`.

You do not need to label every normal note in a long rapid passage.  Label all
suspected dead/flick/body events and a sample of normal attacks across each
recording.  More recordings are more useful than an exact four-way class
split.

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

`labels.jsonl` contains one record per labelled onset:

```json
{"session_id":"abc123","index":12,"onset_s":3.42,"label":"bass_flick","group_id":"d4c0...","updated_at":"2026-11-04T04:10:34+00:00"}
```

A pitched run writes one record per onset with the same `group_id`; Undo removes
the whole group.  Only the onset is stored.  The playback and feature windows
are derived from the candidate record, so `ending_s` is not a human label.

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
