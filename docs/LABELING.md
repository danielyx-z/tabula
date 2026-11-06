# Dead-note labelling

The labeller creates training data for the per-note dead-note classifier.  It
is MIDI-anchored: audio only describes the attack, it never decides which
notes exist.

Open `http://127.0.0.1:8000/label`.  Recent sessions are listed on the same
page; click one to resume, or use **New** to start another.  **Delete** removes
that session's audio, MIDI and labels from this machine.

## Session

Upload a recording.  Upload its MIDI too if you have one; otherwise the local
MuScriptor small model transcribes it.  MuScriptor quantization is off by
default so trailing/percussive detail survives; `TABULA_MUSCRIPTOR_QUANTIZE=1`
and `TABULA_MUSCRIPTOR_GRID=8` (1/32 notes) opt back in.  Sessions live under:

```text
data/labels_dead/<session_id>/
  audio.wav
  transcribed.mid
  candidates.json
  labels.jsonl
  meta.json
```

Candidates are note groups from the MIDI file.  Notes with the same onset (or
within a small chord window) share a group and one playback window.  Every
note is still an independent candidate, so a dead note inside a chord is a
separate label.

MuScriptor writes notes shifted forward so bar 1 lands on tick 0, and records
that shift in a `muscriptor:bar_offset=` marker.  The labeller and classifier
subtract that marker value from their audio times; they do not estimate any
additional offset.  MIDI ticks stay unchanged for the tab model.

## Piano roll

Pitched is the default.  You only mark the dead notes.

- Click a note to toggle dead.
- Drag a rectangle to mark every note intersecting it dead.
- Alt-drag a rectangle to clear everything inside back to pitched.
- Right-click a dead note to clear it.
- Space plays/pauses; U undoes; Escape cancels a drag.
- Mouse wheel zooms around the cursor; `+`/`−` and the slider zoom around the
  visible centre.
- Shift-wheel or the bottom scrollbar scrolls horizontally.  Drag the scrollbar
  thumb, or click its track to jump.
- Clicking an empty spot seeks the audio.

Playback uses a Web Audio `AudioBufferSourceNode` and `AudioContext.currentTime`,
the same clock strategy as MuScriptor's web player.  This avoids the small
media-element clock drift that made the playhead lag the audio.

The playhead and bar lines come from the MIDI tempo map.  No fixed BPM is
assumed.  MuScriptor's `muscriptor:bar_offset=` marker is subtracted from the
audio-facing times used by the roll, playback, classifier features and
inference; the MIDI tick stays unchanged for the tab model.

When you have checked a take, press the small **Mark reviewed** toggle.  Only
reviewed sessions contribute implicit "pitched" negatives to training.  Without that flag, only
explicit dead/keep records are used, so a partially labelled song cannot teach
the model that an unmarked dead note is pitched.

## Training

```bash
PYTHONPATH=src .venv/bin/python3 scripts/train_dead_classifier.py \
  --holdout-session <session_id>
```

The script prints train and holdout precision/recall/confusion at the selected
threshold.  The default threshold search maximizes dead recall subject to a
minimum dead precision (default 0.95) using leave-one-session-out predictions.

Output:

```text
data/models/dead_notes/
  metadata.json
  model.json
```

The web app enables the model automatically when `metadata.json` exists.  A
manual run:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/stitch_tab.py \
  --audio take.wav --midi take.mid --out-dir out/take
```

writes `dead_decisions.json`, `dead_notes.json`, and `song.gp5` with
`NoteType.dead` notes for the online renderer.

## Legacy data

The old `data/labels` sessions were generated from audio/Basic Pitch onsets and
cannot provide per-note chord labels.  They were deleted.  Start sessions fresh
under `data/labels_dead`; do not import old chord labels.
