# Dead-note labelling

The labeller creates training data for the per-note pitched/dead/noise
classifier.  It is MIDI-anchored: audio only describes the attack, it never
decides which notes exist.

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

Pitched is the default.  Choose **Dead**, **Noise** or **Ignore** mode, then
mark the notes.

- Click a note to toggle the selected class.  It does not start playback.
- Drag a rectangle to paint the selected class on every intersecting note.
- Alt-drag a rectangle to clear everything inside back to pitched.
- Right-click a dead, noise or ignore note to clear it.
- **Dead** becomes an `x` in the tab and GP5. **Noise** is a spurious note and is
  removed from the tab output. **Ignore** is for ambiguous notes: it stays
  pitched in the output, but is excluded from classifier training.
- `1` / `2` / `3` switch to Dead / Noise / Ignore.
- Space plays/pauses; Ctrl+Z undoes; Escape cancels a drag.
- Mouse wheel zooms around the cursor; `+`/`−` and the slider zoom around the
  visible centre.
- Shift-wheel or the bottom scrollbar scrolls horizontally.  Drag the scrollbar
  thumb, or click its track to jump.
- The **Mix** slider crossfades the original recording with a live note-level
  MIDI monitor.  Pitched notes use a cached FluidSynth sample; dead notes play
  a short muted-string thump; noise notes are silent.  Changes apply the next
  time you press Play, and to already-scheduled notes during playback.  Keep
  the original audio dominant for attack decisions; use more MIDI to hear the
  transcription.
- Clicking an empty spot seeks the audio.

Playback uses Web Audio `AudioBufferSourceNode` nodes and
`AudioContext.currentTime`, the same clock strategy as MuScriptor's web player.
The original recording and the note-level MIDI monitor share the same clock, so
the playhead does not lag the audio.  The monitor schedules a short lookahead;
toggling a note stops its currently sounding voice immediately and future notes
pick up the new label.

The playhead and bar lines come from the MIDI tempo map.  No fixed BPM is
assumed.  MuScriptor's `muscriptor:bar_offset=` marker is subtracted from the
audio-facing times used by the roll, playback, classifier features and
inference; the MIDI tick stays unchanged for the tab model.

When you have checked a take, press the small **Mark reviewed** toggle.  Only
reviewed sessions contribute implicit "pitched" negatives to training.  Without
that flag, only explicit dead/keep records are used, so a partially labelled
song cannot teach the model that an unmarked dead note is pitched.  Explicit
`ignore` notes are skipped entirely, so they neither train a class nor become
implicit pitched negatives.

## Training

```bash
PYTHONPATH=src .venv/bin/python3 scripts/train_dead_classifier.py \
  --holdout-session <session_id>
```

The script trains a three-class `pitched` / `dead` / `noise` model and prints
per-class precision, recall, F1 and the confusion matrix for the training and
holdout sessions.  Inference thresholds default to `0.80` for both dead and
noise and can be changed with `--dead-threshold` / `--noise-threshold`.

A practical starting set is roughly 200–300 dead positives, 50–100 noise
positives and 2,500+ reviewed notes across at least 8–10 songs, with 2–3 whole
songs held out.  More song diversity matters more than piling all positives
into one recording.

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
