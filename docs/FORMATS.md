# Input / output formats

This page is the reference for every representation that crosses a stage
boundary. All timing is integer ticks at **PPQ 960** (960 ticks = one quarter
note) on a **20-tick grid** (1/48 note).

```
audio file / YouTube URL
        │  basic-pitch
        ▼
MIDI note events  ──────────────►  canonical Note record
        │                              │
        │                              ▼
        │                        shard dataset (training)
        │                              │
        │                              ▼
        │                        encoder tokens ──► T5 ──► decoder tokens
        │                                                    │
        ▼                                                    ▼
   input.mid / input.wav                            raw.mid / raw.wav
                                                            │
                                                            ▼
                                                    final.mid / final.wav
                                                    final_tab.txt / report.json
```

## 1. Audio input

Accepted by `scripts/audio2tab.py`: a local audio file (`.wav`, `.mp3`, …) or a
YouTube URL. Audio is transcribed by basic-pitch into note events; nothing
about the audio itself is stored.

## 2. MIDI input

Accepted by `scripts/midi2tab.py` and `src/cli.py`: `.mid` / `.midi`.

`corpus.parse.midi.parse_midi` reads the file with `pretty_midi`, merges
non-drum tracks by default, filters pitches to 36–96, and rescales the file's
own resolution to **PPQ 960** using its tempo map. String and fret are not
present in MIDI, so they are placeholders (`s=0`, `f=0`).

## 3. Canonical note record

Defined in `src/core/schema.py` and `src/corpus/dataset.py`.

| Field | Type | Meaning |
| --- | --- | --- |
| `t` | int32 | onset in ticks (PPQ 960) |
| `dur` | int32 | duration in ticks, `>= 1` |
| `pitch` | int8 | MIDI pitch, 36–96 |
| `s` | int8 | string, **1 = thickest (low E), 6 = thinnest (high E)** |
| `f` | int8 | fret, 0–24 |

Invariants:

* `pitch == STANDARD[s - 1] + f`
* `STANDARD = (40, 45, 50, 55, 59, 64)` (E2 A2 D3 G3 B3 E4)
* one note per string per onset
* notes sorted by `(t, pitch)`

Numpy representation: 11 bytes per note.

```python
dtype = [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"),
         ("s", "<i1"), ("f", "<i1")]
```

## 4. Shard dataset (training input)

```
data/processed_guitarset/
  manifest.json          contract, counts, reject histogram
  rejects.jsonl          one reason per dropped track
  shard-00000/
    notes.npy            structured array, all tracks concatenated
    offsets.npy          int64[n_tracks + 1] -> slice for track i
    tracks.jsonl         one metadata object per track, same order
```

`tracks.jsonl` fields: `id`, `source`, `song`, `track`, `tuning`,
`tuning_name`, `frets`, `instrument`, `tempo`, `n_notes`, `start`, `end`,
`split`, `bar_ticks`, plus dataset-specific fields.

Splits are assigned per song by a stable hash: 80 % train, 10 % valid,
10 % test. A song never appears in two splits.

Load with:

```python
from corpus.dataset import iter_tracks

for meta, notes in iter_tracks("data/processed_guitarset", split="train"):
    ...  # notes is an 11-byte-record numpy view
```

## 5. Encoder tokens (model input)

Built by `notes_to_events()` in `src/tablature/tokenizer.py`.

```
NOTE_ON<55>  TIME_SHIFT<120>  NOTE_OFF<55>  TIME_SHIFT<60>  NOTE_ON<57> ...
```

* `TIME_SHIFT<n>` advances the clock by `n` ticks. There are only **11**
  possible values: `20, 40, 60, 120, 180, 240, 480, 720, 960, 1920, 3840`.
  Larger shifts are written as several tokens; every multiple of the 20-tick
  grid is representable exactly.
* A shift of 960 costs one token; an odd value such as 340 costs three
  (`240 + 60 + 40`). A 640-tick triplet quarter is `480 + 120 + 40`.
* Note-offs carry their own `TIME_SHIFT` when the clock advances.
* At inference, a 20-note chunk is prefixed with the previous chunk's last
  note as context. `base_t` is the context onset, so events are relative.

## 6. Decoder tokens (model output)

Built by `notes_to_target()` and produced by the model:

```
</s>  TAB<5,0>  TIME_SHIFT<120>  TAB<4,2>  TIME_SHIFT<60> ...
```

* `</s>` is the decoder start token.
* One context `TAB<s,f>` may follow it during inference.
* `TAB<s,f>` only at note onsets. **There are no note-offs.**
* `TIME_SHIFT<n>` is emitted whenever the clock advances.

Because there are no note-offs, `src/tablature/stream.py` infers a note's
duration from the next onset on the same string, capped at 960 ticks.

## 7. Vocabulary

`vocab/guitarset.json` has **246 tokens**:

| Class | Count | Notes |
| --- | --- | --- |
| `<pad>`, `</s>`, `<unk>` | 3 | T5 specials |
| `NOTE_ON<p>` | 41 | pitches seen in GuitarSet |
| `NOTE_OFF<p>` | 41 | same pitches |
| `TIME_SHIFT<n>` | 11 | compositional; see above |
| `TAB<s,f>` | 150 | 6 strings × 25 frets |

Only `NOTE_ON` / `NOTE_OFF` are pruned to the corpus. Every `TAB` and
`TIME_SHIFT` token is always kept, because an unseen duration can be composed
and an unseen pitch cannot be recovered.

Rebuild the vocabulary when a new corpus uses pitches outside the current set:

```bash
python scripts/build_vocab.py --data-dir data/processed_guitarset \
    --out vocab/guitarset.json
```

The checkpoint's `config.vocab_size` must equal the vocabulary size.

## 8. Exported artefacts

`scripts/midi2tab.py` writes everything below into `--out-dir`.

| File | Format | Source |
| --- | --- | --- |
| `input.mid` | MIDI, type 1 | parsed input notes |
| `raw.mid` | MIDI, type 1 | model tokens only |
| `final.mid` | MIDI, type 1 | raw + playability correction |
| `input.wav` | 44.1 kHz mono | synthesis of `input.mid` |
| `raw.wav` | 44.1 kHz mono | synthesis of `raw.mid` |
| `final.wav` | 44.1 kHz mono | synthesis of `final.mid` |
| `final_tab.txt` | ASCII tab | corrected fingering |
| `report.json` | JSON | counts, alignment metrics, post-processing stats |

### WAV synthesis

`src/audio/synth.py` renders each note with Karplus-Strong: a filtered noise
burst and plucked harmonic shape circulate through a delay line, with
string-dependent decay and brightness, a pluck-position comb and a fixed
acoustic-body EQ. It is deterministic for a given note list, so each WAV is
derived only from its own MIDI. It is a synthesis approximation, not a sample
library: it does not reproduce a specific guitar or microphone.

Each MIDI file uses **one track per string** (`string 1` … `string 6`,
program 25 acoustic guitar). A plain MIDI file stores pitch and time only, so
two different fingerings of the same notes would otherwise be byte-identical.
Input MIDI without string information goes on a `pitch only` track.

`report.json`:

```json
{
  "comparison": {
    "input_notes": 133,
    "raw_notes": 130,
    "aligned": 114,
    "pitch_correct": 57,
    "pitch_accuracy": 0.4286,
    "tab_exact": 0,
    "tab_accuracy": 0.0
  },
  "postprocessing": {
    "dropped_impossible": 7,
    "refingered_chord_span": 10,
    "refingered_jump": 31
  },
  "final_notes": 123
}
```

`tab_accuracy` is exact `(string, fret)` agreement with the input's fingering.
For a plain MIDI input there is no ground-truth fingering, so it is only
meaningful when the input came from an annotated dataset.

## 9. Techniques

The canonical record has no articulation field and the vocabulary has no
technique tokens. Hammer-ons, pull-offs, slides, bends, vibrato and palm mute
are **not represented**. GuitarSet does not annotate them either.
