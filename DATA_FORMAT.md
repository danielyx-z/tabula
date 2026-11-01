# Dataset format

The training pipeline consumes **sharded canonical note events**. This is the
contract that a new data source must satisfy.

## Record

```python
Note(t: int, dur: int, pitch: int, s: int, f: int)
```

| Field | Range | Meaning |
| --- | --- | --- |
| `t` | `>= 0` | onset in ticks, PPQ 960 |
| `dur` | `>= 1` | duration in ticks |
| `pitch` | 36–96 | MIDI pitch |
| `s` | 1–6 | **1 = thickest (low E), 6 = thinnest (high E)** |
| `f` | 0–24 | fret |

Required invariants:

1. `pitch == (40, 45, 50, 55, 59, 64)[s - 1] + f`
2. no two notes share `(s, t)`
3. `t` is non-decreasing in the stored array
4. standard tuning only

`scripts/verify.py` checks all of these plus shard bookkeeping.

## Time grid

Everything is quantised to a **20-tick grid** (1/48 note). That represents
triplets (320), dotted values (720) and 32nd notes (120) exactly, unlike a
60-tick grid, which silently discards sub-30-tick gaps.

## Chord grouping

Performed sources stagger a strum across 20–40 ticks. `core.group.group_chords`
collapses onsets inside a window onto the first attack and extends durations so
each note keeps its release tick. The GuitarSet converter enables this by
default with a 40-tick window.

Effect on GuitarSet:

| | before | after |
| --- | --- | --- |
| distinct onsets | 51,666 | 38,052 |
| chord onsets | 8,867 (17.2 %) | 13,520 (35.5 %) |
| mean notes per onset | 1.21 | 1.64 |
| total notes | 62,476 | 62,476 |

**Chord onset ordering:** within a shared onset, notes are sorted strictly by
string ascending: `s=1` (thickest / low E) → `s=6` (thinnest / high E). This
matches how tab is conventionally written and gives a deterministic target
token order for the decoder.

## Shards on disk

```
data/processed_guitarset/
  manifest.json
  rejects.jsonl
  shard-00000/
    notes.npy
    offsets.npy
    tracks.jsonl
```

`notes.npy` dtype:

```python
[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]
```

`offsets.npy` is `int64[n_tracks + 1]`: track `i` is
`notes[offsets[i]:offsets[i + 1]]`.

`tracks.jsonl` line for a GuitarSet track:

```json
{
  "id": 0,
  "source": "00_BN1-129-Eb_comp.jams",
  "song": "00_BN1-129-Eb_comp",
  "performance": "00_BN1-129-Eb",
  "track": 0,
  "dataset": "guitarset",
  "tuning": [40, 45, 50, 55, 59, 64],
  "tuning_name": "E4 B3 G3 D3 A2 E2",
  "frets": 24,
  "instrument": 25,
  "tempo": 129.0,
  "n_notes": 133,
  "start": 0,
  "end": 12345,
  "split": "train",
  "bar_ticks": 3840
}
```

New field: `"performance"` — the underlying performance key (filename with
`_comp` / `_solo` suffix stripped). Splits are assigned per performance.

## Splits

Assigned per **performance**, never per track, by a stable BLAKE2b hash of the
performance key (song name with `_comp` / `_solo` suffix removed):

* `< 80` → `train`
* `80–89` → `valid`
* `90–99` → `test`

Because the key is the performance, `_comp` and `_solo` versions of the same
performance always land in the same split — no train/test leakage.

Produced from the full 360-file GuitarSet annotation set:

```
files: 360, tracks: 360, notes: 62,476
splits: train 272, valid 50, test 38
```

(180 underlying performances; each has a comp and solo take.)

## GuitarSet converter

```bash
python scripts/preprocess_guitarset.py \
    --jams-dir /path/to/guitarSet_annotation \
    --out data/processed_guitarset \
    --chord-window 40
```

Covers the six per-string `note_midi` annotations, maps seconds to beats with
the `beat_position` annotation, snaps to the 20-tick grid, groups strummed
chords, drops same-string same-tick duplicates, and writes the same shard
layout as the GuitarPro converter.

## Adding another source

A converter only has to emit `(t, dur, pitch, s, f)` records and call
`ShardWriter.add(meta, notes_to_array(notes))`. Checklist:

1. Map the source's strings to `s=1` (low E) … `s=6` (high E).
2. Convert time to ticks at PPQ 960. If the source is in seconds, use its beat
   or tempo annotation rather than assuming 120 BPM.
3. Snap to the 20-tick grid.
4. Group strummed onsets with `core.group.group_chords`.
5. Drop or merge same-string same-tick duplicates.
6. Assign `split` per **performance** (not per file).
7. Run `scripts/verify.py --dataset <out>`; it must print
   `all invariants hold`.
8. If the new corpus uses pitches outside the current vocabulary, run
   `scripts/build_vocab.py` and pass the new vocab to training.

## What GuitarSet gives us that GuitarPro did not

GuitarSet is real recorded guitar with per-string MIDI annotations, so every
note already has a true `(string, fret)`. There is no need to infer a fingering
from pitch, which is what made the earlier synthetic corpus musically useless.

## Known gaps

* GAPS ships MusicXML, not JAMS. It is **out of scope** — no converter will be
  added.
* GuitarSet timing is performed, not notated. The beat map keeps it musically
  proportional, but it is still quantised.
* GuitarSet has six-string acoustic guitar only. Extended-range instruments
  would need a different tuning contract.
* No technique annotations (hammer-on, slide, bend, …) exist in GuitarSet or
  in the canonical record.