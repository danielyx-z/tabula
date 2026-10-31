# DadaGP Preprocessing

The `vladsavelyev/guitar_tab` dataset on Hugging Face (~75 MB `data.zip`) contains
~47,000 GuitarPro songs pre-tokenised in **alphaTex**. This document describes
how `scripts/preprocess_dadagp.py` turns them into Tabula's canonical shards.

The converter does **not** implement its own parser. It reuses
`src/corpus/parse/alphatex.py`, the same parser used for the original GuitarPro
corpus, so chords, dotted notes, tuplets, harmonics and rests behave identically
to the canonical pipeline.

## Source Format (alphaTex)

```
\title "Honestly"
\tempo 92
.
\track
\instrument 30
\tuning E5 B4 G4 D4 A3 E3
\frets 24
:16 (0.1 7.2 7.3 5.5) 3.2.8{d} | r | :8 5.6 7.4 |
```

Beat body tokens:

- `:N` — set the current duration to an N-th note (`:1` … `:64`).
- `|` — bar line; bars are measured, not assumed to be 3 or 4 beats.
- `f.s` — note: **fret.string** (string 1 = high E in alphaTex).
- `(f.s f.s …)` — chord; the whole parenthesised group is one beat.
- `f.s.N` — explicit duration override after the note (`3.2.8` = fret 3,
  string 2, 8th note).
- `{d}` / `{dd}` — dotted / double dotted.
- `{tu 3}` / `{tu a b}` — triplet / explicit tuplet ratio.
- `r` — rest; a bar that contains only `r` is a whole-measure rest.
- `{nh}` / `{ah}` — natural / artificial harmonic. These sound at a pitch the
  fret does not imply, so they are dropped rather than mislabelled.
- `{x}` — dead note; dropped.
- `{b …}`, `{s}`, `{v}` and other effects are ignored.

Properties bind to the preceding beat and must be written **without a space**
(`3.2{d}`). Metadata lines start with `\`.

### Octave convention (easy to get wrong)

The corpus labels the high E string as `E5` even though it sounds at MIDI 64
(scientific `E4`). The mapping is `midi = 12 * octave + pitch_class`, not the
scientific `12 * (octave + 1) + pitch_class`. `core.fretboard.parse_tuning`
implements the corpus convention:

```python
parse_tuning("E5 B4 G4 D4 A3 E3")  # -> (40, 45, 50, 55, 59, 64) == STANDARD
```

Using the scientific formula shifts the whole corpus up an octave and makes
`pitch != STANDARD[s - 1] + f`; `scripts/verify.py` rejects that.

## Conversion Pipeline

`scripts/preprocess_dadagp.py` does:

1. **Parse** every `.tex` with `corpus.parse.alphatex.parse_file` (all `\track`
   blocks, not just the first).
2. **Filter** with `corpus.filter.evaluate`:
   - six strings,
   - **standard tuning only** (`STANDARD`),
   - GM guitar programs only (24…32), unless `--allow-non-guitar`,
   - `0 <= fret <= --max-fret` (24, the TAB vocabulary limit),
   - `>= --min-notes` notes and `>= --min-bars` bars,
   - no unparsed characters, no mid-track instrument changes,
   - harmonics below the 20% ratio.
3. **Snap** onsets and durations to the 20-tick grid (`quantise_tick`,
   `quantise_dur`). alphaTex timing is exact; measured on 1,000 files, 99.99%
   already landed on the grid and the rest are tuplet/128th values.
4. **Group** onsets within `--chord-window` (40 ticks) with `core.group.group_chords`.
   This also drops duplicate `(string, tick)` pairs.
5. **Deduplicate** byte-identical note arrays (same performance exported twice).
6. **Split** on the unique `artist/song` key (path relative to `--in-dir`),
   BLAKE2b 80/10/10. The same key always lands in one split.
7. **Write shards** with `ShardWriter`, identical layout to GuitarSet, ordered by
   `(t, s, pitch, f)` so chords list strings in ascending order.

Every dropped track is written to `rejects.jsonl` with a reason.

## Usage

```bash
# Download once (75 MB) and extract:
wget -O data/dadagp_raw/data.zip \
  "https://huggingface.co/datasets/vladsavelyev/guitar_tab/resolve/main/data.zip"
unzip -q data/dadagp_raw/data.zip -d data/dadagp_raw    # -> Artist/Song.tex

# Convert:
python scripts/preprocess_dadagp.py \
    --in-dir data/dadagp_raw \
    --out data/processed_dadagp

# Quick probe:
python scripts/preprocess_dadagp.py --in-dir data/dadagp_raw --limit 1000 \
    --out /tmp/probe_dadagp
```

`--in-dir` also accepts the `.zip` directly. Other flags: `--min-notes` (20),
`--min-bars` (4), `--chord-window` (40), `--max-fret` (24),
`--allow-non-guitar`, `--no-dedupe`, `--shard-notes` (1,000,000), `--overwrite`
(the script refuses to append to an existing shard set unless this is passed).

Output mirrors GuitarSet:

```
data/processed_dadagp/
  manifest.json
  rejects.jsonl
  shard-00000/  notes.npy  offsets.npy  tracks.jsonl
  shard-00001/  ...
```

The `.chunk_cache/` mentioned in older notes is created by the trainer
(`ChunkStore`), not by the preprocessing script.

## Manifest Fields

Per track (`tracks.jsonl`), matching the GuitarSet schema:

```json
{
  "id": 0,
  "source": "Zwan/Zwan_-_Honestly.tex",
  "song": "Zwan/Zwan_-_Honestly",
  "performance": "Zwan/Zwan_-_Honestly",
  "track": 0,
  "dataset": "dadagp",
  "tuning": [40, 45, 50, 55, 59, 64],
  "tuning_name": "E4 B3 G3 D3 A2 E2",
  "frets": 24,
  "instrument": 30,
  "tempo": 92,
  "n_notes": 1234,
  "n_bars": 48,
  "start": 0,
  "end": 123456,
  "split": "train",
  "bar_ticks": 3840,
  "bar_consistency": 0.98,
  "harmonic_notes": 0,
  "dotted_beats": 12,
  "tuplet_beats": 0
}
```

`manifest.json` records PPQ 960, the 20-tick grid, string order, filters,
`notes_on_grid` fractions and `rejects_by_reason`.

## Scale (full build)

Built on `fractal` in **629 s** (single core, 47,299 files):

| Metric | Value |
|--------|-------|
| Files | 47,299 |
| Tracks seen | 188,296 |
| Tracks kept | 63,013 |
| Notes | 60,082,609 |
| Duplicates dropped | 3,073 |
| Shards (1M notes) | 61 |
| Split | train 50,550 / valid 6,278 / test 6,185 |
| Output size | 676 MB |
| Songs (unique keys) | 31,316 |

That is ~960× GuitarSet's 62,476 notes and several hundred times its 529
training chunks. Rejects on the full run:

```
non_standard_tuning 45,323   not_guitar_program 38,799   not_six_string 35,937
same_string_same_tick_after_grouping 7,854   duplicate 3,073   too_few_notes 1,258
invalid_fret_or_string 702   no_tracks 173   harmonic_heavy 120   too_few_bars 55
fret_out_of_mvp_range 16
```

## Filtering Notes

- **All tracks** in a file are considered. DadaGP files have 2–8 tracks; taking
  only `\track` #1 discards most guitar material and sometimes keeps a bass,
  piano or synth track.
- **Standard tuning is mandatory.** The decoder emits `TAB<string,fret>` and
  inference maps pitches through `STANDARD`; a drop-D or Eb track teaches
  pitch→position mappings that are wrong at inference time. GuitarSet has the
  same rule (`corpus/filter.py`).
- **Guitar programs only** (`24…32`). Non-guitar tracks that happen to use six
  strings would otherwise be labelled as guitar.
- **Frets > 24 are rejected**, because the frozen vocabulary has
  `TAB<s,f>` only for `f <= 24`. Letting them through silently maps to `<unk>`.
- Harmonics are excluded because their sounding pitch is not
  `open_string + fret`.

## Vocabulary Compatibility

`vocab/guitarset.json` (248 tokens) was pruned to the pitches GuitarSet happens
to use: `NOTE_ON`/`NOTE_OFF` cover MIDI **40…81**. Standard-tuning DadaGP
reaches MIDI 88 (24th fret on the high E), so a GuitarSet-pruned vocabulary
produces `<unk>` encoder tokens. Measured: 66 UNK tokens in the first 300 probe
tracks.

Before pretraining on DadaGP, build one vocabulary for **both** stages:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_vocab.py \
    --data-dir data/processed_dadagp \
    --max-tracks 5000 \
    --out vocab/canonical.json
```

Measured: `--max-tracks 5000` sees every pitch in **40…88** and produces
**262 tokens** (3 specials + 98 NOTE + 11 TIME_SHIFT + 150 TAB), hash
`0811c6040f3af477`. All TAB and TIME_SHIFT tokens are kept unconditionally, so
the sample size only has to cover the pitch range. The unpruned base vocabulary
(286 tokens, pitches 36…96) is also valid. The same file must be used for
pre-training and GuitarSet fine-tuning, otherwise the token ids do not line up.

The decoder schema is unchanged: `TAB<s,f>` + `TIME_SHIFT` with `s ∈ 1…6`
(1 = low E), `f ∈ 0…24`. The encoder schema is `NOTE_ON`/`NOTE_OFF` +
`TIME_SHIFT` with pitches in the vocabulary.

## Verification

```bash
PYTHONPATH=src .venv/bin/python scripts/verify.py --dataset data/processed_dadagp
PYTHONPATH=src .venv/bin/python tests/test_dadagp.py
```

Full build:

```
shards      61
tracks      63,013
notes       60,082,609
songs       31,316
all invariants hold
notes_on_grid: {"20": 1.0, "60": 0.956, "120": 0.954, "240": 0.932, "480": 0.760}
```

100% of onsets are on the 20-tick grid, so every `TIME_SHIFT` delta is
representable exactly.

## Integration with Training

```bash
# Stage 1 — pre-train on DadaGP:
PYTHONPATH=src .venv/bin/python -m model.train \
  --data-dir data/processed_dadagp \
  --vocab-path vocab/canonical.json \
  --output-dir runs/pretrain_t5_dadagp \
  --batch-size 128 --grad-accum 1 --bf16 \
  --max-steps 30000 --eval-every 1000 \
  --metrics-every 5000 --save-every 2500 --postproc full

# Stage 2 — fine-tune on GuitarSet (same vocab file):
PYTHONPATH=src .venv/bin/python -m model.train \
  --data-dir data/processed_guitarset \
  --vocab-path vocab/canonical.json \
  --output-dir runs/finetune_t5_guitarset \
  --resume-from runs/pretrain_t5_dadagp/best \
  --lr 1e-4 --max-steps 5000 --patience 5
```

On WATcloud the same job script handles both stages:

```bash
DATA_DIR=data/processed_dadagp VOCAB_PATH=vocab/canonical.json \
OUTPUT_DIR=runs/pretrain_t5_dadagp BATCH_SIZE=128 GRAD_ACCUM=1 \
sbatch scripts/train_t5.sh
```

The first epoch pays the chunk-store build once (measured: ~28 min for 50,550
train tracks, ~2 GB of `int32` token arrays); later epochs read the cached
arrays. Budget the job wall time accordingly.

## Caveats

- Split identity is the relative path (`artist/song`). Two byte-identical
  exports are deduplicated; near-duplicates at different paths are not.
- Only the note timeline is used; tempo maps and time signatures are stored
  (`tempo`, `bar_ticks`) but the chunker assumes a steady clock.
- `min_bars = 4` and `min_notes = 20` are deliberately permissive because
  DadaGP is a web-scraped corpus; raise them if quality matters more than size.
