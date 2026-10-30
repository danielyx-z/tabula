# Audio → Tablature Pipeline

Full chain: **any audio file (or YouTube URL) → MIDI notes → guitar tablature**

```
audio file / YouTube URL
    │  yt-dlp (download audio)
    ▼
basic-pitch (Spotify, ONNX)  audio → symbolic MIDI notes
    │  NOTE_ON / NOTE_OFF / TIME_SHIFT
    ▼
fretwire T5 (2.81M params)  MIDI → tablature chunks (20 notes + context)
    │  overlap correction + ±5-note neighbor search
    ▼
ASCII tablature  (optionally MIDI + WAV output)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/audio2tab.py` | Full pipeline: YouTube/audio → tab |
| `scripts/infer_sample.py` | Inspect one inference chunk (tokens, answer, output) |
| `src/fretwire/cli.py` | `midi2tab` CLI: parse MIDI → run T5 → render tab |
| `src/fretwire/parse/midi.py` | MIDI parser: pretty_midi → canonical `Note` events |
| `src/fretwire/render.py` | ASCII tab visualizer |
| `preview.py` | Audio synthesis + WAV/MIDI writers |

## Usage

```bash
# YouTube URL → tablature (full pipeline)
PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py \
    "https://www.youtube.com/watch?v=tyKu0uZS86Q" \
    --checkpoint runs/pretrain_t5_v5/best \
    --beam 1 --width 100 --no-play

# Local audio file → tablature
PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py song.wav \
    --checkpoint runs/pretrain_t5_v5/best --beam 1 --width 100 --no-play

# Also write MIDI + WAV of the predicted tab
PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py song.wav \
    --checkpoint runs/pretrain_t5_v5/best --mid out.mid --wav out.wav --no-play

# MIDI file → tablature (skip audio→MIDI step)
PYTHONPATH=src .venv/bin/python3 -m fretwire.cli song.mid \
    --checkpoint runs/pretrain_t5_v5/best --beam 1 --width 100 --no-play
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `runs/pretrain_t5_v5/best` | Trained T5 checkpoint directory |
| `--vocab` | `vocab_v1_full.json` | Frozen 423-token vocabulary |
| `--beam` | `1` | Beam size (1 = greedy decoding) |
| `--max-len` | `512` | Max decode length (tokens) |
| `--width` | `80` | Terminal width for ASCII tab |
| `--mid` | — | Write predicted tab as MIDI file |
| `--wav` | — | Write predicted tab as WAV audio |
| `--no-play` | — | Don't play audio back |
| `--postproc` | `full` | `none` / `overlap` / `full` post-processing |

## Installation (local machine, RTX 3070)

```bash
.venv/bin/pip install basic-pitch librosa mir_eval resampy onnxruntime pretty_midi yt-dlp
sudo apt-get install -y ffmpeg   # for yt-dlp audio extraction
```

## Example: Laufey — "Valentine" (YouTube URL)

**Input:** `https://www.youtube.com/watch?v=tyKu0uZS86Q` — slow piano ballad, easy for guitar.

basic-pitch detected **1,208 notes** across the full song. The T5 model transcribed them in **61 chunks** of 20 notes each. Post-processing enforced **100% pitch validity**.

Below is a clean 15-second excerpt showing input → output side by side.

### Step 1 — basic-pitch input notes (first 12 of 79, 15-second excerpt)

| start(s) | end(s) | dur(s) | pitch | vel |
|----------|--------|--------|-------|-----|
| 0.209 | 0.650 | 0.441 | 67 | 73 |
| 0.209 | 1.741 | 1.533 | 51 | 81 |
| 0.221 | 0.801 | 0.580 | 58 | 74 |
| 0.221 | 0.882 | 0.662 | 62 | 47 |
| 0.290 | 0.708 | 0.418 | 39 | 58 |
| 0.801 | 1.080 | 0.279 | 68 | 87 |
| 0.801 | 1.776 | 0.975 | 58 | 53 |
| 1.126 | 1.602 | 0.476 | 67 | 88 |
| 1.161 | 1.614 | 0.453 | 39 | 54 |
| 1.695 | 1.869 | 0.174 | 68 | 80 |
| 2.045 | 2.718 | 0.673 | 61 | 72 |
| 2.045 | 2.730 | 0.685 | 67 | 76 |
| 2.056 | 2.207 | 0.151 | 52 | 48 |
| 2.056 | 2.788 | 0.731 | 58 | 84 |
| 3.020 | 3.542 | 0.522 | 55 | 52 |

### Step 2 — Predicted tablature (ASCII output)

```
 E5 |-----------------------------3----------------------------------|---------------
 B4 |-8---------------9---------------------------9------------------|---------------
 G4 |-7------------------------------------------------------12------|---------------
 D4 |-8---------------8--------------------------------------11------|---------------
 A3 |-6-------------------------------------------------------7------|---------------
 E3 |-----0-----------------------0--------------------------18------|---------------

 E5 |-------------------------------------------------|------------------------------
 B4 |------2---------------------------1--------------|11-------9--------------------
 G4 |------0--------------------------13--------------|-5----------------------------
 D4 |-----20---------------------------3--------------|--------13--------------------
 A3 |---------------------------------18--------------|18-------8--------------------
 E3 |-----24---------------------------1--------------|------------------------------

 E5 |----------------------------------|---------------------------------------------
 B4 |----------------------13----------|---------------------------------------------
 G4 |------------------12--------------|---------------------------------------------
 D4 |---------------6-------6----------|---------------------------------------------
 A3 |-----------1----------------------|---------------------------------------------
 E3 |----------22------22--------------|---------------------------------0-----------

 E5 |-------------------|-----------------------------------------3------------------
 B4 |-------------------|---------------------------------1--------------------------
 G4 |-------------------|---------------------------------3-------7-------3----------
 D4 |-------------------|---------6-----------8-------1------------------------------
 A3 |------------8---9--|10---------------6------------------14------17----------10--
 E3 |0------------------|-----------------------------------------0----------18------

 E5 |----|-----------------2-------------------------------------------3--|----------
 B4 |----|---------------------------------1---------------------------3--|-----2----
 G4 |----|-----------------7-------------------------------7--------------|----------
 D4 |----|------------10---------------------------1-----------1----------|----------
 A3 |----|--------11------------------------------------------13----------|----------
 E3 |-6--|-6-------------------0------------------------------------------|----------

 E5 |-----------------------------------2------------------|
 B4 |---1---2-------2-----------3--------------------------|
 G4 |-------------------1----------------------------------|
 D4 |------------------------------------------------------|
 A3 |-----------1------------------------------------------|
 E3 |---------------------------0--------------------------|
```

### Outputs written

| File | Description |
|------|-------------|
| `/tmp/laufey_valentine.mid` | Predicted tablature as Standard MIDI File |
| `/tmp/laufey_valentine.wav` | Synthesized audio of the predicted tab |

## Model details

- **Architecture**: Reduced T5 encoder-decoder (`d_model=128`, `d_ff=1024`, 3 enc/3 dec layers, 4 heads)
- **Parameters**: ~2.81M
- **Vocabulary**: 423 tokens (NOTE_ON/OFF for pitches 36–96, TIME_SHIFT steps, TAB<s,f> for 6 strings × 25 frets)
- **Training chunks**: 512 encoder tokens
- **Inference chunks**: 20 notes + 1 note context carry-over
- **Decoding**: Greedy (beam=1) by default; optional pitch-valid logits mask available but disabled (paper-faithful)
- **Post-processing**: Overlap correction → neighbor search (±5 notes) → pitch correction to 100%
