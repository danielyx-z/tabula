# Tabula — working memory

## Project

Reimplementation of **Fretting-Transformer: Encoder-Decoder Model for MIDI to
Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

- Input: symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`).
- Output: ASCII tab events (`TAB` + `TIME_SHIFT`).
- Model: paper-config T5, 2.79M parameters (`d_model=128`, `d_ff=1024`,
  `d_kv=64`, 3 encoder/3 decoder layers, 4 heads). The project vocabulary
  is 248 tokens for GuitarSet.
- The paper does **not** define a training-time alignment loss, note-count head,
  or pitch-constrained training loss. Do not reintroduce them. Its alignment
  step is post-processing: overlap correction + ±5-note neighbor search.
- Training chunks: 512 encoder tokens. Inference chunks: 20 notes with the
  previous note as context in encoder and decoder.
- Time is quantised to a 20-tick grid (PPQ 960). `TIME_SHIFT` tokens are
  compositional (11 values: 20..3840). Onsets/durations are snapped to this grid.
- Rhythm metrics (onset hit rate, IOI/duration MAE) are computed on raw
  model output; post-processing assigns GT timing and is not rhythm-meaningful.

## Training is user-driven

Do **not** launch long training runs without asking. The local 8 GB GPU is
not used for long training.

OOM is not a design concern. The cluster job requests a GPU shard; use a batch
size that is efficient (defaults are batch 8 × grad-accum 8 = effective 64).

## WATcloud / Slurm

- SSH login: `ssh watcloud`
- Slurm docs: <https://cloud.watonomous.ca/docs/compute-cluster/slurm>
- The Slurm CLI is not on the default login `PATH`; use
  `export PATH=/opt/scripts/bin:$PATH` or `/opt/scripts/bin/sbatch`.
- Submit from the repository root: `mkdir -p logs && sbatch scripts/train_t5.sh`
- Job script: `scripts/train_t5.sh` (edit paths/resources if needed).

Resource choices in the script:

| Resource | Value | Why |
| --- | --- | --- |
| Partition | `compute_dense` | long jobs; 7-day max |
| Time | `2-00:00:00` | training + periodic evaluation |
| GPUs | `--gres=shard:rtx_3090:8192` | 8 GB RTX 3090 shard; plenty for batch 8 |
| CPUs | `8` | tokenisation / dataloading |
| RAM | `32G` | sharded dataset + Python |
| Temp disk | `tmpdisk:20480` | 20 GB `/tmp` |

`shard` is the recommended shared-GPU request. The explicit `rtx_3090` type
avoids the Pascal GTX 1080 node, which does not support bf16. For a dedicated
GPU, replace it with `--gres=gpu:rtx_3090:1`. Do not request `gpu:1` on the
`compute` partition unless the job can actually use most of the card. The
tested node was `thor-slurm1`: RTX 3090, driver 580.178.04, CUDA 13.0.

`--resume-from` restores weights, vocabulary, optimizer/scheduler state and
early-stopping counters, and continues from the saved epoch. A checkpoint
without `trainer_state.pt` falls back to weights/vocabulary only.

Monitor:

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.10T %.10M %.9l %R"
tail -f logs/<jobid>-tabula-t5.out
scancel <jobid>
```

WATcloud notifications (instead of Slurm mail):

```bash
./watcloud subscription <jobid> you@example.com
./watcloud subscription <jobid> --discord
```

## Training status

The old `runs/pretrain_t5_v5` checkpoint (kept only on the workstation) and
its metrics were trained on the discarded synthetic corpus. They are **not comparable** to GuitarSet and the
vocabulary has since changed (248 tokens), so the old checkpoint cannot be
loaded with the current data. Treat it as historical.

A real GuitarSet training run has not been completed yet. The active defaults
are `data/processed_guitarset`, `vocab/guitarset.json` and
`runs/pretrain_t5_guitarset`.

Evaluate a checkpoint without training:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
  --checkpoint runs/pretrain_t5_guitarset/best --tracks 10 --postproc full \
  --context-mode pred
```

## Cluster setup

The cluster needs its own environment and data:

```bash
ssh watcloud
cd ~/tabula_gs                    # new GuitarSet workspace
git clone https://github.com/danielyx-z/tabula.git .
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pretty_midi
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The tested cluster driver is 580.178.04 (CUDA 13.0); the cu121 wheel above
works. If `torch.cuda.is_available()` is false, reinstall the CUDA wheel
matching the compute-node driver (see
<https://pytorch.org/get-started/locally/>) before submitting. The processed
corpus must exist at `data/processed_guitarset` (train / valid / test shards).
Copy `data/processed_guitarset/` and `vocab/guitarset.json` from the fractal
workstation (2 MB total). Verify: vocab size 248, hash `6265d9bb91e68095`.
The GuitarSet shard is small (<2 MB); checkpoints are ~11 MB each.

## Reference training command

```bash
PYTHONPATH=src .venv/bin/python3 -m model.train \
  --data-dir data/processed_guitarset \
  --vocab-path vocab/guitarset.json \
  --output-dir runs/pretrain_t5_guitarset \
  --batch-size 8 \
  --grad-accum 8 \
  --bf16 \
  --max-steps 30000 \
  --eval-every 500 \
  --eval-loss-batches 0 \
  --eval-batches 10 \
  --metrics-every 2500 \
  --save-every 2500 \
  --postproc full
```

`max_steps` counts **optimizer steps** (micro-batches / grad_accum). With
batch 8 and grad-accum 8, 30,000 steps ≈ 240,000 sequences. The dataset
cycles infinitely; `eval_every`, `metrics_every`, and `save_every` count
optimizer steps. No partial grad-accum step occurs.

Evaluation is two-tier:

- every `eval_every`: fast teacher-forced validation loss on all validation
  chunks; used for early stopping and `best/` checkpoint;
- every `metrics_every`: full-track greedy decoding for raw and post-processed
  tab/pitch/rhythm metrics. This is expensive, so it is less frequent.

Use `--eval-beam 1` for the paper-style greedy default. `--grad-checkpointing`
is off by default; enable it only if memory becomes tight.

## Files that matter

- `src/model/model.py` — paper-config T5 factory.
- `src/tablature/tokenizer.py` — v3 event/target encoding.
- `src/model/chunk.py` — 512-token train windows / 20-note inference chunks.
- `src/tablature/postproc.py` — overlap correction + neighbor search.
- `src/tablature/metrics.py` — tab/pitch accuracy, difficulty, **rhythm**.
- `src/model/train.py` — training, validation loss, full metrics.
- `src/model/decode.py` — T5 generation helpers.
- `src/core/quantise.py` — 20-tick grid, shift decomposition, `quantise_notes`,
  **`ticks_to_seconds(tick, tempo_changes, ppq)`** converts tick → seconds
  following a tempo map.
- `src/audio/synth.py` — FluidSynth guitar synthesis. Writes a temporary MIDI
  then renders with `fluidsynth -ni -g 1.0 -R 0 -C 0`. **Critical:** FluidSynth
  writes **stereo** WAVs; `synthesize()` reads `nchan`/`sampwidth` and downmixes
  to mono before trimming, otherwise interleaved L/R frames are read as
  twice as many samples and audio plays 2× slow / half-length. Accepts
  `tempo_changes` and `source_midi_path` to preserve the input tempo map.
- `scripts/midi2tab.py` — `write_tab_midi(path, notes, bpm, ppq,
  source_midi_path, tempo_changes)`. If `source_midi_path` is given, loads that
  MIDI, sets `pm.instruments = []`, appends six string tracks and writes — this
  preserves the source's full tempo map, key signature and time signature.
  `ticks_to_seconds` converts note ticks to seconds using the tempo map.
- `vocab/canonical.json` — frozen 262-token vocab used by the DadaGP pretrain
  and the GuitarSet fine-tune (`runs/finetune_t5_guitarset/best`).
  `vocab/guitarset.json` (248) is the older GuitarSet-only vocab. Do not use
  the unpruned `DEFAULT_VOCAB` (286) for training.
- `src/transcribe/muscriptor.py` — local MuScriptor call. The web app uses the
  small model when the user does not upload MIDI.
- `src/tabfeatures/midi_groups.py` — MIDI-derived attack/chord groups used by
  both the labeller and the dead-note classifier. Candidates never come from
  audio onset detection.
- `src/tabfeatures/labeling/` — `data/labels_dead` sessions, per-note labels,
  reviewed flag, JSONL export.
- `src/tabfeatures/deadnotes/` — MIDI-anchored features and a three-class
  XGBoost model (`pitched` / `dead` / `noise`). It runs after `midi2tab`; dead
  notes go to `dead_notes.json` for `tabfeatures/gp5.py`, noise notes are
  dropped from the tab. `raw.mid` is untouched.
- `webapp/templates/label.html`, `webapp/static/label.js` — piano-roll
  labeller; pitched is default, switch between dead/noise/ignore, click/box-select
  marks the active class. `ignore` stays pitched in the output but is excluded
  from training. Includes session resume/delete, bottom scrollbar,
  zoom controls, and an original-audio/live MIDI-monitor mix. The monitor is
  note-level: dead notes thump, noise is silent, and labels apply live during
  playback. Clicking a label does not audition.
- `scripts/train_dead_classifier.py` — trains from reviewed sessions and prints
  holdout precision/recall/confusion.
- `webapp/templates/index.html`, `webapp/static/app.js`, `webapp/static/style.css`
  — workbench. The session rail is resizable: `#rail-resizer` is a grid
  column (`--rail-width`, 260–560 px, default 360) with pointer drag, keyboard
  arrows/Home/End, double-click reset and `localStorage` key
  `tabula:railWidth`; alphaTab re-renders 140 ms after the drag ends. It is
  hidden below the mobile breakpoint, where `.workspace` is `display: block`.
  `.app-shell` pins `grid-template-columns: minmax(0, 1fr)` so a wide alphaTab
  surface cannot stretch the page sideways. Take titles are editable inline
  (`#take-title`) and saved with `PATCH /api/jobs/{job_id}/title`, which
  rewrites `report.json`, `files.title`, `summary.exported.title` and the GP5
  metadata in place. `#stave-button` toggles tab-only view
  (`display.staveProfile = StaveProfile.Tab`, `api.updateSettings()` +
  `render()`), persisted as `tabula:tabOnly`. The score hides dynamics via
  `notation.elements` (`NotationElement.EffectDynamics -> false`) and hides
  rest glyphs in the rendered SVG (SMuFL U+E4E0..U+E4EF) after every render,
  via a childList MutationObserver on the score element. Space always toggles
  play/pause, even when a toolbar button holds focus (keydown and keyup are
  prevented). PDF export is client-side (vendored jsPDF + svg2pdf) and
  paginates the already-vector score SVGs onto A4. The empty state is
  `templates/_hero_tab.html`: eight real bars of one transcription, fixed
  string labels on the left, bars scrolling under them in a masked marquee
  (two copies, 110 s loop, dashes only — no fake notation, and every line's
  bar lines land on the same column). The "New take" picker is always open;
  `Advanced` (`#advanced-control`) holds beam search, MuScriptor quantization
  (off / 1/16 / 1/32 / 1/64, audio only) and the dead-note threshold; there is
  no "fast draft" checkbox. Uploads are capped at 20 MB (`TABULA_MAX_AUDIO_MB`
  default 20) and the drop zone just says "Audio · N MB max". The top-right
  pill reads "local server" (green) for loopback/LAN/tailnet clients and
  "public demo" (accent) otherwise. The labeller is private: `/label`,
  `/api/label/*` and `/api/monitor/*` return 404 unless the client address is
  loopback, LAN or tailnet (`_is_private_client`), and the footer link is only
  rendered for those callers. The footer links to the GitHub source instead of
  claiming a local-only workbench.

## Fixed bugs (recent)

- **Large audio/MIDI offsets** — alignment used to refine from a zero-offset
  pass with a coarse window of only two grid cells.  A clip whose MIDI carried
  ~0.9 s of extra lead-in therefore matched 45.5% and `stitch_tab.py` refused
  the take as "poorly aligned".  `align_events()` now scans the full
  `--max-offset-sec` range (±10 s) and keeps the shift with the most matched
  onsets, breaking ties by residual spread.  `clip_audio`: 45.5% → 75.8%,
  7 dead notes.  The stitch error path now also reads `stitch.log` instead of
  reporting a bare `Exit code 1`.  (The +0.93 s that estimator found was the
  MuScriptor bar shift, since folded into the MIDI — see the next entry.  A
  healthy pair now reports an offset near zero.)

- **MuScriptor bar offset** — MuScriptor shifts every note forward and records
  the shift in `muscriptor:bar_offset=`. That shift is now **folded into the
  MIDI at intake** (`normalise_bar_offset` in `tabfeatures/midi_groups.py`, for
  both a fresh transcription and an uploaded MIDI) and the marker is dropped,
  so the tab, GP5, dead notes and labeller all sit on the recording's timeline.
  Before this, `clip_audio` rendered a phantom empty first bar, started 0.965 s
  late against the audio, and needed a +0.93 s alignment hack; now the
  alignment offset is −0.03 s, the score is 14 bars instead of 15 and the whole
  take is 20.6 s instead of 22.1 s. Do not subtract the marker anywhere else.
- **Rhythm metrics reported 0 %** — `scripts/midi2tab.py` decoded the
  concatenated multi-chunk token stream as one timeline, so `matched` was
  always 0. It now passes the chunk-stitched notes (`pred_notes=raw`):
  onset hit rate @20/60/120 = 97/98.5/100 %, mean onset error 2.4 ticks.
- **Synthesis quality** — one shared `_fluidsynth_wav` renderer: 44.1 kHz for
  file renders (was 22.05 kHz), reverb 0.18 and gain 0.75 for headroom, then
  peak normalisation to 0.89 with short fades, so six-note chords no longer
  clip and the nylon patch has a little room. Override with `TABULA_SF2_REVERB`,
  `TABULA_SF2_CHORUS`, `TABULA_SF2_GAIN`.
- **MuScriptor model choice** — `TABULA_MUSCRIPTOR_MODEL` still wins; with no
  override the app picks the largest weight set already in the HF cache
  (`default_model()`), else `small`. `medium`/`large` are gated on HuggingFace,
  so `hf auth login` + accepting the licence is the one step needed to raise
  transcriptions above what `small` can do.
- **Labeller playback clock** — an `<audio>` element clock drifted by tens of
  milliseconds against the WAV. Playback now uses an `AudioBufferSourceNode`
  and `AudioContext.currentTime`, matching MuScriptor's own web player.
- **MuScriptor quantization** — pre-quantization is off by default. `midi2tab`
  applies the 20-tick model grid itself. If `TABULA_MUSCRIPTOR_QUANTIZE=1`,
  `TABULA_MUSCRIPTOR_GRID` defaults to 8 (1/32), not 4 (1/16).
- **Main workbench is audio-only** — `/` has no MIDI upload slot. The label page
  keeps optional MIDI upload for labelling a specific transcription.
- **Tempo map flattening** — output `input/raw/final.mid` previously flattened
  to a single constant BPM. Now they load the source MIDI and replace instruments,
  keeping all original tempo changes (e.g. lover_girl.mid: 37 tempo changes,
  74–83 BPM, first 80.9 BPM, duration 176.56 s).
- **2× slow WAV synthesis** — FluidSynth writes stereo; the old code treated the
  interleaved L/R stream as consecutive mono samples, halving the playback rate
  and trimming to half the music. Fixed by downmixing to mono before trim.
  Every WAV (`input/raw/final.wav`) is now derived from its own MIDI at the
  correct speed.
- **Tab measure wrapping** — `notes_to_tab` now wraps at whole-measure
  boundaries so bar lines align across all six strings within each system.
- **Chunk-boundary duplication** — the T5 decoder could generate past its
  20-note inference window, and `scripts/midi2tab.py` only stripped one
  re-emitted context note. Output was therefore not time-sorted, contained
  exact duplicates, and had notes at onsets the input never had. Each chunk's
  output is now trimmed to `base_t < t <= last_input_t`, giving 0 duplicates
  and 0 backwards jumps (lover_girl: 1199 raw notes -> 965, 14 breaks -> 0).
- **Pitch provenance in `final.mid`** — `final.mid` takes onset, duration and
  pitch from the input MIDI (ground truth); the model's string is a
  tie-breaker for a playable position
  (`fret = pitch - tuning[string] - capo`), so every surviving note is pitched
  exactly. `raw.mid` stays uncorrected so the model's own pitch errors are
  still visible. Rare chord-tone drops are listed in the report and change
  coverage, not pitch accuracy.
- **Low-position prior** — `assign_onsets()` assigns each onset's chord
  jointly to unique strings, minimising `(max fret, notes above 6, open
  strings, sum of frets, span, deviation from the model)`. It compares the
  capped assignment (candidate frets <= `--max-fret`, default 6) with the
  uncapped one and keeps the better key, so keeping one note cheap cannot
  force another to fret 14. `fix_chord_spans`, `limit_jumps` and
  `repair_overlaps` are capo-aware and also filter by `--max-fret`.
  On lover_girl: 156 open strings (15.9%), 43 notes above 6 (4.4%, was 104 /
  10.6%), mean fret 2.73 (was 3.25), max fret 9, 980/980 notes.
- **Unplayable input chords** — a piano-derived MIDI can contain six-note
  chords wider than any hand (e.g. Valentine's `[51,58,62,63,67,70]`). The old
  `fix_chord_spans` re-fingered greedily and could move two notes onto one
  string, after which `cap_simultaneous` dropped 42 notes. It is now a *joint*
  per-onset assignment, so it cannot create a string collision; when no
  voicing fits the target span it falls back to the most compact unique-string
  voicing (minimising the maximum fret). `playability_pass` also re-assigns
  any residual collision instead of dropping the note. If the best full
  voicing still reaches fret 10, `drop_unplayable_chord_tones` removes **one
  inner tone**, preferring a duplicated pitch class and a compact span, and
  only if the max drops to <= 9 by at least 2 frets. Valentine drops the
  duplicated Eb (pitch 63) in 21 chords, keeps the third (67), and uses a
  compact `5/7/6/7/5` capo-relative voicing: 729/750 notes (97.2%), max fret
  7, 0 collisions.
- **Automatic capo** — `--capo auto` scans capo
  0..min(6, --max-capo, min_input_pitch-40) (higher would need downtuning) and
  keeps the lexicographically best result:
  max relative fret, notes above 6, dropped notes, open strings, fret sum,
  residual span. `final.json`/`final_tab.txt` frets are capo-relative and the
  tab gets a `Capo: N` header; `raw.*` stays at capo 0. Valentine picks capo
  1: 237/729 open (32.5%), 47 notes above 6 (6.4%), mean fret 2.23, max 7,
  score 58.8 vs 42.8 at capo 0.
- **Dead notes use the muted-guitar program** — GM has a dedicated muted
  electric guitar (program 28), so a dead beat now carries a program change to
  28 and the next beat restores 24 (nylon). The single `x` on the lowest string
  is kept (`merge_dead_notes` collapses a muted chord again), and it sounds for
  its notated length at velocity 95 — a damped chug instead of a silent tick or
  an open low E. Verified through alphaTab's MIDI generator: program changes at
  8880/10560/12000 and notes of 480-543 ticks on channel 0.
  Dead notes keep their length: `apply_predictions` stores the source note's
  `dur` (not `1`), and the
  vendored `alphaTab.core.min.mjs` is patched (`static $k` 30 -> 200) because
  `MidiFileGenerator` otherwise turns a GP dead note into a fixed ~60-tick
  click whatever the notated length. See `webapp/static/alphatab/README.md`;
  re-apply on upgrade.
- **Dead-note threshold is 0.65 in the app** — `data/models/dead_notes` ships
  a 0.9 cutoff, which left two muted plucks in `clip_audio` bar 4 sounding
  pitched (classifier dead scores 0.84 and 0.67). `TABULA_DEAD_THRESHOLD`
  defaults to 0.65 and is passed to `stitch_tab.py --dead-threshold`; set it
  empty to use the model's own threshold. The classifier itself is unchanged —
  the per-note scores are identical across builds.
- **GP5 grid: 32nds, one flat tempo** — the grid is 120 ticks (a 32nd), so
  onsets follow the MIDI within a 32nd (median 20 ticks, max 100) instead of
  being dragged onto 16ths; nothing shorter than a 32nd is written.
  `_rebuild_rests` merges adjacent rest beats and splits them into plain
  values (no dots) so a hidden rest cannot leave a stray augmentation dot
  behind. Playback stays at a single tempo: MuScriptor's
  expressive drifts (< 10 BPM from the last kept value) are dropped, so the
  score does not drift against itself. A note that starts in the last grid step
  of a bar and rings past it is notated on the bar line.
- **No ties / leading-bar trim** — the GP5 writer never emits ties and never
  re-strikes a note. Each MIDI note is one attack at its own tick; a note still
  sounding when a bar line arrives simply ends there (a beat cannot cross a
  measure). Re-striking it invents an attack the performance does not have —
  on `clip_audio` that put a phantom chord at the top of bar 2. Bar-crossing
  sustains need ties to be exact, which are deliberately not written. Leading
  measures with no performance are dropped. Automatic H/P defaults to
  confidence 0.85, gap 0.35 beats, interval 5 semitones and ring tolerance 40;
  chord H/P remains off.
- **GP5 notes at a bar line** — a note that rounds past the final slot of a
  measure used to vanish; it is now clamped into the last slot, and one that
  starts in the final grid step *and* rings past the bar is notated on the bar
  line. Pitched attacks then match MuScriptor's, minus the dead-note onsets.
- **GP5 grid-boundary collision** — a sustained note ending just after a
  measure boundary and the next note on the same string starting a few ticks
  later could round into the same grid slot. Both were written into the same
  beat, putting two notes on one string; alphaTab then reported a bogus
  `'beat count' ... exceeded ... safety threshold` while reading the corrupt
  stream. The GP5 writer now keeps at most one note per string per slot,
  preferring the later attack (and a pitched note on ties).
- **Nylon/classical lead** — all output MIDI string tracks and the synthesis
  temp MIDI use GM program 24 (Acoustic Guitar (nylon) / Live HQ "Concert
  Nylon Guitar") instead of program 25 steel, so the WAVs are nylon.
- **ASCII tab padding** — `render_tab_with_features` used to pad each cell
  with spaces ("6  "); the unused part of a cell is now string dashes ("6--")
  so the text tab has no gaps after notes.
- **One `playability_pass`** — `scripts/midi2tab.py` carried a duplicate copy
  of the pass; it now imports `tablature.playability.playability_pass`, so the
  CLI and library cannot drift apart.
