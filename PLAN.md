# Fretwire — MIDI → guitar tab, MVP plan

> **Implementation note (current):** The code follows the Fretting-Transformer
> paper. Training/inference use 512-token / 20-note chunks with one context
> note, a reduced T5 (2.81M params), and post-processing (overlap correction +
> ±5-note neighbor search). There is no alignment loss, note-count head, or
> pitch-constrained training loss, and constrained decoding is off by default.
> Older sections below that mention K=64/L=8, a soft-diagonal alignment loss,
> or pitch-valid decoding are historical. See [`AGENTS.md`](AGENTS.md) for the
> current training/cluster notes.

**Scope of this document: the MVP only.** Everything about capo, alternate tunings, MIDI
augmentation and audio input is deliberately deferred — see §8 (backlog) so it doesn't get lost.

Reference: Hamberger, Murgul, Schmidt, Heizmann, *Fretting-Transformer: Encoder-Decoder Model for
MIDI to Tablature Transcription*, arXiv:2506.14223 (ICMC 2025).
Secondary: Hsu et al., arXiv:2607.26440 (2026) — constrained decoding + explicit-event tokenization.

Machine: `fractal` — RTX 3070 8 GB, 30 GB free, Ubuntu 24.04, Python 3.12, **no torch yet**.
Project dir: `~/Projects/tabula`. Corpus already downloaded to `data/raw/guitar_tab.zip`.

---

## 0. MVP definition

**In:** standard tuning, no capo, 6-string, symbolic MIDI in → ASCII tab out.
**Out:** capo, alternate tunings, audio input, bends/slides/vibrato, GP file export, web UI.

Definition of done:

- `midi2tab song.mid` prints a readable ASCII tab from a standard-tuning MIDI file
- 100 % pitch validity on the held-out set (every predicted `(string,fret)` sounds the input pitch)
- ≥ 75 % tab accuracy on the held-out set
- difficulty score ≤ the A\* baseline on the same inputs
- a 20-song hand-check that doesn't embarrass anyone

Held-out numbers are comparable to the paper's *"standard tuning without capo"* row (Table 5):
GuitarToday 98.41 / Leduc 72.19 / DadaGP 81.58. That's the reference point — but note GuitarToday's
98 % is meaningless as a benchmark (a no-model lowest-fret baseline scores 98.30 there).

---

## 1. Corrections to the original brief

| Assumption | What the papers say | Decision |
|---|---|---|
| Use pre-trained `t5-small` | The reduced T5 is **trained from scratch**. Text domain-adaptation converged faster but was **4 % worse** on tab accuracy (§4.3) | `T5ForConditionalGeneration` + custom `T5Config`, random init |
| Pretrain DadaGP → fine-tune clean sets | All datasets are **combined** for training (16,451 standard / 129,748 conditioned sequences) | Curriculum (big → clean) **plus** a combined-training control run |
| ~20 notes of context, "strategy unconfirmed" | **20-note inference chunks**; last note's tokens carried into encoder *and* decoder; training on 512-token sequences | Train on 512-token windows; infer in 20-note chunks with one note of encoder/decoder context |
| Fine-tune on Leduc | Leduc-only from scratch **collapses to 0.49 %** tab accuracy (Hsu 2026); DadaGP+Leduc reaches 95.97 % | Never train from scratch on small clean data |
| Post-processing gets pitch to 100 % | **Pitch-valid constrained decoding**: 71.78 → 83.65 % tab accuracy on DadaGP, vs 72.82 % for rule-based repair | Generate normally, then apply overlap correction + ±5-note neighbour search (the paper's post-processing) |

Also: **`TabFormer` is taken** (IBM tabular transformer, 2021). Use `Fretwire` (package `fretwire`).

---

## 2. Data

### 2.1 Source

`vldsavelyev/guitar_tab` on HuggingFace — **open, no gate**, 72 MB zip → 951 MB of alphaTex.
Already on the machine at `data/raw/guitar_tab.zip`.

DadaGP (email request) and Leduc (request from Riley/Edwards) are nice-to-have for the comparison
table; **the MVP does not depend on either**. Send the emails on day 1 anyway — they're async.

### 2.2 Measured corpus facts

`data.zip` is one file per **multi-track song**; the HF dataset card's 188,297 rows are those same
songs split per track. Per track, for the guitar family (GM ids 24–31) and 6 strings:

```
songs (files) ..................... 47,299   (41,210 multi-track)
tracks ............................ 188,297
6-string guitar tracks ............  99,404
  standard tuning .................  68,086   (68.5%)
  non-standard ....................  31,318   (31.5%)   <- dropped from the MVP
  standard + plausible ............  63,886   (62.9M notes)
  notes/track p50/p90/p99 .........  784 / 2,180 / 4,274
```

Top non-standard tunings (high → low, as written), for the record:

```
 8,536  Eb5 Bb4 Gb4 Db4 Ab3 Eb3     half-step down   <- exactly re-mappable to standard
 5,404  E5 B4 G4 D4 A3 D3           drop D           <- NOT re-mappable (different shape space)
 3,715  D5 A4 F4 C4 G3 D3           D standard       <- exactly re-mappable
 2,950  D5 A4 F4 C4 G3 C3           drop C           <- NOT re-mappable
 1,949  C0 C0 C0 C0 C0 C0           invalid/junk     <- drop, not a guitar
```

**63,886 standard-tuning tracks ≈ 28× DadaGP's 2,301 filtered tracks.** Expect ~55–60k after dedupe
and per-artist caps. That is ample for a 2.81 M-parameter model.

### 2.3 You need tab data, not MIDI data

You need **paired** input/target, but only one side has to be collected, because tab determines MIDI:

```
pitch    = tuning[string] + fret          (standard tuning: [64,59,55,50,45,40], high → low)
duration = the written note duration
```

General MIDI corpora (Lakh, MAESTRO, DAW exports) have no string/fret, so they cannot be targets
and teach nothing here. They're only relevant to *input* realism — deferred (§8).

Caveat accepted for now: tab-derived MIDI is unnaturally clean. Expect a measurable drop on real
DAW MIDI; measure it at M5 with a handful of real files rather than discovering it from a user.

### 2.4 Canonical intermediate representation

Parse `.tex` once into JSONL; never re-parse for a tokenization experiment.

```jsonc
{"id":"sha1(...)", "source":"hf-guitar_tab", "song":"Wasteland", "track":3,
 "tuning":[64,59,55,50,45,40], "frets":24, "tempo":75, "num_strings":6,
 "notes":[{"t":0,"dur":480,"pitch":50,"s":5,"f":2}, ...],
 "bars":38, "n_notes":412}
```

Rules that save weeks:

- **PPQ = 960.** Quarter note = 960 ticks, sixteenth = 240.
- Resolve **ties, dots, tuplets** into `(onset, duration)` before writing. This is not optional:
  measured across the corpus, **5.69 M tuplet tags (39.8 % of files contain `{tu}`)** and **5.66 M
  dotted beats (`{d}`)**. Get these wrong and timing is broken on 40 % of the data.

  The alphaTex semantics, confirmed in `AlphaTex1LanguageHandler.ts`:

  | tag | meaning | duration factor |
  |---|---|---|
  | `{d}` | **dotted** (not "dead" — `case 'd': beat.dots = 1`) | × 1.5 |
  | `{dd}` | double dotted (0 occurrences in this corpus) | × 1.75 |
  | `{tu 3}` | triplet → numerator 3, denominator 2 | × 2/3 |
  | `{tu 5}` / `{tu 6}` / `{tu 7}` | → denominator 4 | × 4/5, × 4/6, × 4/7 |
  | `{tu 9}`…`{tu 12}` | → denominator 8 | × 8/n |
  | `{tu a b}` | explicit numerator/denominator | × b/a |

  Dead notes are `{x}`, **not** `{d}` — and there are **0** of them in the corpus, so there is no
  unpitched-note case to handle. `{g}` (ghost) is still pitched: keep it.
- **Repeats** (`\ro`/`\rc`) are *not* expanded in this corpus — expand them or drop the track. Never
  guess the bar order; a wrong order silently teaches nonsense. Log the drop count.
- Deterministic sort: `(t, is_off, pitch, s, f)` — note-offs before note-ons at equal ticks.
- **Drop, don't repair**: `f < 0 || f > frets`, `s ∉ 1..6`, or `pitch != tuning[s] + f` → drop the
  track and log the reason. These are parse bugs, not data quirks.
- **Dedupe by song**, keyed on the normalised pitch-onset sequence. The corpus has many
  covers/duplicates of the same song; near-duplicates across the split are how you fake a 95 % score.
- **Split before any processing that could leak**: 80/10/10 by song identity.
  (The paper's "90/10/10" sums to 110 and is unusable.)

### 2.5 Filters, in order

1. `num_strings == 6` (drops bass, 7-string, ukulele-ish material)
2. GM guitar ids 24–31; verify empirically, don't trust the ids — the corpus looks 1-based
   (24 nylon, 25 steel, 30 distortion), while the paper cites "channel 25 nylon / 26 steel"
3. tuning **exactly** `E5 B4 G4 D4 A3 E3` (this single string match replaces a whole subsystem)
4. tuning must parse to MIDI 0–127 and be strictly descending → kills the `C0 C0…` junk
5. `n_notes >= 50`, `bars >= 4`
6. not percussion: reject tracks whose notes cluster on GM drum pitches or whose `(s,f)` is degenerate
7. per-artist / per-song caps so one artist doesn't dominate

After 1–6: **63,886**. After 7: ~55–60k.

### 2.6 String numbering — read this twice

In this corpus **string 1 = highest-pitched** (verified: drop-D files contain `(0.4 0.5 0.6)` = open
D4/A3/D3, a D5 power chord). The paper and Hsu index from the **lowest** pitch. Both work; mixing
them silently does not — it produces a model that reports 0 % accuracy and a day of debugging.

**Decision: internal `s=1` is the thickest string** (low E, MIDI 40) — identical to the paper's and
Hsu's convention, so published numbers stay comparable. Convert at the parser boundary
(`internal = 7 - alphaTex_string`), in **one** function, with a unit test on a known chord.

**Chord order: ascending pitch, on both sides.** Notes sharing an onset are emitted lowest pitch
first, and this must hold for the *input* `NOTE_ON`s as well as the target `TAB`s — the decoder's
mask advances an input-note cursor in lockstep with the target (see §4.4), so if the two orders
disagree the mask references the wrong pitch. With `s=1` thickest, ascending pitch also happens to
be ascending string index, which keeps chords monotonic in the token stream.

### 2.7 Licensing

The corpus is GP3/4/5 files sourced from a tracker. **Research use only.** Keep `data/` gitignored;
never commit corpus data, derived JSONL, tokenised shards or rendered tabs. Ship code, configs, vocab.

---

## 3. Tokenization

### 3.1 Vocabulary (~350 tokens, no conditioning)

| Family | Count | Notes |
|---|---|---|
| `<pad>`, `</s>`, `<unk>` | 3 | T5 conventions |
| `NOTE_ON<p>`, `NOTE_OFF<p>` | 61 × 2 | p ∈ 36..96 covers guitar |
| `TIME_SHIFT<d>` | ~40 after pruning | see §3.2 |
| `TAB<s,f>` | 6 × 25 = 150 | s ∈ 1..6 (1 = lowest), f ∈ 0..24 |

No `CAPO` tokens, no `TUNING` tokens in the MVP. Build from the training split with
min-frequency pruning, freeze to `vocab_v1.json`, record its hash in every checkpoint.
Keep the vocabulary small — at 2.81 M params the output softmax needs to stay sharp.

### 3.2 Time — lock this now

- PPQ 960. Onsets and durations quantised to a **1/16 grid (60 ticks)**, minimum duration 60.
- `TIME_SHIFT<d>` for `d ∈ {60,120,…,7680}` (1/16 note … 2 bars) + two coarse fallbacks (11520,
  15360). Prune shifts below 0.1 % frequency; expect ~40 live tokens.
- Gaps on the grid → one token; off-grid remainder → ≤3 decomposed tokens, and log the rate
  (if >2 %, the grid is wrong).
- **The training tokenizer, the eval harness and the inference path must use the identical
  quantiser.** This is the contract that later makes MIDI augmentation possible (§8) and the thing
  that silently invalidates every metric if it drifts.

Gate: encode → decode is bit-identical on ≥99.5 % of 1,000 held-out tracks.

### 3.3 Sequence layout

```
encoder: NOTE_ON<55> TIME_SHIFT<240> NOTE_ON<59> TIME_SHIFT<240> NOTE_OFF<55> NOTE_OFF<59> TIME_SHIFT<480> …
decoder: TAB<3,0> TIME_SHIFT<240> TAB<2,1> TIME_SHIFT<240> TIME_SHIFT<480> …
```

Time-ordered global event stream, not per-note blocks. Note-offs at their true offset — that's what
makes duration available to the encoder, and v5 of the paper shows removing it hurts.

Known weakness inherited deliberately: the target has no note-off, so a sustained note and a
re-struck note look identical and "this string is still ringing" is inexpressible. This is exactly
why the paper needs overlap correction. Accepted for the MVP; the explicit-event variant is the
first v2 ablation.

### 3.4 Chunking

- Window of **K = 64 notes** with **L = 8 notes** of left context, capped at 512 tokens.
- Training examples use exactly this windowing. At inference, stride `K−L`, keep predictions for
  notes `[L, K)`, use the first L as context only; dedupe by `(round(onset), pitch)`.
- Report **boundary vs interior** tab accuracy separately. With 20-note chunks and one note of
  carry-over (the paper's setting) boundary errors are a large share of total error and are never
  reported.

### 3.5 Pairs, and the leakage test that saves the project

Both sides are generated from one canonical record, together, at the same time:

- **Input** is built from `(onset, duration, pitch)` only.
- **Target** is built from `(onset, pitch, s, f)`, but only `s` and `f` survive — `pitch` is implied
  by the tuning, so the target carries **no pitch token** at all. `TAB<6,5>` implies pitch 45 because
  `tuning[6] = 40`. That is the entire task, and it is why pitch accuracy is measurable: decode
  TAB → pitch, compare against the input.

Worked example (real corpus file, `1_God/1_God_-_Tali.tex`, standard tuning; a 4-note chord then two
melody notes):

```
s1f5 -> 69   s2f5 -> 64   s3f6 -> 61   s6f5 -> 45        (same onset, t=2400)

INPUT : TIME_SHIFT<2400> NOTE_ON<45> NOTE_ON<61> NOTE_ON<64> NOTE_ON<69>
        TIME_SHIFT<240> NOTE_OFF<45> NOTE_OFF<61> NOTE_OFF<64> NOTE_OFF<69>
        NOTE_ON<71> TIME_SHIFT<240> NOTE_OFF<71> ...

TARGET: TIME_SHIFT<2400> TAB<1,5> TAB<4,6> TAB<5,5> TAB<6,5>
        TIME_SHIFT<240> TAB<6,7> TIME_SHIFT<240> TAB<6,5>

        (ascending pitch 45 < 61 < 64 < 69 == ascending string 1 < 4 < 5 < 6)
```

The two sides differ in length — note-offs exist only on the input — which is why this is an
encoder-decoder task and not sequence tagging. `TIME_SHIFT` is the only family shared by both sides,
with a different meaning on each (stream gap vs. gap between TABs); don't assume symmetry.

Two invariants, both cheap to test and both fatal if violated:

1. **Refingering invariance.** A2 is playable as `s6f5` or `s5f0` — same pitch, different answer.
   Re-encode a track with an alternative valid fingering: the **input token sequence must be
   byte-identical**. If it changes, fingering leaked into the input and the model will learn to copy
   instead of infer — ~99 % validation, collapses on real MIDI.
2. **Type-level isolation.** `encode_input(events: list[PitchEvent])` where `PitchEvent` has no
   `s`/`f` fields. Make leakage unrepresentable rather than trusting discipline.

Also: **trim leading silence** to the first onset (store the offset), or a track with a long intro
burns its 512-token budget on `TIME_SHIFT` tokens.

**Chord ordering.** Notes sharing an onset have no natural order. Fix one canonical order (ascending
pitch) for training *and* inference, and **evaluate chords as sets per onset** — positional
comparison penalises answers that are semantically identical.

---

## 4. Model and training

### 4.1 Architecture

```python
T5Config(d_model=128, d_ff=1024, num_layers=3, num_decoder_layers=3,
         num_heads=4, dropout=0.1, vocab_size=len(tokenizer))
```
≈ **2.81 M parameters**. Random init (no text pre-training).

### 4.2 Training recipe

- Adafactor with a **self-adaptive learning rate** (the paper's choice), dropout 0.1, bf16
  autocast, weight decay 0 by default. `--lr` switches to a fixed-rate cosine schedule.
- Effective batch 64 (gradient accumulation; the model fits 8 GB trivially)
- Teacher-forced cross-entropy on target tokens
- 300 epochs; early stop on **validation loss** and report full tab metrics every 1250 optimizer
  steps (tab accuracy plateaus at the position-ambiguity ceiling while loss keeps falling)
- Budget: minutes per epoch on the 3070; one overnight run for the full schedule.
  **Iterate on a 5 % subset first**, scale only once the pipeline is proven.

### 4.3 Two stages

1. **Pre-train** on ~55–60k tracks. This is where the `pitch → position` prior and the token
   grammar are learned. It is also *required*: small-data-only training collapses (§1).
2. **Fine-tune** on the clean slice (Leduc if access arrives, else the hand-checked set), LR
   1e-4–2e-4, hard early stop — 2–3 epochs is generous at that size. Transpose augmentation
   ±1..5 semitones if the clean set is under ~2k sequences (55 songs × 11 = 605 in Hsu's setup).
3. Run the **combined-training control** too (stage 1 data + clean data shuffled). You need both
   numbers to know which regime you're in.

### 4.4 Inference: pitch-valid constrained decoding *(historical)*

> The paper does **not** use this. It generates normally and applies the
> overlap-correction + neighbour-search post-processing described in §3.5.

At each `TAB` step the source pitch `p` is known (you're replaying the input stream), so:

```
T(p) = { (s,f) : tuning[s] + f == p,  s ∈ 1..6,  f ∈ 0..24 }
logits[TAB<s,f> ∉ T(p)] = -inf
```

Because tuning is fixed, `T(p)` is a **precomputed static table** — a dict lookup, no arithmetic at
decode time. Implement as an HF `LogitsProcessor` driven by an input-note cursor.

Measured effect in the literature: FT 71.78 → 83.65 % tab accuracy on DadaGP; rule-based
post-processing only reaches 72.82 % *and* can desynchronise timing.

Then, as a safety net only:

- **overlap repair** — two notes on the same string with intersecting intervals → reassign the later
  one to the cheapest valid position
- **neighbour search** — match each input note to the closest predicted note within ±5 notes
  (mostly redundant once the mask is in place)

Optional, cheap: beam search 4–8 with length normalisation.

### 4.5 Evaluation harness — build before the first real training run

| Metric | Definition |
|---|---|
| Pitch accuracy | fraction of predicted TABs whose implied pitch equals the source pitch |
| Tab accuracy | fraction matching ground-truth `(s,f)` |
| Difficulty | mean of `fret_stretch + locality(α=0.25) + vertical_stretch` (paper §3.6) |
| Error decomposition | timing / pitch / position / correct — the single most informative diagnostic |
| Validity | every TAB token must exist on the fretboard (must be 100 %) |
| Boundary vs interior | tab accuracy split by position in the chunk |
| Playability violations | same-string overlap, chord span > 5 frets, > 4 simultaneous notes |

Baselines, cheapest first: (1) lowest-viable-fret, ~50 lines — the "is the model learning at all"
floor (98.3 % on GuitarToday, 79.2 % on DadaGP); (2) A\* (Burlet & Fujinaga) or `tutts`'s Viterbi,
for the playability comparison.

**Always report the split with the number.** Published "DadaGP" tab accuracy ranges 71.5–81.6 %
depending on filtering; a bare number is not a result.

Expected ceiling: ~85 % tab accuracy. Position ambiguity, not pitch, is the binding constraint —
Hsu decomposes DadaGP error into timing 2.6 % / pitch 7.4 % / **position 13.0 %**.

---

## 5. Repo layout

```
fretwire/
  data/                 # gitignored: zip, .tex, jsonl, tokenised shards
  src/fretwire/
    parse/alphatex.py   # .tex -> canonical events
    schema.py           # NoteEvent, Track, tuning/fretboard validation
    fretboard.py        # Fretboard(tuning) - the ONLY place pitch<->(s,f) lives
    filter.py           # §2.5 rules, dedupe, split
    tokenizer.py        # vocab build, encode/decode
    quantise.py         # the 960 PPQ / 60-tick contract (§3.2)
    chunk.py            # §3.4 windowing + stitching
    model.py            # T5Config factory
    train.py            # config-driven pre-train / fine-tune
    decode.py           # LogitsProcessor + beam search
    postproc.py         # overlap repair, neighbour search
    metrics.py          # §4.5
    render.py           # ASCII tab (+ .tex later)
    cli.py              # midi2tab
  configs/              # one yaml per run: vocab hash + git sha recorded
  tests/                # round-trip, fretboard math, mask correctness, chunk stitching
  PLAN.md
```

Non-negotiables: `data/` gitignored; every run records config + vocab hash + git sha; fretboard and
quantiser each live in exactly one module with tests.

---

## 6. Milestones

| # | Deliverable | Gate | Est. |
|---|---|---|---|
| **M0** | env + repo + stats | torch sees the 3070; corpus stats reproduce §2.2 | 0.5 d |
| **M1** | `.tex` → canonical JSONL | ≥60k standard-tuning tracks; 100 tracks eyeballed against source; invariant violations < 0.1 % | 2–3 d |
| **M2** | tokenizer + quantiser | ≥99.5 % bit-identical round-trip on 1k tracks; vocab frozen and hashed | 1–2 d |
| **M3** | baseline model + eval harness | 5 % subset trained; pitch ≥95 %, tab ≥65 % on holdout; full metric table produced | 2 d |
| **M4** | constrained decoding + repair | 100 % pitch validity; tab +5 pts over M3; boundary/interior split reported | 1 d |
| **M5** | full pre-train, fine-tune, CLI | 55–60k-track run; fine-tune; `midi2tab` renders ASCII; 20-song playability spot-check; real-MIDI smoke test | 3–5 d |

**MVP ≈ 2–2.5 weeks.** M3 already gives a working end-to-end (if mediocre) system, so the demo exists
early and everything after is measurable improvement.

Send the DadaGP and Leduc access emails on day 1 — they're other people's inboxes, and Leduc is the
only benchmark that actually discriminates.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Leduc / DadaGP access never arrives | MVP doesn't need them: hold out standard-tuning tracks + a 20-song hand-checked set |
| Repeat / technique semantics mis-parsed | Expand or drop repeats; assert invariants; eyeball 100 tracks at M1 |
| Position ambiguity caps tab accuracy ~85 % | Expected. Report error decomposition so "right pitch, wrong string" is visibly distinct from "wrong note" |
| Corpus partly algorithmically generated | Compare against the lowest-fret baseline per split; DadaGP's 79 % baseline score shows how algorithmic a split is |
| Vocab / split drift between runs | Freeze vocab, commit split files, hash both into every config |
| 30 GB disk | 951 MB corpus + 1–2 GB JSONL + shards fits; delete zips after unpacking |
| Train/inference quantiser drift | One `quantise.py`, used by all three paths, with a test that they agree |

---

## 8. Deferred (v2 backlog, in priority order)

1. **Pitch-valid mask with conditioning** — capo + alternate tunings. `T(p)` becomes a function of
   `(tuning, capo)` instead of a static table. Capo supervision must be synthetic: the corpus has
   **0/47,299** capo directives (§2.2). Differentiator vs MIDI-to-Tab, which supports neither.
2. **Explicit-event target** (Hsu 2026): `NOTE_ON_PITCH` / `NOTE_OFF_PITCH` in the decoder output
   too — +5.3 % tab accuracy on DadaGP and it is what stops small-data collapse. Longer targets.
3. **MIDI-domain augmentation**: onset jitter ±20 ms → re-quantise on the §3.2 grid, tempo 0.75–1.5×,
   velocity, input-side noise (bass/drum tracks), forced overlap. Deferred; the quantiser contract
   is already locked so this can be added without invalidating prior numbers.
4. **Uniform-transposition data recovery**: Eb/D/Db-standard tracks (≈15k, §2.2) are exact
   transpositions of standard — transpose pitches, keep `(s,f)`. Plus ~15 % data for ~20 lines.
5. **`.gp` / `.tex` export, audio → tab, bend/slide/vibrato tokens.**

---

## References

1. Hamberger, Murgul, Schmidt, Heizmann. *Fretting-Transformer.* arXiv:2506.14223, ICMC 2025.
2. Hsu, Wang, Hong, Chen. *Explicit Note-Event Tokenization and Pitch-Validity Constrained Decoding
   for MIDI-to-Tablature Transcription.* arXiv:2607.26440, 2026.
3. Edwards, Riley, Sarmento, Dixon. *MIDI-to-Tab.* arXiv:2408.05024, ISMIR 2024.
4. Sarmento et al. *DadaGP.* ISMIR 2021.
5. Riley, Edwards, Dixon. *High Resolution Guitar Transcription via Domain Adaptation.* ICASSP 2024.
6. Burlet, Fujinaga. *Robotaba Guitar Tablature Transcription Framework.* ISMIR 2013. (A\* baseline)
