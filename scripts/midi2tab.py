#!/usr/bin/env python3
"""MIDI -> tablature, with honest artefacts.

    python scripts/midi2tab.py song.mid \
        --checkpoint runs/pretrain_t5_guitarset/best \
        --out-dir out/song

Writes, all derived independently:

    input.mid    the parsed input notes
    raw.mid      the model's own output: onsets from generated TIME_SHIFT
                 tokens, pitches from generated TAB tokens
    final.mid    raw after pitch-preserving playability correction
    *.wav        synthesis of each of the above
    final_tab.txt ASCII tab of the corrected fingering
    report.json  note counts, alignment metrics, and rhythm metrics

MIDI cannot store a fingering, so each output is written as one track per
string.  That keeps the string/fret assignment in the file and lets raw and
final differ even when they sound the same.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pretty_midi
import torch
from transformers import T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from audio.synth import DEFAULT_SF2, synthesize  # noqa: E402
from audio.wav import write_wav  # noqa: E402
from model.chunk import ChunkConfig, make_inference_chunks  # noqa: E402
from model.decode import constrained_beam_search  # noqa: E402
from core.fretboard import STANDARD, positions  # noqa: E402
from core.quantise import quantise_notes, ticks_to_seconds  # noqa: E402
from tablature.metrics import compute_difficulty, compute_rhythm_metrics  # noqa: E402
from corpus.parse.midi import parse_midi  # noqa: E402
from tablature.playability import SOFT_FRET, playability_pass  # noqa: E402
from tablature.postproc import DecodedNote  # noqa: E402
from tablature.render import render_tab  # noqa: E402
from tablature.stream import decode_tab_stream  # noqa: E402
from tablature.tokenizer import PAD_TOKEN, Vocab  # noqa: E402

NOTE_DTYPE = [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]


def to_array(notes: list[dict]) -> np.ndarray:
    arr = np.empty(len(notes), dtype=NOTE_DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (int(n["t"]), int(n["dur"]), int(n["pitch"]), int(n["s"]), int(n["f"]))
    return arr


def write_tab_midi(
    path: Path,
    notes: list[dict],
    bpm: float = 120.0,
    ppq: int = 960,
    source_midi_path: Path | None = None,
    tempo_changes: list[tuple[float, float]] | None = None,
) -> None:
    """One track per string, so the fingering is stored in the file.

    If ``source_midi_path`` is given, that file is loaded and its instruments
    are replaced with our string tracks, preserving the original tempo map,
    key signature, and time signature.  Note times are converted from ticks
    to seconds using ``tempo_changes`` (or a constant ``bpm`` if not given).
    """
    if source_midi_path is not None:
        pm = pretty_midi.PrettyMIDI(str(source_midi_path))
        pm.instruments = []  # drop original instruments, keep tempo map etc.
    else:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tracks = {}
    for s in [0, *range(1, 7)]:
        name = "pitch only" if s == 0 else f"string {s}"
        inst = pretty_midi.Instrument(program=24, name=name)  # GM nylon/classical
        pm.instruments.append(inst)
        tracks[s] = inst
    for n in notes:
        s = int(n["s"])
        if s not in tracks:
            s = 0  # input MIDI has no string information
        start_sec = ticks_to_seconds(int(n["t"]), tempo_changes, ppq)
        end_sec = ticks_to_seconds(int(n["t"]) + max(1, int(n["dur"])), tempo_changes, ppq)
        tracks[s].notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=int(n["pitch"]),
                start=start_sec,
                end=end_sec,
            )
        )
    pm.write(str(path))


def generate(
    model,
    vocab,
    notes,
    beam: int,
    pitch_constraint: bool,
    max_len: int,
    device: str,
    max_fret: int | None = None,
    fret_penalty: float = 0.0,
    on_progress: Callable[[int, int, list[dict]], None] | None = None,
) -> list[dict]:
    # Quantise input notes to training distribution (20-tick grid, 40-tick chord window).
    notes = quantise_notes(notes)
    chunks = make_inference_chunks(notes, vocab, "input", ChunkConfig(max_tokens=max_len))
    pad_id = vocab[PAD_TOKEN]
    raw: list[dict] = []
    all_tokens: list[str] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
        attn = (enc != pad_id).long()
        prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=device)
        max_new = min(max_len, max(16, 8 * chunk.meta["num_notes"] + 16))

        with torch.no_grad(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            gen = constrained_beam_search(
                model,
                enc,
                vocab,
                num_beams=max(1, beam),
                max_length=max_len,
                use_pitch_constraint=pitch_constraint,
                decoder_input_ids=prompt,
                attention_mask=attn,
                max_new_tokens=max_new,
                prompt_tabs=0,
                max_fret=max_fret,
                fret_penalty=fret_penalty,
            )

        tokens = vocab.decode(gen[0, prompt.size(1) :].tolist())
        all_tokens.extend(tokens)
        chunk_notes = decode_tab_stream(tokens, base_t=chunk.meta["base_t"])

        # Trim this chunk's output to its own note window.  The model is free
        # to keep generating past the 20-note window, and the prompt repeats
        # the previous note; without trimming those bleed into the next chunk
        # (duplicate notes, backwards time jumps).
        start = chunk.meta["note_start"]
        end = chunk.meta["note_end"]
        base_t = chunk.meta["base_t"]
        if start > 0:
            # Anything at or before the context note is a re-emitted prompt.
            chunk_notes = [n for n in chunk_notes if n["t"] > base_t]
        else:
            chunk_notes = [n for n in chunk_notes if n["t"] >= base_t]
        # Never occupy the next chunk's input onsets.
        last_t = notes[end - 1].t
        chunk_notes = [n for n in chunk_notes if n["t"] <= last_t]

        raw.extend(chunk_notes)
        if on_progress is not None:
            on_progress(chunk_index, len(chunks), raw)

    return raw, all_tokens


def final_from_input(
    raw: list[dict],
    input_notes,
    max_fret: int | None = None,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    capo: int = 0,
) -> list[dict]:
    """Build the final artefact: input pitch/timing + a playable fingering.

    The input MIDI is ground truth for pitch and time.  The model's string is
    used as a tie-breaker, but the post-processor prefers the lowest
    capo-relative fret (open strings naturally win) whenever that is strictly
    better.  The fret is derived from the input pitch, so pitch is exact by
    construction.  Notes the model missed fall back to the lowest valid
    position.
    """
    raw_by_t: dict[int, list[dict]] = defaultdict(list)
    for n in raw:
        raw_by_t[n["t"]].append(n)
    raw_times = sorted(raw_by_t)

    def nearest(t: int) -> list[dict]:
        if not raw_times:
            return []
        i = bisect.bisect_left(raw_times, t)
        best_t = None
        for j in (i - 1, i):
            if 0 <= j < len(raw_times):
                if best_t is None or abs(raw_times[j] - t) < abs(best_t - t):
                    best_t = raw_times[j]
        return raw_by_t[best_t] if best_t is not None else []

    used: dict[int, set[int]] = {}
    out: list[dict] = []
    for gt in input_notes:
        cands = raw_by_t.get(gt.t)
        if not cands:
            near = nearest(gt.t)
            cands = near if near and abs(near[0]["t"] - gt.t) <= 480 else []
        taken = used.setdefault(gt.t, set())

        # The model's string hint: same pitch first, then any free string.
        s_model = None
        for n in cands:
            if n["pitch"] == gt.pitch and n["s"] not in taken:
                s_model = n["s"]
                break
        if s_model is None:
            for n in cands:
                if n["s"] not in taken:
                    s_model = n["s"]
                    break
        if s_model is None and cands:
            s_model = cands[0]["s"]

        cand_pos = positions(gt.pitch, tuning, frets, capo)
        if not cand_pos:
            # Pitch outside the instrument's range; keep a valid-looking
            # placeholder so downstream code cannot crash.
            out.append(
                {"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": 1, "f": 0}
            )
            continue
        if max_fret is not None:
            low = [c for c in cand_pos if c[1] <= max_fret]
            cand_pos = low or cand_pos
        free = [c for c in cand_pos if c[0] not in taken]
        cand_pos = free or cand_pos
        s, f = min(
            cand_pos,
            key=lambda c: (
                c[1],                          # lowest fret; open first
                0 if c[0] == s_model else 1,   # keep model string on ties
                c[0],                          # low string on ties
            ),
        )
        taken.add(s)
        out.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})
    return out


def pitch_coverage(input_notes, pred: list[dict]) -> dict:
    """Per-onset pitch-multiset coverage of the input by ``pred``."""
    gt_by_t: dict[int, Counter] = defaultdict(Counter)
    for n in input_notes:
        gt_by_t[n.t][n.pitch] += 1
    pred_by_t: dict[int, Counter] = defaultdict(Counter)
    for n in pred:
        pred_by_t[n["t"]][n["pitch"]] += 1

    total = correct = 0
    for t, counts in gt_by_t.items():
        for pitch, cnt in counts.items():
            total += cnt
            correct += min(cnt, pred_by_t[t][pitch])
    return {
        "input_notes": total,
        "pred_notes": len(pred),
        "pitch_correct": correct,
        "pitch_accuracy": round(correct / total, 4) if total else 0.0,
    }


def compare(input_notes, raw) -> dict:
    """Greedy nearest-onset alignment, purely descriptive."""
    used: set[int] = set()
    aligned = pitch_hits = tab_hits = 0
    for gt in input_notes:
        best_j = None
        best_dt = None
        for j, p in enumerate(raw):
            if j in used:
                continue
            dt = abs(p["t"] - gt.t)
            if best_dt is None or dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is None or best_dt is None or best_dt > 480:
            continue
        used.add(best_j)
        aligned += 1
        p = raw[best_j]
        pitch_hits += int(p["pitch"] == gt.pitch)
        tab_hits += int(p["s"] == gt.s and p["f"] == gt.f)
    n = len(input_notes)
    return {
        "input_notes": n,
        "raw_notes": len(raw),
        "aligned": aligned,
        "pitch_correct": pitch_hits,
        "pitch_accuracy": round(pitch_hits / n, 4) if n else 0.0,
        "tab_exact": tab_hits,
        "tab_accuracy": round(tab_hits / n, 4) if n else 0.0,
    }


def _capo_key(final: list[dict], stats: dict) -> tuple:
    """Lexicographic playability key for one capo (lower is better)."""
    frets = [n["f"] for n in final]
    if not frets:
        return (0, 0, 0, 0, 0, 0)
    by_t: dict[int, list[int]] = defaultdict(list)
    for n in final:
        by_t[n["t"]].append(n["f"])
    span = max((max(v) - min(v) for v in by_t.values()), default=0)
    dropped = len(stats.get("dropped_chord_tones", [])) + stats.get(
        "dropped_impossible", 0
    )
    return (
        max(frets),                                # lowest maximum fret
        sum(1 for f in frets if f > SOFT_FRET),   # then fewest high frets
        dropped,                                   # avoid drops unless they help
        sum(1 for f in frets if f != 0),           # prefer open strings
        sum(frets),                                # lower overall position
        max(0, span - 4),                          # residual span violation
    )


def _capo_trial(capo: int, final: list[dict], stats: dict, key: tuple) -> dict:
    """Human-readable stats and a 0-100 score for one capo candidate."""
    frets = [n["f"] for n in final]
    n = len(frets)
    if not n:
        return {"capo": capo, "feasible": True, "notes": 0, "score": 0.0}
    open_notes = sum(1 for f in frets if f == 0)
    high = sum(1 for f in frets if f > SOFT_FRET)
    dropped = len(stats.get("dropped_chord_tones", [])) + stats.get(
        "dropped_impossible", 0
    )
    open_rate = open_notes / n
    high_rate = high / n
    mean_fret = sum(frets) / n
    maxf = max(frets)
    drop_rate = dropped / (n + dropped) if (n + dropped) else 0.0
    score = 100.0 * (
        0.45 * open_rate
        + 0.30 * (1.0 - high_rate)
        + 0.15 * (1.0 - min(mean_fret, 12.0) / 12.0)
        + 0.10 * (1.0 - min(maxf, 12.0) / 12.0)
    ) - 10.0 * drop_rate
    return {
        "capo": capo,
        "feasible": True,
        "notes": n,
        "open_notes": open_notes,
        "open_rate": round(open_rate, 4),
        "high_notes": high,
        "high_rate": round(high_rate, 4),
        "mean_fret": round(mean_fret, 3),
        "max_fret": maxf,
        "dropped": dropped,
        "score": round(max(0.0, score), 1),
        "key": list(key),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input MIDI file")
    ap.add_argument("--checkpoint", default="runs/pretrain_t5_guitarset/best")
    ap.add_argument("--vocab", default="vocab/guitarset.json")
    ap.add_argument("--out-dir", default="out/midi2tab")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument(
        "--sf2",
        default=DEFAULT_SF2,
        help="SoundFont used for WAV synthesis (default: Live HQ Natural GM V3)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=120,
        help="max chars per tab line (default: 160)",
    )
    ap.add_argument(
        "--pitch-constraint",
        action="store_true",
        default=False,
        help="enable pitch-valid / low-position constrained decoding (opt-in; "
        "the model is not trained for it, so off by default)",
    )
    ap.add_argument(
        "--max-fret",
        type=int,
        default=6,
        help="softly avoid capo-relative frets above this (default: 6)",
    )
    ap.add_argument(
        "--capo",
        default="auto",
        help="capo selection: auto, off, or a fixed 0-6 (default: auto)",
    )
    ap.add_argument(
        "--max-capo",
        type=int,
        default=6,
        help="highest capo to consider (default: 6)",
    )
    ap.add_argument(
        "--no-drop-chord-tones",
        action="store_true",
        help="never drop an inner chord tone, even for an unplayable voicing",
    )
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help="skip WAV synthesis (MIDI, JSON and text tab are still written)",
    )
    ap.add_argument(
        "--progress-file",
        default=None,
        help="write partial decoding progress as JSON (for live preview)",
    )
    ap.add_argument(
        "--fret-penalty",
        type=float,
        default=0.3,
        help="soft bias -lambda*fret toward lower positions (default: 0.3)",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(args.vocab)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device).eval()
    if device == "cuda":
        model.to(torch.bfloat16)

    midi = parse_midi(args.input)
    notes = sorted(midi.notes, key=lambda n: (n.t, n.pitch, n.s, n.f))
    if not notes:
        raise SystemExit("no notes in input")

    # Quantise input to training grid so the model sees familiar timings.
    notes = quantise_notes(notes)

    progress_path = Path(args.progress_file) if args.progress_file else None
    preview_capo = 0
    if str(args.capo).strip().lower() not in {"auto", "off"}:
        try:
            preview_capo = int(args.capo)
        except (TypeError, ValueError):
            preview_capo = 0

    def write_progress(payload: dict) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_name(f".{progress_path.name}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(progress_path)

    def on_chunk(done: int, total: int, partial: list[dict]) -> None:
        preview_inputs = [
            DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
            for n in partial
        ]
        preview = final_from_input(
            partial,
            preview_inputs,
            max_fret=args.max_fret,
            capo=preview_capo,
        )
        nodes = [
            DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
            for n in preview
        ]
        write_progress(
            {
                "stage": "tab",
                "final": False,
                "chunks_done": done,
                "chunks_total": total,
                "notes": len(partial),
                "capo": preview_capo,
                "auto_capo": str(args.capo).strip().lower() == "auto",
                "message": f"Decoded chunk {done}/{total}…",
                "tab": render_tab(nodes, width=args.width, capo=preview_capo),
            }
        )

    raw, gen_tokens = generate(
        model,
        vocab,
        notes,
        args.beam,
        args.pitch_constraint,
        args.max_len,
        device,
        max_fret=args.max_fret,
        fret_penalty=args.fret_penalty,
        on_progress=on_chunk if progress_path is not None else None,
    )
    # Capo scan.  A capo at c raises the lowest open string to 40+c, so a note
    # below that would need downtuning; only capos up to min_pitch-40 are
    # feasible.  The model is not re-run: each candidate is a cheap re-fingering
    # of the same raw string hints.  The final artefact uses the best one.
    if not notes:
        raise SystemExit("input MIDI has no notes")
    min_pitch = min(n.pitch for n in notes)
    max_capo = min(6, args.max_capo)
    capo_max = min(max_capo, max(0, min_pitch - STANDARD[0]))
    if args.capo == "off":
        capos = [0]
    elif args.capo == "auto":
        capos = list(range(capo_max + 1))
    else:
        try:
            fixed = int(args.capo)
        except ValueError:
            raise SystemExit(
                f"--capo must be auto, off, or 0-{max_capo} "
                f"(got {args.capo!r})"
            )
        if not 0 <= fixed <= max_capo:
            raise SystemExit(f"--capo must be 0-{max_capo}")
        if fixed > capo_max:
            raise SystemExit(
                f"capo {fixed} would need downtuning "
                f"(lowest input pitch {min_pitch}); max feasible is {capo_max}"
            )
        capos = [fixed]

    capo_trials = []
    best = None
    for c in capos:
        seed = final_from_input(raw, notes, max_fret=args.max_fret, capo=c)
        final_c, stats_c = playability_pass(
            seed,
            max_fret=args.max_fret,
            capo=c,
            allow_drop=not args.no_drop_chord_tones,
        )
        key = _capo_key(final_c, stats_c)
        capo_trials.append(_capo_trial(c, final_c, stats_c, key))
        if best is None or key < best[0]:
            best = (key, c, final_c, stats_c)
    if best is None:
        raise SystemExit("no feasible capo")
    _, capo, final, proc_stats = best

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Preserve the input file's tempo map so output timing matches the source.
    _pm_in = pretty_midi.PrettyMIDI(str(args.input))
    _tc_in = _pm_in.get_tempo_changes()
    tempo_changes = list(zip(_tc_in[0].tolist(), _tc_in[1].tolist())) if _tc_in[1].size else None

    in_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in notes]
    write_tab_midi(out / "input.mid", in_dicts, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)
    write_tab_midi(out / "raw.mid", raw, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)
    write_tab_midi(out / "final.mid", final, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)

    if not args.no_audio:
        for name, ds in (("input", in_dicts), ("raw", raw), ("final", final)):
            write_wav(out / f"{name}.wav",
                      synthesize(to_array(ds), args.bpm,
                                 sf2_path=args.sf2,
                                 source_midi_path=args.input, tempo_changes=tempo_changes))

    # Convert notes to Note objects for rhythm metrics.
    from core.schema import Note
    gt_notes = [Note(t=n.t, dur=n.dur, pitch=n.pitch, s=n.s, f=n.f) for n in notes]
    pred_tabs = [(n["s"], n["f"]) for n in raw]
    # Multi-chunk inference restarts TIME_SHIFT at each chunk's own base time,
    # so the concatenated token stream cannot be decoded as one timeline.
    # Measure rhythm from the chunk-stitched notes instead.
    rhythm = compute_rhythm_metrics(
        pred_tabs, gen_tokens, gt_notes, vocab, pred_notes=raw
    )

    nodes = [DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"]) for n in final]
    tab_text = render_tab(nodes, width=args.width, capo=capo)
    (out / "final_tab.txt").write_text(tab_text + "\n", encoding="utf-8")
    write_progress(
        {
            "stage": "tab",
            "final": True,
            "chunks_done": len(make_inference_chunks(notes, vocab, "input", ChunkConfig(max_tokens=args.max_len))),
            "notes": len(final),
            "capo": capo,
            "auto_capo": str(args.capo).strip().lower() == "auto",
            "message": "Finalising the tab…",
            "tab": tab_text,
        }
    )

    raw_nodes = [DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"]) for n in raw]
    (out / "raw_tab.txt").write_text(render_tab(raw_nodes, width=args.width) + "\n", encoding="utf-8")

    (out / "input.json").write_text(json.dumps(in_dicts, indent=2) + "\n", encoding="utf-8")
    (out / "raw.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    (out / "final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    report = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "beam": args.beam,
        "pitch_constraint": bool(args.pitch_constraint),
        "max_fret": args.max_fret,
        "soft_fret": SOFT_FRET,
        "fret_penalty": args.fret_penalty,
        "capo": capo,
        "capo_feasible_max": capo_max,
        "capo_scores": capo_trials,
        "dropped_chord_tones": proc_stats.get("dropped_chord_tones", []),
        "soundfont": None if args.no_audio else str(args.sf2),
        "comparison": compare(notes, raw),
        "final_comparison": pitch_coverage(notes, final),
        "rhythm": rhythm.to_dict(),
        "postprocessing": proc_stats,
        "final_notes": len(final),
        "raw_difficulty": round(compute_difficulty(nodes), 3),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())