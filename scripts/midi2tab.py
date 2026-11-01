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
    raw_tab.txt  ASCII tab of the model's uncorrected fingering
    final_tab.txt ASCII tab of the corrected fingering
    input.json / raw.json / final.json
                 machine-readable note lists for comparison
    report.json  note counts, alignment metrics, and rhythm metrics

MIDI cannot store a fingering, so each output is written as one track per
string.  That keeps the string/fret assignment in the file and lets raw and
final differ even when they sound the same.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from transformers import T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from audio.synth import synthesize  # noqa: E402
from audio.wav import write_wav  # noqa: E402
from model.chunk import ChunkConfig, make_inference_chunks  # noqa: E402
from model.decode import constrained_beam_search  # noqa: E402
from core.fretboard import STANDARD  # noqa: E402
from core.quantise import quantise_notes, ticks_to_seconds  # noqa: E402
from tablature.metrics import compute_difficulty, compute_rhythm_metrics  # noqa: E402
from corpus.parse.midi import parse_midi  # noqa: E402
from tablature.playability import (  # noqa: E402
    cap_simultaneous,
    fix_chord_spans,
    limit_jumps,
)
from tablature.postproc import DecodedNote, repair_overlaps  # noqa: E402
from tablature.render import render_tab  # noqa: E402
from tablature.stream import decode_tab_stream  # noqa: E402
from tablature.tokenizer import PAD_TOKEN, Vocab, tab_token  # noqa: E402

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
        inst = pretty_midi.Instrument(program=25, name=name)
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


def generate(model, vocab, notes, beam: int, pitch_constraint: bool, max_len: int, device: str,
             max_fret: int | None = 12, fret_penalty: float = 0.0) -> list[dict]:
    # Quantise input notes to training distribution (20-tick grid, 40-tick chord window).
    notes = quantise_notes(notes)
    chunks = make_inference_chunks(notes, vocab, "input", ChunkConfig(max_tokens=max_len))
    pad_id = vocab[PAD_TOKEN]
    raw: list[dict] = []
    all_tokens: list[str] = []

    prev_generated: dict | None = None
    for chunk in chunks:
        enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
        attn = (enc != pad_id).long()
        # A pitch-only input has no TAB token for the context note.  The paper's
        # carry-over context is the *previous generated* note, so replace the
        # decoder prompt's context token after the first chunk.
        dec_tokens = list(chunk.dec_tokens)
        if chunk.meta["has_context"] and prev_generated is not None:
            dec_tokens[-1] = vocab[tab_token(prev_generated["s"], prev_generated["f"])]
        prompt = torch.tensor([dec_tokens], dtype=torch.long, device=device)
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

        start = chunk.meta["note_start"]
        if start > 0 and chunk_notes and prev_generated is not None:
            first = chunk_notes[0]
            if (
                first["s"] == prev_generated["s"]
                and first["f"] == prev_generated["f"]
                and first["t"] == chunk.meta["base_t"]
            ):
                chunk_notes = chunk_notes[1:]

        raw.extend(chunk_notes)
        if chunk_notes:
            prev_generated = chunk_notes[-1]

    return raw, all_tokens


def playability_pass(raw: list[dict]) -> tuple[list[dict], dict]:
    nodes = [
        DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in raw
    ]
    nodes, dropped_a = cap_simultaneous(nodes)
    nodes = repair_overlaps(nodes)  # no ground truth: pitch is preserved
    nodes, chord_moves = fix_chord_spans(nodes)
    nodes, jump_moves = limit_jumps(nodes)
    nodes = repair_overlaps(nodes)
    nodes, dropped_b = cap_simultaneous(nodes)
    final = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in nodes]
    stats = {
        "dropped_impossible": dropped_a + dropped_b,
        "refingered_chord_span": chord_moves,
        "refingered_jump": jump_moves,
    }
    return final, stats


def compare(input_notes, raw) -> dict:
    """Greedy nearest-onset alignment, purely descriptive."""
    used: set[int] = set()
    aligned = pitch_hits = tab_hits = 0
    has_strings = any(int(gt.s) > 0 for gt in input_notes)
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
        if has_strings:
            tab_hits += int(p["s"] == gt.s and p["f"] == gt.f)
    n = len(input_notes)
    return {
        "input_notes": n,
        "raw_notes": len(raw),
        "aligned": aligned,
        "pitch_correct": pitch_hits,
        "pitch_accuracy": round(pitch_hits / n, 4) if n else 0.0,
        "input_has_strings": has_strings,
        "tab_exact": tab_hits if has_strings else None,
        "tab_accuracy": round(tab_hits / n, 4) if n and has_strings else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input MIDI file")
    ap.add_argument("--checkpoint", default="runs/finetune_t5_guitarset/best")
    ap.add_argument("--vocab", default="vocab/canonical.json")
    ap.add_argument("--out-dir", default="out/midi2tab")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument(
        "--pitch-constraint",
        dest="pitch_constraint",
        action="store_true",
        default=True,
        help="force every TAB pitch to match an encoder NOTE_ON",
    )
    ap.add_argument(
        "--no-pitch-constraint",
        dest="pitch_constraint",
        action="store_false",
        help="disable pitch/low-position masking",
    )
    ap.add_argument(
        "--max-fret",
        type=int,
        default=12,
        help="prefer positions at or below this fret when possible; "
             "-1 disables the low-position rule",
    )
    ap.add_argument(
        "--fret-penalty",
        type=float,
        default=0.0,
        help="soft penalty (-lambda*fret) among allowed positions",
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

    raw, gen_tokens = generate(
        model,
        vocab,
        notes,
        args.beam,
        args.pitch_constraint,
        args.max_len,
        device,
        max_fret=None if args.max_fret < 0 else args.max_fret,
        fret_penalty=args.fret_penalty,
    )
    final, proc_stats = playability_pass(raw)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Preserve the input file's tempo map so output timing matches the source.
    _pm_in = pretty_midi.PrettyMIDI(str(args.input))
    tc_in = _pm_in.get_tempo_changes()
    tempo_changes = list(zip(tc_in[0].tolist(), tc_in[1].tolist())) if tc_in[1].size else None

    in_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in notes]
    write_tab_midi(out / "input.mid", in_dicts, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)
    write_tab_midi(out / "raw.mid", raw, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)
    write_tab_midi(out / "final.mid", final, bpm=args.bpm,
                   source_midi_path=args.input, tempo_changes=tempo_changes)

    for name, ds in (("input", in_dicts), ("raw", raw), ("final", final)):
        write_wav(out / f"{name}.wav",
                  synthesize(to_array(ds), args.bpm, tempo_changes=tempo_changes,
                             source_midi_path=args.input))

    # Convert notes to Note objects for rhythm metrics.
    from core.schema import Note
    gt_notes = [Note(t=n.t, dur=n.dur, pitch=n.pitch, s=n.s, f=n.f) for n in notes]
    pred_tabs = [(n["s"], n["f"]) for n in raw]
    rhythm = compute_rhythm_metrics(pred_tabs, gen_tokens, gt_notes, vocab, pred_notes=raw)

    nodes = [DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"]) for n in final]
    tab_text = render_tab(nodes, width=args.width)
    (out / "final_tab.txt").write_text(tab_text + "\n", encoding="utf-8")

    raw_nodes = [DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"]) for n in raw]
    (out / "raw_tab.txt").write_text(render_tab(raw_nodes, width=args.width) + "\n", encoding="utf-8")

    # Machine-readable note lists: each is derived from its own stage.
    (out / "input.json").write_text(json.dumps(in_dicts, indent=2) + "\n", encoding="utf-8")
    (out / "raw.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    (out / "final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    report = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "beam": args.beam,
        "pitch_constraint": bool(args.pitch_constraint),
        "max_fret": args.max_fret,
        "fret_penalty": args.fret_penalty,
        "comparison": compare(notes, raw),
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