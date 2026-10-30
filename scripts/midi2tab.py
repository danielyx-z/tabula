#!/usr/bin/env python3
"""MIDI -> tablature, with honest artefacts.

    python scripts/midi2tab.py song.mid \
        --checkpoint runs/pretrain_t5_v5/best \
        --out-dir out/song

Writes, all derived independently:

    input.mid    the parsed input notes
    raw.mid      the model's own output: onsets from generated TIME_SHIFT
                 tokens, pitches from generated TAB tokens
    final.mid    raw after pitch-preserving playability correction
    *.wav        synthesis of each of the above
    final_tab.txt ASCII tab of the corrected fingering
    report.json  note counts and alignment metrics

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

from preview import synthesize, write_wav  # noqa: E402
from fretwire.chunk import ChunkConfig, make_inference_chunks  # noqa: E402
from fretwire.decode import constrained_beam_search  # noqa: E402
from fretwire.fretboard import STANDARD  # noqa: E402
from fretwire.metrics import compute_difficulty  # noqa: E402
from fretwire.parse.midi import parse_midi  # noqa: E402
from fretwire.playability import (  # noqa: E402
    cap_simultaneous,
    fix_chord_spans,
    limit_jumps,
)
from fretwire.postproc import DecodedNote, repair_overlaps  # noqa: E402
from fretwire.render import render_tab  # noqa: E402
from fretwire.stream import decode_tab_stream  # noqa: E402
from fretwire.tokenizer import PAD_TOKEN, Vocab  # noqa: E402

NOTE_DTYPE = [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]


def to_array(notes: list[dict]) -> np.ndarray:
    arr = np.empty(len(notes), dtype=NOTE_DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (int(n["t"]), int(n["dur"]), int(n["pitch"]), int(n["s"]), int(n["f"]))
    return arr


def write_tab_midi(path: Path, notes: list[dict], bpm: float = 120.0, ppq: int = 960) -> None:
    """One track per string, so the fingering is stored in the file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tracks = {}
    for s in [0, *range(1, 7)]:
        name = "pitch only" if s == 0 else f"string {s}"
        inst = pretty_midi.Instrument(program=25, name=name)
        pm.instruments.append(inst)
        tracks[s] = inst
    spb = 60.0 / bpm
    for n in notes:
        s = int(n["s"])
        if s not in tracks:
            s = 0  # input MIDI has no string information
        tracks[s].notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=int(n["pitch"]),
                start=n["t"] / ppq * spb,
                end=(n["t"] + max(1, n["dur"])) / ppq * spb,
            )
        )
    pm.write(str(path))


def generate(model, vocab, notes, beam: int, pitch_constraint: bool, max_len: int, device: str) -> list[dict]:
    chunks = make_inference_chunks(notes, vocab, "input", ChunkConfig(max_tokens=max_len))
    pad_id = vocab[PAD_TOKEN]
    raw: list[dict] = []

    for chunk in chunks:
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
            )

        tokens = vocab.decode(gen[0, prompt.size(1) :].tolist())
        chunk_notes = decode_tab_stream(tokens, base_t=chunk.meta["base_t"])

        start = chunk.meta["note_start"]
        if start > 0 and chunk_notes:
            ctx = notes[start - 1]
            first = chunk_notes[0]
            if (
                first["s"] == ctx.s
                and first["f"] == ctx.f
                and first["t"] == chunk.meta["base_t"]
            ):
                chunk_notes = chunk_notes[1:]

        raw.extend(chunk_notes)

    return raw


def playability_pass(raw: list[dict]) -> tuple[list[dict], dict]:
    nodes = [
        DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in raw
    ]
    nodes, dropped_a = cap_simultaneous(nodes)
    nodes = repair_overlaps(nodes)  # no ground truth: pitch is preserved
    nodes, chord_moves = fix_chord_spans(nodes)
    nodes, jump_moves = limit_jumps(nodes)
    nodes, dropped_b = cap_simultaneous(nodes)
    nodes = repair_overlaps(nodes)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input MIDI file")
    ap.add_argument("--checkpoint", default="runs/pretrain_t5_v5/best")
    ap.add_argument("--vocab", default="vocab_v1_full.json")
    ap.add_argument("--out-dir", default="out/midi2tab")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument(
        "--pitch-constraint",
        action="store_true",
        help="force every TAB pitch to match an encoder NOTE_ON",
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

    raw = generate(model, vocab, notes, args.beam, args.pitch_constraint, args.max_len, device)
    final, proc_stats = playability_pass(raw)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    in_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in notes]
    write_tab_midi(out / "input.mid", in_dicts, bpm=args.bpm)
    write_tab_midi(out / "raw.mid", raw, bpm=args.bpm)
    write_tab_midi(out / "final.mid", final, bpm=args.bpm)

    for name, ds in (("input", in_dicts), ("raw", raw), ("final", final)):
        write_wav(out / f"{name}.wav", synthesize(to_array(ds), args.bpm))

    nodes = [DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"]) for n in final]
    tab_text = render_tab(nodes, width=args.width)
    (out / "final_tab.txt").write_text(tab_text + "\n", encoding="utf-8")

    report = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "beam": args.beam,
        "pitch_constraint": bool(args.pitch_constraint),
        "comparison": compare(notes, raw),
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
