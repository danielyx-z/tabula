#!/usr/bin/env python3
"""Honest MIDI -> tab run.

input.mid  : the parsed input notes, unchanged.
raw.mid    : the model's own predictions.  Onsets come from the generated
             TIME_SHIFT tokens, pitches from the generated TAB tokens.
             No ground-truth timing or pitch is used anywhere.
final.mid  : the same raw notes after overlap repair.  Overlap repair may
             move a note to another position *of its own predicted pitch*.
             It never substitutes a ground-truth pitch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from preview import synthesize, write_smf, write_wav
import pretty_midi
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search
from fretwire.fretboard import STANDARD, pitch_of
from fretwire.metrics import compute_difficulty
from fretwire.parse.midi import parse_midi
from fretwire.playability import cap_simultaneous, fix_chord_spans, limit_jumps
from fretwire.postproc import DecodedNote, repair_overlaps
from fretwire.render import render_tab
from fretwire.stream import decode_tab_stream
from fretwire.tokenizer import EOS_TOKEN, PAD_TOKEN, Vocab

IN_MIDI = "/tmp/lover_girl.mid"
OUT = "/tmp/honest_out"
os.makedirs(OUT, exist_ok=True)

USE_PITCH_CONSTRAINT = False  # True = force each TAB pitch to match an encoder NOTE_ON

NOTE_DTYPE = [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]


def to_arr(notes: list[dict]) -> np.ndarray:
    arr = np.empty(len(notes), dtype=NOTE_DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (int(n["t"]), int(n["dur"]), int(n["pitch"]), int(n["s"]), int(n["f"]))
    return arr


def write_tab_midi(path: str, notes: list[dict], bpm: float = 120.0, ppq: int = 960) -> None:
    """Write one MIDI track per guitar string so the fingering is stored.

    A normal MIDI file only holds pitch and time, so two tabs with the same
    notes but different fingerings would be byte-identical.  Putting each
    string on its own track keeps the fingering in the file.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tracks = {}
    for s in range(1, 7):
        inst = pretty_midi.Instrument(program=25, name=f"string {s}")
        pm.instruments.append(inst)
        tracks[s] = inst

    spb = 60.0 / bpm
    for n in notes:
        start = n["t"] / ppq * spb
        end = (n["t"] + max(1, n["dur"])) / ppq * spb
        tracks[int(n["s"])].notes.append(
            pretty_midi.Note(
                velocity=90, pitch=int(n["pitch"]), start=start, end=end
            )
        )
    pm.write(path)


def main() -> None:
    vocab = Vocab.load("vocab_v1_full.json")
    model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
    model.to("cuda").eval()
    model.to(torch.bfloat16)

    notes = parse_midi(IN_MIDI).notes
    notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
    print(f"input notes: {len(notes)}")

    chunks = make_inference_chunks(notes, vocab, "lover_girl", ChunkConfig(max_tokens=512))
    print(f"chunks: {len(chunks)}")

    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    raw: list[dict] = []
    for ci, chunk in enumerate(chunks):
        enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device="cuda")
        attn = (enc != pad_id).long()
        prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device="cuda")
        max_new = min(512, max(16, 8 * chunk.meta["num_notes"] + 16))

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = constrained_beam_search(
                model,
                enc,
                vocab,
                num_beams=4,
                max_length=512,
                use_pitch_constraint=USE_PITCH_CONSTRAINT,
                decoder_input_ids=prompt,
                attention_mask=attn,
                max_new_tokens=max_new,
                prompt_tabs=0,
            )

        ids = gen[0, prompt.size(1):].tolist()
        toks = vocab.decode(ids)
        chunk_notes = decode_tab_stream(toks, base_t=chunk.meta["base_t"])

        # Generation starts after the prompt, which already carries the context
        # TAB.  If the model re-emits that same note at the context tick, drop it.
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

        print(
            f"  chunk {ci + 1}/{len(chunks)}: expected {chunk.meta['num_notes']}, "
            f"got {len(chunk_notes)} notes"
        )
        raw.extend(chunk_notes)

    print(f"\nraw predicted notes: {len(raw)}")

    # Post-processing: pgitch-preserving playability fixes, then a final
    # overlap/duplicate pass so the result is always physically playable.
    nodes = [
        DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in raw
    ]
    nodes, dropped = cap_simultaneous(nodes)
    nodes = repair_overlaps(nodes)  # no gt_notes -> pitch is preserved
    nodes, moved = fix_chord_spans(nodes)
    nodes, jumped = limit_jumps(nodes)
    nodes, dropped2 = cap_simultaneous(nodes)
    nodes = repair_overlaps(nodes)
    final = [
        {"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in nodes
    ]

    print(
        f"final notes: {len(final)} "
        f"(dropped {dropped + dropped2} impossible duplicates, "
        f"re-fingered {moved + jumped} notes)"
    )

    # Diagnostics: how far apart are the model's chords?
    by_t: dict[int, list[int]] = {}
    for n in raw:
        by_t.setdefault(n["t"], []).append(n["f"])
    big = [max(v) - min(v) for v in by_t.values() if len(v) > 1 and max(v) - min(v) > 4]
    print(f"chords with >4-fret span: {len(big)} (max {max(big) if big else 0})")

    # ---------------- write MIDI + WAV, each from its own note list ----------------
    in_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in notes]

    write_smf(f"{OUT}/input.mid", in_dicts)
    write_tab_midi(f"{OUT}/raw.mid", raw)
    write_tab_midi(f"{OUT}/final.mid", final)
    import json
    json.dump(raw, open(f"{OUT}/raw_notes.json", "w"))
    json.dump(final, open(f"{OUT}/final_notes.json", "w"))
    print(f"\nwrote {OUT}/input.mid ({len(in_dicts)} notes)")
    print(f"wrote {OUT}/raw.mid ({len(raw)} notes)")
    print(f"wrote {OUT}/final.mid ({len(final)} notes)")

    for name, ds in (("input", in_dicts), ("raw", raw), ("final", final)):
        samples = synthesize(to_arr(ds), 120.0)
        write_wav(f"{OUT}/{name}.wav", samples)
        print(f"wrote {OUT}/{name}.wav ({len(samples) / 44100:.1f}s)")

    # ---------------- honest comparison ----------------
    print("\n=== comparison (greedy nearest-onset alignment) ===")
    used: set[int] = set()
    pitch_hits = tab_hits = aligned = 0
    for gt in notes:
        best_j, best_dt = None, None
        for j, p in enumerate(raw):
            if j in used:
                continue
            dt = abs(p["t"] - gt.t)
            if best_dt is None or dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is None or best_dt is None or best_dt > 480:
            continue
        used.add(best_j)
        p = raw[best_j]
        aligned += 1
        if p["pitch"] == gt.pitch:
            pitch_hits += 1
        if p["s"] == gt.s and p["f"] == gt.f:
            tab_hits += 1

    print(f"aligned notes: {aligned}/{len(notes)}")
    print(f"pitch correct: {pitch_hits}/{len(notes)} ({pitch_hits / len(notes) * 100:.1f}%)")
    print(f"tab exact    : {tab_hits}/{len(notes)} ({tab_hits / len(notes) * 100:.1f}%)")
    print(f"raw difficulty: {compute_difficulty(nodes):.2f}")

    tab_text = render_tab(nodes, width=100)
    with open(f"{OUT}/final_tab.txt", "w") as fh:
        fh.write(tab_text + "\n")
    print(f"\nwrote {OUT}/final_tab.txt")
    print(tab_text[:4000])


if __name__ == "__main__":
    main()
