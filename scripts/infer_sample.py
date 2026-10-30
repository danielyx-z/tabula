#!/usr/bin/env python3
"""Infer one 20-note chunk and print its tokens, target, output and tab.

The output is deliberately verbose: it shows exactly what the encoder and
decoder see, the teacher-forcing answer, the generated tokens, and the
post-processed tablature.  It is a debugging/demo companion to
``scripts/eval_checkpoint.py``.

Example:
    .venv/bin/python3 scripts/infer_sample.py \
        --checkpoint runs/pretrain_t5_guitarset/best \
        --track-id 93 --chunk 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model.chunk import ChunkConfig, make_inference_chunks  # noqa: E402
from corpus.dataset import iter_tracks  # noqa: E402
from core.fretboard import STANDARD, pitch_of  # noqa: E402
from core.quantise import quantise_notes  # noqa: E402
from tablature.metrics import (  # noqa: E402
    align_predictions,
    compute_rhythm_metrics,
    decode_predictions,
)
from tablature.postproc import DecodedNote, postprocess  # noqa: E402
from core.schema import Note  # noqa: E402
from tablature.tokenizer import (  # noqa: E402
    EOS_TOKEN,
    PAD_TOKEN,
    Vocab,
    notes_to_target,
)
from preview import build_columns, render_all  # noqa: E402


def records_to_notes(rows) -> list[Note]:
    return [
        Note(
            t=int(r["t"]),
            dur=int(r["dur"]),
            pitch=int(r["pitch"]),
            s=int(r["s"]),
            f=int(r["f"]),
        )
        for r in rows
    ]


def load_track(data_dir: str, split: str, track_id: int | None, ordinal: int):
    for i, (meta, rows) in enumerate(iter_tracks(data_dir, split=split)):
        if track_id is not None:
            if int(meta.get("id", -1)) != track_id:
                continue
        elif i != ordinal:
            continue
        return meta, records_to_notes(rows)
    raise SystemExit(
        f"track {track_id if track_id is not None else ordinal} not found"
    )


def token_rows(ids: list[int], vocab: Vocab) -> str:
    """Compact, aligned token dump: ``id:token`` in up to four columns."""
    cells = [f"{i:>3}:{vocab.id_to_token.get(i, '那些')}" for i in ids]
    width = max((len(c) for c in cells), default=1) + 2
    per_row = max(1, min(4, 118 // width))
    lines = []
    for start in range(0, len(cells), per_row):
        row = cells[start : start + per_row]
        lines.append("  ".join(c.ljust(width) for c in row).rstrip())
    return "\n".join(lines)


def strip_eos(ids: list[int], eos_id: int) -> list[int]:
    if eos_id in ids:
        return ids[: ids.index(eos_id) + 1]
    return ids


def to_array(notes: list[Note] | list[DecodedNote]) -> np.ndarray:
    arr = np.empty(len(notes), dtype=[("t", "<i4"), ("dur", "<i4"),
                                      ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
    for i, n in enumerate(notes):
        arr[i] = (int(n.t), int(n.dur), int(n.pitch), int(n.s), int(n.f))
    return arr


def render(notes: list[Note] | list[DecodedNote], tuning, width: int) -> str:
    if not notes:
        return "(no notes)"
    arr = to_array(notes)
    start = int(arr["t"].min())
    end = int((arr["t"] + np.maximum(arr["dur"], 1)).max()) + 1
    columns = build_columns(arr, start, end)
    return "\n".join(render_all(columns, tuning, width))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/pretrain_t5_guitarset/best"))
    ap.add_argument("--vocab", default=str(ROOT / "vocab/guitarset.json"))
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed_guitarset"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--track-id", type=int, default=93)
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)
    meta, notes = load_track(args.data_dir, args.split, args.track_id, args.ordinal)
    # Sort GT to match the chunker's sort order: (t, s, pitch, f)
    notes = sorted(notes, key=lambda n: (n.t, n.s, n.pitch, n.f))
    # Quantise to training distribution
    notes = quantise_notes(notes)
    chunks = make_inference_chunks(
        notes, vocab, meta.get("id", args.track_id), ChunkConfig(max_tokens=args.max_len)
    )
    if not chunks:
        raise SystemExit("track has no notes")
    if not 0 <= args.chunk < len(chunks):
        raise SystemExit(f"chunk must be 0..{len(chunks) - 1}")
    chunk = chunks[args.chunk]
    start = int(chunk.meta["note_start"])
    end = int(chunk.meta["note_end"])
    gt = notes[start:end]
    context = notes[start - 1] if start > 0 else None
    tuning = tuple(meta.get("tuning", STANDARD))

    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(args.device).eval()
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        model.to(torch.bfloat16)

    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]
    enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=args.device)
    attention = (enc != pad_id).long()
    prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=args.device)
    max_new = min(args.max_len, max(16, 6 * len(gt) + 16))

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
    ):
        generated = model.generate(
            input_ids=enc,
            attention_mask=attention,
            decoder_input_ids=prompt,
            max_new_tokens=max_new,
            num_beams=args.beam,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
    gen_ids = strip_eos(generated[0, prompt.size(1) :].tolist(), eos_id)
    gen_tokens = vocab.decode(gen_ids)
    pred_tabs = decode_predictions(gen_ids, vocab)
    post_notes = postprocess(pred_tabs, gt, mode="full", tuning=tuning)
    post_tabs = [(n.s, n.f) for n in post_notes]

    raw_counts = align_predictions(pred_tabs, gt)
    post_counts = align_predictions(post_tabs, gt)
    rhythm = compute_rhythm_metrics(pred_tabs, gen_tokens, gt, vocab)

    # Target answer tokens, without the prompt/EOS prefix used for training.
    target_strings = notes_to_target(gt, base_t=int(chunk.meta["base_t"]))
    target_ids = vocab.encode(target_strings) + [eos_id]
    prompt_strings = [vocab.id_to_token[i] for i in chunk.dec_tokens]

    print("=" * 100)
    print("SAMPLE")
    print("=" * 100)
    print(f"split:       {args.split}")
    print(f"track id:    {meta.get('id')}   song: {meta.get('song')}")
    print(f"track notes: {len(notes)}   chunk: {args.chunk}/{len(chunks) - 1}   "
          f"notes [{start}:{end}] ({len(gt)} notes)")
    print(f"checkpoint:  {args.checkpoint}")
    print(f"device:      {args.device}   bf16: {use_bf16}   beam: {args.beam}")
    if context is not None:
        print(f"context:     t={context.t} dur={context.dur} pitch={context.pitch} "
              f"tab=TAB<{context.s},{context.f}>")
    else:
        print("context:     none")

    print()
    print("=" * 100)
    print("1. TOKENS FED TO THE MODEL")
    print("=" * 100)
    print(f"Encoder input: {len(chunk.enc_tokens)} tokens "
          f"(NOTE_ON/NOTE_OFF/TIME_SHIFT, context + {len(gt)} target notes)")
    print(token_rows(chunk.enc_tokens, vocab))
    print()
    print(f"Decoder prompt: {len(chunk.dec_tokens)} tokens "
          f"(start symbol + context TAB; generation begins here)")
    print(token_rows(chunk.dec_tokens, vocab))

    print()
    print("=" * 100)
    print("2. ANSWER (teacher-forcing target)")
    print("=" * 100)
    print(f"Target tokens: {len(target_ids)} tokens "
          f"({len(target_strings)} target tokens + EOS)")
    print(token_rows(target_ids, vocab))
    print()
    print(f"Target tablature ({len(gt)} notes):")
    print(render(gt, tuning, args.width))

    print()
    print("=" * 100)
    print("3. MODEL OUTPUT")
    print("=" * 100)
    print(f"Generated tokens: {len(gen_ids)} tokens")
    print(token_rows(gen_ids, vocab))
    print()
    print(f"Generated TAB events ({len(pred_tabs)}): {pred_tabs}")
    print()
    print("Per-note comparison (raw and post-processed against the answer):")
    print(f"{'#':>3} {'t':>7} {'pitch':>5} {'answer':>10} {'raw':>10} {'post':>10}")
    for i, gt_n in enumerate(gt):
        raw_sf = pred_tabs[i] if i < len(pred_tabs) else None
        post_n = post_notes[i]
        mark = "" if raw_sf == (gt_n.s, gt_n.f) else "  <- raw differs"
        print(
            f"{i:>3} {gt_n.t:>7} {gt_n.pitch:>5} "
            f"{f'({gt_n.s},{gt_n.f})':>10} "
            f"{str(raw_sf):>10} {f'({post_n.s},{post_n.f})':>10}{mark}"
        )

    print()
    print(f"Raw tablature ({len(pred_tabs)} predicted notes):")
    raw_notes: list[DecodedNote] = []
    for i, tab in enumerate(pred_tabs):
        if i < len(gt):
            t, dur, pitch = gt[i].t, gt[i].dur, gt[i].pitch
        else:
            last = gt[-1]
            t = last.t + (i - len(gt) + 1) * 60
            dur, pitch = 60, pitch_of(tab[0], tab[1], tuning)
        raw_notes.append(DecodedNote(t=t, dur=dur, pitch=pitch, s=tab[0], f=tab[1]))
    print(render(raw_notes, tuning, args.width))
    print()
    print(f"Post-processed tablature ({len(post_notes)} notes):")
    print(render(post_notes, tuning, args.width))

    print()
    print("=" * 100)
    print("4. SAMPLE ACCURACY")
    print("=" * 100)
    print(f"Raw pitch: {raw_counts.pitch_accuracy():.4f}   "
          f"raw tab: {raw_counts.tab_accuracy():.4f}   "
          f"raw valid: {raw_counts.validity():.4f}")
    print(f"Post pitch: {post_counts.pitch_accuracy():.4f}   "
          f"post tab: {post_counts.tab_accuracy():.4f}   "
          f"post valid: {post_counts.validity():.4f}")
    print(f"Raw exact: {sum(1 for i, t in enumerate(pred_tabs) if i < len(gt) and t == (gt[i].s, gt[i].f))}/{len(gt)}   "
          f"Post exact: {sum(1 for i, t in enumerate(post_tabs) if t == (gt[i].s, gt[i].f))}/{len(gt)}")
    print()
    print("Rhythm metrics (raw):")
    print(f"  Matched/Missing/Extra: {rhythm.matched}/{rhythm.missing}/{rhythm.extra}")
    print(f"  Onset Hit @20/60/120: {rhythm.onset_hit_rate(20):.4f}/{rhythm.onset_hit_rate(60):.4f}/{rhythm.onset_hit_rate(120):.4f}")
    print(f"  Median Onset Error:   {rhythm.median_onset_error():.1f} ticks")
    print(f"  Median IOI Error:     {rhythm.median_ioi_error():.1f} ticks")
    print(f"  Median Dur Error:     {rhythm.median_dur_error():.1f} ticks")


if __name__ == "__main__":
    main()