"""Chunking that follows the Fretting-Transformer paper.

* Training: sequences are filled up to ``max_tokens`` (512) with notes.
* Inference: chunks of about 20 notes.  The last note of the previous chunk
  is repeated at the start of both the encoder and the decoder, which is the
  paper's context carry-over rule.
* Chords (notes with the same onset) are never split across a chunk boundary.
  A chunk may therefore contain a few notes more or fewer than 20.

The decoder is prompted with the previous note's TAB token only.  The encoder
keeps the previous note's complete ON/OFF events, so the model still sees its
duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.schema import Note
from tablature.tokenizer import EOS_TOKEN, Vocab, notes_to_events, notes_to_target, tab_token

# Paper values.
K = 20                 # inference window size, in notes
CONTEXT_NOTES = 1      # previous chunk's last note
L = CONTEXT_NOTES      # backwards-compatible alias
MAX_TOKENS = 512       # paper training sequence length


@dataclass(frozen=True)
class Chunk:
    """One encoder/decoder example."""

    enc_tokens: list[int]          # encoder input ids
    dec_tokens: list[int]          # teacher-forcing input or inference prompt
    labels: list[int]              # -100 where the loss must be ignored
    meta: dict[str, Any]           # track id, note window, prompt length, ...


@dataclass(frozen=True)
class ChunkConfig:
    k: int = K
    context_notes: int = CONTEXT_NOTES
    max_tokens: int = MAX_TOKENS
    # 0 = grow training windows until the token cap; >0 is useful for debugging.
    training_chunk_notes: int = 0


CFG = ChunkConfig()


def _onset_bounds(notes: list[Note], index: int) -> tuple[int, int]:
    """Return the half-open index range of notes sharing ``notes[index].t``."""
    t = notes[index].t
    lo = index
    while lo > 0 and notes[lo - 1].t == t:
        lo -= 1
    hi = index + 1
    while hi < len(notes) and notes[hi].t == t:
        hi += 1
    return lo, hi


def _avoid_chord_split(notes: list[Note], start: int, end: int, total: int) -> int:
    """Move a proposed chunk boundary so an onset group is not cut in half.

    If the chord straddles ``end``, end the chunk *before* the chord when that
    leaves at least one note, otherwise extend the chunk through the chord.
    """
    if end <= start or end >= total:
        return end
    if notes[end - 1].t != notes[end].t:
        return end
    group_start, group_end = _onset_bounds(notes, end)
    if group_start > start:
        return group_start
    return group_end


def _build_chunk(
    notes: list[Note],
    start: int,
    end: int,
    vocab: Vocab,
    track_id: Any,
    cfg: ChunkConfig,
    *,
    training: bool,
) -> Chunk:
    current = notes[start:end]
    context = (
        notes[start - 1]
        if start > 0 and cfg.context_notes > 0
        else None
    )
    enc_notes = ([context] if context is not None else []) + current
    base_t = context.t if context is not None else current[0].t

    enc_ids = vocab.encode(notes_to_events(enc_notes, base_t=base_t))
    target_ids = vocab.encode(notes_to_target(current, base_t=base_t))

    eos_id = vocab[EOS_TOKEN]
    context_ids = (
        [vocab[tab_token(context.s, context.f)]] if context is not None else []
    )

    if training:
        dec_tokens = [eos_id] + context_ids + target_ids
        labels = [-100] * len(context_ids) + target_ids + [eos_id]
    else:
        # Generation prompt: start symbol plus the previous note's TAB.
        dec_tokens = [eos_id] + context_ids
        labels = []

    return Chunk(
        enc_tokens=enc_ids,
        dec_tokens=dec_tokens,
        labels=labels,
        meta={
            "track_id": track_id,
            "note_start": start,
            "note_end": end,
            "num_notes": len(current),
            "base_t": base_t,
            "has_context": context is not None,
            "context_s": context.s if context is not None else 0,
            "context_f": context.f if context is not None else 0,
            "prompt_len": len(dec_tokens),
        },
    )


def _fits(
    notes: list[Note],
    start: int,
    end: int,
    vocab: Vocab,
    track_id: Any,
    cfg: ChunkConfig,
) -> bool:
    chunk = _build_chunk(notes, start, end, vocab, track_id, cfg, training=True)
    return (
        len(chunk.enc_tokens) <= cfg.max_tokens
        and len(chunk.dec_tokens) <= cfg.max_tokens
    )


def make_training_chunks(
    notes: list[Note],
    vocab: Vocab,
    track_id: Any,
    cfg: ChunkConfig = CFG,
) -> list[Chunk]:
    """Fill 512-token training examples with as many notes as fit."""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n.t, n.s, n.pitch, n.f))
    total = len(notes)
    chunks: list[Chunk] = []
    start = 0

    while start < total:
        if cfg.training_chunk_notes > 0:
            end = min(start + cfg.training_chunk_notes, total)
            end = _avoid_chord_split(notes, start, end, total)
            # Re-check fit after chord-boundary adjustment.
            if not _fits(notes, start, end, vocab, track_id, cfg):
                # Fall back to the pre-adjustment boundary (which we know fit).
                end = min(start + cfg.training_chunk_notes, total)
            chunks.append(
                _build_chunk(notes, start, end, vocab, track_id, cfg, training=True)
            )
            start = end
            continue

        # A single note must fit; otherwise skip it rather than loop forever.
        if not _fits(notes, start, start + 1, vocab, track_id, cfg):
            start += 1
            continue

        # Exponential growth followed by a binary search for the last end that
        # still fits.  This avoids re-tokenising the window for every note.
        step = 1
        best = start + 1
        while start + step <= total and _fits(
            notes, start, start + step, vocab, track_id, cfg
        ):
            best = start + step
            step *= 2

        lo = best + 1
        hi = min(start + step, total)
        while lo <= hi:
            mid = (lo + hi) // 2
            if _fits(notes, start, mid, vocab, track_id, cfg):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # A token boundary can fall inside a strummed chord.  Move it to the
        # chord edge so the decoder never learns half a chord.
        best = _avoid_chord_split(notes, start, best, total)
        # Re-check fit after chord-boundary adjustment; binary search ensures
        # the pre-adjustment boundary fits.
        if not _fits(notes, start, best, vocab, track_id, cfg):
            # Find the last fitting boundary <= best.
            for end in range(best, start, -1):
                if _fits(notes, start, end, vocab, track_id, cfg):
                    best = end
                    break
        chunks.append(
            _build_chunk(notes, start, best, vocab, track_id, cfg, training=True)
        )
        start = best

    return chunks


def make_inference_chunks(
    notes: list[Note],
    vocab: Vocab,
    track_id: Any,
    cfg: ChunkConfig = CFG,
) -> list[Chunk]:
    """Split a track into 20-note chunks with one note of context carry-over."""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n.t, n.s, n.pitch, n.f))
    total = len(notes)
    chunks: list[Chunk] = []
    start = 0
    while start < total:
        end = min(start + max(1, cfg.k), total)
        end = _avoid_chord_split(notes, start, end, total)
        chunk = _build_chunk(notes, start, end, vocab, track_id, cfg, training=False)
        # A pathological 20-note chunk can exceed the encoder budget; the
        # paper's chunks normally fit comfortably, so truncation is a safety net.
        if len(chunk.enc_tokens) > cfg.max_tokens:
            chunk = Chunk(
                enc_tokens=chunk.enc_tokens[: cfg.max_tokens],
                dec_tokens=chunk.dec_tokens,
                labels=[],
                meta=chunk.meta,
            )
        chunks.append(chunk)
        start = end
    return chunks


