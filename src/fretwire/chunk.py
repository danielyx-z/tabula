"""Chunking that follows the Fretting-Transformer paper.

* Training: sequences are filled up to ``max_tokens`` (512) with notes.
* Inference: chunks of 20 notes.  The last note of the previous chunk is
  repeated at the start of both the encoder and the decoder, which is the
  paper's context carry-over rule.

The decoder is prompted with the previous note's TAB token only.  The encoder
keeps the previous note's complete ON/OFF events, so the model still sees its
duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import Note
from .tokenizer import EOS_TOKEN, Vocab, notes_to_events, notes_to_target, tab_token

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

    notes = sorted(notes, key=lambda n: (n.t, n.pitch, n.s, n.f))
    total = len(notes)
    chunks: list[Chunk] = []
    start = 0

    while start < total:
        if cfg.training_chunk_notes > 0:
            end = min(start + cfg.training_chunk_notes, total)
            chunk = _build_chunk(
                notes, start, end, vocab, track_id, cfg, training=True
            )
            if (
                len(chunk.enc_tokens) <= cfg.max_tokens
                and len(chunk.dec_tokens) <= cfg.max_tokens
            ):
                chunks.append(chunk)
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

    notes = sorted(notes, key=lambda n: (n.t, n.pitch, n.s, n.f))
    chunks: list[Chunk] = []
    for start in range(0, len(notes), max(1, cfg.k)):
        end = min(start + cfg.k, len(notes))
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
    return chunks


def make_chunks_from_notes(
    notes: list[Note],
    vocab: Vocab,
    track_id: Any,
    cfg: ChunkConfig = CFG,
) -> list[Chunk]:
    """Backwards-compatible alias for :func:`make_training_chunks`."""
    return make_training_chunks(notes, vocab, track_id, cfg)
