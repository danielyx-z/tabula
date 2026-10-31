"""Pre-tokenised, in-memory chunk store for training.

Chunking is the expensive part of the training pipeline: every epoch used to
re-read each track, rebuild ``Note`` objects and run a token-budget search
(exponential growth + binary search) that re-tokenises the same window many
times.  Measured on the GuitarSet shard that is ~47 ms of CPU per training
batch, against ~37 ms of GPU time per batch — so the data path alone was
roughly halving throughput.

This module tokenises every training chunk **once**, flattens the result into
three ``int32`` arrays (encoder ids, decoder ids, labels) with offsets, and
serves batches by slicing.  Serving cost drops to ~0.4 ms per batch, and the
store is persisted next to the dataset so restarts and evaluation reuse it.

The store keeps dynamic per-batch padding (``collate_fn``), because padding to
a fixed 512 would waste ~25% of encoder and ~42% of decoder compute.

Cache invalidation is content-based: the file name includes a fingerprint of
the shard files (size + mtime), the vocabulary hash, the token cap and the
chunker version, so stale caches are never silently reused.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from corpus.dataset import iter_tracks, records_to_notes, shard_dirs
from model.chunk import ChunkConfig, make_training_chunks
from tablature.tokenizer import Vocab

# Bump whenever chunking semantics change so old caches are discarded.
CHUNK_CACHE_VERSION = 1


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class ChunkStore:
    """Flat token arrays for a split's training chunks.

    ``enc``/``dec``/``lab`` hold every chunk back to back; ``*_off`` are the
    ``n + 1`` boundaries into them.  Chunk ``i`` is ``enc[enc_off[i]:enc_off[i+1]]``.
    """

    enc: np.ndarray
    dec: np.ndarray
    lab: np.ndarray
    enc_off: np.ndarray
    dec_off: np.ndarray
    lab_off: np.ndarray
    track_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.enc_off) - 1

    @property
    def nbytes(self) -> int:
        return self.enc.nbytes + self.dec.nbytes + self.lab.nbytes

    def lengths(self) -> tuple[np.ndarray, np.ndarray]:
        return np.diff(self.enc_off), np.diff(self.dec_off)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            enc=self.enc,
            dec=self.dec,
            lab=self.lab,
            enc_off=self.enc_off,
            dec_off=self.dec_off,
            lab_off=self.lab_off,
            track_ids=self.track_ids,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ChunkStore":
        with np.load(path) as data:
            return cls(
                enc=data["enc"],
                dec=data["dec"],
                lab=data["lab"],
                enc_off=data["enc_off"],
                dec_off=data["dec_off"],
                lab_off=data["lab_off"],
                track_ids=data["track_ids"],
            )


def _cache_key(
    data_dir: str | Path,
    split: str,
    vocab: Vocab,
    max_len: int,
    max_tracks: int,
    max_chunks_per_track: int,
) -> str:
    """Fingerprint everything that can change the chunked token stream."""
    h = hashlib.blake2b(digest_size=8)
    h.update(f"v{CHUNK_CACHE_VERSION}|{split}|{max_len}".encode())
    h.update(f"|{vocab.hash}|{max_tracks}|{max_chunks_per_track}".encode())
    for shard in shard_dirs(data_dir):
        for name in ("notes.npy", "offsets.npy", "tracks.jsonl"):
            f = shard / name
            st = f.stat()
            h.update(f"{shard.name}/{name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _iter_chunks(
    data_dir: str | Path,
    split: str,
    vocab: Vocab,
    max_len: int,
    max_tracks: int,
    max_chunks_per_track: int,
):
    cfg = ChunkConfig(max_tokens=max_len)
    for idx, (meta, notes_arr) in enumerate(iter_tracks(data_dir, split=split)):
        if max_tracks and idx >= max_tracks:
            break
        notes = records_to_notes(notes_arr)
        if not notes:
            continue
        chunks = make_training_chunks(
            notes, vocab, meta.get("id", idx), cfg
        )
        if max_chunks_per_track:
            chunks = chunks[:max_chunks_per_track]
        for chunk in chunks:
            if not chunk.enc_tokens or not chunk.labels:
                continue
            yield meta.get("id", idx), chunk


def build_chunk_store(
    data_dir: str | Path,
    split: str,
    vocab: Vocab,
    max_len: int,
    max_tracks: int = 0,
    max_chunks_per_track: int = 0,
) -> ChunkStore:
    """Tokenise every training chunk for ``split`` into flat arrays."""
    enc_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    enc_lens: list[int] = []
    dec_lens: list[int] = []
    lab_lens: list[int] = []
    track_ids: list[int] = []

    for track_id, chunk in _iter_chunks(
        data_dir, split, vocab, max_len, max_tracks, max_chunks_per_track
    ):
        enc_parts.append(np.asarray(chunk.enc_tokens, dtype=np.int32))
        dec_parts.append(np.asarray(chunk.dec_tokens, dtype=np.int32))
        lab_parts.append(np.asarray(chunk.labels, dtype=np.int32))
        enc_lens.append(len(chunk.enc_tokens))
        dec_lens.append(len(chunk.dec_tokens))
        lab_lens.append(len(chunk.labels))
        track_ids.append(track_id)

    def flat(parts: list[np.ndarray], lens: list[int]):
        arr = (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
        )
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        return arr, offsets

    enc, enc_off = flat(enc_parts, enc_lens)
    dec, dec_off = flat(dec_parts, dec_lens)
    lab, lab_off = flat(lab_parts, lab_lens)

    # Track ids are usually ints (both preprocessors write `"id": n_kept`), but
    # tolerate other JSON scalars rather than crashing on an exotic corpus.
    try:
        ids_arr = np.asarray(track_ids, dtype=np.int64)
    except (TypeError, ValueError):
        ids_arr = np.asarray(track_ids, dtype=str)

    return ChunkStore(
        enc=enc,
        dec=dec,
        lab=lab,
        enc_off=enc_off,
        dec_off=dec_off,
        lab_off=lab_off,
        track_ids=ids_arr,
    )


def get_chunk_store(
    data_dir: str | Path,
    split: str,
    vocab: Vocab,
    max_len: int,
    max_tracks: int = 0,
    max_chunks_per_track: int = 0,
    *,
    use_disk_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> ChunkStore:
    """Return the chunk store for ``split``, building and caching as needed.

    Disk caching is best-effort: a read-only or missing cache directory falls
    back to building in memory so training never fails over a cache write.
    """
    if not use_disk_cache:
        return build_chunk_store(
            data_dir, split, vocab, max_len, max_tracks, max_chunks_per_track
        )

    key = _cache_key(
        data_dir, split, vocab, max_len, max_tracks, max_chunks_per_track
    )
    root = Path(cache_dir) if cache_dir else Path(data_dir) / ".chunk_cache"
    path = root / f"{split}-{max_len}-{key}.npz"

    if path.exists():
        try:
            store = ChunkStore.load(path)
            _log(f"chunk cache: loaded {len(store)} {split} chunks from {path}")
            return store
        except Exception as exc:  # corrupt/partial file -> rebuild
            _log(f"chunk cache: ignoring unreadable {path} ({exc})")

    store = build_chunk_store(
        data_dir, split, vocab, max_len, max_tracks, max_chunks_per_track
    )
    try:
        store.save(path)
        _log(
            f"chunk cache: built {len(store)} {split} chunks "
            f"({store.nbytes / 1e6:.1f} MB tokens) -> {path}"
        )
    except OSError as exc:
        _log(f"chunk cache: could not write {path} ({exc}); keeping in memory")
    return store
