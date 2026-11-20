"""Sharded, memory-mappable storage for canonical note events.

Layout::

    data/processed_guitarset/
        manifest.json
        shard-00000/
            notes.npy      structured array, 11 bytes/note, zero-copy mmap
            offsets.npy    int64[n_tracks + 1] -> slice into notes.npy
            tracks.jsonl   one metadata object per track, same order as offsets

The structured dtype is the point: a training loop memory-maps the shard and
slices per track without materialising Python objects::

    notes = np.load("shard-00000/notes.npy", mmap_mode="r")
    offsets = np.load("shard-00000/offsets.npy")
    track = notes[offsets[i]:offsets[i + 1]]

Notes are sorted by (onset, string) and carry (t, dur, pitch, s, f).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from core.schema import Note

NOTE_DTYPE = np.dtype(
    [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]
)


def records_to_notes(notes_arr) -> list[Note]:
    """Convert a structured ``notes.npy`` slice into ``Note`` objects."""
    return [
        Note(
            t=int(row["t"]),
            dur=int(row["dur"]),
            pitch=int(row["pitch"]),
            s=int(row["s"]),
            f=int(row["f"]),
        )
        for row in notes_arr
    ]


def notes_to_array(notes) -> np.ndarray:
    arr = np.empty(len(notes), dtype=NOTE_DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (n.t, n.dur, n.pitch, n.s, n.f)
    return arr


class ShardWriter:
    """Append tracks, flush to disk every ``max_notes`` notes."""

    def __init__(self, root: str | Path, max_notes: int = 1_000_000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_notes = max_notes
        self.shard_index = len(list(self.root.glob("shard-*")))
        self._reset()

    def _reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._offsets: list[int] = [0]
        self._metas: list[dict] = []

    @property
    def total_notes(self) -> int:
        return self._offsets[-1]

    def add(self, meta: dict, notes: np.ndarray) -> None:
        self._chunks.append(notes)
        self._offsets.append(self._offsets[-1] + len(notes))
        self._metas.append(meta)
        if self._offsets[-1] >= self.max_notes:
            self.flush()

    def flush(self) -> None:
        if not self._metas:
            return
        out = self.root / f"shard-{self.shard_index:05d}"
        out.mkdir(parents=True, exist_ok=True)
        arr = np.concatenate(self._chunks) if self._chunks else np.empty(0, NOTE_DTYPE)
        np.save(out / "notes.npy", arr)
        np.save(out / "offsets.npy", np.asarray(self._offsets, dtype=np.int64))
        with open(out / "tracks.jsonl", "w", encoding="utf-8") as fh:
            for meta in self._metas:
                fh.write(json.dumps(meta, separators=(",", ":")) + "\n")
        self.shard_index += 1
        self._reset()
        del arr

    def close(self) -> None:
        self.flush()


def shard_dirs(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("shard-*") if p.is_dir())


def load_shard(path: str | Path):
    path = Path(path)
    notes = np.load(path / "notes.npy", mmap_mode="r")
    offsets = np.load(path / "offsets.npy")
    with open(path / "tracks.jsonl", encoding="utf-8") as fh:
        metas = [json.loads(line) for line in fh if line.strip()]
    return notes, offsets, metas


def iter_tracks(
    root: str | Path,
    split: str | None = None,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield ``(metadata, notes_view)`` across all shards, optionally by split.

    ``shuffle`` randomises shard order and track order within each shard. This
    matters when a consumer only reads a fixed prefix of the corpus: a fixed
    order would keep revisiting the same first tracks. Validation/evaluation
    should leave it off. Training does not need it any more -- the chunk store
    is built in a fixed order and shuffled per epoch at batch time.
    """
    shards = shard_dirs(root)
    rng = np.random.default_rng(seed) if shuffle else None
    if rng is not None:
        shards = [shards[i] for i in rng.permutation(len(shards))]

    for shard in shards:
        notes, offsets, metas = load_shard(shard)
        order = rng.permutation(len(metas)) if rng is not None else range(len(metas))
        for i in order:
            meta = metas[int(i)]
            if split is not None and meta.get("split") != split:
                continue
            yield meta, notes[offsets[int(i)] : offsets[int(i) + 1]]



def write_manifest(root: str | Path, manifest: dict) -> None:
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
