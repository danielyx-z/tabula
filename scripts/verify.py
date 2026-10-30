#!/usr/bin/env python3
"""Invariant checks over a built dataset.

    python scripts/verify.py --dataset data/processed

Checks, in order of how much they would hurt if they broke:

1. ``pitch == tuning[s - 1] + fret`` for every note
2. no two notes on the same string at the same tick (unplayable chord)
3. frets inside the MVP vocabulary, strings in 1..6
4. songs never appear in more than one split
5. every shard is internally consistent (offsets, metadata, note count)
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fretwire.dataset import NOTE_DTYPE, load_shard, shard_dirs  # noqa: E402
from fretwire.fretboard import STANDARD  # noqa: E402

TUNING = np.asarray(STANDARD, dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data/processed"))
    ap.add_argument("--max-fret", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.dataset)
    shards = shard_dirs(root)
    if not shards:
        raise SystemExit(f"no shards in {root}")

    problems: collections.Counter[str] = collections.Counter()
    song_split: dict[str, str] = {}
    tracks = notes_total = 0

    for shard in shards:
        notes, offsets, metas = load_shard(shard)
        if len(offsets) != len(metas) + 1:
            problems["offset_meta_mismatch"] += 1
        if offsets[-1] != len(notes):
            problems["note_count_mismatch"] += 1
        if notes.dtype != NOTE_DTYPE:
            problems["dtype_mismatch"] += 1

        for i, meta in enumerate(metas):
            tracks += 1
            view = notes[offsets[i] : offsets[i + 1]]
            notes_total += len(view)
            if meta["n_notes"] != len(view):
                problems["metadata_n_notes_mismatch"] += 1

            prev = song_split.setdefault(meta["song"], meta["split"])
            if prev != meta["split"]:
                problems["song_in_two_splits"] += 1

            if not len(view):
                problems["empty_track"] += 1
                continue

            p = view["pitch"].astype(np.int32)
            s = view["s"].astype(np.int32)
            f = view["f"].astype(np.int32)
            t = view["t"].astype(np.int64)

            if np.any(p != TUNING[np.clip(s - 1, 0, 5)] + f):
                problems["pitch_not_derivable"] += 1
            if np.any((s < 1) | (s > 6)):
                problems["bad_string"] += 1
            if np.any((f < 0) | (f > args.max_fret)):
                problems["bad_fret"] += 1
            if np.any(t < 0) or np.any(view["dur"] < 1):
                problems["bad_time"] += 1
            key = s.astype(np.int64) * 10_000_000 + t
            if len(np.unique(key)) != len(key):
                problems["two_notes_one_string_at_once"] += 1
            if np.any(np.diff(t) < 0):
                problems["notes_not_time_ordered"] += 1

    print(f"shards      {len(shards)}")
    print(f"tracks      {tracks:,}")
    print(f"notes       {notes_total:,}")
    print(f"songs       {len(song_split):,}")
    if problems:
        print("\nproblems:")
        for name, count in problems.most_common():
            print(f"  {name:38s} {count:,}")
        return 1
    print("\nall invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
