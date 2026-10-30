#!/usr/bin/env python3
"""Build a frozen vocabulary from a shard dataset.

    python scripts/build_vocab.py --data-dir data/processed_guitarset_guitarset \
        --out vocab_guitarset.json

NOTE_ON / NOTE_OFF / TIME_SHIFT tokens are kept only if they occur at least
``--min-freq`` times.  All TAB tokens and the special tokens are always kept.
Rebuild the vocab whenever the new corpus uses pitches or timing values that
the old vocab does not contain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import iter_tracks  # noqa: E402
from tablature.tokenizer import build_vocab_from_tracks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed_guitarset"))
    ap.add_argument("--out", default=str(ROOT / "vocab.json"))
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()

    vocab = build_vocab_from_tracks(
        iter_tracks(args.data_dir),
        min_freq=args.min_freq,
        max_tracks=args.max_tracks,
    )
    vocab.save(args.out)

    pitches = [t for t in vocab.tokens if t.startswith("NOTE_ON<")]
    shifts = [t for t in vocab.tokens if t.startswith("TIME_SHIFT<")]
    print(f"vocab size : {vocab.size}")
    print(f"hash       : {vocab.hash}")
    print(f"NOTE_ON    : {len(pitches)}")
    print(f"TIME_SHIFT : {len(shifts)}")
    print(f"wrote      : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
