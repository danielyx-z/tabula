#!/usr/bin/env python3
"""Parse the GuitarPro corpus into the sharded canonical dataset.

    python scripts/preprocess.py --out data/processed
    python scripts/preprocess.py --limit 2000 --out /tmp/probe   # quick check

Every dropped track is written to ``rejects.jsonl`` with a reason, and the
manifest records the full pipeline contract (PPQ, grid, tuning, string order).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fretwire.dataset import NOTE_DTYPE, ShardWriter, notes_to_array, write_manifest  # noqa: E402
from fretwire.filter import FilterConfig, evaluate  # noqa: E402
from fretwire.fretboard import STANDARD, format_tuning  # noqa: E402
from fretwire.parse.alphatex import ParseError, parse_file  # noqa: E402

PPQ = 960
GRID = 60  # 1/16 note


def song_key(source: str) -> str:
    return source[: -len(".tex")] if source.endswith(".tex") else source


def split_of(song: str) -> str:
    h = int.from_bytes(hashlib.blake2b(song.encode(), digest_size=8).digest(), "big")
    bucket = h % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def digest(notes: np.ndarray) -> bytes:
    return hashlib.blake2b(notes.tobytes(), digest_size=16).digest()


def iter_sources(path: Path):
    if path.is_dir():
        for p in sorted(path.rglob("*.tex")):
            yield str(p), p.read_bytes()
        return
    with zipfile.ZipFile(path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".tex")):
            yield name, zf.read(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/raw/guitar_tab.zip"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--limit", type=int, default=0, help="max .tex files to process")
    ap.add_argument("--min-notes", type=int, default=50)
    ap.add_argument("--shard-notes", type=int, default=1_000_000)
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = FilterConfig(min_notes=args.min_notes)
    writer = ShardWriter(out, max_notes=args.shard_notes)
    rejects = open(out / "rejects.jsonl", "w", encoding="utf-8")
    seen: set[bytes] = set()

    reasons: collections.Counter[str] = collections.Counter()
    flags: collections.Counter[str] = collections.Counter()
    bar_hist: collections.Counter[int] = collections.Counter()
    per_split: collections.Counter[str] = collections.Counter()
    consistencies: list[float] = []
    n_files = n_tracks = n_kept = n_notes = n_dupes = 0
    on_grid = collections.Counter()
    t0 = time.time()

    for name, raw in iter_sources(source):
        if args.limit and n_files >= args.limit:
            break
        n_files += 1
        try:
            text = raw.decode("utf-8", "replace")
            tracks = parse_file(text, name, song_key(name))
        except (ParseError, ValueError, IndexError) as exc:
            reasons["parse_error"] += 1
            rejects.write(json.dumps({"source": name, "reason": "parse_error",
                                      "detail": str(exc)[:160]}) + "\n")
            continue

        for track in tracks:
            n_tracks += 1
            verdict = evaluate(track, cfg)
            if verdict is not None:
                reasons[verdict[0]] += 1
                rejects.write(json.dumps({"source": name, "track": track.index,
                                          "reason": verdict[0], "detail": verdict[1][:160]}) + "\n")
                continue

            arr = notes_to_array(track.notes)
            arr = arr[np.lexsort((arr["pitch"], arr["t"]))]  # ascending pitch in chords

            key = digest(arr)
            if key in seen:
                reasons["duplicate"] += 1
                n_dupes += 1
                continue
            seen.add(key)

            ticks = arr["t"].astype(np.int64)
            for grid in (20, 60, 120, 240, 480):
                on_grid[grid] += int((ticks % grid == 0).sum())

            bar_counts = collections.Counter(round(b / GRID) for b in track.bars)
            bar_mode, bar_votes = bar_counts.most_common(1)[0] if bar_counts else (0, 0)
            bar_consistency = bar_votes / len(track.bars) if track.bars else 0.0

            for flag in track.flags:
                flags[flag.split(":")[0]] += 1
            for length in track.bars:
                bar_hist[round(length / GRID)] += 1

            split = split_of(track.song)
            meta = {
                "id": n_kept,
                "source": track.source,
                "song": track.song,
                "track": track.index,
                "tuning": list(track.tuning),
                "tuning_name": format_tuning(track.tuning),
                "frets": track.frets,
                "instrument": track.instrument,
                "tempo": track.tempo,
                "n_notes": len(arr),
                "n_bars": len(track.bars),
                "start": int(arr["t"][0]) if len(arr) else 0,
                "end": int((arr["t"] + arr["dur"]).max()) if len(arr) else 0,
                "split": split,
                "bar_ticks": bar_mode * GRID,
                "bar_consistency": round(bar_consistency, 3),
                "harmonic_notes": track.stats.get("harmonic", 0),
                "dotted_beats": track.stats.get("dotted", 0),
                "tuplet_beats": track.stats.get("tuplet", 0),
            }
            writer.add(meta, arr)
            n_kept += 1
            n_notes += len(arr)
            per_split[split] += 1
            if track.bars:
                consistencies.append(bar_consistency)

        if n_files % 2000 == 0:
            print(f"  {n_files} files, {n_kept} tracks, {n_notes} notes, "
                  f"{time.time() - t0:.1f}s", file=sys.stderr)

    writer.close()
    rejects.close()

    total_bars = sum(bar_hist.values())
    four_four = bar_hist.get(64, 0) + bar_hist.get(63, 0) + bar_hist.get(65, 0)
    manifest = {
        "format": "fretwire-canonical-v1",
        "source": str(source.name),
        "ppq": PPQ,
        "grid": GRID,
        "string_order": "1 = thickest (low E), 6 = thinnest (high E)",
        "tuning": list(STANDARD),
        "tuning_name": format_tuning(STANDARD),
        "note_dtype": NOTE_DTYPE.descr,
        "filters": {
            "min_notes": args.min_notes,
            "min_bars": cfg.min_bars,
            "max_harmonic_ratio": cfg.max_harmonic_ratio,
            "tuning": "standard only",
        },
        "counts": {
            "files": n_files,
            "tracks_seen": n_tracks,
            "tracks_kept": n_kept,
            "notes": n_notes,
            "duplicates": n_dupes,
            "shards": writer.shard_index,
            "per_split": dict(sorted(per_split.items())),
        },
        "rejects_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "bar_lengths_in_16ths": {str(k): v for k, v in sorted(bar_hist.items())},
        "bars_total": total_bars,
        "bars_4_4_ratio": round(four_four / total_bars, 4) if total_bars else 0.0,
        "notes_on_grid": {
            str(g): round(on_grid[g] / n_notes, 4) for g in sorted(on_grid)
        } if n_notes else {},
        "tracks_bar_consistency_ge_0_9": sum(1 for c in consistencies if c >= 0.9),
        "tracks_with_bars": len(consistencies),
    }
    write_manifest(out, manifest)

    print(json.dumps({k: manifest[k] for k in
                      ("counts", "rejects_by_reason", "bars_4_4_ratio")}, indent=2))
    print(f"wrote {out} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
