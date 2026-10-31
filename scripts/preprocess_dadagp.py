#!/usr/bin/env python3
"""Convert DadaGP alphaTex (.tex) files into Tabula's canonical shards.

DadaGP ships GuitarPro songs pre-tokenised as alphaTex.  This converter reuses
the canonical alphaTex parser (``corpus.parse.alphatex``) and the same admission
rules as the GuitarSet converter:

* six strings, **standard tuning only** (the TAB vocabulary is tuning-agnostic,
  so admitting drop/alternate tunings silently poisons pitch -> position),
* guitar programs only (GM 24..32),
* frets inside the vocabulary range (0..24),
* 20-tick grid, i.e. exactly what ``TIME_SHIFT`` can express,
* onsets within ``--chord-window`` collapsed into one column.

Timing: alphaTex durations are exact (measured: 99.99% already on the 20-tick
grid); the remaining tuplet/128th values are snapped with ``quantise_tick`` so
the encoder and decoder share one clock.

    python scripts/preprocess_dadagp.py --in-dir data/dadagp_raw --out data/processed_dadagp
    python scripts/preprocess_dadagp.py --in-dir data/dadagp_raw --limit 1000 --out /tmp/probe
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import NOTE_DTYPE, ShardWriter, notes_to_array, write_manifest  # noqa: E402
from corpus.filter import FilterConfig, evaluate  # noqa: E402
from corpus.parse.alphatex import ParseError, parse_file  # noqa: E402
from core.fretboard import STANDARD, format_tuning  # noqa: E402
from core.group import group_chords  # noqa: E402
from core.quantise import GRID, PPQ, quantise_dur, quantise_tick  # noqa: E402
from core.schema import Note, Track  # noqa: E402
from tablature.tokenizer import MAX_FRET as VOCAB_MAX_FRET  # noqa: E402

GRID_CHECKS = (20, 60, 120, 240, 480)


def song_key(source: str) -> str:
    """``Artist/Song.tex`` -> ``Artist/Song`` (stable, unique per performance)."""
    return source[: -len(".tex")] if source.endswith(".tex") else source


def split_of(song: str) -> str:
    """Stable 80/10/10 split keyed on the underlying performance."""
    h = int.from_bytes(hashlib.blake2b(song.encode(), digest_size=8).digest(), "big")
    bucket = h % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def digest(notes: np.ndarray) -> bytes:
    return hashlib.blake2b(notes.tobytes(), digest_size=16).digest()


def convert_track(track: Track, chord_window: int = 40) -> tuple[list[Note], int]:
    """Snap a parsed track to the 20-tick grid and group strummed chords.

    Returns the notes (sorted by ``(t, s, pitch, f)``, ascending string inside a
    chord) and the number of same-string/same-tick duplicates dropped.
    """
    snapped = [
        Note(
            t=quantise_tick(n.t, GRID),
            dur=quantise_dur(n.dur, GRID),
            pitch=n.pitch,
            s=n.s,
            f=n.f,
        )
        for n in track.notes
    ]
    notes, dropped = group_chords(snapped, chord_window, grid=GRID)
    if chord_window > 0:
        return notes, dropped

    # group_chords() returns early when grouping is disabled, so enforce the
    # one-note-per-string-per-tick invariant here as well.
    seen: set[tuple[int, int]] = set()
    unique: list[Note] = []
    for n in notes:
        key = (n.s, n.t)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(n)
    return unique, dropped


def iter_sources(path: Path):
    """Yield ``(relative_name, bytes)`` from a directory tree or a zip archive."""
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".tex")):
                yield name, zf.read(name)
        return
    for p in sorted(path.rglob("*.tex")):
        yield p.relative_to(path).as_posix(), p.read_bytes()


def bar_ticks(track: Track) -> int:
    """Modal bar length in ticks (falls back to one 4/4 bar)."""
    if not track.bars:
        return PPQ * 4
    counts = collections.Counter(round(b / GRID) * GRID for b in track.bars)
    return counts.most_common(1)[0][0]


def bar_consistency(track: Track) -> float:
    if not track.bars:
        return 0.0
    counts = collections.Counter(round(b / GRID) for b in track.bars)
    return round(counts.most_common(1)[0][1] / len(track.bars), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Root of extracted DadaGP alphaTex (artist/song.tex) or a .zip")
    ap.add_argument("--out", default=str(ROOT / "data/processed_dadagp"))
    ap.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    ap.add_argument("--min-notes", type=int, default=20)
    ap.add_argument("--min-bars", type=int, default=4)
    ap.add_argument("--chord-window", type=int, default=40,
                    help="collapse onsets within this many ticks (0 disables)")
    ap.add_argument("--max-fret", type=int, default=VOCAB_MAX_FRET,
                    help=f"highest fret to admit (vocabulary limit: {VOCAB_MAX_FRET})")
    ap.add_argument("--allow-non-guitar", action="store_true",
                    help="keep non-GM-guitar programs (default: guitar only)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep byte-identical note arrays (default: drop duplicates)")
    ap.add_argument("--shard-notes", type=int, default=1_000_000)
    ap.add_argument("--overwrite", action="store_true",
                    help="clear existing shards/manifest in --out before writing")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"--in-dir does not exist: {in_dir}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(p for p in out.glob("shard-*") if p.is_dir())
    if existing and not args.overwrite:
        raise SystemExit(
            f"{out} already contains {len(existing)} shard(s); "
            "pass --overwrite or choose a new --out"
        )
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
        for name in ("manifest.json", "rejects.jsonl"):
            (out / name).unlink(missing_ok=True)

    if args.max_fret > VOCAB_MAX_FRET:
        raise SystemExit(
            f"--max-fret {args.max_fret} exceeds the TAB vocabulary limit "
            f"({VOCAB_MAX_FRET}); such notes would train as <unk>"
        )

    cfg = FilterConfig(
        tuning=STANDARD,
        min_notes=args.min_notes,
        min_bars=args.min_bars,
        require_guitar_program=not args.allow_non_guitar,
        max_fret=args.max_fret,
    )

    writer = ShardWriter(out, max_notes=args.shard_notes)
    rejects_fh = open(out / "rejects.jsonl", "w", encoding="utf-8")
    reasons: collections.Counter[str] = collections.Counter()
    per_split: collections.Counter[str] = collections.Counter()
    on_grid: collections.Counter[int] = collections.Counter()
    seen: set[bytes] = set()
    n_files = n_tracks = n_kept = n_notes = n_dupes = 0
    t0 = time.time()

    for name, raw in iter_sources(in_dir):
        if args.limit and n_files >= args.limit:
            break
        n_files += 1
        if n_files % 2000 == 0:
            print(f"  {n_files} files, {n_kept} tracks, {n_notes} notes, "
                  f"{time.time() - t0:.1f}s", file=sys.stderr)

        song = song_key(name)
        try:
            text = raw.decode("utf-8", "replace")
            tracks = parse_file(text, name, song)
        except (ParseError, ValueError, IndexError, KeyError) as exc:
            reasons["parse_error"] += 1
            rejects_fh.write(json.dumps({"source": name, "reason": "parse_error",
                                         "detail": str(exc)[:160]}) + "\n")
            continue

        if not tracks:
            reasons["no_tracks"] += 1
            rejects_fh.write(json.dumps({"source": name, "reason": "no_tracks"}) + "\n")
            continue

        for track in tracks:
            n_tracks += 1
            verdict = evaluate(track, cfg)
            if verdict is not None:
                reasons[verdict[0]] += 1
                rejects_fh.write(json.dumps({"source": name, "track": track.index,
                                             "reason": verdict[0],
                                             "detail": verdict[1][:160]}) + "\n")
                continue

            notes, dropped = convert_track(track, args.chord_window)
            if dropped:
                reasons["same_string_same_tick_after_grouping"] += dropped
            if not notes:
                reasons["empty_after_grouping"] += 1
                continue

            arr = notes_to_array(notes)
            # Canonical order: onset, then ascending string inside a chord.
            arr = arr[np.lexsort((arr["s"], arr["t"]))]

            if not args.no_dedupe:
                key = digest(arr)
                if key in seen:
                    reasons["duplicate"] += 1
                    n_dupes += 1
                    continue
                seen.add(key)

            ticks = arr["t"].astype(np.int64)
            for grid in GRID_CHECKS:
                on_grid[grid] += int((ticks % grid == 0).sum())

            split = split_of(song)
            meta = {
                "id": n_kept,
                "source": name,
                "song": song,
                "performance": song,
                "track": track.index,
                "dataset": "dadagp",
                "tuning": list(track.tuning),
                "tuning_name": format_tuning(track.tuning),
                "frets": track.frets,
                "instrument": track.instrument,
                "tempo": track.tempo,
                "n_notes": len(arr),
                "n_bars": len(track.bars),
                "start": int(arr["t"][0]),
                "end": int((arr["t"] + arr["dur"]).max()),
                "split": split,
                "bar_ticks": bar_ticks(track),
                "bar_consistency": bar_consistency(track),
                "harmonic_notes": track.stats.get("harmonic", 0),
                "dotted_beats": track.stats.get("dotted", 0),
                "tuplet_beats": track.stats.get("tuplet", 0),
            }
            writer.add(meta, arr)
            n_kept += 1
            n_notes += len(arr)
            per_split[split] += 1

    writer.close()
    rejects_fh.close()

    manifest = {
        "format": "tabula-canonical-v1",
        "source": "dadagp",
        "ppq": PPQ,
        "grid": GRID,
        "string_order": "1 = thickest (low E), 6 = thinnest (high E)",
        "tuning": list(STANDARD),
        "tuning_name": format_tuning(STANDARD),
        "note_dtype": NOTE_DTYPE.descr,
        "filters": {
            "min_notes": args.min_notes,
            "min_bars": args.min_bars,
            "chord_window_ticks": args.chord_window,
            "max_fret": args.max_fret,
            "tuning": "standard only",
            "require_guitar_program": not args.allow_non_guitar,
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
        "notes_on_grid": {
            str(g): round(on_grid[g] / n_notes, 6) for g in sorted(on_grid)
        } if n_notes else {},
    }
    write_manifest(out, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {out} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
