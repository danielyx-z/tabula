#!/usr/bin/env python3
"""Convert GuitarSet JAMS annotations into Tabula's canonical shards.

GuitarSet annotates each of the six strings separately, so this converter
produces real ``(t, dur, pitch, s, f)`` records rather than guessing fingerings.

    python scripts/preprocess_guitarset.py \
        --jams-dir /path/to/guitarSet_annotation \
        --out data/processed_guitarset

Timing: GuitarSet is performed, not quantised.  Onsets are mapped from seconds
to *beats* with the ``beat_position`` annotation, then to ticks at PPQ 960.
Onsets and durations are snapped to a 20-tick grid so the encoder's TIME_SHIFT
vocabulary can represent every delta exactly.

A strum spreads a chord over 20–40 ticks.  ``--chord-window`` (default 40)
collapses onsets inside that window to the first attack, which is how a
human reads the result: one chord, not a fast arpeggio.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import ShardWriter, notes_to_array, write_manifest  # noqa: E402
from core.fretboard import STANDARD, format_tuning  # noqa: E402
from core.group import group_chords  # noqa: E402
from core.schema import Note  # noqa: E402

PPQ = 960
GRID = 20  # ticks; the encoder's TIME_SHIFT grid
MIN_PITCH = 36
MAX_PITCH = 96


def split_of(song: str) -> str:
    """Stable 80/10/10 split keyed on the performance name."""
    h = int.from_bytes(hashlib.blake2b(song.encode(), digest_size=8).digest(), "big")
    bucket = h % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def _beat_map(jam: dict) -> tuple[np.ndarray, np.ndarray] | None:
    rows: list[tuple[float, float]] = []
    for ann in jam["annotations"]:
        if ann.get("namespace") != "beat_position":
            continue
        for e in ann["data"]:
            v = e["value"]
            beat = (v["measure"] - 1) * v["num_beats"] + (v["position"] - 1)
            rows.append((float(e["time"]), float(beat)))
    if len(rows) < 2:
        return None
    rows.sort()
    times = np.asarray([r[0] for r in rows], dtype=np.float64)
    beats = np.asarray([r[1] for r in rows], dtype=np.float64)
    # Drop duplicate timestamps (different annotations can share a tick).
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, beats = times[keep], beats[keep]
    if len(times) < 2:
        return None
    return times, beats


def _to_beat(t: float, beat_map: tuple[np.ndarray, np.ndarray] | None) -> float:
    if beat_map is None:
        return t  # fallback: one second == one beat
    times, beats = beat_map
    if t <= times[0]:
        slope = (beats[1] - beats[0]) / (times[1] - times[0])
        return float(beats[0] + (t - times[0]) * slope)
    if t >= times[-1]:
        slope = (beats[-1] - beats[-2]) / (times[-1] - times[-2])
        return float(beats[-1] + (t - times[-1]) * slope)
    return float(np.interp(t, times, beats))


def _snap(t: float) -> int:
    return int(round(t / GRID) * GRID)


def parse_jams(path: Path) -> tuple[list[Note], dict, list[str]]:
    jam = json.loads(path.read_text(encoding="utf-8"))
    beat_map = _beat_map(jam)

    tempo = 120.0
    for ann in jam["annotations"]:
        if ann.get("namespace") == "tempo" and ann["data"]:
            tempo = float(ann["data"][0]["value"])
            break

    notes: list[Note] = []
    rejects: list[str] = []
    string_index = 0

    for ann in jam["annotations"]:
        if ann.get("namespace") != "note_midi":
            continue
        string_index += 1
        if string_index > len(STANDARD):
            break
        open_pitch = STANDARD[string_index - 1]
        for e in ann["data"]:
            pitch = int(round(float(e["value"])))
            fret = pitch - open_pitch
            if not (MIN_PITCH <= pitch <= MAX_PITCH):
                rejects.append(f"pitch_out_of_range:{pitch}")
                continue
            if not (0 <= fret <= 24):
                rejects.append(f"fret_out_of_range:{fret}")
                continue
            b0 = _to_beat(float(e["time"]), beat_map)
            b1 = _to_beat(float(e["time"]) + float(e["duration"]), beat_map)
            t = _snap(b0 * PPQ)
            dur = max(GRID, _snap((b1 - b0) * PPQ))
            if t < 0:
                continue
            notes.append(Note(t=t, dur=dur, pitch=pitch, s=string_index, f=fret))

    # One note per string per onset; GuitarSet occasionally digitisations share
    # a tick after snapping.
    notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
    seen: set[tuple[int, int]] = set()
    unique: list[Note] = []
    for n in notes:
        key = (n.s, n.t)
        if key in seen:
            rejects.append("same_string_same_tick")
            continue
        seen.add(key)
        unique.append(n)

    meta = {
        "tempo": tempo,
        "duration_s": float(jam.get("file_metadata", {}).get("duration", 0.0)),
    }
    return unique, meta, rejects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jams-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "data/processed_guitarset"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-notes", type=int, default=20)
    ap.add_argument("--chord-window", type=int, default=40,
                    help="collapse onsets within this many ticks (0 disables)")
    ap.add_argument("--shard-notes", type=int, default=1_000_000)
    args = ap.parse_args()

    jams_dir = Path(args.jams_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(out, max_notes=args.shard_notes)
    rejects_fh = open(out / "rejects.jsonl", "w", encoding="utf-8")
    reasons: collections.Counter[str] = collections.Counter()
    per_split: collections.Counter[str] = collections.Counter()
    n_files = n_kept = n_notes = 0
    t0 = time.time()

    for path in sorted(jams_dir.glob("*.jams")):
        if args.limit and n_files >= args.limit:
            break
        n_files += 1
        song = path.stem
        try:
            notes, info, rejects = parse_jams(path)
        except (ValueError, KeyError, TypeError) as exc:
            reasons["parse_error"] += 1
            rejects_fh.write(json.dumps({"source": path.name, "reason": "parse_error",
                                         "detail": str(exc)[:160]}) + "\n")
            continue

        for reason in rejects:
            reasons[reason.split(":")[0]] += 1
            rejects_fh.write(json.dumps({"source": path.name, "reason": reason}) + "\n")

        if len(notes) < args.min_notes:
            reasons["too_few_notes"] += 1
            continue

        notes, merged = group_chords(notes, args.chord_window)
        if merged:
            reasons["same_string_same_tick_after_grouping"] += merged

        arr = notes_to_array(notes)
        arr = arr[np.lexsort((arr["pitch"], arr["t"]))]
        split = split_of(song)

        meta = {
            "id": n_kept,
            "source": path.name,
            "song": song,
            "track": 0,
            "dataset": "guitarset",
            "tuning": list(STANDARD),
            "tuning_name": format_tuning(STANDARD),
            "frets": 24,
            "instrument": 25,
            "tempo": round(info["tempo"], 3),
            "n_notes": len(arr),
            "start": int(arr["t"][0]),
            "end": int((arr["t"] + arr["dur"]).max()),
            "split": split,
            "bar_ticks": PPQ * 4,
        }
        writer.add(meta, arr)
        n_kept += 1
        n_notes += len(arr)
        per_split[split] += 1

    writer.close()
    rejects_fh.close()

    manifest = {
        "format": "tabula-canonical-v1",
        "source": "guitarset",
        "ppq": PPQ,
        "grid": GRID,
        "string_order": "1 = thickest (low E), 6 = thinnest (high E)",
        "tuning": list(STANDARD),
        "tuning_name": format_tuning(STANDARD),
        "filters": {"min_notes": args.min_notes, "chord_window_ticks": args.chord_window},
        "counts": {
            "files": n_files,
            "tracks_kept": n_kept,
            "notes": n_notes,
            "shards": writer.shard_index,
            "per_split": dict(sorted(per_split.items())),
        },
        "rejects_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
    write_manifest(out, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {out} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
