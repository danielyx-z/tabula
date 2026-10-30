#!/usr/bin/env python3
"""Audit the GuitarSet shards against their JAMS sources.

Checks performed:

* every processed track's notes are exactly the JAMS ``note_midi`` records
  mapped through the timing pipeline (seconds -> beats -> ticks at PPQ 960,
  snapped to the 20-tick grid, chord-grouped);
* invariants: pitches match ``tuning[s-1] + f``, frets in [0, 24],
  onsets/durations are grid multiples, no negative times, no duplicate
  (string, onset), tracks sorted chronologically;
* no performance (comp/solo take) is split across train/valid/test;
* vocabulary covers every NOTE_ON/NOTE_OFF/TIME_SHIFT/TAB token used by the
  corpus;
* no training or inference chunk boundary splits an onset group (chord).

Run from the repo root:

    python scripts/audit_guitarset.py --jams-dir /tmp/guitarSet_annotation \
        --data-dir data/processed_guitarset --vocab vocab/guitarset.json

Exit status is nonzero on any failure.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import iter_tracks  # noqa: E402
from core.fretboard import STANDARD, format_tuning  # noqa: E402
from core.schema import Note  # noqa: E402
from core.quantise import GRID  # noqa: E402
from model.chunk import ChunkConfig, make_inference_chunks, make_training_chunks  # noqa: E402
from tablature.tokenizer import (  # noqa: E402
    EOS_TOKEN, UNK_TOKEN, Vocab, build_base_vocab, notes_to_events,
    notes_to_target, parse_tab,
)

FAILURES: list[str] = []


def fail(name: str, detail: str = "") -> None:
    FAILURES.append(f"{name}: {detail}")
    print(f"FAIL {name}: {detail}")


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}{' -- ' + detail if detail else ''}")
    else:
        fail(name, detail)


def performance_key(song: str) -> str:
    for suffix in ("_comp", "_solo"):
        if song.endswith(suffix):
            return song[: -len(suffix)]
    return song


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "guitarset_converter", ROOT / "scripts/preprocess_guitarset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_timing(jams_dir: Path, data_dir: Path) -> None:
    """Recompute JAMS notes and diff against shards."""
    mod = load_converter()
    stats: dict[str, dict] = {}
    mismatches = 0
    total_files = 0
    all_ok = True

    for path in sorted(jams_dir.glob("*.jams")):
        song = path.stem
        total_files += 1
        try:
            jam_notes, info, rejects = mod.parse_jams(path)
        except Exception as exc:  # noqa: BLE001
            fail("parse", f"{song}: {exc}")
            all_ok = False
            continue

        # The shards store chord-grouped notes; recompute the same grouping.
        grouped_jam, merged = mod.group_chords([n for n in jam_notes], 40)

        # Find matching shard track by source name.
        found = None
        for meta, arr in iter_tracks(str(data_dir)):
            if meta.get("source") == path.name:
                found = (meta, arr)
                break

        if found is None:
            fail("missing_shard", f"{song} has no processed track")
            all_ok = False
            continue
        meta, arr = found

        shard_notes = [
            Note(int(r["t"]), int(r["dur"]), int(r["pitch"]), int(r["s"]), int(r["f"]))
            for r in arr
        ]
        stats[song] = {
            "jams_notes": len(jam_notes),
            "shard_notes": len(shard_notes),
            "grouped_notes": len(grouped_jam),
            "merged_by_grouping": merged,
            "splits": meta["split"],
        }

        # Compare as sorted sets so within-onset ordering can't false-alarm.
        gj_key = sorted(grouped_jam, key=lambda n: (n.t, n.s, n.pitch, n.f))
        ss_key = sorted(shard_notes, key=lambda n: (n.t, n.s, n.pitch, n.f))
        if gj_key != ss_key:
            mismatches += 1
            only_jams = [n for n in grouped_jam if n not in shard_notes]
            only_shard = [n for n in shard_notes if n not in grouped_jam]
            fail("note_mismatch", f"{song}: jams {len(jam_notes)} shard {len(shard_notes)} "
                  f"only_jams {len(only_jams)} only_shard {len(only_shard)}")
            if only_jams:
                print("   only_jams:", [(n.t, n.pitch, n.s, n.f) for n in only_jams[:5]])
            if only_shard:
                print("   only_shard:", [(n.t, n.pitch, n.s, n.f) for n in only_shard[:5]])
            all_ok = False

    check("all tracks present and notes match (content)", all_ok,
          f"{mismatches}/{total_files} files differed")


def audit_invariants(data_dir: Path) -> None:
    bad = collections.Counter()
    tracks = notes = 0
    t_lo = t_hi = None
    dur_lo = dur_hi = None

    for meta, arr in iter_tracks(str(data_dir)):
        tracks += 1
        notes += len(arr)
        prev_t = -1
        seen = set()
        for r in arr:
            t, dur, pitch, s, f = (
                int(r["t"]), int(r["dur"]), int(r["pitch"]),
                int(r["s"]), int(r["f"]),
            )
            t_lo = t if t_lo is None else min(t_lo, t)
            t_hi = t if t_hi is None else max(t_hi, t)
            dur_lo = dur if dur_lo is None else min(dur_lo, dur)
            dur_hi = dur if dur_hi is None else max(dur_hi, dur)

            if pitch != STANDARD[s - 1] + f:
                bad["pitch_formula"] += 1
            if not (0 <= f <= 24):
                bad["fret_range"] += 1
            if t < 0 or dur <= 0 or t % GRID or dur % GRID:
                bad["grid"] += 1
            if t < prev_t:
                bad["order"] += 1
            prev_t = t
            if (s, t) in seen:
                bad["duplicate_s_t"] += 1
            seen.add((s, t))
        if len(arr) < 20:
            bad["too_few_notes"] += 1

    check("pitch == tuning[s-1] + f", bad["pitch_formula"] == 0, str(bad["pitch_formula"]))
    check("frets in [0, 24]", bad["fret_range"] == 0, str(bad["fret_range"]))
    check("onsets/durs grid multiples (20 ticks)", bad["grid"] == 0, str(bad["grid"]))
    check("tracks sorted by onset", bad["order"] == 0, str(bad["order"]))
    check("no duplicate (string, onset)", bad["duplicate_s_t"] == 0,
          str(bad["duplicate_s_t"]))
    check("min notes per track", bad["too_few_notes"] == 0,
          str(bad["too_few_notes"]))
    print(f"tracks {tracks} notes {notes}")
    print(f"onset ticks {t_lo}..{t_hi}  dur ticks {dur_lo}..{dur_hi}")


def audit_splits(jams_dir: Path, data_dir: Path) -> None:
    splits_by_performance = collections.defaultdict(set)
    tracks_by_performance = collections.defaultdict(list)
    for meta, _ in iter_tracks(str(data_dir)):
        tracks_by_performance[meta["performance"]].append(meta)
        splits_by_performance[meta["performance"]].add(meta["split"])

    mixed = {k: v for k, v in splits_by_performance.items() if len(v) > 1}
    check("no performance split across train/valid/test", not mixed,
          f"{len(mixed)} mixed: {list(mixed.items())[:6]}")

    per_split_counts = collections.Counter()
    for tracks in tracks_by_performance.values():
        splits = {m["split"] for m in tracks}
        if len(splits) == 1:
            per_split_counts[splits.pop()] += len(tracks)
        else:
            for m in tracks:
                per_split_counts[m["split"]] += 1
    n_tracks = sum(len(v) for v in tracks_by_performance.values())
    print("tracks per split:", dict(sorted(per_split_counts.items())))
    print("performances:", len(tracks_by_performance), "tracks:", n_tracks)


def audit_vocab(data_dir: Path, vocab_path: Path) -> None:
    vocab = Vocab.load(vocab_path)
    base = build_base_vocab()
    missing_events = missing_targets = oov = 0

    for meta, arr in iter_tracks(str(data_dir)):
        notes = [
            Note(int(r["t"]), int(r["dur"]), int(r["pitch"]), int(r["s"]), int(r["f"]))
            for r in arr
        ]
        notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
        for tok in notes_to_events(notes, base_t=notes[0].t):
            if tok not in vocab.token_to_id:
                missing_events += 1
                if missing_events <= 3:
                    print("   missing event token:", tok)
        for tok in notes_to_target(notes, base_t=notes[0].t):
            if tok not in vocab.token_to_id:
                missing_targets += 1
                if missing_targets <= 3:
                    print("   missing target token:", tok)
        for tok in notes_to_events(notes, base_t=notes[0].t) + notes_to_target(
            notes, base_t=notes[0].t
        ):
            if tok not in vocab.token_to_id:
                oov += 1

    used_pitch_tokens = set()
    for meta, arr in iter_tracks(str(data_dir)):
        for r in arr:
            used_pitch_tokens.add(f"NOTE_ON<{int(r['pitch'])}>")
            used_pitch_tokens.add(f"NOTE_OFF<{int(r['pitch'])}>")

    check("vocab covers all encoder events", missing_events == 0, str(missing_events))
    check("vocab covers all decoder targets", missing_targets == 0, str(missing_targets))
    print("vocab size", vocab.size, "| base", len(base),
          "| pitches in base:", len([t for t in base if t.startswith("NOTE_ON")]),
          "| used pitch tokens:", len(used_pitch_tokens))


def audit_chunks(data_dir: Path, vocab: Vocab) -> None:
    train_split = infer_split = 0
    train_chunks = infer_chunks = 0
    examples = []
    bad_oov = 0

    for meta, arr in iter_tracks(str(data_dir)):
        notes = [
            Note(int(r["t"]), int(r["dur"]), int(r["pitch"]), int(r["s"]), int(r["f"]))
            for r in arr
        ]
        notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
        total = len(notes)

        for c in make_training_chunks(notes, vocab, meta["id"], ChunkConfig()):
            train_chunks += 1
            s, e = c.meta["note_start"], c.meta["note_end"]
            if e < total and notes[e - 1].t == notes[e].t:
                train_split += 1
                if len(examples) < 8:
                    examples.append((meta["song"], s, e, notes[e - 1].t))
            for tok_id in c.enc_tokens:
                if tok_id == vocab[UNK_TOKEN]:
                    bad_oov += 1

        for c in make_inference_chunks(notes, vocab, meta["id"], ChunkConfig()):
            infer_chunks += 1
            s, e = c.meta["note_start"], c.meta["note_end"]
            if e < total and notes[e - 1].t == notes[e].t:
                infer_split += 1

    check("training chunks never split chords", train_split == 0,
          f"{train_split}/{train_chunks}")
    check("inference chunks never split chords", infer_split == 0,
          f"{infer_split}/{infer_chunks}")
    check("chunk encoder tokens all in vocab", bad_oov == 0, str(bad_oov))
    if examples:
        print("  first splits:", examples[:6])


def audit_timing_map(jams_dir: Path) -> None:
    mod = load_converter()
    worst_extrap = worst_slope_dev = 0.0
    n_checked = 0
    tempo_bad = 0
    for path in sorted(jams_dir.glob("*.jams")):
        jam = json.loads(path.read_text(encoding="utf-8"))
        bmap = mod._beat_map(jam)
        if bmap is None:
            continue
        times, beats = bmap
        slope = np.diff(beats) / np.diff(times)
        tempo = 120.0
        for ann in jam["annotations"]:
            if ann.get("namespace") == "tempo" and ann["data"]:
                tempo = float(ann["data"][0]["value"])
                break
        dev = abs(float(np.median(slope)) - tempo / 60.0) / (tempo / 60.0)
        worst_slope_dev = max(worst_slope_dev, dev)
        n_checked += 1
        if dev > 0.05:
            tempo_bad += 1
        ann_times = [float(e["time"]) for ann in jam["annotations"]
                     if ann.get("namespace") == "note_midi" for e in ann["data"]]
        if ann_times:
            first_extrap = max(0.0, times[0] - min(ann_times))
            last_extrap = max(0.0, max(ann_times) - times[-1])
            worst_extrap = max(worst_extrap, first_extrap, last_extrap)

    check("beat-map slope matches tempo (<5% dev)", tempo_bad == 0,
          f"{tempo_bad}/{n_checked} files, worst {worst_slope_dev*100:.2f}%")
    print(f"beat-map files {n_checked}, worst extrap beyond beat map {worst_extrap:.3f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit GuitarSet shards vs JAMS sources")
    ap.add_argument("--jams-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", default=str(ROOT / "vocab/guitarset.json"))
    args = ap.parse_args()

    jams_dir = Path(args.jams_dir)
    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab)

    print("=== 1. timing & notes (JAMS vs shards) ===")
    audit_timing(jams_dir, data_dir)

    print("\n=== 2. invariants ===")
    audit_invariants(data_dir)

    print("\n=== 3. splits ===")
    audit_splits(jams_dir, data_dir)

    print("\n=== 4. vocab coverage ===")
    audit_vocab(data_dir, vocab_path)

    print("\n=== 5. chunk boundaries ===")
    vocab = Vocab.load(vocab_path)
    audit_chunks(data_dir, vocab)

    print("\n=== 6. beat-map sanity ===")
    audit_timing_map(jams_dir)

    print("\n" + "=" * 40)
    if FAILURES:
        print(f"AUDIT FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES[:30]:
            print("  -", f)
        return 1
    print("AUDIT PASSED: no mistakes found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
