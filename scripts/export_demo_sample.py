#!/usr/bin/env python3
"""Export one GuitarSet track as a pitch-only input MIDI plus ground truth.

The input file deliberately contains no string/fret information: it is only
what a normal MIDI file would carry.  Ground truth is written separately so a
model output can be compared without leaking fingering into the input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import iter_tracks  # noqa: E402
from tablature.postproc import DecodedNote  # noqa: E402
from tablature.render import render_tab  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed_guitarset")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--track-id", type=int, default=6)
    ap.add_argument("--out-dir", default="out/demo_track6")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = None
    rows = None
    for m, r in iter_tracks(args.data_dir, split=args.split):
        if int(m.get("id", -1)) == args.track_id:
            meta, rows = m, r
            break
    if meta is None or rows is None:
        raise SystemExit(f"track {args.track_id} not found in {args.split}")

    bpm = float(meta.get("tempo", 120.0))
    ppq = 960
    spb = 60.0 / bpm
    notes = sorted(
        (int(t), int(d), int(p), int(s), int(f))
        for t, d, p, s, f in rows
    )

    # 1. Pitch-only input MIDI: no string or fret information.
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=25, name="pitch only")
    pm.instruments.append(inst)
    for t, d, p, s, f in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=p,
                start=t / ppq * spb,
                end=(t + max(1, d)) / ppq * spb,
            )
        )
    pm.write(str(out / "input.mid"))

    # 2. Ground-truth MIDI: one track per string, kept separate from input.
    gpm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tracks = {}
    for s in range(1, 7):
        tr = pretty_midi.Instrument(program=25, name=f"string {s}")
        gpm.instruments.append(tr)
        tracks[s] = tr
    for t, d, p, s, f in notes:
        tracks[s].notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=p,
                start=t / ppq * spb,
                end=(t + max(1, d)) / ppq * spb,
            )
        )
    gpm.write(str(out / "ground_truth.mid"))

    gt = [
        {"t": t, "dur": d, "pitch": p, "s": s, "f": f}
        for t, d, p, s, f in notes
    ]
    (out / "ground_truth.json").write_text(
        json.dumps({"meta": meta, "notes": gt}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "input.json").write_text(
        json.dumps(
            [
                {"t": t, "dur": d, "pitch": p, "s": 0, "f": 0}
                for t, d, p, s, f in notes
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    nodes = [
        DecodedNote(t=t, dur=d, pitch=p, s=s, f=f)
        for t, d, p, s, f in notes
    ]
    (out / "ground_truth_tab.txt").write_text(
        render_tab(nodes, width=100) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "track_id": args.track_id,
                "split": args.split,
                "source": meta.get("source"),
                "bpm": bpm,
                "notes": len(notes),
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
