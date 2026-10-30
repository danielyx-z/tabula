#!/usr/bin/env python3
"""Convert GuitarSet JAMS annotation to WAV using fretwire's synthesizer."""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/daniel/Projects/tabula/src")
from preview import synthesize, write_wav

# Open string pitches for standard tuning (string 1 = thickest = low E)
OPEN_PITCH = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}


def jam_to_notes(jams_path):
    """Parse a GuitarSet JAMS file into Note array."""
    with open(jams_path) as f:
        d = json.load(f)

    notes = []
    string_idx = 1
    for ann in d["annotations"]:
        if ann.get("namespace") != "note_midi":
            continue
        if string_idx > 6:
            break
        data = ann["data"]
        if isinstance(data, dict):
            times = data["time"]
            durations = data["duration"]
            values = data["value"]
        else:
            times = [e["time"] for e in data]
            durations = [e["duration"] for e in data]
            values = [e["value"] for e in data]

        open_pitch = OPEN_PITCH[string_idx]
        for t, dur, pitch in zip(times, durations, values):
            midi_pitch = round(pitch)
            fret = midi_pitch - open_pitch
            if 0 <= fret <= 24:
                notes.append({
                    "t": round(t * 960),      # seconds to ticks (PPQ=960)
                    "dur": round(dur * 960),
                    "pitch": midi_pitch,
                    "s": string_idx,
                    "f": fret,
                })
        string_idx += 1

    notes.sort(key=lambda n: (n["t"], n["pitch"]))
    return notes


def main():
    jams_path = "/tmp/guitarSet_annotation/00_BN1-129-Eb_comp.jams"
    notes = jam_to_notes(jams_path)
    print(f"Parsed {len(notes)} notes from {jams_path}")

    arr = np.empty(len(notes), dtype=[("t", "<i4"), ("dur", "<i4"),
                                       ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
    for i, n in enumerate(notes):
        arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])

    samples = synthesize(arr, 120.0)
    write_wav("/tmp/guitarset_sample.wav", samples)
    print(f"Saved /tmp/guitarset_sample.wav, {len(samples)/44100:.1f}s")


if __name__ == "__main__":
    main()
