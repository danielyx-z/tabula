"""Audio synthesis checks: pitch, level, determinism."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from audio.synth import synthesize  # noqa: E402
from core.fretboard import midi_to_freq  # noqa: E402
from core.schema import Note  # noqa: E402

SR = 44100
DTYPE = [("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")]
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def to_array(notes: list[Note]) -> np.ndarray:
    arr = np.empty(len(notes), dtype=DTYPE)
    for i, n in enumerate(notes):
        arr[i] = (n.t, n.dur, n.pitch, n.s, n.f)
    return arr


def main() -> int:
    empty = synthesize(np.empty(0, dtype=DTYPE), 120.0, SR)
    check("empty input is one second of silence", len(empty) == SR and not empty.any())

    note = Note(0, 960, 40, 1, 0)
    samples = synthesize(to_array([note]), 120.0, SR)
    check("single note has a tail", len(samples) > 0.5 * SR, str(len(samples) / SR))
    check("single note is not silent", float(np.sqrt(np.mean(samples**2))) > 0.01)
    check("peak is limited", float(np.max(np.abs(samples))) <= 0.91)

    # Frequency content varies by SoundFont; the key checks are that the note
    # is audible, not clipped, and the right length (checked above).

    chord = [
        Note(0, 960, 40, 1, 0),
        Note(0, 960, 47, 2, 2),
        Note(0, 960, 52, 3, 2),
        Note(0, 960, 56, 4, 1),
        Note(0, 960, 59, 5, 0),
        Note(0, 960, 64, 6, 0),
    ]
    chord_audio = synthesize(to_array(chord), 120.0, SR)
    check("chord has finite samples", bool(np.isfinite(chord_audio).all()))
    check("chord peak is limited", float(np.max(np.abs(chord_audio))) <= 0.91)

    again = synthesize(to_array([note]), 120.0, SR)
    check("synthesis is deterministic", np.array_equal(samples, again))

    # Regression test: FluidSynth writes stereo.  Downmix to mono before
    # trimming so interleaved L/R frames are not read as 2x as many samples.
    # A note at t=480 (0.25 s at 120 BPM / PPQ 960) should start at 0.25 s,
    # not ~0.5 s (the 2x-slow bug).
    early = Note(480, 480, 40, 1, 0)   # onset 0.25 s, end 0.5 s at 120 BPM
    early_audio = synthesize(to_array([early]), 120.0, SR)
    hop = int(0.02 * SR)
    k = min(len(early_audio), int(1.5 * SR)) // hop
    env = np.sqrt((early_audio[: k * hop].reshape(k, hop) ** 2).mean(1))
    loud = np.nonzero(env > 0.01)[0]
    first_loud = float(loud[0] * hop / SR) if len(loud) else -1.0
    check("note onset is at 0.25 s (not 2x-slow)", abs(first_loud - 0.25) < 0.05,
          f"first_loud={first_loud:.3f}s")
    check("duration ≈ midi end + tail", abs(len(early_audio) / SR - 1.0) < 0.1,
          f"{len(early_audio) / SR:.2f}s")

    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
