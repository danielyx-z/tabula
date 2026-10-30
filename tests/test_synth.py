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

    segment = samples[int(0.15 * SR) : int(0.65 * SR)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / SR)
    peak = float(freqs[int(np.argmax(spectrum))])
    check(
        "E2 fundamental is in tune",
        abs(peak - midi_to_freq(40)) < 2.5,
        f"{peak:.1f} Hz vs {midi_to_freq(40):.1f} Hz",
    )

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

    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
