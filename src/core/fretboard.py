"""Fretboard geometry.

The single place where pitch <-> (string, fret) conversions happen.

Convention (matches the paper / Hsu et al. and the corpus conversion in
``parse.alphatex``): string 1 is the THICKEST (lowest-pitched) string.
Standard tuning is therefore (40, 45, 50, 55, 59, 64) for strings 1..6.

Scientific pitch notation: MIDI 60 = C4 (middle C).
"""

from __future__ import annotations

import re

PC_BY_NAME = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10,
    "BB": 10, "B": 11, "CB": 11,
}
NAME_BY_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

STANDARD = (40, 45, 50, 55, 59, 64)

_TUNING_TOKEN = re.compile(r"([A-Ga-g][#b]?)(\d)")


def parse_tuning(text: str) -> tuple[int, ...]:
    """Parse an alphaTex tuning string.

    alphaTex writes tunings high string first, e.g. ``E5 B4 G4 D4 A3 E3``.
    Returns MIDI pitches ordered from string 1 (thickest/lowest) upward.

    Note the corpus labels the high E as ``E5`` because it derives the octave
    as ``midi // 12``; MIDI 64 -> ``E5``. Keep that mapping, it is lossless.
    """
    pitches = []
    for token in text.split():
        m = _TUNING_TOKEN.fullmatch(token)
        if not m:
            raise ValueError(f"bad tuning token: {token!r}")
        name = m.group(1).upper()
        if name not in PC_BY_NAME:
            raise ValueError(f"bad note name in tuning: {token!r}")
        pitches.append(12 * int(m.group(2)) + PC_BY_NAME[name])
    if not pitches:
        raise ValueError("empty tuning")
    return tuple(reversed(pitches))


def format_tuning(tuning: tuple[int, ...]) -> str:
    """Render a tuning back to alphaTex order (high string first).

    Uses scientific pitch notation: MIDI 60 = C4.
    Standard tuning renders as: ``E4 B3 G3 D3 A2 E2``.
    """
    return " ".join(pitch_name(p) for p in reversed(tuning))


def format_tuning_low_to_high(tuning: tuple[int, ...]) -> str:
    """Render a tuning low string first (string 1 to 6).

    Uses scientific pitch notation: MIDI 60 = C4.
    Standard tuning renders as: ``E2 A2 D3 G3 B3 E4``.
    """
    return " ".join(pitch_name(p) for p in tuning)


def format_tuning_alphatex(tuning: tuple[int, ...]) -> str:
    """Render a tuning back to alphaTex order (high string first).

    Uses alphaTex pitch notation: MIDI 60 = C5 (original corpus convention).
    Standard tuning renders as: ``E5 B4 G4 D4 A3 E3``.
    """
    return " ".join(pitch_name_alphatex(p) for p in reversed(tuning))


def pitch_name(pitch: int) -> str:
    """Scientific pitch notation: MIDI 60 = C4."""
    return f"{NAME_BY_PC[pitch % 12]}{pitch // 12 - 1}"


def pitch_name_alphatex(pitch: int) -> str:
    """AlphaTex pitch notation: MIDI 60 = C5 (original corpus convention)."""
    return f"{NAME_BY_PC[pitch % 12]}{pitch // 12}"


def pitch_of(s: int, f: int, tuning: tuple[int, ...], capo: int = 0) -> int:
    return tuning[s - 1] + f + capo


def positions(
    pitch: int, tuning: tuple[int, ...], frets: int, capo: int = 0
) -> list[tuple[int, int]]:
    """All physically valid (string, fret) realisations of ``pitch``.

    Used by the constrained-decoding mask and by the refingering test.
    """
    out = []
    for s in range(1, len(tuning) + 1):
        f = pitch - tuning[s - 1] - capo
        if 0 <= f <= frets:
            out.append((s, f))
    return out


def is_playable(pitch: int, tuning: tuple[int, ...], frets: int, capo: int = 0) -> bool:
    return bool(positions(pitch, tuning, frets, capo))


def midi_to_freq(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)
