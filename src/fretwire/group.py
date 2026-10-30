"""Chord grouping for performed (non-quantised) sources.

A strum spreads the notes of one chord over tens of ticks.  Tablature shows
them in one column, so converters should collapse onsets inside a small window
onto the first attack.  Durations are extended to preserve each note's release.
"""

from __future__ import annotations

from .quantise import GRID
from .schema import Note


def group_chords(
    notes: list[Note],
    window: int = 40,
    grid: int = GRID,
) -> tuple[list[Note], int]:
    """Collapse onsets within ``window`` ticks onto the first attack.

    ``notes`` is sorted and mutated in place.  Returns the notes and the number
    of same-string same-tick duplicates dropped after grouping.
    """
    if not notes or window <= 0:
        return notes, 0

    notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
    grouped: list[Note] = []
    i = 0
    while i < len(notes):
        anchor = notes[i].t
        j = i
        while j < len(notes) and notes[j].t - anchor <= window:
            j += 1
        for n in notes[i:j]:
            end = n.t + n.dur
            n.t = anchor
            n.dur = max(grid, end - anchor)
        grouped.extend(notes[i:j])
        i = j

    grouped.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
    seen: set[tuple[int, int]] = set()
    unique: list[Note] = []
    dropped = 0
    for n in grouped:
        key = (n.s, n.t)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(n)
    return unique, dropped
