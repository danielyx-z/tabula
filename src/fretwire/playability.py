"""Pitch-preserving playability fixes for generated tablature.

Nothing in this module looks at ground-truth notes.  Every adjustment either
moves a note to a different (string, fret) that produces *the same pitch* or
drops a physically impossible duplicate.  The music is never rewritten.
"""

from __future__ import annotations

from collections import defaultdict

from .fretboard import STANDARD, positions
from .postproc import DecodedNote


def fix_chord_spans(
    notes: list[DecodedNote],
    max_span: int = 4,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
) -> tuple[list[DecodedNote], int]:
    """Re-finger chords whose frets are farther apart than a hand can reach.

    Returns the notes and the number of notes whose string/fret changed.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        groups[n.t].append(i)

    moved = 0
    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        ordered = sorted(notes[i].f for i in idxs)
        if ordered[-1] - ordered[0] <= max_span:
            continue

        center = ordered[len(ordered) // 2]
        used_strings: set[int] = set()
        # Fix the worst outlier first so later choices see the settled notes.
        for i in sorted(idxs, key=lambda j: -abs(notes[j].f - center)):
            n = notes[i]
            cands = [
                (s, f)
                for s, f in positions(n.pitch, tuning, frets)
                if s not in used_strings and abs(f - center) <= max_span
            ]
            if cands:
                s, f = min(
                    cands, key=lambda c: (abs(c[1] - center), abs(c[0] - n.s))
                )
            else:
                s, f = n.s, n.f
            used_strings.add(s)
            if (s, f) != (n.s, n.f):
                n.s, n.f = s, f
                moved += 1

    return notes, moved


def limit_jumps(
    notes: list[DecodedNote],
    max_jump: int = 5,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
) -> tuple[list[DecodedNote], int]:
    """Re-finger notes that require an unreachable fret jump from the previous note.

    A note is only moved when an alternative position of the *same pitch*
    shortens the jump, so pitch and timing are untouched.
    """
    order = sorted(range(len(notes)), key=lambda i: (notes[i].t, notes[i].s))
    moved = 0
    prev_f: int | None = None
    prev_s: int | None = None

    # Strings already occupied at each onset; a move must not create an
    # impossible two-notes-on-one-string chord.
    occupied: dict[int, set[int]] = defaultdict(set)
    for n in notes:
        occupied[n.t].add(n.s)

    for i in order:
        n = notes[i]
        if prev_f is not None and abs(n.f - prev_f) > max_jump:
            free = occupied[n.t] - {n.s}
            cands = [
                (s, f)
                for s, f in positions(n.pitch, tuning, frets)
                if s not in free and abs(f - prev_f) < abs(n.f - prev_f)
            ]
            if cands:
                s, f = min(
                    cands,
                    key=lambda c: abs(c[1] - prev_f) + 0.25 * abs(c[0] - (prev_s or n.s)),
                )
                occupied[n.t].discard(n.s)
                occupied[n.t].add(s)
                n.s, n.f = s, f
                moved += 1
        prev_f, prev_s = n.f, n.s

    return notes, moved


def cap_simultaneous(
    notes: list[DecodedNote],
    max_notes: int = 6,
) -> tuple[list[DecodedNote], int]:
    """Drop physically impossible simultaneous notes.

    Two notes on the same string at the same onset cannot both sound, and a
    hand plays at most one note per string.  Keep the first occurrence on each
    string and drop the rest.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        groups[n.t].append(i)

    drop: set[int] = set()
    for _, idxs in groups.items():
        seen: set[int] = set()
        for i in idxs:
            if notes[i].s in seen or len(seen) >= max_notes:
                drop.add(i)
            else:
                seen.add(notes[i].s)

    if not drop:
        return notes, 0
    return [n for i, n in enumerate(notes) if i not in drop], len(drop)
