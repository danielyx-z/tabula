"""Pitch-preserving playability fixes for generated tablature.

Nothing in this module looks at ground-truth notes.  Every adjustment either
moves a note to a different (string, fret) that produces *the same pitch* or
drops a physically impossible duplicate.  The music is never rewritten.

Added:
* ``assign_onsets`` — joint per-onset string/fret assignment that prefers
  low positions and enforces a fret cap per onset (Layer 2 of the low-position
  prior).  Runs after the span/jump repairs so the cap is enforced last.
* ``max_fret`` filtering on ``fix_chord_spans`` and ``limit_jumps`` so those
  passes never push a note above the cap.
"""

from __future__ import annotations

from collections import defaultdict
import itertools

from core.fretboard import STANDARD, positions
from .postproc import DecodedNote, repair_overlaps


def fix_chord_spans(
    notes: list[DecodedNote],
    max_span: int = 4,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    max_fret: int | None = None,
) -> tuple[list[DecodedNote], int]:
    """Re-finger chords whose frets are farther apart than a hand can reach.

    When ``max_fret`` is set, candidate positions above it are excluded unless
    no lower position exists for the pitch.  Returns the notes and the number
    of notes whose string/fret changed.
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
                if s not in used_strings
                and abs(f - center) <= max_span
                and (max_fret is None or f <= max_fret)
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
    max_fret: int | None = None,
) -> tuple[list[DecodedNote], int]:
    """Re-finger notes that require an unreachable fret jump from the previous note.

    A note is only moved when an alternative position of the *same pitch*
    shortens the jump, so pitch and timing are untouched.  When ``max_fret``
    is set, candidate positions above it are excluded.
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
                if s not in free
                and abs(f - prev_f) < abs(n.f - prev_f)
                and (max_fret is None or f <= max_fret)
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


def assign_onsets(
    notes: list[DecodedNote],
    max_fret: int | None = None,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    max_span: int = 4,
) -> tuple[list[DecodedNote], int]:
    """Joint per-onset string/fret assignment preferring low positions.

    For every onset, assigns each note a (string, fret) realisation of the
    same pitch such that no two notes share a string.  When ``max_fret`` is
    set, positions above it are only used when no position at or below it
    exists for the pitch (so genuinely high notes are still playable).

    Among feasible assignments prefers (in order):
      1. fewest notes above max_fret,
      2. lowest maximum fret in the onset,
      3. lowest sum of frets,
      4. least deviation from the predicted position.

    When ``max_fret`` is None, only same-string collisions at an onset are
    resolved; predicted positions are otherwise left untouched.

    Returns the notes and the number of notes whose string/fret changed.
    """
    if not notes:
        return notes, 0

    if max_fret is None:
        return _resolve_collisions(notes, tuning, frets), 0

    by_onset: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        by_onset[n.t].append(i)

    out = list(notes)
    moved = 0

    for t in sorted(by_onset):
        idxs = by_onset[t]

        if len(idxs) == 1:
            i = idxs[0]
            n = out[i]
            cands = [c for c in positions(n.pitch, tuning, frets) if c[1] <= max_fret]
            if not cands:
                cands = positions(n.pitch, tuning, frets)
            cur = (n.s, n.f)
            if cur in cands:
                chosen = cur
            else:
                chosen = min(cands, key=lambda c: (c[1], c[0], abs(c[0] - n.s)))
            if chosen != (n.s, n.f):
                n.s, n.f = chosen
                moved += 1
            continue

        # Candidate positions per note, capped when possible.
        cand_lists = []
        for i in idxs:
            n = out[i]
            cands = [c for c in positions(n.pitch, tuning, frets) if c[1] <= max_fret]
            if not cands:
                cands = positions(n.pitch, tuning, frets)
            cand_lists.append(cands)

        predicted = [(out[i].s, out[i].f) for i in idxs]
        best_key = None
        best_combo = None
        for combo in itertools.product(*cand_lists):
            strings = [c[0] for c in combo]
            if len(set(strings)) != len(strings):
                continue
            frets_used = [c[1] for c in combo]
            span = max(frets_used) - min(frets_used)
            above = sum(1 for f in frets_used if f > max_fret)
            key = (
                above,
                max(frets_used),
                sum(frets_used),
                span,
                sum(0 if c == p else 1 for c, p in zip(combo, predicted)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_combo = combo

        if best_combo is None:
            # Fallback: greedy lowest fret per note with unique strings.
            used: set[int] = set()
            best_combo = []
            for cands in cand_lists:
                chosen = None
                for c in sorted(cands, key=lambda c: (c[1], c[0])):
                    if c[0] not in used:
                        chosen = c
                        break
                if chosen is None:
                    chosen = cands[0]
                used.add(chosen[0])
                best_combo.append(chosen)

        for i, (s, f) in zip(idxs, best_combo):
            n = out[i]
            if (n.s, n.f) != (s, f):
                n.s, n.f = s, f
                moved += 1

    return out, moved


def _resolve_collisions(
    notes: list[DecodedNote],
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
) -> list[DecodedNote]:
    """No-cap mode: resolve same-string collisions at each onset only."""
    by_onset: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        by_onset[n.t].append(i)
    out = list(notes)
    for t, idxs in by_onset.items():
        if len(idxs) <= 1:
            continue
        used: set[int] = set()
        for i in idxs:
            n = out[i]
            if n.s in used:
                chosen = None
                for c in sorted(positions(n.pitch, tuning, frets), key=lambda c: (c[1], c[0])):
                    if c[0] not in used:
                        chosen = c
                        break
                if chosen is not None:
                    n.s, n.f = chosen
            used.add(n.s)
    return out


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


def playability_pass(
    raw: list[dict],
    max_fret: int | None = None,
    max_span: int = 4,
    max_jump: int = 5,
) -> tuple[list[dict], dict]:
    nodes = [
        DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in raw
    ]
    nodes, dropped_a = cap_simultaneous(nodes)
    # Layer 2: joint per-onset assignment enforcing the fret cap and string
    # uniqueness before the span/jump repairs.
    nodes, onset_moves = assign_onsets(nodes, max_fret=max_fret, max_span=max_span)
    nodes = repair_overlaps(nodes, max_fret=max_fret)  # no ground truth: pitch is preserved
    nodes, chord_moves = fix_chord_spans(nodes, max_span=max_span, max_fret=max_fret)
    nodes, jump_moves = limit_jumps(nodes, max_jump=max_jump, max_fret=max_fret)
    nodes = repair_overlaps(nodes, max_fret=max_fret)
    nodes, dropped_b = cap_simultaneous(nodes)
    final = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in nodes]
    stats = {
        "dropped_impossible": dropped_a + dropped_b,
        "refingered_chord_span": chord_moves,
        "refingered_jump": jump_moves,
        "refingered_onset": onset_moves,
    }
    return final, stats
