"""Pitch-preserving playability fixes for generated tablature.

Nothing in this module looks at ground-truth notes.  Every adjustment moves a
note to a different (string, fret) that produces *the same pitch*, or removes a
physically impossible duplicate.  One deliberate exception: when a chord's
best full voicing requires an extreme fret, ``drop_unplayable_chord_tones`` may
remove a single inner tone so the rest of the chord becomes playable.  The
caller records those drops in the report.

Capo
----
Frets in this module are *capo-relative*: the sounding pitch is
``tuning[string] + capo + fret``.  ``capo=0`` is the normal no-capo case.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import itertools

from core.fretboard import STANDARD, positions
from .postproc import DecodedNote, repair_overlaps

# Soft ceiling for tab numbers.  Frets above this are used only when a pitch
# has no lower position (e.g. the very top of the neck).
SOFT_FRET = 6

# Chord-tone dropping policy: only consider it when the best full voicing
# reaches DROP_TRIGGER, accept only if the reduced voicing is at most DROP_MAX
# and improves the maximum fret by at least DROP_GAIN.
DROP_TRIGGER = 10
DROP_MAX = 9
DROP_GAIN = 2


def _onset_key(combo, predicted, high_fret: int = SOFT_FRET):
    """Lexicographic key for one onset (lower is better).

    Order: lowest maximum fret, fewest notes above the soft cap, fewest
    non-open strings, lowest fret sum, smallest span, least deviation from the
    predicted position.  Open strings are deliberately after the maximum fret:
    an open string is nice, but not at the cost of a higher fret elsewhere.
    """
    frets = [f for _, f in combo]
    return (
        max(frets),
        sum(1 for f in frets if f > high_fret),
        sum(1 for f in frets if f != 0),
        sum(frets),
        max(frets) - min(frets),
        sum(0 if c == p else 1 for c, p in zip(combo, predicted)),
    )


def _span_key(combo, predicted, high_fret: int = SOFT_FRET):
    """Key for the span repair: fix the span first, then the same preferences."""
    frets = [f for _, f in combo]
    return (
        max(frets) - min(frets),
        max(frets),
        sum(1 for f in frets if f > high_fret),
        sum(1 for f in frets if f != 0),
        sum(frets),
        sum(0 if c == p else 1 for c, p in zip(combo, predicted)),
    )


def _compact_key(combo, predicted):
    """Key for the fallback when no voicing fits the span."""
    frets = [f for _, f in combo]
    return (
        max(frets),
        max(frets) - min(frets),
        sum(1 for f in frets if f != 0),
        sum(frets),
        sum(0 if c == p else 1 for c, p in zip(combo, predicted)),
    )


def _candidate_lists(
    notes: list[DecodedNote],
    idxs: list[int],
    tuning: tuple[int, ...],
    frets: int,
    capo: int,
    max_fret: int | None,
    cap: bool = True,
) -> list[list[tuple[int, int]]]:
    """Per-note positions, optionally dropping positions above ``max_fret``."""
    out: list[list[tuple[int, int]]] = []
    for i in idxs:
        n = notes[i]
        cands = positions(n.pitch, tuning, frets, capo)
        if cap and max_fret is not None:
            low = [c for c in cands if c[1] <= max_fret]
            if low:
                cands = low
        out.append(cands)
    return out


def _best_assignment(
    notes: list[DecodedNote],
    idxs: list[int],
    tuning: tuple[int, ...],
    frets: int,
    capo: int,
    max_fret: int | None,
    high_fret: int = SOFT_FRET,
    cap: bool = True,
    span_first: bool = False,
):
    """Best unique-string assignment for one onset, or ``(None, None)``."""
    cand_lists = _candidate_lists(notes, idxs, tuning, frets, capo, max_fret, cap)
    if any(not c for c in cand_lists):
        return None, None
    predicted = [(notes[i].s, notes[i].f) for i in idxs]
    key_fn = _span_key if span_first else _onset_key
    best_key = None
    best_combo = None
    for combo in itertools.product(*cand_lists):
        strings = [c[0] for c in combo]
        if len(set(strings)) != len(strings):
            continue
        key = key_fn(combo, predicted, high_fret)
        if best_key is None or key < best_key:
            best_key = key
            best_combo = combo
    return best_combo, best_key


def fix_chord_spans(
    notes: list[DecodedNote],
    max_span: int = 4,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    max_fret: int | None = None,
    capo: int = 0,
) -> tuple[list[DecodedNote], int]:
    """Re-finger chords whose frets are farther apart than a hand can reach.

    Each chord is re-assigned *jointly* to unique strings, so a move can never
    put two notes on the same string.  When no unique-string voicing fits
    ``max_span``, it falls back to the most compact unique-string voicing over
    all positions (minimising the maximum fret) instead of creating a
    collision; if even that is impossible the chord is left untouched.
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
        cand_lists: list[list[tuple[int, int]]] = []
        for i in idxs:
            n = notes[i]
            cands = [
                (s, f)
                for s, f in positions(n.pitch, tuning, frets, capo)
                if abs(f - center) <= max_span
                and (max_fret is None or f <= max_fret)
            ]
            # Keep the current position as a fallback so a note that cannot
            # reach the target window does not make the whole chord
            # unsolvable; the unique-string check still applies.
            cand_lists.append(cands or [(n.s, n.f)])

        predicted = [(notes[i].s, notes[i].f) for i in idxs]
        best_key = None
        best_combo = None
        for combo in itertools.product(*cand_lists):
            strings = [c[0] for c in combo]
            if len(set(strings)) != len(strings):
                continue
            key = _span_key(combo, predicted)
            if best_key is None or key < best_key:
                best_key = key
                best_combo = combo

        if best_combo is None:
            # No unique-string voicing inside the span.  Choose the most
            # compact achievable voicing over all positions rather than
            # leaving an extreme fret (e.g. a piano chord with one note at
            # fret 22); unique strings are still required.
            full_lists = [
                positions(notes[i].pitch, tuning, frets, capo) for i in idxs
            ]
            for combo in itertools.product(*full_lists):
                strings = [c[0] for c in combo]
                if len(set(strings)) != len(strings):
                    continue
                key = _compact_key(combo, predicted)
                if best_key is None or key < best_key:
                    best_key = key
                    best_combo = combo

        if best_combo is None:
            # Truly no unique-string voicing; leave the chord as-is.
            continue

        for i, (s, f) in zip(idxs, best_combo):
            n = notes[i]
            if (n.s, n.f) != (s, f):
                n.s, n.f = s, f
                moved += 1

    return notes, moved


def limit_jumps(
    notes: list[DecodedNote],
    max_jump: int = 5,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    max_fret: int | None = None,
    capo: int = 0,
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
                for s, f in positions(n.pitch, tuning, frets, capo)
                if s not in free
                and abs(f - prev_f) < abs(n.f - prev_f)
                and (max_fret is None or f <= max_fret)
            ]
            if cands:
                s, f = min(
                    cands,
                    key=lambda c: (
                        abs(c[1] - prev_f),
                        c[1],
                        0 if c[1] <= SOFT_FRET else 1,
                        0 if c[1] == 0 else 1,
                        0.25 * abs(c[0] - (prev_s or n.s)),
                        c[0],
                    ),
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
    capo: int = 0,
    high_fret: int = SOFT_FRET,
) -> tuple[list[DecodedNote], int]:
    """Joint per-onset string/fret assignment preferring playable positions.

    For every onset, assigns each note a (string, fret) realisation of the
    same pitch such that no two notes share a string.  When ``max_fret`` is
    set, the capped assignment (candidate frets at or below the cap) is
    compared with the uncapped one; keeping a note on a low fret can force
    another note far up the neck, so the key below chooses the better voicing
    even if it uses a fret above the cap.

    Among feasible assignments prefers (in order):
      1. lowest maximum fret in the onset,
      2. fewest notes above ``high_fret`` (the soft cap, default 6),
      3. fewest non-open strings,
      4. lowest sum of frets,
      5. smallest span,
      6. least deviation from the predicted position.

    When ``max_fret`` is None, only same-string collisions at an onset are
    resolved; predicted positions are otherwise left untouched.

    Returns the notes and the number of notes whose string/fret changed.
    """
    if not notes:
        return notes, 0

    if max_fret is None:
        return _resolve_collisions(notes, tuning, frets, capo), 0

    by_onset: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        by_onset[n.t].append(i)

    out = list(notes)
    moved = 0

    for t in sorted(by_onset):
        idxs = by_onset[t]

        best_combo, best_key = _best_assignment(
            out, idxs, tuning, frets, capo, max_fret, high_fret, cap=True
        )
        # Also consider positions above the cap: keeping the low-fret
        # candidates for one note can force another note far up the neck
        # (e.g. 66 and 73 both wanting the high E string).  The full
        # assignment may use more notes above 6 but a much lower maximum, and
        # the max-first key chooses correctly.
        full_combo, full_key = _best_assignment(
            out, idxs, tuning, frets, capo, max_fret, high_fret, cap=False
        )
        if full_combo is not None and (
            best_combo is None or full_key < best_key
        ):
            best_combo = full_combo
        if best_combo is None:
            # Truly infeasible (>6 distinct-string pitches at one onset):
            # greedy lowest-fret assignment; cap_simultaneous drops the rest.
            used: set[int] = set()
            best_combo = []
            for i in idxs:
                n = out[i]
                cands = positions(n.pitch, tuning, frets, capo)
                chosen = None
                for c in sorted(
                    cands, key=lambda c: (c[1], 0 if c[1] == 0 else 1, c[0])
                ):
                    if c[0] not in used:
                        chosen = c
                        break
                if chosen is None:
                    chosen = cands[0] if cands else (n.s, n.f)
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
    capo: int = 0,
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
                for c in sorted(
                    positions(n.pitch, tuning, frets, capo),
                    key=lambda c: (c[1], 0 if c[1] == 0 else 1, c[0]),
                ):
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


def drop_unplayable_chord_tones(
    notes: list[DecodedNote],
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    capo: int = 0,
    max_fret: int | None = None,
    high_fret: int = SOFT_FRET,
) -> tuple[list[DecodedNote], list[dict]]:
    """Drop at most one inner tone from a chord whose full voicing is extreme.

    A drop is considered only when the best full voicing reaches
    ``DROP_TRIGGER`` (default 10).  Inner voices are tried before the bass or
    melody, and a candidate is accepted only when the remaining notes reach at
    most ``DROP_MAX`` (default 9) and improve the maximum fret by at least
    ``DROP_GAIN`` (default 2).  This keeps the drop rare and preserves the
    outer voices whenever possible.

    Returns the surviving notes and a list describing the dropped notes.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        groups[n.t].append(i)

    keep = set(range(len(notes)))
    dropped: list[dict] = []

    for _, idxs in groups.items():
        idxs = [i for i in idxs if i in keep]
        if len(idxs) < 4:
            continue

        full_combo, _ = _best_assignment(
            notes, idxs, tuning, frets, capo, max_fret, high_fret, cap=False
        )
        if full_combo is None:
            continue
        full_pos = dict(zip(idxs, full_combo))
        full_max = max(f for _, f in full_combo)
        if full_max < DROP_TRIGGER:
            continue

        pitches = [notes[i].pitch for i in idxs]
        lo, hi = min(pitches), max(pitches)
        pc_counts = Counter(p % 12 for p in pitches)
        duplicate = {i for i in idxs if pc_counts[notes[i].pitch % 12] > 1}
        inner = [i for i in idxs if lo < notes[i].pitch < hi]
        extremes = [i for i in idxs if notes[i].pitch in (lo, hi)]

        chosen = None
        # Try inner voices first.  Only fall back to the bass/melody when no
        # inner voice can rescue the chord.
        for group in (inner, extremes):
            for i in group:
                rem = [j for j in idxs if j != i]
                combo, key = _best_assignment(
                    notes, rem, tuning, frets, capo, max_fret, high_fret, cap=False
                )
                if combo is None:
                    continue
                rem_max = max(f for _, f in combo)
                if rem_max > DROP_MAX or full_max - rem_max < DROP_GAIN:
                    continue
                # After the maximum fret and span, prefer dropping a pitch
                # class that is already duplicated in the chord: that changes
                # the harmony least.  A more compact hand shape still wins.
                cand_key = (
                    key[0],                      # max fret
                    key[4],                      # span (hand stretch)
                    0 if i in duplicate else 1,  # duplicated pitch class
                    key[1],                      # notes above the soft cap
                    key[2],                      # non-open strings
                    key[3],                      # sum of frets
                    key[5],                      # deviation
                )
                cand = (cand_key, i, combo)
                if chosen is None or cand[0] < chosen[0]:
                    chosen = cand
            if chosen is not None:
                break

        if chosen is None:
            continue

        _, drop_i, combo = chosen
        remaining = [j for j in idxs if j != drop_i]
        for j, (s, f) in zip(remaining, combo):
            notes[j].s, notes[j].f = s, f
        keep.discard(drop_i)
        s_drop, f_drop = full_pos[drop_i]
        dropped.append(
            {
                "t": notes[drop_i].t,
                "pitch": notes[drop_i].pitch,
                "s": s_drop,
                "f": f_drop,
            }
        )

    return [n for i, n in enumerate(notes) if i in keep], dropped


def playability_pass(
    raw: list[dict],
    max_fret: int | None = None,
    max_span: int = 4,
    max_jump: int = 5,
    capo: int = 0,
    allow_drop: bool = True,
) -> tuple[list[dict], dict]:
    """Pitch-preserving playability and low-position correction.

    Notes are never dropped unless a single onset really has more notes than
    the guitar has strings, or one inner tone of an otherwise unplayable chord
    is removed (see ``drop_unplayable_chord_tones``).  Assign onsets first
    (unique strings + fret cap), then the span/jump/overlap repairs, and
    finally a collision safety net.
    """
    nodes = [
        DecodedNote(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in raw
    ]
    # Assign onsets first: it makes strings unique and applies the fret cap.
    nodes, onset_moves = assign_onsets(
        nodes, max_fret=max_fret, max_span=max_span, capo=capo
    )
    nodes = repair_overlaps(nodes, max_fret=max_fret, capo=capo)
    nodes, chord_moves = fix_chord_spans(
        nodes, max_span=max_span, max_fret=max_fret, capo=capo
    )
    dropped_chord: list[dict] = []
    if allow_drop:
        nodes, dropped_chord = drop_unplayable_chord_tones(
            nodes, max_fret=max_fret, capo=capo
        )
    nodes, jump_moves = limit_jumps(
        nodes, max_jump=max_jump, max_fret=max_fret, capo=capo
    )
    nodes = repair_overlaps(nodes, max_fret=max_fret, capo=capo)
    # Safety net: if a later pass still left two notes on one string at the
    # same onset, reassign those onsets jointly instead of dropping the note.
    seen: dict[int, set[int]] = defaultdict(set)
    colliding_t: set[int] = set()
    for n in nodes:
        if n.s in seen[n.t]:
            colliding_t.add(n.t)
        seen[n.t].add(n.s)
    if colliding_t:
        assign_onsets(
            [n for n in nodes if n.t in colliding_t],
            max_fret=max_fret,
            max_span=max_span,
            capo=capo,
        )
    nodes, dropped = cap_simultaneous(nodes)
    final = [
        {"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f}
        for n in nodes
    ]
    stats = {
        "dropped_impossible": dropped,
        "dropped_chord_tones": dropped_chord,
        "refingered_chord_span": chord_moves,
        "refingered_jump": jump_moves,
        "refingered_onset": onset_moves,
    }
    return final, stats
