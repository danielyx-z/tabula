"""alphaTex (.tex) parser -> canonical note events.

The corpus is a GuitarPro export in alphaTab's alphaTex text format:

    \\title "Wasteland"
    \\tempo 75
    .
    \\track
    \\instrument 30
    \\tuning E5 B4 G4 D4 A3 D3
    \\frets 24
    :8 (0.4 0.5) (2.2 0.4 0.5).4 2.2.4 (2.2{d} 2.4{d} 2.5{d}).4 |

Notes are ``fret.string[.duration]``. Durations are per beat and are modified by
beat properties that the exporter repeats on every note of a chord:

    {d}      dotted          x1.5
    {dd}     double dotted   x1.75
    {tu 3}   triplet         2/3      (5->4/5, 6->4/6, 7->4/7, 9..12->8/n)
    {tu a b} explicit        b/a

Pitch is implied by the tab: ``pitch = tuning[string - 1] + fret``.
Harmonic notes ({nh}, {ah}) sound at a pitch their fret does not imply, so they
are dropped rather than mislabelled.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from core.fretboard import parse_tuning, pitch_of
from core.schema import Note, Track

DURATIONS = {1: 3840, 2: 1920, 4: 960, 8: 480, 16: 240, 32: 120, 64: 60, 128: 30}

# {tu 3} -> 3:2 etc, mirroring AlphaTex1LanguageHandler._getTupletDenominator
TUPLET_DENOMINATORS = {3: 2, 5: 4, 6: 4, 7: 4, 9: 8, 10: 8, 11: 8, 12: 8}

_DIRECTIVE = re.compile(r"^\s*\\(\w+)\s*(.*)$")
_TRACK_START = re.compile(r"^\s*\\track\b", re.MULTILINE)

# Guitar programs in the corpus (1-based GM guitar family, plus harmonics).
GUITAR_PROGRAMS = set(range(24, 33))


class ParseError(ValueError):
    pass


def _read_props(s: str, i: int) -> tuple[list[str], int]:
    """Read a ``{...}`` property block. ``i`` points at '{'."""
    j = s.find("}", i)
    if j < 0:
        return [], len(s)
    return s[i + 1 : j].split(), j + 1


def _read_note(s: str, i: int) -> tuple[int, int, int]:
    """Read ``fret.string``. ``i`` points at the first digit."""
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    if j >= len(s) or s[j] != ".":
        raise ParseError(f"expected '.' after fret at {i}: {s[i:i + 12]!r}")
    k = j + 1
    while k < len(s) and s[k].isdigit():
        k += 1
    if k == j + 1:
        raise ParseError(f"missing string number at {i}: {s[i:i + 12]!r}")
    return int(s[i:j]), int(s[j + 1 : k]), k


def _tuplet_ratio(props: list[str]) -> float:
    if "tu" not in props:
        return 1.0
    k = props.index("tu")
    nums = [int(x) for x in props[k + 1 : k + 3] if x.isdigit()]
    if len(nums) == 1:
        num = nums[0]
        den = TUPLET_DENOMINATORS.get(num)
        return den / num if den else 1.0
    if len(nums) == 2:
        num, den = nums
        return den / num if num else 1.0
    return 1.0


def parse_body(
    body: str, tuning: tuple[int, ...], frets: int
) -> tuple[list[Note], list[float], set[str], dict[str, int]]:
    """Parse the bar/beat body of one track.

    Returns (notes, bar_durations_in_ticks, flags, counters).

    A bar whose whole content is a bare ``r`` is a whole-measure rest: the
    exporter writes ``r |`` rather than ``:1 r |``. Those bars are snapped to
    the track's modal bar length after the fact, because their printed length
    depends on whatever ``:N`` happened to be active.
    """
    notes: list[Note] = []
    bars: list[float] = []
    bar_meta: list[tuple[float, float, bool]] = []   # start, length, lone rest
    flags: set[str] = set()
    counts = {"beats": 0, "dotted": 0, "tuplet": 0, "harmonic": 0, "dead": 0,
              "bad_string": 0, "bad_fret": 0, "lone_rest_bars": 0}

    i, n = 0, len(body)
    t = 0.0
    bar_start = 0.0
    default = DURATIONS[8]
    bar_beats = 0
    bar_only_rest = True
    bar_explicit = False

    def close_bar(end: float) -> None:
        nonlocal bar_beats, bar_only_rest, bar_explicit
        length = end - bar_start
        if length > 0.5:
            bars.append(length)
            bar_meta.append((bar_start, length, bar_beats == 1 and bar_only_rest and not bar_explicit))
        bar_beats = 0
        bar_only_rest = True
        bar_explicit = False

    while i < n:
        ch = body[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "|":
            close_bar(t)
            bar_start = t
            i += 1
            continue

        if ch == ":":
            j = i + 1
            while j < n and body[j].isdigit():
                j += 1
            num = int(body[i + 1 : j]) if j > i + 1 else 0
            default = DURATIONS.get(num, default)
            i = j
            continue

        props: list[str] = []
        pairs: list[tuple[int, int]] = []
        counts["beats"] += 1
        bar_beats += 1

        # leading properties, e.g. "{tu 3} 4.3"
        while i < n and body[i] == "{":
            p, i = _read_props(body, i)
            props += p

        if i < n and body[i] == "(":
            i += 1
            while i < n and body[i] != ")":
                c = body[i]
                if c.isspace():
                    i += 1
                    continue
                if c == "{":
                    p, i = _read_props(body, i)
                    props += p
                    continue
                if c.isdigit():
                    f, st, i = _read_note(body, i)
                    pairs.append((f, st))
                    while i < n and body[i] == "{":
                        p, i = _read_props(body, i)
                        props += p
                    continue
                i += 1
            i += 1  # consume ')'
        elif i < n and body[i] == "r":
            i += 1
        elif i < n and body[i].isdigit():
            f, st, i = _read_note(body, i)
            pairs.append((f, st))
            bar_only_rest = False
            while i < n and body[i] == "{":
                p, i = _read_props(body, i)
                props += p
        else:
            flags.add(f"skipped:{body[i]!r}")
            i += 1
            continue

        while i < n and body[i] == "{":
            p, i = _read_props(body, i)
            props += p

        explicit = None
        if i < n and body[i] == ".":
            j = i + 1
            while j < n and body[j].isdigit():
                j += 1
            if j > i + 1:
                explicit = int(body[i + 1 : j])
                bar_explicit = True
                i = j
            else:
                i += 1

        while i < n and body[i] == "{":
            p, i = _read_props(body, i)
            props += p

        base = DURATIONS.get(explicit, default) if explicit else default
        if "dd" in props:
            dur = base * 1.75
            counts["dotted"] += 1
        elif "d" in props:
            dur = base * 1.5
            counts["dotted"] += 1
        else:
            dur = float(base)
        ratio = _tuplet_ratio(props)
        if ratio != 1.0:
            counts["tuplet"] += 1
            dur *= ratio

        if pairs:
            bar_only_rest = False
            if "x" in props:
                counts["dead"] += 1
            elif "nh" in props or "ah" in props:
                counts["harmonic"] += len(pairs)
            else:
                for f, st in pairs:
                    if not (1 <= st <= len(tuning)):
                        counts["bad_string"] += 1
                        continue
                    if not (0 <= f <= frets):
                        counts["bad_fret"] += 1
                        continue
                    notes.append(
                        Note(
                            t=round(t),
                            dur=max(1, round(dur)),
                            pitch=pitch_of(st, f, tuning),
                            s=st,
                            f=f,
                        )
                    )

        t += dur

    close_bar(t)

    # Bare-`r` bars are whole-measure rests: snap them to the track's modal bar
    # length and move everything after them.
    if len(bar_meta) > 2 and any(m[2] for m in bar_meta):
        lengths = [round(l / 60) * 60 for _, l, lone in bar_meta if not lone and l >= 240]
        if lengths:
            mode, votes = Counter(lengths).most_common(1)[0]
            if votes >= 2:
                cuts: list[tuple[float, float]] = []
                cum = 0.0
                for idx, (start, length, lone) in enumerate(bar_meta):
                    if lone and abs(mode - length) >= 1:
                        cum += mode - length
                        cuts.append((start + length, cum))
                        bars[idx] = mode
                        counts["lone_rest_bars"] += 1
                if cuts:
                    cut_ticks = [c[0] for c in cuts]
                    for note in notes:
                        k = bisect_right(cut_ticks, note.t)
                        if k:
                            note.t = max(0, round(note.t + cuts[k - 1][1]))

    for key in ("bad_string", "bad_fret"):
        if counts[key]:
            flags.add(key)
    return notes, bars, flags, counts


def parse_file(text: str, source: str, song: str) -> list[Track]:
    """Parse a (possibly multi-track) alphaTex file into Tracks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    segments = _TRACK_START.split(text)
    preamble, segments = segments[0], segments[1:]
    if not segments:                       # no \track marker: treat as one track
        segments = [preamble]
        preamble = ""

    score_tempo = 120
    for line in preamble.split("\n"):
        m = _DIRECTIVE.match(line)
        if m and m.group(1) == "tempo":
            score_tempo = _as_int(m.group(2), 120)

    tracks: list[Track] = []
    for idx, seg in enumerate(segments):
        tuning: tuple[int, ...] | None = None
        frets = 24
        instrument = -1
        tempo = score_tempo
        instrument_changes = 0
        body_lines: list[str] = []

        for line in seg.split("\n"):
            stripped = line.strip()
            m = _DIRECTIVE.match(line)
            if m:
                key, rest = m.group(1), m.group(2).strip()
                if key == "tuning":
                    try:
                        tuning = parse_tuning(rest)
                    except ValueError:
                        tuning = None
                elif key == "frets":
                    frets = _as_int(rest, 24)
                elif key == "instrument":
                    instrument_changes += 1
                    instrument = _as_int(rest, -1)
                elif key == "tempo":
                    tempo = _as_int(rest, tempo)
                continue
            if stripped and stripped != ".":
                body_lines.append(line)

        if tuning is None:
            continue

        notes, bars, flags, counts = parse_body(" ".join(body_lines), tuning, frets)
        if instrument_changes > 1:
            flags.add("instrument_change")
        tracks.append(
            Track(
                source=source,
                song=song,
                index=idx,
                tuning=tuning,
                frets=frets,
                instrument=instrument,
                tempo=tempo,
                notes=notes,
                bars=bars,
                flags=flags,
                stats=counts,
            )
        )
    return tracks


def _as_int(text: str, default: int) -> int:
    m = re.match(r"-?\d+", text.strip())
    return int(m.group()) if m else default
