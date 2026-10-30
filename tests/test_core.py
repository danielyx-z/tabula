"""Plain-assert tests: ``python tests/test_core.py`` (no pytest needed)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fretwire.fretboard import (  # noqa: E402
    STANDARD,
    format_tuning,
    parse_tuning,
    pitch_name,
    pitch_of,
    positions,
)
from fretwire.parse.alphatex import parse_body, parse_file  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def test_tuning() -> None:
    check("tuning parses standard", parse_tuning("E5 B4 G4 D4 A3 E3") == STANDARD,
          str(parse_tuning("E5 B4 G4 D4 A3 E3")))
    check("tuning roundtrips", parse_tuning(format_tuning(STANDARD)) == STANDARD,
          format_tuning(STANDARD))
    check("high string is s=6", pitch_of(6, 0, STANDARD) == 64)
    check("low string is s=1", pitch_of(1, 0, STANDARD) == 40)
    check("E at 12th fret = 76", pitch_of(6, 12, STANDARD) == 76)
    check("drop D parses", parse_tuning("E5 B4 G4 D4 A3 D3")[0] == 38)


def test_positions() -> None:
    check("E2 open only on string 1", positions(40, STANDARD, 24) == [(1, 0)])
    check("E4 has six homes", len(positions(64, STANDARD, 24)) == 6,
          str(positions(64, STANDARD, 24)))
    check("out of range is unplayable", positions(30, STANDARD, 24) == [])
    check("pitch names match corpus", pitch_name(64) == "E5", pitch_name(40) == "E3")


def test_body() -> None:
    tune = STANDARD

    notes, bars, flags, counts = parse_body(":4 0.6 0.5 0.4 0.3 |", tune, 24)
    check("four quarters", [n.dur for n in notes] == [960] * 4, str([n.dur for n in notes]))
    check("bar sums to 4/4", bars == [3840], str(bars))
    check("string->pitch", [(n.s, n.pitch) for n in notes] ==
          [(6, 64), (5, 59), (4, 55), (3, 50)], str([(n.s, n.pitch) for n in notes]))

    notes, _, _, _ = parse_body(":8 0.6{d} 0.6 0.6 |", tune, 24)
    check("dotted = x1.5", notes[0].dur == 720, str(notes[0].dur))

    notes, _, _, _ = parse_body(":8 0.6{dd} 0.6 |", tune, 24)
    check("double dotted = x1.75", notes[0].dur == 840, str(notes[0].dur))

    notes, bars, _, _ = parse_body(":8 3.2{tu 3} 3.2{tu 3} 3.2{tu 3} |", tune, 24)
    check("triplet eighths", [n.dur for n in notes] == [320, 320, 320],
          str([n.dur for n in notes]))
    check("triplet bar is 3 eighths", abs(bars[0] - 960) < 1, str(bars))

    notes, _, _, _ = parse_body(":8 0.6{tu 5} 0.6 |", tune, 24)
    check("quintuplet uses 4/5", notes[0].dur == 384, str(notes[0].dur))

    notes, bars, _, _ = parse_body(":4 (3.4{d} 3.5{d} 1.6{d}).4 0.1 |", tune, 24)
    check("dotted chord", notes[0].dur == 1440 and len(notes) == 4, str([n.dur for n in notes]))
    check("chord pitches", [n.pitch for n in notes] == [58, 62, 65, 40],
          str([n.pitch for n in notes]))

    notes, _, flags, counts = parse_body(":4 12.1{nh} 0.1 |", tune, 24)
    check("harmonic dropped", len(notes) == 1 and counts["harmonic"] == 1,
          f"{len(notes)} {counts}")

    notes, _, flags, counts = parse_body(":4 0.9 0.1 |", tune, 24)
    check("bad string flagged", "bad_string" in flags and counts["bad_string"] == 1, str(flags))

    notes, _, flags, counts = parse_body(":4 30.1 0.1 |", tune, 24)
    check("bad fret flagged", "bad_fret" in flags and counts["bad_fret"] == 1, str(flags))

    notes, bars, _, _ = parse_body(":16 r.4 0.6 |", tune, 24)
    check("explicit rest duration", notes[0].t == 960, str(notes[0].t))

    body = ":4 0.6 0.6 0.6 0.6 | r | 0.6 0.6 0.6 0.6 | 0.6 0.6 0.6 0.6 |"
    notes, bars, _, counts = parse_body(body, tune, 24)
    check("bare rest fills the bar", [round(b) for b in bars] == [3840, 3840, 3840, 3840], str(bars))
    check("notes after a fixed rest shift", notes[4].t == 7680, str(notes[4].t))


def test_file() -> None:
    text = """\\title "Two tracks"
\\tempo 90
.
\\track
\\instrument 30
\\tuning E5 B4 G4 D4 A3 E3
\\frets 24
:4 0.6 0.5 0.4 0.3 |
\\track
\\instrument 34
\\tuning E5 B4 G4 D4 A3 E3
\\frets 24
:4 0.1 0.1 0.1 0.1 |
"""
    tracks = parse_file(text, "x.tex", "x")
    check("two tracks parsed", len(tracks) == 2, str(len(tracks)))
    check("track metadata", tracks[0].instrument == 30 and tracks[1].instrument == 34,
          f"{tracks[0].instrument} {tracks[1].instrument}")
    check("score tempo inherited", tracks[1].tempo == 90, str(tracks[1].tempo))
    check("track 2 notes", tracks[1].n_notes == 4, str(tracks[1].n_notes))


def main() -> int:
    test_tuning()
    test_positions()
    test_body()
    test_file()
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
