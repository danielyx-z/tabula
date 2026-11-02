"""Playability/capo regression tests (``pytest tests/test_playability.py``)."""

from __future__ import annotations

from core.fretboard import STANDARD, positions
from core.schema import Note
from tablature.playability import playability_pass
from tablature.render import render_tab


def test_positions_are_capo_relative() -> None:
    assert positions(41, STANDARD, 24, 0) == [(1, 1)]
    assert positions(41, STANDARD, 24, 1) == [(1, 0)]
    assert positions(41, STANDARD, 24, 2) == []


def test_single_note_prefers_open_string() -> None:
    raw = [{"t": 0, "dur": 480, "pitch": 64, "s": 5, "f": 5}]
    final, _ = playability_pass(raw, max_fret=6)
    assert (final[0]["s"], final[0]["f"]) == (6, 0)


def test_capo_threading_makes_first_fret_open() -> None:
    raw = [{"t": 0, "dur": 480, "pitch": 41, "s": 1, "f": 1}]
    final, _ = playability_pass(raw, max_fret=6, capo=1)
    assert (final[0]["s"], final[0]["f"]) == (1, 0)


def test_uncapped_assignment_avoids_extreme_fret() -> None:
    # Both pitches want the high E string: capping 66 to fret 2 would force
    # 73 up to the B string at fret 14.  The full assignment (73 at E9, 66 at
    # B7) has a much lower maximum fret.
    raw = [
        {"t": 0, "dur": 480, "pitch": 73, "s": 5, "f": 14},
        {"t": 0, "dur": 480, "pitch": 66, "s": 6, "f": 2},
    ]
    final, _ = playability_pass(raw, max_fret=6)
    by_pitch = {n["pitch"]: n for n in final}
    assert max(n["f"] for n in final) == 9
    assert (by_pitch[73]["s"], by_pitch[73]["f"]) == (6, 9)
    assert (by_pitch[66]["s"], by_pitch[66]["f"]) == (5, 7)


def test_unplayable_six_note_chord_drops_one_inner_tone() -> None:
    pitches = [51, 58, 62, 63, 67, 70]
    strings = [1, 2, 3, 4, 5, 6]
    frets = [11, 13, 12, 8, 8, 6]
    raw = [
        {"t": 0, "dur": 480, "pitch": p, "s": s, "f": f}
        for p, s, f in zip(pitches, strings, frets)
    ]
    for capo in (0, 1):
        final, stats = playability_pass(raw, max_fret=6, capo=capo)
        assert len(final) == 5
        assert len(stats["dropped_chord_tones"]) == 1
        # Pitch 63 duplicates the Eb of pitch 51; 67 (the third) survives.
        assert stats["dropped_chord_tones"][0]["pitch"] == 63
        assert 67 in {n["pitch"] for n in final}
        assert max(n["f"] for n in final) <= 9
        used = [n["s"] for n in final]
        assert len(used) == len(set(used))


def test_tab_shows_capo_and_relative_frets() -> None:
    notes = [Note(t=0, dur=480, pitch=41, s=1, f=0)]
    text = render_tab(notes, width=80, capo=1)
    assert text.splitlines()[0] == "Capo: 1"
    assert "E2|0" in text
    assert not render_tab(notes, width=80).startswith("Capo:")
