"""Plain-assert tests for the DadaGP conversion: ``python tests/test_dadagp.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.filter import FilterConfig, evaluate  # noqa: E402
from corpus.parse.alphatex import parse_file  # noqa: E402
from core.fretboard import STANDARD, parse_tuning  # noqa: E402
from core.quantise import GRID  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "preprocess_dadagp", ROOT / "scripts" / "preprocess_dadagp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A chord, an explicit 8th with a dotted property, a plain 16th, a rest bar and
# an 8th-note pair: the real corpus grammar, not a toy subset.
SAMPLE = r"""
\title "Probe"
\tempo 100
.
\track
\instrument 25
\tuning E5 B4 G4 D4 A3 E3
\frets 24
:16 (0.1 2.3 2.4 0.5) 3.2.8{d} 4.4 | r | :8 5.6 7.4 |
"""


def _track(text: str = SAMPLE):
    tracks = parse_file(text, "Artist/Song.tex", "Artist/Song")
    assert len(tracks) == 1, len(tracks)
    return tracks[0]


def test_octave_convention() -> None:
    check("corpus E5 == MIDI 64", parse_tuning("E5 B4 G4 D4 A3 E3") == STANDARD,
          str(parse_tuning("E5 B4 G4 D4 A3 E3")))
    check("parsed tuning is STANDARD", tuple(_track().tuning) == STANDARD,
          str(_track().tuning))


def test_chord_preserved() -> None:
    chord = [n for n in _track().notes if n.t == 0]
    check("chord keeps 4 notes", len(chord) == 4, str(len(chord)))
    check("chord strings 1,3,4,5", sorted(n.s for n in chord) == [1, 3, 4, 5],
          str(sorted(n.s for n in chord)))
    check("chord frets 0,2,2,0",
          [n.f for n in sorted(chord, key=lambda n: n.s)] == [0, 2, 2, 0],
          str([n.f for n in sorted(chord, key=lambda n: n.s)]))


def test_conversion_contract() -> None:
    dadagp = _load_script()
    notes, dropped = dadagp.convert_track(_track())
    check("nothing dropped", dropped == 0, str(dropped))
    check("notes survive conversion", bool(notes), str(len(notes)))
    check("onsets on 20-tick grid", all(n.t % GRID == 0 for n in notes))
    check("durations on 20-tick grid",
          all(n.dur % GRID == 0 and n.dur >= GRID for n in notes))
    check("pitch == STANDARD[s-1] + f",
          all(n.pitch == STANDARD[n.s - 1] + n.f for n in notes))
    check("sorted by (t, s)",
          notes == sorted(notes, key=lambda n: (n.t, n.s, n.pitch, n.f)))
    first_t = notes[0].t
    first = [n.s for n in notes if n.t == first_t]
    check("chord ordered by ascending string", first == sorted(first), str(first))


def test_dotted_duration() -> None:
    dadagp = _load_script()
    notes, _ = dadagp.convert_track(_track())
    dotted = [n for n in notes if n.s == 2 and n.f == 3]
    check("dotted note present", bool(dotted))
    check("dotted 8th == 720 ticks", bool(dotted) and dotted[0].dur == 720,
          str(dotted[0].dur if dotted else None))


def test_filters() -> None:
    check("standard guitar accepted",
          evaluate(_track(), FilterConfig(min_notes=1, min_bars=1)) is None)
    drop_d = _track(SAMPLE.replace("E5 B4 G4 D4 A3 E3", "E5 B4 G4 D4 A3 D3"))
    verdict = evaluate(drop_d, FilterConfig(min_notes=1, min_bars=1))
    check("non-standard tuning rejected",
          verdict is not None and verdict[0] == "non_standard_tuning", str(verdict))


def test_split() -> None:
    dadagp = _load_script()
    keys = [f"Artist/Song{i}" for i in range(200)]
    splits = {k: dadagp.split_of(k) for k in keys}
    check("split is deterministic", splits == {k: dadagp.split_of(k) for k in keys})
    check("split covers all three buckets",
          set(splits.values()) == {"train", "valid", "test"}, str(set(splits.values())))
    check("song key strips .tex",
          dadagp.song_key("Artist/Song.tex") == "Artist/Song")


def main() -> int:
    test_octave_convention()
    test_chord_preserved()
    test_conversion_contract()
    test_dotted_duration()
    test_filters()
    test_split()
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
