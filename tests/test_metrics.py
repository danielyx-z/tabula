"""Plain-assert tests: ``python tests/test_metrics.py`` (no pytest needed)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.fretboard import STANDARD, pitch_of  # noqa: E402
from tablature.metrics import (  # noqa: E402
    aggregate_counts,
    align_predictions,
    check_playability,
    difficulty_terms,
)
from tablature.postproc import _first_valid, postprocess  # noqa: E402
from core.quantise import SHIFT_VOCAB, decompose_shift  # noqa: E402
from core.schema import Note  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def test_quantise() -> None:
    valid = set(SHIFT_VOCAB)
    bad: list[tuple] = []
    # Every multiple of the 20-tick grid must decompose exactly.
    for shift in range(20, 40_001, 20):
        parts = decompose_shift(shift)
        if any(part not in valid for part in parts):
            bad.append((shift, parts))
            break
        if sum(parts) != shift:
            bad.append((shift, parts, sum(parts)))
            break
    check("shift decomposition is exact and stays in vocab", not bad, str(bad[:1]))
    check("zero shift is empty", decompose_shift(0) == [])
    check("two-bar shift", sum(decompose_shift(23040)) == 23040)
    check("shift token count is small", len(SHIFT_VOCAB) == 11, str(SHIFT_VOCAB))
    check(
        "640 ticks is a quarter-note triplet",
        decompose_shift(640) == [480, 120, 40],
        str(decompose_shift(640)),
    )


def test_first_valid() -> None:
    check("lowest-fret fallback for E4", _first_valid(64) == (6, 0), str(_first_valid(64)))
    check("lowest-fret fallback for E2", _first_valid(40) == (1, 0), str(_first_valid(40)))
    check("lowest-fret fallback for A2", _first_valid(45) == (2, 0), str(_first_valid(45)))


def test_align_boundary() -> None:
    gt = [
        Note(0, 480, 64, 6, 0),
        Note(960, 480, 66, 5, 2),
        Note(1920, 480, 67, 5, 3),
        Note(2880, 480, 69, 5, 5),
    ]
    boundary = [True, False, False, True]
    exact = align_predictions([(6, 0), (5, 2), (5, 3), (5, 5)], gt, boundary)
    check("exact predictions", exact.tab_correct == 4, str(exact))
    check("boundary split", (exact.boundary_correct, exact.boundary_total) == (2, 2), str(exact))
    check("interior split", (exact.interior_correct, exact.interior_total) == (2, 2), str(exact))

    short = align_predictions([(6, 0), (5, 3), (5, 5)], gt, boundary)
    check("missing note keeps denominator", short.total_notes == 4, str(short))
    check("missing note only costs its own slot", short.tab_correct == 1, str(short))


def test_postprocess() -> None:
    gt = [
        Note(0, 480, 64, 6, 0),
        Note(480, 480, 66, 5, 2),
        Note(960, 480, 67, 5, 3),
    ]
    out = postprocess([(6, 0)], gt, mode="full")
    check("full post-processing restores note count", len(out) == 3, str(len(out)))
    check(
        "full post-processing preserves pitch",
        all(pitch_of(n.s, n.f, STANDARD) == n.pitch for n in out),
        str([(n.s, n.f, n.pitch) for n in out]),
    )
    check("fallback uses an open string", (out[1].s, out[1].f) == (6, 2), str((out[1].s, out[1].f)))


def test_coverage_provenance() -> None:
    gt = [
        Note(0, 480, 64, 6, 0),
        Note(480, 480, 66, 5, 2),
        Note(960, 480, 67, 5, 3),
        Note(1440, 480, 69, 5, 5),
    ]
    out = postprocess([(6, 0), (5, 2)], gt, mode="full")
    flags = [n.from_prediction for n in out]
    check(
        "padding is marked as fallback",
        flags == [True, True, False, False],
        str(flags),
    )

    # A predicted string that cannot sound the input pitch must fall back.
    far = [Note(0, 480, 88, 6, 24)]
    out_far = postprocess([(1, 0)], far, mode="full")
    check(
        "unplayable predicted string is a fallback",
        out_far[0].from_prediction is False,
        str(out_far[0]),
    )

    counts = align_predictions([(6, 0), (5, 2)], gt, [True, False, False, True])
    counts.covered_notes = 2
    counts.covered_tab_correct = counts.tab_correct
    check(
        "coverage uses the GT denominator",
        abs(counts.coverage() - 0.5) < 1e-9,
        str(counts.coverage()),
    )
    check(
        "tab@pred uses the covered denominator",
        abs(counts.tab_accuracy_at_pred() - 1.0) < 1e-9,
        str(counts.tab_accuracy_at_pred()),
    )


def test_playability() -> None:
    gt = [Note(0, 480, 64, 6, 0), Note(0, 480, 65, 6, 1)]
    overlap, span, simult = check_playability([(6, 0), (6, 1)], gt)
    check("same-string chord counted", overlap == 1, str((overlap, span, simult)))
    check("small chord span accepted", span == 0, str(span))
    check("two-note chord accepted", simult == 0, str(simult))


def test_difficulty() -> None:
    terms = difficulty_terms(
        [Note(0, 960, 40, 1, 0), Note(960, 960, 88, 6, 24)]
    )
    check("one transition", len(terms) == 1, str(terms))
    check("paper's maximum is 18.5", abs(terms[0] - 18.5) < 1e-9, str(terms))


def test_group_chords() -> None:
    from core.group import group_chords

    notes = [
        Note(560, 340, 56, 3, 6),
        Note(560, 400, 60, 4, 5),
        Note(580, 380, 63, 5, 4),
        Note(1300, 280, 68, 6, 4),
    ]
    grouped, dropped = group_chords(notes, window=40)
    check("strum collapses to one onset", grouped[0].t == grouped[2].t == 560)
    check("chord keeps all notes", len(grouped) == 4 and dropped == 0)
    check("late note stays separate", grouped[3].t == 1300)
    check("release tick preserved", grouped[2].t + grouped[2].dur == 960)


def test_fret_diagnostics() -> None:
    gt = [
        Note(0, 480, 64, 6, 0),
        Note(480, 480, 66, 5, 2),
        Note(960, 480, 67, 5, 3),
    ]
    counts = align_predictions([(6, 0), (6, 2), (5, 4)], gt)
    check(
        "string accuracy counts string-only matches",
        abs(counts.string_accuracy() - 2 / 3) < 1e-9,
        str(counts.string_accuracy()),
    )
    check(
        "fret accuracy counts fret-only matches",
        abs(counts.fret_accuracy() - 2 / 3) < 1e-9,
        str(counts.fret_accuracy()),
    )
    check(
        "mean fret over predictions",
        abs(counts.mean_fret() - 2.0) < 1e-9,
        str(counts.mean_fret()),
    )
    check("no high frets", counts.high_fret_rate() == 0.0, str(counts.fret_hist))

    high = align_predictions([(3, 17), (4, 22), (6, 0)], gt)
    check(
        "high fret count at threshold 15",
        high.high_fret_count() == 2,
        str(high.fret_hist),
    )
    check(
        "high fret rate",
        abs(high.high_fret_rate() - 2 / 3) < 1e-9,
        str(high.high_fret_rate()),
    )
    check(
        "fret histogram bins",
        high.fret_hist[0] == 1
        and high.fret_hist[17] == 1
        and high.fret_hist[22] == 1,
        str(high.fret_hist),
    )

    merged = aggregate_counts([counts, high])
    check(
        "aggregated fret histogram and count",
        merged.fret_hist[17] == 1
        and merged.fret_hist[22] == 1
        and merged.pred_fret_count == 6,
        str((merged.pred_fret_count, merged.fret_hist)),
    )


def main() -> int:
    test_quantise()
    test_first_valid()
    test_align_boundary()
    test_postprocess()
    test_coverage_provenance()
    test_playability()
    test_difficulty()
    test_group_chords()
    test_fret_diagnostics()
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
