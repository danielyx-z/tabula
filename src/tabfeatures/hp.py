"""Conservative hammer-on / pull-off inference.

Only adjacent notes on the same string are considered.  The rules are
deliberately strict: an inferred articulation is useful, a wrong automatic
hammer-on changes the performance.  Everything is written to
``hp_review.json`` so it can be checked.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def infer_hp(
    notes: list[dict[str, Any]],
    *,
    max_gap_ticks: int = 480,
    max_interval: int = 7,
    ring_tolerance: int = 60,
    skip_chords: bool = True,
    grid_ticks: int = 240,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    """Return ``{(tick, string): feature}`` and a review list."""
    if max_gap_ticks <= 0:
        return {}, []

    active = [note for note in notes if not note.get("_drop")]
    onset_counts = Counter(int(note["t"]) for note in active)

    by_string: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in active:
        by_string[int(note["s"])].append(note)
    for string in by_string:
        by_string[string].sort(key=lambda n: (int(n["t"]), int(n["f"])))

    features: dict[tuple[int, int], dict[str, Any]] = {}
    review: list[dict[str, Any]] = []

    for string, sequence in by_string.items():
        for first, second in zip(sequence, sequence[1:]):
            t1 = int(first["t"])
            t2 = int(second["t"])
            gap = t2 - t1
            if gap <= 0 or gap > int(max_gap_ticks):
                continue
            if int(grid_ticks) > 0 and int(round(t1 / grid_ticks)) == int(
                round(t2 / grid_ticks)
            ):
                # Both notes would render in the same 1/16 cell.
                continue
            if t2 > t1 + int(first.get("dur", 0)) + int(ring_tolerance):
                continue

            interval = abs(int(second["pitch"]) - int(first["pitch"]))
            if interval < 1 or interval > int(max_interval):
                continue
            if int(first.get("f", 0)) <= 0 or int(second.get("f", 0)) <= 0:
                continue
            if skip_chords and (
                onset_counts[t1] > 1 or onset_counts[t2] > 1
            ):
                continue
            if (t1, string) in features or (t2, string) in features:
                continue

            kind = "h" if int(second["pitch"]) > int(first["pitch"]) else "p"
            confidence = 0.5 + 0.5 * (1.0 - gap / float(max_gap_ticks))
            if interval > 5:
                confidence *= 0.8
            confidence = round(max(0.0, min(1.0, confidence)), 6)

            features[(t2, string)] = {
                "hammer": True,
                "hp": kind,
                "source": "inferred",
                "confidence": confidence,
            }
            review.append(
                {
                    "from": {
                        "t": t1,
                        "s": string,
                        "f": int(first["f"]),
                        "pitch": int(first["pitch"]),
                    },
                    "to": {
                        "t": t2,
                        "s": string,
                        "f": int(second["f"]),
                        "pitch": int(second["pitch"]),
                    },
                    "kind": kind,
                    "gap_ticks": gap,
                    "interval_semitones": interval,
                    "confidence": confidence,
                }
            )

    return features, review
