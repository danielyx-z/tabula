"""Place dead notes on strings without disturbing the existing fingering."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _slot(tick: int, grid_ticks: int) -> int:
    return int(round(int(tick) / int(grid_ticks)))


def _free_strings(
    occupied: set[tuple[int, int]], tick: int, grid_ticks: int
) -> list[int]:
    slot = _slot(tick, grid_ticks)
    return [s for s in range(1, 7) if (slot, s) not in occupied]


def _nearest_string(notes: list[dict[str, Any]], tick: int) -> int | None:
    if not notes:
        return None
    nearest = min(notes, key=lambda n: (abs(int(n["t"]) - tick), int(n["s"])))
    return int(nearest["s"])


def insert_dead_notes(
    notes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    strings: str = "lowest",
    grid_ticks: int = 240,
    min_confidence: float = 0.5,
    max_per_tick: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn unmatched audio candidates into dead-note records.

    ``notes`` is the Tabula ``final.json`` note list.  The returned dead notes
    are separate records; ``final.json`` is never mutated.

    String policy:
      * ``lowest``  -- try string 1 first, then the next free string.
      * ``nearest`` -- try the string of the nearest pitched note first.
      * ``all``     -- mark every free string at that tick (a dead strum).

    If every string is already occupied in that grid slot the candidate is
    skipped and reported in ``stats['skipped']``.
    """
    if strings not in {"lowest", "nearest", "all"}:
        raise ValueError("strings must be one of: lowest, nearest, all")
    if max_per_tick < 1:
        raise ValueError("max_per_tick must be >= 1")

    # Occupancy is measured on the 1/16 (or chosen) grid, not on exact ticks.
    # Tabula's final.json can sit on a 20-tick grid, so a note at tick 260 and a
    # dead note at 240 would otherwise both land in the same rendered slot.
    active_notes = [note for note in notes if not note.get("_drop")]
    occupied: set[tuple[int, int]] = {
        (_slot(int(note["t"]), grid_ticks), int(note["s"]))
        for note in active_notes
    }

    candidates: list[tuple[int, float, dict[str, Any]]] = []
    filtered = Counter()
    for event in events:
        if event.get("matched"):
            filtered["matched_note_attack"] += 1
            continue
        if event.get("class") == "pitched":
            filtered["pitched_event"] += 1
            continue
        confidence = float(event.get("confidence", 0.0) or 0.0)
        if confidence < float(min_confidence):
            filtered["below_confidence"] += 1
            continue
        tick = int(event.get("tick", event.get("raw_tick", -1)))
        if tick < 0:
            filtered["negative_tick"] += 1
            continue
        candidates.append((tick, confidence, event))

    # A single audio transient can produce multiple detected onsets.  Keep the
    # strongest candidate per tick instead of stacking dead notes.
    by_tick: dict[int, tuple[int, float, dict[str, Any]]] = {}
    for tick, confidence, event in candidates:
        previous = by_tick.get(tick)
        if previous is None or confidence > previous[1]:
            by_tick[tick] = (tick, confidence, event)
    duplicate_ticks = len(candidates) - len(by_tick)

    dead_notes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    skipped_reasons = Counter()

    for tick in sorted(by_tick):
        _, confidence, event = by_tick[tick]
        free = _free_strings(occupied, tick, grid_ticks)
        if not free:
            skipped_reasons["all_strings_occupied"] += 1
            skipped.append(
                {
                    "tick": tick,
                    "time": event.get("time"),
                    "confidence": round(confidence, 6),
                    "reason": "all_strings_occupied",
                }
            )
            continue

        if strings == "nearest":
            preferred = _nearest_string(active_notes, tick)
            order = (
                [preferred] + [s for s in range(1, 7) if s != preferred]
                if preferred is not None
                else list(range(1, 7))
            )
            free_ordered = [s for s in order if s in free]
        else:
            free_ordered = free

        selected = free_ordered if strings == "all" else free_ordered[:1]
        if not selected:
            skipped_reasons["all_strings_occupied"] += 1
            skipped.append(
                {
                    "tick": tick,
                    "time": event.get("time"),
                    "confidence": round(confidence, 6),
                    "reason": "all_strings_occupied",
                }
            )
            continue

        for string in selected[: max(1, int(max_per_tick))]:
            dead_notes.append(
                {
                    "t": int(tick),
                    "dur": int(grid_ticks),
                    "s": int(string),
                    "f": 0,
                    "kind": "dead",
                    "source": "audio",
                    "confidence": round(float(confidence), 6),
                }
            )
            occupied.add((_slot(tick, grid_ticks), string))

    stats = {
        "candidates": len(candidates),
        "inserted": len(dead_notes),
        "skipped": len(skipped),
        "skipped_reasons": dict(skipped_reasons),
        "filtered": dict(filtered),
        "duplicate_ticks_collapsed": duplicate_ticks,
        "strings_policy": strings,
        "min_confidence": float(min_confidence),
    }
    return dead_notes, {**stats, "skipped_events": skipped}


def trim_notes_for_dead(
    notes: list[dict[str, Any]],
    dead_notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shorten pitched notes that are ringing when a dead note is struck.

    A dead note on a string mutes that string.  Without trimming, the GP5 file
    would contain a sustained tie and a dead note on the same string at the same
    time, which is not a valid single-string performance.  ``final.json`` on
    disk is never touched; only the in-memory records are adjusted.
    """
    by_string: dict[int, list[int]] = {}
    for dead in dead_notes:
        string = int(dead["s"])
        by_string.setdefault(string, []).append(int(dead["t"]))
    for string in by_string:
        by_string[string].sort()

    for note in notes:
        if note.get("_drop"):
            continue
        string = int(note["s"])
        start = int(note["t"])
        end = start + max(0, int(note.get("dur", 0)))
        for dead_tick in by_string.get(string, []):
            if dead_tick <= start:
                continue
            if dead_tick >= end:
                break
            note["dur"] = max(1, dead_tick - start)
            break
    return notes


def merge_dead_notes(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge dead-note groups, keeping the highest-confidence entry per slot."""
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for group in groups:
        for note in group:
            key = (int(note["t"]), int(note["s"]))
            previous = merged.get(key)
            if previous is None or float(note.get("confidence", 0.0) or 0.0) > float(
                previous.get("confidence", 0.0) or 0.0
            ):
                merged[key] = note
    return [merged[key] for key in sorted(merged)]
