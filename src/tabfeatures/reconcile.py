"""Reconcile audio onsets against MIDI note onsets.

A percussive onset that coincides with a MIDI note is a normal pluck attack and
must not become a dead note.  Only unmatched onsets are candidates.  Matching is
done in tick space, using the MIDI file's own tempo map, with a tolerance of
half a grid cell by default.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

from .ticks import PPQ, seconds_to_tick, snap_tick


def read_midi_notes(
    midi_path: str | Path,
    *,
    program_filter: set[int] | None = None,
) -> tuple[pretty_midi.PrettyMIDI, list[dict[str, Any]]]:
    """Return the PrettyMIDI object and every non-drum note onset.

    Onsets are converted to PPQ-960 ticks so they can be compared directly with
    the audio onsets and with the Tabula ``final.json`` note times.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    onsets: list[dict[str, Any]] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        if program_filter is not None and instrument.program not in program_filter:
            continue
        for note in instrument.notes:
            onsets.append(
                {
                    "tick": seconds_to_tick(pm, float(note.start)),
                    "time": round(float(note.start), 6),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "program": int(instrument.program),
                    "track": instrument.name or "",
                }
            )
    onsets.sort(key=lambda item: (item["tick"], item["pitch"]))
    return pm, onsets


def _nearest_onset(
    midi_onsets: list[dict[str, Any]],
    tick: int,
) -> dict[str, Any] | None:
    if not midi_onsets:
        return None
    ticks = [item["tick"] for item in midi_onsets]
    idx = bisect_left(ticks, tick)
    candidates: list[int] = []
    if idx < len(ticks):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(ticks[i] - tick))
    return midi_onsets[best]


def _match_pass(
    events: list[dict[str, Any]],
    pm: pretty_midi.PrettyMIDI,
    midi_onsets: list[dict[str, Any]],
    *,
    offset_sec: float,
    tolerance_ticks: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        tick = seconds_to_tick(pm, float(event["time"]) + offset_sec)
        nearest = _nearest_onset(midi_onsets, tick)
        if nearest is not None and abs(nearest["tick"] - tick) <= tolerance_ticks:
            out.append(
                {
                    "event": event,
                    "tick": tick,
                    "matched": True,
                    "nearest": nearest,
                    "residual": int(tick - nearest["tick"]),
                }
            )
        else:
            out.append(
                {
                    "event": event,
                    "tick": tick,
                    "matched": False,
                    "nearest": None,
                    "residual": None,
                }
            )
    return out


def align_events(
    events: list[dict[str, Any]],
    pm: pretty_midi.PrettyMIDI,
    midi_onsets: list[dict[str, Any]],
    *,
    grid_ticks: int = PPQ // 4,
    tolerance_ticks: int | None = None,
    auto_offset: bool = True,
    max_offset_sec: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match audio events to MIDI onsets and attach tick positions.

    The returned events are copies with ``tick``, ``raw_tick``, ``snap_error``,
    ``matched``, ``matched_note``, ``class`` and ``confidence`` fields.  The
    alignment summary is JSON-serialisable.
    """
    tolerance = (
        int(tolerance_ticks)
        if tolerance_ticks is not None and int(tolerance_ticks) > 0
        else max(30, int(grid_ticks) // 2)
    )
    tolerance = max(1, tolerance)

    offset_sec = 0.0
    warnings: list[str] = []

    # Use a coarse tolerance for the offset-estimation pass.  A MIDI rendered
    # with a leading-silence trim can be shifted by more than half a grid cell,
    # which would otherwise make every onset look unmatched.
    coarse_tolerance = max(tolerance, 2 * int(grid_ticks))
    first_pass = _match_pass(
        events,
        pm,
        midi_onsets,
        offset_sec=offset_sec,
        tolerance_ticks=coarse_tolerance,
    )
    matched_first = [item for item in first_pass if item["matched"]]

    if auto_offset and matched_first:
        offsets = [
            float(item["event"]["time"]) - float(item["nearest"]["time"])
            for item in matched_first
        ]
        # ``offsets`` are audio_time - midi_time.  The correction added to the
        # audio timeline is the negative of that.
        candidate = -float(np.median(offsets)) if offsets else 0.0
        if abs(candidate) <= float(max_offset_sec):
            offset_sec = candidate
        else:
            warnings.append(
                f"estimated offset {candidate:.3f}s exceeds limit "
                f"{float(max_offset_sec):.3f}s; not applied"
            )

    matched_pass = _match_pass(
        events,
        pm,
        midi_onsets,
        offset_sec=offset_sec,
        tolerance_ticks=tolerance,
    )

    enriched: list[dict[str, Any]] = []
    residuals: list[int] = []
    for item in matched_pass:
        event = dict(item["event"])
        # Keep the timbre decision made before reconciliation.  A matched
        # onset is not automatically a plucked note: a body knock can coincide
        # with a MIDI note the transcriber invented.  The non-note validator
        # and H/P gate consume these fields.
        event["audio_class"] = str(event.get("class") or "")
        event["audio_confidence"] = float(event.get("confidence", 0.0) or 0.0)
        event["audio_strength"] = float(event.get("strength", 0.0) or 0.0)
        raw_tick = int(item["tick"])
        snapped = snap_tick(raw_tick, grid_ticks)
        matched = bool(item["matched"])
        nearest = item["nearest"]
        event["raw_tick"] = raw_tick
        event["tick"] = snapped
        event["snap_error"] = int(snapped - raw_tick)
        event["matched"] = matched
        event["matched_note"] = (
            {
                "tick": int(nearest["tick"]),
                "time": float(nearest["time"]),
                "pitch": int(nearest["pitch"]),
                "program": int(nearest["program"]),
                "track": nearest.get("track", ""),
            }
            if matched and nearest is not None
            else None
        )
        if matched:
            event["class"] = "note_attack"
            event["confidence"] = 0.0
            residuals.append(abs(int(item["residual"] or 0)))
        enriched.append(event)

    total = len(enriched)
    matched_count = sum(1 for event in enriched if event.get("matched"))
    match_rate = (matched_count / total) if total else 0.0

    if total == 0:
        warnings.append("no percussive onsets detected")
    elif match_rate < 0.30:
        warnings.append(
            f"low audio/MIDI match rate ({matched_count}/{total}); "
            "check that the audio and MIDI cover the same performance"
        )
    if abs(offset_sec) > 0.05:
        warnings.append(
            f"applied global audio/MIDI offset of {offset_sec:+.3f}s"
        )
    if len(matched_first) < 3 and auto_offset:
        warnings.append(
            f"only {len(matched_first)} matched onset(s); offset estimate is weak"
        )

    alignment = {
        "applied_offset_sec": round(offset_sec, 6),
        "grid_ticks": int(grid_ticks),
        "tolerance_ticks": int(tolerance),
        "total_events": total,
        "matched_events": matched_count,
        "unmatched_events": total - matched_count,
        "match_rate": round(match_rate, 6),
        "midi_note_count": len(midi_onsets),
        "median_residual_ticks": (
            round(float(np.median(residuals)), 3) if residuals else None
        ),
        "max_residual_ticks": max(residuals) if residuals else None,
        "warnings": warnings,
    }
    return enriched, alignment
