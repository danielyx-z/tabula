"""General non-note / percussion validation for MIDI-derived tab notes.

The MIDI model can put a pitch on a knock, a muted hit, a pick click, or a
body tap.  Rather than trying to classify every kind of percussion separately,
this module asks whether the audio actually contains a stable pitched note at
the expected MIDI pitch.  Anything that fails confidently is represented as a
dead note (``x``) on the same string; ambiguous cases stay as notes and are
written to a review file.

The decision is based on signal evidence, never on a song-specific pitch,
string, timestamp, or section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence import (
    AudioAnalysis,
    best_event_for_note,
    event_audio_class,
    non_note_score,
)


def detect_non_notes(
    notes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    analysis: AudioAnalysis | None = None,
    audio_path: str | Path | None = None,
    grid_ticks: int = 240,
    min_confidence: float = 0.60,
    review_confidence: float = 0.45,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return notes with ``_drop`` markers, replacement dead notes, and review.

    ``notes`` are copied before modification.  ``final.json`` on disk is never
    touched.  High-confidence non-notes are marked ``_drop`` and get a dead
    note on the same string.  Medium-confidence candidates are left as notes
    but listed in the review payload.
    """
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if review_confidence < 0.0 or review_confidence > 1.0:
        raise ValueError("review_confidence must be between 0 and 1")

    if analysis is None and audio_path is not None:
        analysis = AudioAnalysis.load(audio_path)

    result = [dict(note) for note in notes]
    dead_notes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen_dead: set[tuple[int, int]] = set()

    candidates = 0
    no_event = 0
    converted = 0
    for note in result:
        if note.get("_drop"):
            continue
        event = best_event_for_note(note, events, grid_ticks=grid_ticks)
        if event is None:
            no_event += 1
            continue
        candidates += 1
        pitch_support = (
            analysis.pitch_support(float(event["time"]), int(note["pitch"]))
            if analysis is not None
            else 0.5
        )
        score = non_note_score(event, pitch_support)
        item = {
            "note": {
                "t": int(note["t"]),
                "s": int(note["s"]),
                "f": int(note.get("f", 0)),
                "pitch": int(note["pitch"]),
                "dur": int(note.get("dur", 0)),
            },
            "event": {
                "time": round(float(event.get("time", 0.0)), 6),
                "class": event_audio_class(event),
                "confidence": round(
                    float(
                        event.get(
                            "audio_confidence", event.get("confidence", 0.0)
                        )
                        or 0.0
                    ),
                    6,
                ),
                "strength": round(float(event.get("strength", 0.0) or 0.0), 6),
                "harmonic_ratio": round(float(event.get("harmonic_ratio", 0.0) or 0.0), 6),
                "voiced_prob": round(float(event.get("voiced_prob", 0.0) or 0.0), 6),
                "pitch_support": round(float(pitch_support), 6),
            },
            "score": round(float(score), 6),
        }
        if score >= review_confidence:
            review.append(item)
        if score < min_confidence:
            continue

        note["_drop"] = True
        key = (int(note["t"]), int(note["s"]))
        if key not in seen_dead:
            dead_notes.append(
                {
                    "t": int(note["t"]),
                    "dur": max(1, int(grid_ticks)),
                    "s": int(note["s"]),
                    "f": 0,
                    "kind": "dead",
                    "source": "audio_non_note",
                    "confidence": round(float(score), 6),
                }
            )
            seen_dead.add(key)
        converted += 1

    review.sort(key=lambda item: (-float(item["score"]), int(item["note"]["t"])))
    stats = {
        "candidates": candidates,
        "converted": converted,
        "reviewed": len(review),
        "no_event": no_event,
        "min_confidence": float(min_confidence),
        "review_confidence": float(review_confidence),
    }
    return result, dead_notes, review, stats
