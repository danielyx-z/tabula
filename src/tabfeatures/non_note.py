"""Validate MIDI-derived notes against the source audio.

The transcriber can put a pitch on a knock, a muted hit, a pick click, or a
body tap.  This module asks whether the audio actually contains a stable
pitched note at the expected MIDI pitch.  There are three outcomes:

* a percussive event on a single note replaces that note with ``x`` on the
  same string;
* a percussive event on a chord keeps the pitched chord and adds one ``x``
  on the lowest free string, which renders beneath the chord;
* ambiguous cases stay as notes and are written to a review file.

The decision is based on signal evidence, never on a song-specific pitch,
string, timestamp, or section.  ``final.json`` on disk is never mutated.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .dead import lowest_free_string
from .evidence import (
    AudioAnalysis,
    best_event_for_note,
    event_audio_class,
    non_note_score,
)
from .ticks import PPQ


def _slot(tick: int, grid_ticks: int) -> int:
    return int(round(int(tick) / int(grid_ticks)))


def _key(note: dict[str, Any]) -> tuple[int, int]:
    return int(note["t"]), int(note["s"])


def _note_payload(note: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": int(note["t"]),
        "s": int(note["s"]),
        "f": int(note.get("f", 0)),
        "pitch": int(note["pitch"]),
        "dur": int(note.get("dur", 0)),
    }


def _event_payload(
    event: dict[str, Any],
    *,
    score: float | None = None,
    pitch_support: float | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time": round(float(event.get("time", 0.0)), 6),
        "class": event_audio_class(event),
        "confidence": round(
            float(
                event.get("audio_confidence", event.get("confidence", 0.0))
                or 0.0
            ),
            6,
        ),
        "strength": round(float(event.get("strength", 0.0) or 0.0), 6),
        "harmonic_ratio": round(
            float(event.get("harmonic_ratio", 0.0) or 0.0), 6
        ),
        "voiced_prob": round(float(event.get("voiced_prob", 0.0) or 0.0), 6),
    }
    if pitch_support is not None:
        payload["pitch_support"] = round(float(pitch_support), 6)
    if score is not None:
        payload["score"] = round(float(score), 6)
    if action is not None:
        payload["action"] = action
    return payload


def _review_item(
    note: dict[str, Any],
    event: dict[str, Any],
    score: float,
    *,
    action: str,
    reason: str | None = None,
    pitch_support: float | None = None,
) -> dict[str, Any]:
    item = {
        "note": _note_payload(note),
        "event": _event_payload(
            event,
            score=score,
            pitch_support=pitch_support,
            action=action,
        ),
        "score": round(float(score), 6),
        "action": action,
    }
    if reason is not None:
        item["reason"] = reason
    return item


def detect_non_notes(
    notes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    analysis: AudioAnalysis | None = None,
    audio_path: str | Path | None = None,
    grid_ticks: int = 240,
    min_confidence: float = 0.60,
    review_confidence: float = 0.45,
    slap_min_pitch_support: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return notes with ``_drop`` markers, replacement/additive dead notes, and review.

    A single percussive note is marked ``_drop`` and replaced by a dead note on
    its original string.  A percussive chord with real pitch support is kept;
    one dead note is placed on the lowest free string so it renders beneath the
    chord.  Medium-confidence candidates are left as notes but listed in the
    review payload.
    """
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if review_confidence < 0.0 or review_confidence > 1.0:
        raise ValueError("review_confidence must be between 0 and 1")
    if slap_min_pitch_support < 0.0:
        raise ValueError("slap_min_pitch_support must be >= 0")

    if analysis is None and audio_path is not None:
        analysis = AudioAnalysis.load(audio_path)

    result = [dict(note) for note in notes]
    dead_notes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen_dead: set[tuple[int, int]] = set()
    scores: dict[tuple[int, int], float] = {}
    events_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    converted_keys: set[tuple[int, int]] = set()
    handled_keys: set[tuple[int, int]] = set()
    slap_slots: set[int] = set()

    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in result:
        if note.get("_drop"):
            continue
        by_slot[_slot(int(note["t"]), grid_ticks)].append(note)

    # A note may align to an onset in the neighbouring grid slot, so match per
    # note rather than per event.  This keeps repeated notes usable even when
    # the detected transient is up to one grid cell away.
    candidates: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any], float, float]] = {}
    for note in result:
        if note.get("_drop"):
            continue
        event = best_event_for_note(note, events, grid_ticks=grid_ticks)
        if event is None:
            continue
        if event_audio_class(event) not in {"dead", "percussive"}:
            # Keep pitched events in the score map so the repeated-run fallback
            # can still catch a pitched outlier surrounded by non-notes.
            scores[_key(note)] = 0.0
            events_by_key[_key(note)] = event
            continue
        pitch_support = (
            analysis.pitch_support(float(event["time"]), int(note["pitch"]))
            if analysis is not None
            else 0.5
        )
        score = non_note_score(event, pitch_support)
        candidates[_key(note)] = (note, event, pitch_support, score)

    candidate_keys = set(candidates)
    no_event = sum(1 for note in result if _key(note) not in candidate_keys)

    converted = 0
    slap = 0
    skipped = 0
    for key in sorted(
        candidates,
        key=lambda item: (
            _slot(item[0], grid_ticks),
            int(item[0]),
            int(item[1]),
        ),
    ):
        note, event, pitch_support, score = candidates[key]
        slot = _slot(int(note["t"]), grid_ticks)
        if note.get("_drop"):
            continue
        group = [item for item in by_slot.get(slot, []) if not item.get("_drop")]
        if not group:
            continue

        scores[key] = score
        events_by_key[key] = event
        if score < min_confidence:
            if score >= review_confidence:
                review.append(
                    _review_item(
                        note,
                        event,
                        score,
                        action="review",
                        pitch_support=pitch_support,
                    )
                )
            continue

        group_support = {
            _key(item): (
                analysis.pitch_support(float(event["time"]), int(item["pitch"]))
                if analysis is not None
                else 0.5
            )
            for item in group
        }
        max_support = max(group_support.values()) if group_support else 0.0
        is_slap = (
            analysis is not None
            and len(group) >= 2
            and max_support >= float(slap_min_pitch_support)
        )

        if is_slap:
            # A slap strum is a real chord plus a percussive attack.  Keep the
            # chord and put one x on the lowest free string beneath it.
            handled_keys.update(_key(item) for item in group)
            if slot not in slap_slots:
                string = lowest_free_string(
                    result,
                    dead_notes,
                    int(note["t"]),
                    grid_ticks=grid_ticks,
                )
                if string is None:
                    skipped += 1
                    review.append(
                        _review_item(
                            note,
                            event,
                            score,
                            action="slap",
                            reason="no_free_string",
                            pitch_support=pitch_support,
                        )
                    )
                    continue
                dead_notes.append(
                    {
                        "t": int(note["t"]),
                        "dur": max(1, int(grid_ticks)),
                        "s": int(string),
                        "f": 0,
                        "kind": "dead",
                        "source": "audio_slap",
                        "confidence": round(float(score), 6),
                    }
                )
                slap_slots.add(slot)
                slap += 1
                if score >= review_confidence:
                    review.append(
                        _review_item(
                            note,
                            event,
                            score,
                            action="slap",
                            pitch_support=pitch_support,
                        )
                    )
            continue

        # Single-note non-note: replace the invented pitch with x on its
        # original string.  A chord with no pitch support follows the same
        # policy for the matched note.
        note["_drop"] = True
        handled_keys.add(key)
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
        converted_keys.add(key)
        converted += 1
        review.append(
            _review_item(
                note,
                event,
                score,
                action="replace",
                pitch_support=pitch_support,
            )
        )

    # Context fallback for a short repeated run.  A body tap can occasionally
    # excite a stable pitch on one hit while the surrounding hits are clearly
    # non-note events.  This is a general run rule: same string and pitch,
    # short gaps, at least four confident non-notes, and a majority of the run.
    # It is only a fallback; isolated repetitive notes are never touched.
    by_pitch: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for note in result:
        key = _key(note)
        if key in scores:
            by_pitch[(int(note["s"]), int(note["pitch"]))].append(note)

    max_run_gap = max(1, int(grid_ticks) * 2)
    context_converted = 0
    for group in by_pitch.values():
        group.sort(key=lambda item: int(item["t"]))
        run: list[dict[str, Any]] = []
        for note in group:
            if run and int(note["t"]) - int(run[-1]["t"]) > max_run_gap:
                run = []
            run.append(note)
            if len(run) < 5:
                continue
            run_keys = [_key(item) for item in run]
            confident = [key for key in run_keys if key in converted_keys]
            if len(confident) < 4:
                continue
            if len(confident) / len(run_keys) < 0.60:
                continue
            if int(run[-1]["t"]) - int(run[0]["t"]) < 2 * PPQ:
                continue
            for item in run:
                key = _key(item)
                if key in converted_keys or key in handled_keys or item.get("_drop"):
                    continue
                score = float(scores.get(key, 0.0))
                item["_drop"] = True
                if key not in seen_dead:
                    dead_notes.append(
                        {
                            "t": int(item["t"]),
                            "dur": max(1, int(grid_ticks)),
                            "s": int(item["s"]),
                            "f": 0,
                            "kind": "dead",
                            "source": "audio_non_note_run",
                            "confidence": 0.60,
                            "reason": "repeated_non_note_run",
                        }
                    )
                    seen_dead.add(key)
                event = events_by_key.get(key)
                if event is not None:
                    review.append(
                        _review_item(
                            item,
                            event,
                            score,
                            action="replace",
                            reason="repeated_non_note_run",
                        )
                    )
                converted_keys.add(key)
                handled_keys.add(key)
                context_converted += 1
                converted += 1

    review.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["note"]["t"]),
            int(item["note"]["s"]),
        )
    )
    stats = {
        "candidates": len(candidates),
        "converted": converted,
        "slap": slap,
        "skipped": skipped,
        "context_converted": context_converted,
        "reviewed": len(review),
        "no_event": no_event,
        "min_confidence": float(min_confidence),
        "review_confidence": float(review_confidence),
        "slap_min_pitch_support": float(slap_min_pitch_support),
    }
    return result, dead_notes, review, stats
