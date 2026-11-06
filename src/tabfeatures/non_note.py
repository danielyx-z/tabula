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
from statistics import median
from typing import Any, Callable

from .dead import lowest_free_string
from .evidence import (
    AudioAnalysis,
    best_event_for_note,
    event_audio_class,
    nearest_unmatched_event_for_note,
    non_note_score,
)
from .note_presence import basic_pitch_votes_for_notes
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
    unmatched_min_confidence: float = 0.50,
    unmatched_max_harmonic_ratio: float = 0.70,
    review_confidence: float = 0.45,
    slap_min_pitch_support: float = 0.12,
    presence: Any | None = None,
    basic_pitch_notes: list[dict[str, Any]] | None = None,
    tick_seconds: Callable[[int], float] | None = None,
    offset_sec: float = 0.0,
    presence_dead_threshold: float = 0.65,
    presence_pitched_threshold: float = 0.70,
    run_presence_ratio: float = 0.90,
    basic_pitch_tolerance_s: float = 0.12,
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
    if unmatched_min_confidence < 0.0 or unmatched_min_confidence > 1.0:
        raise ValueError("unmatched_min_confidence must be between 0 and 1")
    if unmatched_max_harmonic_ratio < 0.0 or unmatched_max_harmonic_ratio > 1.0:
        raise ValueError("unmatched_max_harmonic_ratio must be between 0 and 1")
    if review_confidence < 0.0 or review_confidence > 1.0:
        raise ValueError("review_confidence must be between 0 and 1")
    if slap_min_pitch_support < 0.0:
        raise ValueError("slap_min_pitch_support must be >= 0")
    if presence_dead_threshold < 0.0 or presence_dead_threshold > 1.0:
        raise ValueError("presence_dead_threshold must be between 0 and 1")
    if presence_pitched_threshold < 0.0 or presence_pitched_threshold > 1.0:
        raise ValueError("presence_pitched_threshold must be between 0 and 1")
    if run_presence_ratio < 0.0:
        raise ValueError("run_presence_ratio must be >= 0")

    if presence is not None or basic_pitch_notes is not None:
        return _detect_with_presence(
            notes,
            events,
            analysis=analysis,
            audio_path=audio_path,
            presence=presence,
            basic_pitch_notes=basic_pitch_notes,
            grid_ticks=grid_ticks,
            tick_seconds=tick_seconds,
            offset_sec=offset_sec,
            min_confidence=min_confidence,
            unmatched_min_confidence=unmatched_min_confidence,
            unmatched_max_harmonic_ratio=unmatched_max_harmonic_ratio,
            review_confidence=review_confidence,
            slap_min_pitch_support=slap_min_pitch_support,
            presence_dead_threshold=presence_dead_threshold,
            presence_pitched_threshold=presence_pitched_threshold,
            run_presence_ratio=run_presence_ratio,
            basic_pitch_tolerance_s=basic_pitch_tolerance_s,
        )

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
    used_event_ticks: set[int] = set()
    unmatched_skipped_harmonic = 0

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
            event = nearest_unmatched_event_for_note(
                note, events, grid_ticks=grid_ticks
            )
        if event is None:
            continue
        if (
            not event.get("matched")
            and float(event.get("harmonic_ratio", 0.0) or 0.0)
            > float(unmatched_max_harmonic_ratio)
        ):
            # A strong harmonic onset is probably a real note that merely
            # failed alignment, not a dead-string attack.
            unmatched_skipped_harmonic += 1
            continue
        event_tick = int(event.get("tick", -1))
        if event_tick >= 0:
            used_event_ticks.add(event_tick)
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
        conversion_threshold = (
            min_confidence if event.get("matched") else unmatched_min_confidence
        )
        if score < conversion_threshold:
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
            bool(event.get("matched"))
            and analysis is not None
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
        "unmatched_min_confidence": float(unmatched_min_confidence),
        "unmatched_max_harmonic_ratio": float(unmatched_max_harmonic_ratio),
        "unmatched_skipped_harmonic": unmatched_skipped_harmonic,
        "review_confidence": float(review_confidence),
        "slap_min_pitch_support": float(slap_min_pitch_support),
        "used_event_ticks": sorted(used_event_ticks),
    }
    return result, dead_notes, review, stats


def _pick_group_event(events: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    available = [event for event in events if event is not None]
    if not available:
        return None
    available.sort(
        key=lambda event: (
            bool(event.get("matched")),
            float(event.get("audio_confidence", 0.0) or 0.0),
            float(event.get("strength", 0.0) or 0.0),
        )
    )
    return available[-1]


def _event_review_event(event: dict[str, Any] | None) -> dict[str, Any]:
    if event is not None:
        return event
    return {
        "time": 0.0,
        "class": "presence",
        "audio_confidence": 0.0,
        "strength": 0.0,
        "harmonic_ratio": 0.0,
        "voiced_prob": 0.0,
    }


def _detect_with_presence(
    notes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    analysis: AudioAnalysis | None,
    audio_path: str | Path | None,
    presence: Any | None,
    basic_pitch_notes: list[dict[str, Any]] | None,
    grid_ticks: int,
    tick_seconds: Callable[[int], float] | None,
    offset_sec: float,
    min_confidence: float,
    unmatched_min_confidence: float,
    unmatched_max_harmonic_ratio: float,
    review_confidence: float,
    slap_min_pitch_support: float,
    presence_dead_threshold: float,
    presence_pitched_threshold: float,
    run_presence_ratio: float,
    basic_pitch_tolerance_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use Basic Pitch note presence as the per-note decision.

    The uploaded MIDI is the source of candidate notes.  Basic Pitch is run on
    the same audio.  For every uploaded note:

    * Basic Pitch has the same note at that time -> keep it pitched;
    * Basic Pitch has no note there -> replace it with ``x`` on its string;
    * Basic Pitch was not run -> keep the event-heuristic decision.

    Chords are handled per string.  If some strings are detected and others are
    missing, only the missing strings become ``x``.  A fully pitched percussive
    chord is a slap strum and gets one ``x`` below it.
    """
    result, dead_notes, review, base_stats = detect_non_notes(
        notes,
        events,
        analysis=analysis,
        audio_path=audio_path,
        grid_ticks=grid_ticks,
        min_confidence=min_confidence,
        unmatched_min_confidence=unmatched_min_confidence,
        unmatched_max_harmonic_ratio=unmatched_max_harmonic_ratio,
        review_confidence=review_confidence,
        slap_min_pitch_support=slap_min_pitch_support,
    )
    if basic_pitch_notes is None:
        return result, dead_notes, review, base_stats

    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in result:
        by_slot[_slot(int(note["t"]), grid_ticks)].append(note)

    event_by_key: dict[tuple[int, int], dict[str, Any] | None] = {}
    time_by_key: dict[tuple[int, int], float | None] = {}
    used_event_ticks: set[int] = set(base_stats.get("used_event_ticks", []))
    for note in result:
        key = _key(note)
        event = best_event_for_note(
            note, events, grid_ticks=grid_ticks
        ) or nearest_unmatched_event_for_note(
            note, events, grid_ticks=grid_ticks
        )
        event_by_key[key] = event
        if event is not None:
            event_tick = int(event.get("tick", -1))
            if event_tick >= 0:
                used_event_ticks.add(event_tick)
        if tick_seconds is not None:
            time_by_key[key] = float(tick_seconds(int(note["t"]))) - float(offset_sec)
        elif event is not None:
            time_by_key[key] = float(event.get("time", 0.0) or 0.0)
        else:
            time_by_key[key] = None

    bp_votes = basic_pitch_votes_for_notes(
        result,
        basic_pitch_notes,
        time_by_key=time_by_key,
        tolerance_s=basic_pitch_tolerance_s,
    )
    group_size = {slot: len(group) for slot, group in by_slot.items()}

    dead_by_key = {
        (int(dead["t"]), int(dead["s"])): dead
        for dead in dead_notes
    }
    note_by_key = {_key(note): note for note in result}
    absent_dead = 0
    present_kept = 0

    for key, bp in bp_votes.items():
        note = note_by_key.get(key)
        if note is None:
            continue
        if group_size.get(_slot(int(note["t"]), grid_ticks), 0) != 1:
            # Basic Pitch is a single-note fallback.  Polyphonic chords keep
            # the event/presence decision because Basic Pitch misses notes in
            # dense chords.
            continue
        if bp is False:
            absent_dead += 1
            note["_drop"] = True
            existing = dead_by_key.get(key)
            if existing is None:
                dead_notes.append(
                    {
                        "t": int(note["t"]),
                        "dur": max(1, int(grid_ticks)),
                        "s": int(note["s"]),
                        "f": 0,
                        "kind": "dead",
                        "source": "basic_pitch_absence",
                        "confidence": 1.0,
                        "reason": "basic_pitch_absent",
                    }
                )
                dead_by_key[key] = dead_notes[-1]
            else:
                existing["source"] = "basic_pitch_absence"
                existing["confidence"] = 1.0
                existing["reason"] = "basic_pitch_absent"
            review.append(
                _review_item(
                    note,
                    _event_review_event(event_by_key.get(key)),
                    1.0,
                    action="replace",
                    reason="basic_pitch_absent",
                )
            )
        elif bp is True:
            if note.get("_drop"):
                note.pop("_drop", None)
                present_kept += 1
            if key in dead_by_key:
                dead_notes = [
                    dead
                    for dead in dead_notes
                    if (int(dead["t"]), int(dead["s"])) != key
                ]
                dead_by_key.pop(key, None)

    slap = int(base_stats.get("slap", 0) or 0)

    review.sort(
        key=lambda item: (
            int(item["note"]["t"]),
            int(item["note"]["s"]),
        )
    )
    stats = {
        **base_stats,
        "basic_pitch_notes": len(basic_pitch_notes),
        "basic_pitch_absent_dead": absent_dead,
        "basic_pitch_present_kept": present_kept,
        "slap": slap,
        "used_event_ticks": sorted(used_event_ticks),
        "basic_pitch_tolerance_s": float(basic_pitch_tolerance_s),
        "presence_dead_threshold": float(presence_dead_threshold),
        "presence_pitched_threshold": float(presence_pitched_threshold),
        "run_presence_ratio": float(run_presence_ratio),
    }
    return result, dead_notes, review, stats
