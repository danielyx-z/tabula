"""Runtime inference for the tiny XGBoost event model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import (
    analysis_after,
    analysis_before,
    extract_event_features,
    load_audio,
)
from .model import EventModel


def predict_events(
    audio_path: str | Path,
    events: list[dict[str, Any]],
    model: EventModel,
    *,
    sr: int = 22050,
    before_s: float | None = None,
    after_s: float | None = None,
) -> list[dict[str, Any]]:
    """Score every aligned onset in ``events``.

    Each feature window stops before the next onset.  Fast passages therefore
    get shorter windows instead of overlapping the following note.
    """
    y, actual_sr = load_audio(audio_path, sr=sr)
    max_after = (
        float(after_s)
        if after_s is not None
        else float(model.feature_window.get("after_s", 0.30))
    )
    window_before = (
        float(before_s)
        if before_s is not None
        else float(model.feature_window.get("before_s", 0.02))
    )

    ordered: list[tuple[float, int, dict[str, Any]]] = []
    for position, event in enumerate(events):
        try:
            onset_s = float(event.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        ordered.append((onset_s, position, event))
    ordered.sort(key=lambda item: (item[0], item[1]))

    scored: list[tuple[int, dict[str, Any]]] = []
    for order_index, (onset_s, position, event) in enumerate(ordered):
        next_onset = (
            ordered[order_index + 1][0]
            if order_index + 1 < len(ordered)
            else None
        )
        previous_onset = (
            ordered[order_index - 1][0] if order_index > 0 else None
        )
        features = extract_event_features(
            y,
            actual_sr,
            onset_s,
            before_s=analysis_before(
                onset_s,
                previous_onset,
                max_before_s=window_before,
            ),
            after_s=analysis_after(
                onset_s,
                next_onset,
                max_after_s=max_after,
            ),
        )
        heads = model.predict(features)
        scored.append(
            (
                position,
                {
                    "index": event.get("index"),
                    "time": round(onset_s, 6),
                    "tick": event.get("tick"),
                    "matched": bool(event.get("matched")),
                    "matched_note": event.get("matched_note"),
                    "heads": {
                        key: round(float(value), 6)
                        for key, value in heads.items()
                    },
                },
            )
        )
    scored.sort(key=lambda item: item[0])
    return [prediction for _, prediction in scored]


def _slot(tick: int, grid_ticks: int) -> int:
    return int(round(int(tick) / int(grid_ticks)))


def _occupied(
    notes: list[dict[str, Any]],
    dead_notes: list[dict[str, Any]],
    grid_ticks: int,
) -> set[tuple[int, int]]:
    occupied = {
        (_slot(int(note["t"]), grid_ticks), int(note["s"]))
        for note in notes
        if not note.get("_drop")
    }
    occupied.update(
        {(_slot(int(note["t"]), grid_ticks), int(note["s"])) for note in dead_notes}
    )
    return occupied


def _lowest_free(
    occupied: set[tuple[int, int]],
    tick: int,
    grid_ticks: int,
) -> int | None:
    slot = _slot(tick, grid_ticks)
    for string in range(1, 7):
        if (slot, string) not in occupied:
            return string
    return None


def apply_event_predictions(
    notes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    grid_ticks: int = 240,
    thresholds: dict[str, float] | None = None,
    review_margin: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Add model-detected dead/flick/percussive events without deleting notes.

    ``dead_string`` and ``bass_flick`` share one ``dead_note`` head.  That head
    is additive: the melody note stays and the ``X`` goes on the lowest free
    string.  Body percussion is additive in the same way, with a staccato dot.
    """
    from .schema import DEFAULT_THRESHOLDS

    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    result = [dict(note) for note in notes]
    dead_notes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    additive = 0
    skipped = 0

    ordered = sorted(
        predictions,
        key=lambda item: (
            int(item.get("tick", 0) or 0),
            float(item.get("time", 0.0) or 0.0),
        ),
    )
    for prediction in ordered:
        tick_value = prediction.get("tick")
        if tick_value is None:
            skipped += 1
            continue
        tick = int(tick_value)
        if tick < 0:
            skipped += 1
            continue
        heads = {
            key: float(value)
            for key, value in (prediction.get("heads") or {}).items()
        }
        body = heads.get("body_percussion", 0.0)
        dead = heads.get("dead_note", 0.0)
        score = max(body, dead)
        if score < max(
            0.0,
            min(
                float(limits["dead_note"]),
                float(limits["body_percussion"]),
            )
            - float(review_margin),
        ):
            continue

        is_body = body >= float(limits["body_percussion"])
        is_dead = dead >= float(limits["dead_note"])
        if not is_body and not is_dead:
            review.append(
                {
                    "tick": tick,
                    "time": prediction.get("time"),
                    "heads": prediction.get("heads"),
                    "reason": "event_model_below_threshold",
                }
            )
            continue

        occupied = _occupied(result, dead_notes, grid_ticks)
        string = _lowest_free(occupied, tick, grid_ticks)
        if string is None:
            review.append(
                {
                    "tick": tick,
                    "time": prediction.get("time"),
                    "heads": prediction.get("heads"),
                    "reason": "no_free_string",
                    "kind": "body_percussion" if is_body else "dead_note",
                }
            )
            skipped += 1
            continue
        dead_notes.append(
            {
                "t": tick,
                "dur": int(grid_ticks),
                "s": int(string),
                "f": 0,
                "kind": "body_percussion" if is_body else "dead",
                "staccato": bool(is_body),
                "source": (
                    "event_model_body_percussion"
                    if is_body
                    else "event_model_dead_note"
                ),
                "confidence": round(float(body if is_body else dead), 6),
            }
        )
        additive += 1

    review.sort(
        key=lambda item: max(
            float(item.get("heads", {}).get("dead_note", 0.0)),
            float(item.get("heads", {}).get("body_percussion", 0.0)),
        ),
        reverse=True,
    )
    stats = {
        "enabled": True,
        "predictions": len(predictions),
        "dead_notes": len(dead_notes),
        "converted": 0,
        "additive": additive,
        "reviewed": len(review),
        "skipped": skipped,
        "thresholds": limits,
    }
    return result, dead_notes, review, stats
