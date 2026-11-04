"""Runtime inference for the tiny XGBoost event model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import extract_event_features, load_audio
from .model import EventModel


def _event_pitch_string(event: dict[str, Any]) -> tuple[int | None, int | None]:
    matched = event.get("matched_note") or {}
    pitch = matched.get("pitch", event.get("pitch"))
    string = event.get("s", event.get("string"))
    try:
        pitch_value = int(pitch) if pitch is not None else None
    except (TypeError, ValueError):
        pitch_value = None
    try:
        string_value = int(string) if string is not None else None
    except (TypeError, ValueError):
        string_value = None
    return pitch_value, string_value


def predict_events(
    audio_path: str | Path,
    events: list[dict[str, Any]],
    model: EventModel,
    *,
    sr: int = 22050,
    before_s: float | None = None,
    after_s: float | None = None,
) -> list[dict[str, Any]]:
    """Score every aligned onset in ``events``."""
    y, actual_sr = load_audio(audio_path, sr=sr)
    predictions: list[dict[str, Any]] = []
    for event in events:
        try:
            onset_s = float(event.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        pitch, string = _event_pitch_string(event)
        features = extract_event_features(
            y,
            actual_sr,
            onset_s,
            pitch=pitch,
            string=string,
            before_s=(
                float(before_s)
                if before_s is not None
                else float(model.feature_window.get("before_s", 0.03))
            ),
            after_s=(
                float(after_s)
                if after_s is not None
                else float(model.feature_window.get("after_s", 0.35))
            ),
        )
        heads = model.predict(features)
        predictions.append(
            {
                "index": event.get("index"),
                "time": round(onset_s, 6),
                "tick": event.get("tick"),
                "matched": bool(event.get("matched")),
                "matched_note": event.get("matched_note"),
                "pitch": pitch,
                "string": string,
                "heads": {key: round(float(value), 6) for key, value in heads.items()},
            }
        )
    return predictions


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
    """Add model-detected dead/percussive events without deleting melody notes.

    A matched note is only dropped for a plain ``dead_string`` prediction when
    the model is not also confident that it is pitched.  Flicks and body
    percussion are additive: the melody note stays and the ``X`` goes on the
    lowest free string.
    """
    from .schema import DEFAULT_THRESHOLDS

    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    result = [dict(note) for note in notes]
    dead_notes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    converted = 0
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
        flick = heads.get("bass_flick", 0.0)
        dead = heads.get("dead_string", 0.0)
        pitched = heads.get("pitched", 0.0)
        score = max(body, flick, dead)
        if score < max(0.0, float(limits["dead_string"]) - float(review_margin)):
            continue

        is_body = body >= float(limits["body_percussion"])
        is_flick = flick >= float(limits["bass_flick"])
        is_dead = dead >= float(limits["dead_string"])
        if score < max(
            float(limits["dead_string"]),
            float(limits["bass_flick"]),
            float(limits["body_percussion"]),
        ):
            review.append(
                {
                    "tick": tick,
                    "time": prediction.get("time"),
                    "heads": prediction.get("heads"),
                    "reason": "event_model_below_threshold",
                }
            )
            continue

        matched = bool(prediction.get("matched"))
        matched_note = prediction.get("matched_note") or {}
        if is_body or is_flick:
            occupied = _occupied(result, dead_notes, grid_ticks)
            string = _lowest_free(occupied, tick, grid_ticks)
            if string is None:
                review.append(
                    {
                        "tick": tick,
                        "time": prediction.get("time"),
                        "heads": prediction.get("heads"),
                        "reason": "no_free_string",
                        "kind": (
                            "body_percussion" if is_body else "bass_flick"
                        ),
                    }
                )
                skipped += 1
                continue
            kind = "body_percussion" if is_body else "dead"
            dead_notes.append(
                {
                    "t": tick,
                    "dur": int(grid_ticks),
                    "s": int(string),
                    "f": 0,
                    "kind": kind,
                    "staccato": bool(is_body),
                    "source": (
                        "event_model_body_percussion"
                        if is_body
                        else "event_model_bass_flick"
                    ),
                    "confidence": round(float(body if is_body else flick), 6),
                }
            )
            additive += 1
            continue

        if is_dead:
            if matched and pitched < 0.50:
                # Drop the MIDI note that the model says is a dead-string hit.
                target_string = prediction.get("string")
                target_pitch = matched_note.get("pitch")
                dropped = False
                for note in result:
                    if int(note.get("t", -1)) != tick or note.get("_drop"):
                        continue
                    if (
                        target_string is not None
                        and int(note.get("s", -1)) != int(target_string)
                    ):
                        continue
                    if (
                        target_pitch is not None
                        and int(note.get("pitch", -1)) != int(target_pitch)
                    ):
                        continue
                    note["_drop"] = True
                    target_string = int(note.get("s", -1))
                    dropped = True
                    break
                occupied = _occupied(result, dead_notes, grid_ticks)
                string = (
                    int(target_string)
                    if target_string is not None
                    and (_slot(tick, grid_ticks), int(target_string)) not in occupied
                    else _lowest_free(occupied, tick, grid_ticks)
                )
                if string is None:
                    review.append(
                        {
                            "tick": tick,
                            "time": prediction.get("time"),
                            "heads": prediction.get("heads"),
                            "reason": "no_free_string",
                            "kind": "dead_string",
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
                        "kind": "dead",
                        "source": (
                            "event_model_dead_string"
                            if dropped
                            else "event_model_dead_string_additive"
                        ),
                        "confidence": round(float(dead), 6),
                    }
                )
                if dropped:
                    converted += 1
                else:
                    additive += 1
            else:
                # No matched note, or the model still hears a real pitch:
                # preserve the note and do not invent a same-string X.
                occupied = _occupied(result, dead_notes, grid_ticks)
                string = _lowest_free(occupied, tick, grid_ticks)
                if string is None:
                    review.append(
                        {
                            "tick": tick,
                            "time": prediction.get("time"),
                            "heads": prediction.get("heads"),
                            "reason": "no_free_string",
                            "kind": "dead_string",
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
                        "kind": "dead",
                        "source": "event_model_dead_string",
                        "confidence": round(float(dead), 6),
                    }
                )
                additive += 1

    review.sort(key=lambda item: float(item.get("heads", {}).get("dead_string", 0.0)), reverse=True)
    stats = {
        "enabled": True,
        "predictions": len(predictions),
        "dead_notes": len(dead_notes),
        "converted": converted,
        "additive": additive,
        "reviewed": len(review),
        "skipped": skipped,
        "thresholds": limits,
    }
    return result, dead_notes, review, stats
