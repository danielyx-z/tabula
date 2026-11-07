"""Apply dead-note predictions to a Tabula ``final.json`` note list."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tabfeatures.midi_groups import load_note_groups

from .features import FeatureConfig, extract_group_rows
from .model import DeadNoteModel


def _key(note: dict[str, Any]) -> tuple[int, int]:
    return int(note["t"]), int(note["s"])


def _match_input_to_final(
    input_notes: list[dict[str, Any]],
    final_notes: list[dict[str, Any]],
    *,
    max_tick_distance: int = 60,
) -> dict[str, dict[str, Any]]:
    """Map MIDI note ids to the corresponding ``final.json`` notes.

    ``midi2tab`` derives final pitches and times from the input, but quantises
    them to its 20-tick grid and may drop an unplayable chord tone.  Match by
    pitch and nearest tick, one final note per input note.
    """
    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in final_notes:
        if note.get("_drop"):
            continue
        by_pitch[int(note.get("pitch", -1))].append(note)
    for notes in by_pitch.values():
        notes.sort(key=lambda item: int(item["t"]))

    used: set[int] = set()
    mapping: dict[str, dict[str, Any]] = {}
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for input_note in input_notes:
        note_id = str(input_note.get("note_id") or "")
        pitch = int(input_note.get("pitch", -1))
        tick = int(input_note.get("tick", -1))
        if not note_id or tick < 0:
            continue
        for final_note in by_pitch.get(pitch, []):
            distance = abs(int(final_note["t"]) - tick)
            if distance <= int(max_tick_distance):
                candidates.append((distance, tick, note_id, final_note))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _distance, _tick, note_id, final_note in candidates:
        if note_id in mapping:
            continue
        identity = id(final_note)
        if identity in used:
            continue
        mapping[note_id] = final_note
        used.add(identity)
    return mapping


def apply_predictions(
    final_notes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    input_notes: list[dict[str, Any]],
    *,
    max_tick_distance: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Relabel dead predictions in place on a copy of ``final_notes``."""
    result = [dict(note) for note in final_notes]
    mapping = _match_input_to_final(
        input_notes, result, max_tick_distance=max_tick_distance
    )
    dead_notes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    unmatched = 0
    noise_count = 0
    already_dead: set[tuple[int, int]] = set()
    for prediction in predictions:
        note_id = str(prediction.get("note_id") or "")
        final_note = mapping.get(note_id)
        item = {
            **prediction,
            "matched": final_note is not None,
        }
        if final_note is None:
            unmatched += 1
            decisions.append(item)
            continue
        item.update(
            {
                "final_t": int(final_note["t"]),
                "final_s": int(final_note["s"]),
                "final_pitch": int(final_note["pitch"]),
            }
        )
        decisions.append(item)
        label = str(prediction.get("label") or "")
        if label not in {"dead", "noise"}:
            continue
        key = _key(final_note)
        if key in already_dead:
            continue
        final_note["_drop"] = True
        already_dead.add(key)
        if label == "noise":
            noise_count += 1
            continue
        dead_notes.append(
            {
                "t": int(final_note["t"]),
                "dur": 1,
                "s": int(final_note["s"]),
                "f": 0,
                "kind": "dead",
                "source": "dead_classifier",
                "confidence": round(float(prediction.get("score", 0.0)), 6),
            }
        )

    dead_notes.sort(key=lambda item: (int(item["t"]), int(item["s"])))
    stats = {
        "enabled": True,
        "predictions": len(predictions),
        "dead": len(dead_notes),
        "noise": noise_count,
        "unmatched": unmatched,
        "input_notes": len(input_notes),
        "final_notes": len(final_notes),
    }
    return result, dead_notes, decisions, stats


def apply_dead_classifier(
    final_notes: list[dict[str, Any]],
    *,
    audio_path: str | Path,
    midi_path: str | Path,
    model: DeadNoteModel,
    threshold: float | None = None,
    config: FeatureConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the model on a MIDI file and relabel its matching final notes."""
    groups_payload = load_note_groups(
        midi_path,
        chord_window_ms=(config or model.config).chord_window_ms,
        duration_s=None,
    )
    groups = groups_payload.get("groups", [])
    input_notes = [
        note for group in groups for note in group.get("notes", [])
    ]
    rows = extract_group_rows(
        audio_path,
        groups,
        config=config or model.config,
    )
    thresholds = dict(model.thresholds)
    if threshold is not None:
        thresholds["dead"] = float(threshold)
    predictions = model.predict(rows, thresholds=thresholds)
    result, dead_notes, decisions, stats = apply_predictions(
        final_notes,
        predictions,
        input_notes,
    )
    stats.update(
        {
            "model": str(model.metadata.get("trained_at") or ""),
            "threshold": float(thresholds.get("dead", 0.8)),
            "dead_threshold": float(thresholds.get("dead", 0.8)),
            "noise_threshold": float(thresholds.get("noise", 0.8)),
            "groups": len(groups),
            "score_min": min(
                (float(item.get("score", 0.0)) for item in predictions), default=0.0
            ),
            "score_max": max(
                (float(item.get("score", 0.0)) for item in predictions), default=0.0
            ),
        }
    )
    return result, dead_notes, decisions, stats
