"""Manual articulation overlay.

The overlay is intentionally small and text-editable.  It is applied after the
automatic dead-note insertion and H/P inference, so a manual entry always wins.

Example ``articulations.json``::

    {
      "version": 1,
      "notes": [
        {"t": 480, "s": 3, "f": 7, "hammer": true},
        {"t": 960, "s": 3, "f": 9, "slide": "legatoSlideTo"},
        {"t": 1440, "s": 3, "f": 9, "vibrato": true},
        {"t": 1920, "s": 3, "f": 9, "bend": [[0, 0], [12, 2]]},
        {"t": 2400, "s": 2, "f": 5, "dead": true}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SLIDE_ALIASES: dict[str, str] = {
    "shift": "shiftSlideTo",
    "shiftSlideTo": "shiftSlideTo",
    "legato": "legatoSlideTo",
    "legatoSlideTo": "legatoSlideTo",
    "slide": "legatoSlideTo",
    "outDownwards": "outDownwards",
    "down": "outDownwards",
    "outUpwards": "outUpwards",
    "up": "outUpwards",
    "intoFromBelow": "intoFromBelow",
    "fromBelow": "intoFromBelow",
    "intoFromAbove": "intoFromAbove",
    "fromAbove": "intoFromAbove",
}

BEND_TYPES = {
    "none",
    "bend",
    "bendRelease",
    "bendReleaseBend",
    "prebend",
    "prebendRelease",
    "dip",
    "dive",
    "releaseUp",
    "invertedDip",
    "return_",
    "releaseDown",
}


def empty_overlay() -> dict[str, Any]:
    return {"version": 1, "notes": []}


def load_overlay(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return empty_overlay()
    overlay_path = Path(path)
    if not overlay_path.exists():
        return empty_overlay()
    data = json.loads(overlay_path.read_text())
    if isinstance(data, list):
        return {"version": 1, "notes": data}
    if not isinstance(data, dict):
        raise ValueError("articulations overlay must be a JSON object or list")
    notes = data.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError("articulations overlay 'notes' must be a list")
    return {"version": int(data.get("version", 1)), "notes": notes}


def _normalise_slide(value: Any) -> str | None:
    if value is None or value is False:
        return None
    if not isinstance(value, str):
        return None
    return SLIDE_ALIASES.get(value, value if value in SLIDE_ALIASES.values() else None)


def _validate_bend(value: Any) -> Any:
    if value is None or value is False:
        return None
    if isinstance(value, dict):
        points = value.get("points", [])
        bend_type = value.get("type", "bend")
    elif isinstance(value, list):
        points = value
        bend_type = "bend"
    else:
        raise ValueError("bend must be a list of [position, value] points or an object")

    if not isinstance(points, list) or not points:
        raise ValueError("bend needs at least one [position, value] point")
    cleaned: list[list[int]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"invalid bend point {point!r}")
        position, bend_value = int(point[0]), int(point[1])
        if not 0 <= position <= 12:
            raise ValueError(f"bend position must be 0..12, got {position}")
        if not 0 <= bend_value <= 12:
            raise ValueError(f"bend value must be 0..12, got {bend_value}")
        cleaned.append([position, bend_value])
    if bend_type not in BEND_TYPES:
        raise ValueError(f"unknown bend type {bend_type!r}")
    return {"type": bend_type, "points": cleaned}


def apply_overlay(
    notes: list[dict[str, Any]],
    dead_notes: list[dict[str, Any]],
    features: dict[tuple[int, int], dict[str, Any]],
    overlay: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    list[str],
]:
    """Apply manual entries and return updated copies plus error strings."""
    entries = overlay.get("notes", []) if isinstance(overlay, dict) else overlay
    if not isinstance(entries, list):
        raise ValueError("articulations overlay 'notes' must be a list")

    index: dict[tuple[int, int], dict[str, Any]] = {}
    for note in notes:
        key = (int(note["t"]), int(note["s"]))
        if key in index:
            # Keep the first; duplicate string/onset pairs are already invalid.
            continue
        index[key] = note

    dead_index = {(int(d["t"]), int(d["s"])) for d in dead_notes}
    errors: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"overlay entry is not an object: {entry!r}")
            continue
        try:
            tick = int(entry["t"])
            string = int(entry["s"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"overlay entry needs integer 't' and 's': {entry!r}")
            continue
        if tick < 0:
            errors.append(f"overlay tick must be >= 0, got {tick}")
            continue
        if string not in range(1, 7):
            errors.append(f"overlay string must be 1..6, got {string}")
            continue

        key = (tick, string)
        note = index.get(key)
        is_dead = bool(entry.get("dead")) or entry.get("kind") == "dead"

        if note is None:
            if not is_dead:
                errors.append(
                    f"no note at t={tick} s={string}; "
                    "check final.json timing or add 'dead': true for a new dead note"
                )
                continue
            if key not in dead_index:
                dead_notes.append(
                    {
                        "t": tick,
                        "dur": int(entry.get("dur", 240)),
                        "s": string,
                        "f": 0,
                        "kind": "dead",
                        "source": "manual",
                        "confidence": 1.0,
                    }
                )
                dead_index.add(key)
            continue

        if "f" in entry and int(entry["f"]) != int(note.get("f", -1)):
            errors.append(
                f"fret mismatch at t={tick} s={string}: overlay says "
                f"{entry['f']}, final.json says {note.get('f')}"
            )
            continue

        if is_dead:
            note["_drop"] = True
            if key not in dead_index:
                dead_notes.append(
                    {
                        "t": tick,
                        "dur": int(entry.get("dur", note.get("dur", 240))),
                        "s": string,
                        "f": 0,
                        "kind": "dead",
                        "source": "manual",
                        "confidence": 1.0,
                    }
                )
                dead_index.add(key)
            features.pop(key, None)
            continue

        # A manual override can keep a note the general non-note validator
        # marked for removal.  Remove only the automatic replacement so an
        # unrelated unmatched percussion dead note at the same slot survives.
        keep_non_note = (
            bool(entry.get("keep"))
            or entry.get("non_note") is False
            or entry.get("dead") is False
        )
        if keep_non_note and note.get("_drop"):
            note.pop("_drop", None)
            dead_notes[:] = [
                dead
                for dead in dead_notes
                if not (
                    int(dead["t"]) == tick
                    and int(dead["s"]) == string
                    and dead.get("source") == "audio_non_note"
                )
            ]
            dead_index.discard(key)

        feature = dict(features.get(key, {}))

        wants_hammer = bool(entry.get("hammer"))
        wants_pull = bool(entry.get("pull"))
        hp_value = entry.get("hp")
        clear_hp = hp_value in {"none", "off", "clear", "remove"}
        explicit_hammer_off = (
            "hammer" in entry
            and entry.get("hammer") is False
            and "pull" not in entry
            and "hp" not in entry
        )
        if wants_hammer and wants_pull:
            errors.append(
                f"entry at t={tick} s={string} sets both hammer and pull"
            )
            continue
        if hp_value in {"h", "hammer", "hammer-on"}:
            wants_hammer = True
            hp_value = "h"
        elif hp_value in {"p", "pull", "pull-off"}:
            wants_pull = True
            hp_value = "p"
        elif hp_value is not None and not clear_hp and hp_value not in {"h", "p"}:
            errors.append(f"unknown hp value {hp_value!r} at t={tick} s={string}")
            continue

        if clear_hp or explicit_hammer_off:
            feature.pop("hammer", None)
            feature.pop("hp", None)
            feature["source"] = "manual"
        elif wants_hammer or wants_pull:
            feature["hammer"] = True
            feature["hp"] = "p" if wants_pull else "h"
            feature["source"] = "manual"

        if "slide" in entry:
            slide = _normalise_slide(entry.get("slide"))
            if entry.get("slide") not in {None, False} and slide is None:
                errors.append(
                    f"unknown slide {entry.get('slide')!r} at t={tick} s={string}"
                )
                continue
            if slide is not None:
                feature["slide"] = slide
                feature["source"] = "manual"

        if "vibrato" in entry:
            feature["vibrato"] = bool(entry.get("vibrato"))
            if feature["vibrato"]:
                feature["source"] = "manual"

        if "bend" in entry:
            try:
                bend = _validate_bend(entry.get("bend"))
            except ValueError as exc:
                errors.append(f"bad bend at t={tick} s={string}: {exc}")
                continue
            if bend is not None:
                feature["bend"] = bend
                feature["source"] = "manual"

        if any(name != "source" for name in feature):
            features[key] = feature
        else:
            features.pop(key, None)

    return notes, dead_notes, features, errors
