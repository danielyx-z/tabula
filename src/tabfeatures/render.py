"""ASCII tab rendering with dead notes and articulation markers.

The existing Tabula renderer shows only fret numbers.  This renderer keeps the
same 1/16 grid and string order but reserves three characters per slot so
``x``, ``5h`` and ``12p`` fit without shifting the bar lines.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.fretboard import STANDARD, pitch_name

from .ticks import PPQ, bar_beat_label

SLOT_TICKS = PPQ // 4
SLOTS_PER_BAR = 16
CELL_WIDTH = 3


def _slide_glyph(slide: Any) -> str:
    name = str(slide)
    if name in {"outDownwards", "down", "intoFromAbove", "fromAbove"}:
        return "\\"
    return "/"


def _cell_text(note: dict[str, Any], feature: dict[str, Any] | None) -> str:
    text = str(int(note.get("f", 0)))
    feature = feature or {}
    if feature.get("hp"):
        return text + str(feature["hp"])
    if feature.get("slide"):
        return text + _slide_glyph(feature["slide"])
    if feature.get("bend"):
        return text + "b"
    if feature.get("vibrato"):
        return text + "~"
    return text


def _feature_words(feature: dict[str, Any]) -> str:
    parts: list[str] = []
    if feature.get("hp") == "h":
        parts.append("hammer-on")
    elif feature.get("hp") == "p":
        parts.append("pull-off")
    if feature.get("slide"):
        parts.append(f"slide {feature['slide']}")
    if feature.get("bend"):
        parts.append("bend")
    if feature.get("vibrato"):
        parts.append("vibrato")
    return ", ".join(parts)


def render_tab_with_features(
    notes: list[dict[str, Any]],
    dead_notes: list[dict[str, Any]] | None = None,
    features: dict[tuple[int, int], dict[str, Any]] | None = None,
    *,
    capo: int = 0,
    width: int = 120,
    ppq: int = PPQ,
    slot_ticks: int | None = None,
    slots_per_bar: int = SLOTS_PER_BAR,
    cell_width: int = CELL_WIDTH,
    show_feature_detail: bool = True,
) -> str:
    """Render a complete ASCII tab.

    Notes are dictionaries from ``final.json`` (``t``, ``dur``, ``pitch``,
    ``s``, ``f``).  Dead notes use the same shape but ``s``/``t`` only.  Feature
    keys are ``(tick, string)``.
    """
    active_notes = [note for note in notes if not note.get("_drop")]
    dead = list(dead_notes or [])
    features = features or {}

    if slot_ticks is None:
        slot_ticks = max(1, int(ppq) // 4)
    slot_ticks = max(1, int(slot_ticks))
    all_ticks = [int(n["t"]) for n in active_notes] + [int(d["t"]) for d in dead]
    if not all_ticks:
        return ""

    max_slot = max(int(round(tick / slot_ticks)) for tick in all_ticks)
    n_bars = max_slot // int(slots_per_bar) + 1
    stride = int(slots_per_bar) * int(cell_width) + 1

    rows: dict[int, list[str]] = {
        string: ["-"] * (n_bars * stride) for string in range(1, 7)
    }
    for bar in range(n_bars):
        rows[1][bar * stride + int(slots_per_bar) * int(cell_width)] = "|"
        for string in range(2, 7):
            rows[string][bar * stride + int(slots_per_bar) * int(cell_width)] = "|"

    def write_cell(string: int, tick: int, text: str) -> None:
        slot = int(round(tick / slot_ticks))
        bar = slot // int(slots_per_bar)
        relative = slot % int(slots_per_bar)
        position = bar * stride + relative * int(cell_width)
        # Pad the unused part of the cell with string dashes, not spaces: a
        # note should read "6--" on the line, not "6  " with a gap.
        for offset, char in enumerate(text[: int(cell_width)]):
            rows[string][position + offset] = char

    by_string: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in active_notes:
        by_string[int(note["s"])].append(note)
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    for string, sequence in by_string.items():
        sequence.sort(key=lambda n: int(n["t"]))
        for first, second in zip(sequence, sequence[1:]):
            previous[(int(second["t"]), string)] = (int(first["t"]), string)

    # H/P features are stored on the destination note (the note being hammered
    # onto or pulled off to).  For readability put the marker on the previous
    # note's cell: ``5h 7`` rather than ``5 7h``.
    hp_marker_on: dict[tuple[int, int], str] = {}
    for (tick, string), feature in features.items():
        hp = feature.get("hp")
        if hp not in {"h", "p"}:
            continue
        key = (int(tick), int(string))
        predecessor = previous.get(key)
        if predecessor is not None and int(tick) - predecessor[0] <= 2 * slot_ticks:
            hp_marker_on[predecessor] = hp
        else:
            hp_marker_on[key] = hp

    for note in sorted(active_notes, key=lambda n: (int(n["t"]), int(n["s"]))):
        key = (int(note["t"]), int(note["s"]))
        display_feature = dict(features.get(key, {}))
        if key in hp_marker_on:
            display_feature["hp"] = hp_marker_on[key]
        elif display_feature.get("hp"):
            display_feature.pop("hp", None)
        write_cell(
            int(note["s"]),
            int(note["t"]),
            _cell_text(note, display_feature),
        )

    for dead_note in sorted(dead, key=lambda d: (int(d["t"]), int(d["s"]))):
        is_body = bool(
            dead_note.get("staccato")
            or dead_note.get("kind") == "body_percussion"
        )
        write_cell(
            int(dead_note["s"]),
            int(dead_note["t"]),
            "x." if is_body else "x",
        )

    label_width = 3
    bars_per_system = max(
        1, (max(20, int(width)) - label_width - 1) // max(1, stride)
    )
    lines: list[str] = []
    if capo:
        lines.append(f"Capo: {capo}")
        lines.append("")

    for first_bar in range(0, n_bars, bars_per_system):
        start = first_bar * stride
        end = min(first_bar + bars_per_system, n_bars) * stride
        for string in range(6, 0, -1):
            label = pitch_name(STANDARD[string - 1])
            lines.append(
                f"{label:>{label_width}}|{''.join(rows[string][start:end])}"
            )
        lines.append("")

    legend: list[str] = []
    if dead:
        legend.append(f"Dead notes: {len(dead)} (x)")
    if any(d.get("staccato") or d.get("kind") == "body_percussion" for d in dead):
        legend.append("x. = body percussion")
    if any(feature.get("hp") for feature in features.values()):
        legend.append("h = hammer-on, p = pull-off")
    if any(feature.get("slide") for feature in features.values()):
        legend.append("/ \\ = slide")
    if any(feature.get("bend") for feature in features.values()):
        legend.append("b = bend")
    if any(feature.get("vibrato") for feature in features.values()):
        legend.append("~ = vibrato")
    if legend:
        lines.append("Features: " + " | ".join(legend))

    if show_feature_detail:
        detail: list[str] = []
        for (tick, string), feature in sorted(features.items()):
            words = _feature_words(feature)
            if not words:
                continue
            detail.append(
                f"  bar {bar_beat_label(tick)} string {string}: {words}"
            )
        if detail:
            lines.append("")
            lines.append("Feature detail:")
            lines.extend(detail)

    return "\n".join(lines).rstrip() + "\n"
