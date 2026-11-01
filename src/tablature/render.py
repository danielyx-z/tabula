"""ASCII tab rendering."""

from __future__ import annotations

from dataclasses import dataclass

from core.fretboard import STANDARD, pitch_name
from core.schema import Note

PPQ = 960


@dataclass
class TabSystem:
    """One system (line group) of tab."""
    strings: list[str]   # 6 string lines, high to low


def notes_to_tab(
    notes: list[Note],
    ppq: int = PPQ,
    slots_per_bar: int = 16,   # 16 sixteenths = one bar of 4/4
    width: int = 120,
) -> list[TabSystem]:
    if not notes:
        return []

    ticks_per_slot = ppq // 4  # one 16th note (240 at PPQ=960)

    def slot(t: int) -> int:
        return round(t / ticks_per_slot)

    # Align to absolute bar boundaries (bar 0 starts at tick 0), not to the
    # first note's position.  This preserves pickup bars and offbeat starts.
    first_bar = 0
    last_bar = max(slot(n.t) for n in notes) // slots_per_bar
    n_bars = last_bar - first_bar + 1

    col_w = 3 if any(n.f >= 10 for n in notes) else 2
    bar_w = slots_per_bar * col_w
    stride = bar_w + 1  # +1 for the bar line

    total = n_bars * stride

    def blank() -> list[str]:
        row = ["-"] * total
        for b in range(n_bars):
            row[b * stride + bar_w] = "|"
        return row

    rows = {s: blank() for s in range(1, 7)}

    for n in notes:
        abs_slot = slot(n.t)
        bar = abs_slot // slots_per_bar
        rel = abs_slot - bar * slots_per_bar
        pos = bar * stride + rel * col_w
        for i, ch in enumerate(str(n.f)):  # left-justified
            rows[n.s][pos + i] = ch

    # Wrap on whole bars
    label_w = 3
    bars_per_sys = max(1, (width - label_w - 1) // stride)
    systems = []
    for b0 in range(0, n_bars, bars_per_sys):
        a = b0 * stride
        z = min(b0 + bars_per_sys, n_bars) * stride
        strings = [
            f"{pitch_name(STANDARD[s - 1]):>{label_w}}|{''.join(rows[s][a:z])}"
            for s in range(6, 0, -1)
        ]
        systems.append(TabSystem(strings=strings))
    return systems


def render_tab(notes: list[Note], width: int = 120) -> str:
    lines = []
    for sys in notes_to_tab(notes, width=width):
        lines.extend(sys.strings)
        lines.append("")
    return "\n".join(lines)


def midi_to_tab(midi_path: str) -> str:
    """Placeholder: MIDI -> tab (requires model inference)."""
    return f"# midi2tab {midi_path}\n# Not implemented yet - needs model checkpoint"
