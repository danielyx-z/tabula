"""ASCII tab rendering."""

from __future__ import annotations

from dataclasses import dataclass

from core.fretboard import STANDARD, pitch_name
from core.schema import Note


@dataclass
class TabSystem:
    """One system (line group) of tab."""
    strings: list[str]    # 6 strings, each a single line of tab
    bar_lines: list[int]  # character positions of bar lines
    beat_width: int = 4   # chars per 16th note


def notes_to_tab(
    notes: list[Note],
    beats_per_bar: int = 16,
    beat_width: int = 4,
    width: int = 80,
) -> list[TabSystem]:
    """
    Convert notes to ASCII tab systems.
    Returns list of systems (each fits terminal width).
    """
    if not notes:
        return []

    # Sort by time
    notes = sorted(notes, key=lambda n: (n.t, n.pitch))

    # PPQ=960, 16th=60 ticks
    TICKS_PER_16TH = 60

    # Find time range
    start_t = min(n.t for n in notes)
    end_t = max(n.t + n.dur for n in notes)

    # Quantize to 16th grid
    start_16th = start_t // TICKS_PER_16TH
    end_16th = (end_t + TICKS_PER_16TH - 1) // TICKS_PER_16TH
    total_16ths = end_16th - start_16th

    # Measures
    beats_per_measure = beats_per_bar
    measures = total_16ths // beats_per_measure + 1

    # Build one continuous string per string
    measure_width = beats_per_measure * beat_width + 1
    tab_lines = {s: "" for s in range(1, 7)}
    for m in range(measures):
        for s in range(1, 7):
            tab_lines[s] += "-" * (beats_per_measure * beat_width) + "|"

    # Place notes
    for n in notes:
        pos_16th = (n.t - start_16th * TICKS_PER_16TH) // TICKS_PER_16TH
        measure = pos_16th // beats_per_measure
        beat_in_measure = pos_16th % beats_per_measure
        char_pos = measure * measure_width + beat_in_measure * beat_width

        fret_str = str(n.f).rjust(2)
        line = tab_lines[n.s]
        if char_pos + len(fret_str) <= len(line):
            tab_lines[n.s] = line[:char_pos] + fret_str + line[char_pos + len(fret_str):]

    # Build systems (wrap at ``width`` chars)
    systems = []
    chars_per_system = width
    total_chars = measures * measure_width

    for sys_start in range(0, total_chars, chars_per_system):
        sys_end = min(sys_start + chars_per_system, total_chars)
        sys_strings = []
        for s in range(6, 0, -1):  # Display high to low
            line = tab_lines[s][sys_start:sys_end]
            sys_strings.append(f"{pitch_name(STANDARD[s-1]):>3} |{line}")
        systems.append(TabSystem(
            strings=sys_strings,
            bar_lines=[],
            beat_width=beat_width,
        ))

    return systems


def render_tab(notes: list[Note], width: int = 80) -> str:
    """Render notes as ASCII tab string."""
    systems = notes_to_tab(notes, width=width)
    lines = []
    for sys in systems:
        for s in sys.strings:
            lines.append(s)
        lines.append("")  # blank line between systems
    return "\n".join(lines)


def midi_to_tab(midi_path: str) -> str:
    """Placeholder: MIDI -> tab (requires model inference)."""
    return f"# midi2tab {midi_path}\n# Not implemented yet - needs model checkpoint"
