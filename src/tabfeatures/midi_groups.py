"""Group MIDI note events into attacks/chords for labelling and inference.

The candidate identity comes from MIDI, never from audio onset detection.
Audio is used later only to describe the attack around the MIDI time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mido
import pretty_midi

PPQ = 960
BAR_OFFSET_MARKER = "muscriptor:bar_offset="
DEFAULT_CHORD_WINDOW_MS = 30.0


def read_bar_offset(midi_path: str | Path) -> float:
    """Return MuScriptor's written bar-alignment delay, in seconds.

    The MIDI writer shifts every note forward so bar 1 lands on tick 0 and
    records that shift in a marker.  The original audio starts that much
    earlier, so audio-aligned playback/labelling must subtract it.
    """
    try:
        midi = mido.MidiFile(str(midi_path))
    except Exception:
        return 0.0
    for track in midi.tracks:
        for message in track:
            if message.type == "marker" and str(message.text).startswith(BAR_OFFSET_MARKER):
                try:
                    return float(str(message.text)[len(BAR_OFFSET_MARKER):])
                except ValueError:
                    return 0.0
    return 0.0


def _project_tick(pm: pretty_midi.PrettyMIDI, seconds: float) -> int:
    """Convert a MIDI-file time to the project's PPQ-960 tick grid."""
    return int(round(float(pm.time_to_tick(float(seconds))) * PPQ / pm.resolution))


def _note_payload(note: pretty_midi.Note, rank: int) -> dict[str, Any]:
    return {
        "pitch": int(note.pitch),
        "velocity": int(note.velocity),
        "start_s": round(float(note.start), 6),
        "end_s": round(float(note.end), 6),
        "duration_s": round(max(0.0, float(note.end) - float(note.start)), 6),
        "chord_rank": int(rank),
    }


def load_note_groups(
    midi_path: str | Path,
    *,
    chord_window_ms: float = DEFAULT_CHORD_WINDOW_MS,
    lead_in_s: float = 0.18,
    tail_s: float = 0.38,
    next_guard_s: float = 0.02,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Return chord/attack groups derived from a MIDI file.

    Notes whose starts are within ``chord_window_ms`` of a group's first note
    share a group.  Every group carries one entry per MIDI note, so a dead note
    inside a chord is an independent label row.
    """
    path = Path(midi_path)
    pm = pretty_midi.PrettyMIDI(str(path))

    raw: list[pretty_midi.Note] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        raw.extend(instrument.notes)
    raw.sort(key=lambda note: (float(note.start), int(note.pitch)))
    if not raw:
        raise ValueError(f"no non-drum notes in {path}")

    window_s = max(0.001, float(chord_window_ms) / 1000.0)
    grouped: list[list[pretty_midi.Note]] = []
    group_start = 0.0
    for note in raw:
        if not grouped or float(note.start) - group_start > window_s:
            grouped.append([note])
            group_start = float(note.start)
        else:
            grouped[-1].append(note)

    total_duration = (
        float(duration_s)
        if duration_s is not None and float(duration_s) > 0
        else float(pm.get_end_time())
    )
    bar_offset = read_bar_offset(path)
    audio_shift = -float(bar_offset)
    starts = [min(float(note.start) for note in group) for group in grouped]

    groups: list[dict[str, Any]] = []
    for index, group in enumerate(grouped):
        group_start = starts[index] + audio_shift
        next_start = (
            starts[index + 1] + audio_shift
            if index + 1 < len(starts)
            else total_duration
        )
        group_end = max(float(note.end) for note in group) + audio_shift
        playback_end = min(
            total_duration,
            max(group_start + float(tail_s), group_end + 0.15),
            next_start - float(next_guard_s),
        )
        if playback_end <= group_start:
            playback_end = min(total_duration, group_start + 0.12)
        playback_start = max(0.0, group_start - float(lead_in_s))

        ordered = sorted(group, key=lambda note: (int(note.pitch), float(note.start)))
        notes: list[dict[str, Any]] = []
        previous_pitch: int | None = None
        group_id = f"g{index:06d}"
        for rank, note in enumerate(ordered):
            payload = _note_payload(note, rank)
            payload.update(
                {
                    "note_id": f"{group_id}:n{rank}",
                    "group_id": group_id,
                    "tick": _project_tick(pm, float(note.start)),
                    "time_s": round(float(note.start) + audio_shift, 6),
                    "interval_to_lower": (
                        int(note.pitch) - previous_pitch
                        if previous_pitch is not None
                        else 0
                    ),
                }
            )
            previous_pitch = int(note.pitch)
            notes.append(payload)

        groups.append(
            {
                "group_id": group_id,
                "tick": _project_tick(pm, starts[index]),
                "time_s": round(group_start, 6),
                "start_s": round(playback_start, 6),
                "end_s": round(playback_end, 6),
                "next_time_s": round(float(next_start), 6),
                "chord_size": len(notes),
                "notes": notes,
            }
        )

    try:
        beats = [
            round(max(0.0, float(value) - bar_offset), 6)
            for value in pm.get_beats()
        ]
    except Exception:
        beats = []
    try:
        downbeats = [
            round(max(0.0, float(value) - bar_offset), 6)
            for value in pm.get_downbeats()
        ]
    except Exception:
        downbeats = []

    tempo_times, tempo_values = pm.get_tempo_changes()
    return {
        "duration": round(total_duration, 6),
        "bar_offset_s": round(bar_offset, 6),
        "beats": beats,
        "downbeats": downbeats,
        "tempo_changes": [
            [round(float(time_s), 6), round(float(bpm), 6)]
            for time_s, bpm in zip(tempo_times, tempo_values)
        ],
        "time_signatures": [
            [round(float(ts.time), 6), int(ts.numerator), int(ts.denominator)]
            for ts in pm.time_signature_changes
        ],
        "groups": groups,
    }


def flatten_notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every note in a :func:`load_note_groups` payload."""
    return [
        note
        for group in payload.get("groups", [])
        for note in group.get("notes", [])
    ]
