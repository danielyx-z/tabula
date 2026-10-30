"""MIDI parser -> canonical note events.

Extracts note events from a MIDI file (single track or multi-track) and
returns them in the same Note format used by the rest of the pipeline:
onset (ticks, PPQ 960), duration (ticks), MIDI pitch, string, fret.

For a pure MIDI->tab tool we don't know the string/fret yet; the model
predicts those.  This parser therefore returns notes with placeholder
string/fret (0, 0); the downstream encoder only needs onset, duration,
and pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pretty_midi

from core.schema import Note


PPQ = 960


@dataclass
class MidiTrack:
    """One MIDI track converted to our internal representation."""
    name: str
    program: int
    is_drum: bool
    notes: list[Note]
    tempo_changes: list[tuple[float, float]]  # (time_sec, bpm)
    time_signatures: list[tuple[float, int, int]]  # (time_sec, numerator, denominator)
    key_signatures: list[tuple[float, int]]  # (time_sec, key_number)


def _ticks_from_seconds(pm: pretty_midi.PrettyMIDI, seconds: float) -> int:
    """Convert seconds to ticks at PPQ 960, following the tempo map.

    pretty_midi's time_to_tick() returns ticks at the file's own resolution
    (e.g. 220 for basic-pitch output).  Scale to our standard PPQ=960 so
    TIME_SHIFT tokens match the training distribution.
    """
    ticks_at_file_res = pm.time_to_tick(seconds)
    return int(round(ticks_at_file_res * PPQ / pm.resolution))


def _seconds_from_ticks(pm: pretty_midi.PrettyMIDI, ticks: int) -> float:
    return pm.tick_to_time(ticks)


def parse_midi(
    path: str | Path,
    *,
    target_program: Optional[int] = None,
    merge_tracks: bool = True,
    min_duration_ticks: int = 1,
    max_pitch: int = 96,
    min_pitch: int = 36,
) -> list[MidiTrack] | MidiTrack:
    """
    Parse a MIDI file into canonical note events.

    Parameters
    ----------
    path : Path to .mid/.midi file
    target_program : If set, only keep tracks with this GM program (0-127).
                     If None, keep all non-drum tracks.
    merge_tracks : If True and target_program selects multiple tracks (or all),
                   merge them into a single MidiTrack. If False, return list.
    min_duration_ticks : Drop notes shorter than this (after quantisation).
    max_pitch / min_pitch : Pitch range to keep (inclusive).

    Returns
    -------
    Single MidiTrack if merge_tracks, else list[MidiTrack].
    """
    pm = pretty_midi.PrettyMIDI(str(path))

    tracks: list[MidiTrack] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if target_program is not None and inst.program != target_program:
            continue

        # Collect notes with exact tick timing
        track_notes: list[Note] = []
        for note in inst.notes:
            pitch = note.pitch
            if not (min_pitch <= pitch <= max_pitch):
                continue

            onset_tick = _ticks_from_seconds(pm, note.start)
            end_tick = _ticks_from_seconds(pm, note.end)
            dur = end_tick - onset_tick
            if dur < min_duration_ticks:
                continue

            track_notes.append(Note(
                t=onset_tick,
                dur=dur,
                pitch=pitch,
                s=0,   # placeholder; model predicts this
                f=0,
            ))

        if not track_notes:
            continue

        # Sort by onset, then pitch
        track_notes.sort(key=lambda n: (n.t, n.pitch))

        # Extract tempo/time signature info
        tempo_times, tempo_bpms = pm.get_tempo_changes()
        tempo_changes = list(zip(tempo_times, tempo_bpms))

        ts_changes = []
        for ts in pm.time_signature_changes:
            ts_changes.append((ts.time, ts.numerator, ts.denominator))

        ks_changes = []
        for ks in pm.key_signature_changes:
            ks_changes.append((ks.time, ks.key_number))

        tracks.append(MidiTrack(
            name=inst.name or f"program_{inst.program}",
            program=inst.program,
            is_drum=False,
            notes=track_notes,
            tempo_changes=tempo_changes,
            time_signatures=ts_changes,
            key_signatures=ks_changes,
        ))

    if not tracks:
        raise ValueError(f"No suitable tracks found in {path}")

    if merge_tracks:
        # Merge all tracks: concatenate notes, sort globally
        all_notes: list[Note] = []
        for tr in tracks:
            all_notes.extend(tr.notes)
        all_notes.sort(key=lambda n: (n.t, n.pitch))

        # Use the first track's tempo/time-sig as representative
        first = tracks[0]
        return MidiTrack(
            name=f"merged_{len(tracks)}_tracks",
            program=first.program,
            is_drum=False,
            notes=all_notes,
            tempo_changes=first.tempo_changes,
            time_signatures=first.time_signatures,
            key_signatures=first.key_signatures,
        )

    return tracks


def midi_to_encoder_tokens(
    notes: list[Note],
    vocab,
    base_t: int = 0,
    snap_shifts: bool = False,
) -> list[int]:
    """
    Convert a list of Notes to encoder token IDs (NOTE_ON, NOTE_OFF, TIME_SHIFT).

    This mirrors `notes_to_events` but accepts Notes with placeholder s/f.
    """
    from tablature.tokenizer import notes_to_events
    # notes_to_events expects Note objects with .t, .dur, .pitch, .s, .f
    # It ignores s/f for encoder.  We just need to ensure the Note objects
    # have the right fields.
    event_tokens = notes_to_events(notes, base_t=base_t, snap_shifts=snap_shifts)
    return vocab.encode(event_tokens)


__all__ = ["parse_midi", "MidiTrack", "midi_to_encoder_tokens", "PPQ"]