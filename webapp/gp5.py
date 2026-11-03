"""Export final tablature notes to a Guitar Pro 5 file.

The web UI renders this file with alphaTab, which is a browser-based Guitar
Pro renderer and player.  The exporter keeps the MIDI tempo/time-signature
map, quantises the tab to the same 1/16-note grid used by the ASCII renderer,
and writes tied notes where a note or chord is sustained across a beat.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import guitarpro as gp
import mido

PPQ = 960
# Guitar Pro files start their first measure at one quarter note into the
# timeline.  We keep that convention and shift all source ticks by GP_START.
GP_START = 960
GRID_MAX = 240  # 1/16 note

# Standard note lengths, longest first.  (ticks, duration denominator, dotted)
_DURATIONS = (
    (3840, 1, False),
    (2880, 2, True),
    (1920, 2, False),
    (1440, 4, True),
    (960, 4, False),
    (720, 8, True),
    (480, 8, False),
    (360, 16, True),
    (240, 16, False),
    (180, 32, True),
    (120, 32, False),
    (90, 64, True),
    (60, 64, False),
)

# Our canonical string numbering is 1 = low E, 6 = high E.  Guitar Pro and
# alphaTab use 1 = high E, 6 = low E.
_TUNING_BY_CANONICAL_STRING = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}
_GP_STRINGS = [gp.GuitarString(number, value) for number, value in (
    (1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40)
)]


def _clean_title(value: str) -> str:
    value = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in value)
    value = " ".join(value.split()).strip()
    return (value or "Uploaded MIDI")[:80]


def _scale_tick(tick: int, source_ppq: int) -> int:
    return int(round(tick * PPQ / source_ppq))


def read_timing(midi_path: str | Path) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    """Return tempo and time-signature maps in 960-PPQ ticks."""
    mid = mido.MidiFile(str(midi_path))
    tempi: dict[int, float] = {0: 120.0}
    signatures: dict[int, tuple[int, int]] = {0: (4, 4)}
    tick = 0
    for message in mido.merge_tracks(mid.tracks):
        tick += message.time
        scaled = _scale_tick(tick, mid.ticks_per_beat)
        if message.type == "set_tempo":
            tempi[scaled] = 60_000_000.0 / message.tempo
        elif message.type == "time_signature":
            if message.numerator > 0 and message.denominator > 0:
                signatures[scaled] = (message.numerator, message.denominator)
    return dict(sorted(tempi.items())), dict(sorted(signatures.items()))


def _measure_plan(max_tick: int, signatures: dict[int, tuple[int, int]]):
    """Build (source_start, length, numerator, denominator) measures."""
    changes = dict(sorted(signatures.items()))
    starts = list(changes)
    measures = []
    start = 0
    while start < max_tick or not measures:
        index = bisect_right(starts, start) - 1
        numerator, denominator = changes[starts[index]] if index >= 0 else (4, 4)
        length = int(numerator * (4 * PPQ / denominator))
        if length <= 0:
            numerator, denominator, length = 4, 4, 4 * PPQ
        measures.append((start, length, numerator, denominator))
        start += length
    return measures


def _grid_for(length: int) -> int:
    grid = math.gcd(length, GRID_MAX)
    if grid <= 0:
        grid = GRID_MAX
    # The tablature pipeline quantises to 20 ticks.  Never go below that.
    if grid < 20:
        grid = math.gcd(length, 20) or 20
    return grid


def _split_durations(slots: int, grid: int) -> list[tuple[gp.Duration, int]]:
    """Split a run of slots into standard note values (longest first)."""
    remaining = slots * grid
    result: list[tuple[gp.Duration, int]] = []
    for ticks, denominator, dotted in _DURATIONS:
        if ticks % grid != 0:
            continue
        while ticks <= remaining:
            result.append((gp.Duration(denominator, isDotted=dotted), ticks // grid))
            remaining -= ticks
    if remaining:
        raise ValueError(
            f"cannot represent {remaining} ticks on a {grid}-tick grid"
        )
    return result


def _resolve_string_overlaps(records: list[dict]) -> int:
    """Keep at most one sounding note per string at any time."""
    by_string: dict[int, list[dict]] = {}
    for record in records:
        by_string.setdefault(int(record["s"]), []).append(record)

    kept: list[dict] = []
    adjusted = 0
    for sequence in by_string.values():
        sequence.sort(key=lambda r: (int(r["t"]), int(r["f"])))
        resolved: list[dict] = []
        for record in sequence:
            if resolved and record["t"] == resolved[-1]["t"]:
                adjusted += 1
                continue
            resolved.append(record)
        for previous, following in zip(resolved, resolved[1:]):
            end = previous["t"] + previous["dur"]
            if end > following["t"]:
                previous["dur"] = max(1, following["t"] - previous["t"])
                adjusted += 1
        kept.extend(resolved)

    records[:] = sorted(kept, key=lambda n: (n["t"], n["s"], n["f"]))
    return adjusted


def _seconds_between(tempi: dict[int, float], start_tick: int, end_tick: int) -> float:
    tick = start_tick
    current = 120.0
    total = 0.0
    for change_tick, bpm in sorted(tempi.items()):
        if change_tick <= start_tick:
            current = bpm
            continue
        if change_tick >= end_tick:
            break
        total += (change_tick - tick) * 60.0 / (current * PPQ)
        tick = change_tick
        current = bpm
    total += (end_tick - tick) * 60.0 / (current * PPQ)
    return total


def _note_value(note: dict, capo: int) -> int | None:
    try:
        pitch = int(note["pitch"])
        string = int(note["s"])
        fret = int(note["f"])
    except (KeyError, TypeError, ValueError):
        return None
    if string not in _TUNING_BY_CANONICAL_STRING:
        return None
    if fret < 0 or fret > 24:
        return None
    if _TUNING_BY_CANONICAL_STRING[string] + capo + fret != pitch:
        return None
    return fret


def build_gp5(
    notes: Iterable[dict],
    timing_midi: str | Path,
    out_path: str | Path,
    *,
    capo: int = 0,
    title: str = "Uploaded MIDI",
) -> dict:
    """Write a GP5 file and return a small summary.

    ``notes`` are the records from the pipeline's final.json:
    ``{"t": int, "dur": int, "pitch": int, "s": int, "f": int}`` where
    ``s`` is 1 = low E and ``f`` is capo-relative.
    """
    capo = max(0, int(capo))
    valid: list[dict] = []
    skipped = 0
    for raw in notes:
        value = _note_value(raw, capo)
        if value is None:
            skipped += 1
            continue
        valid.append({
            "t": int(raw["t"]),
            "dur": max(1, int(raw["dur"])),
            "pitch": int(raw["pitch"]),
            "s": int(raw["s"]),
            "f": value,
        })
    valid.sort(key=lambda n: (n["t"], n["s"], n["f"]))
    overlap_trimmed = _resolve_string_overlaps(valid)

    tempi, signatures = read_timing(timing_midi)
    max_tick = max((n["t"] + n["dur"] for n in valid), default=0)
    plan = _measure_plan(max_tick, signatures)

    initial_tempo = int(round(next(iter(tempi.values()), 120.0)))
    song = gp.Song(title=_clean_title(title), artist="Tabula", tempo=initial_tempo)
    # Song() creates a default track and measure; replace them with ours.
    song.measureHeaders = []
    song.tracks = []

    track = gp.Track(song, name="Guitar", offset=capo, fretCount=24)
    track.strings = [gp.GuitarString(s.number, s.value) for s in _GP_STRINGS]
    track.channel.instrument = 24  # Concert Nylon Guitar
    song.tracks.append(track)

    total_beats = 0
    total_notes = 0
    for number, (start, length, numerator, denominator) in enumerate(plan, 1):
        song.newMeasure()
        header = song.measureHeaders[-1]
        header.number = number
        header.start = GP_START + start
        header.timeSignature = gp.TimeSignature(numerator, gp.Duration(denominator))
        measure = track.measures[-1]
        voice = measure.voices[0]
        end = start + length
        grid = _grid_for(length)
        slots = length // grid

        # Notes that overlap this measure, with quantised local slots.
        local = []
        local_info: dict[int, tuple[int, int, bool]] = {}
        for index, note in enumerate(valid):
            note_end = note["t"] + note["dur"]
            if note["t"] >= end or note_end <= start:
                continue
            first = int(round((max(note["t"], start) - start) / grid))
            last = int(round((min(note_end, end) - start) / grid))
            first = max(0, min(slots, first))
            last = max(first + 1, min(slots, last))
            info = (first, last, note["t"] < start)
            local.append((index, *info))
            local_info[index] = info

        # Tempo boundary slots are forced run boundaries so a tempo change
        # inside a sustained note can be represented at the closest 1/16.
        tempo_by_slot: dict[int, float] = {}
        for change_tick, bpm in tempi.items():
            if not (start <= change_tick < end):
                continue
            slot = int(round((change_tick - start) / grid))
            if 0 <= slot < slots:
                tempo_by_slot[slot] = bpm
        tempo_slots = set(tempo_by_slot)

        slot_active: list[tuple[int, ...]] = []
        for slot in range(slots):
            slot_active.append(tuple(
                index for index, first, last, _before in local
                if first <= slot < last
            ))

        # Coalesce equal active sets into runs, splitting at tempo changes.
        runs = []
        run_start = 0
        for slot in range(1, slots + 1):
            changed = (
                slot == slots
                or slot_active[slot] != slot_active[slot - 1]
                or slot in tempo_slots
            )
            if changed:
                runs.append((run_start, slot, slot_active[slot - 1]))
                run_start = slot

        beat_by_slot: dict[int, gp.Beat] = {}
        for run_start, run_end, active in runs:
            part_start = run_start
            for duration, part_slots in _split_durations(run_end - run_start, grid):
                beat = gp.Beat(
                    voice,
                    duration=duration,
                    start=GP_START + start + part_start * grid,
                )
                beat_by_slot.setdefault(part_start, beat)
                if active:
                    beat.status = gp.BeatStatus.normal
                    for index in sorted(
                        active, key=lambda i: (valid[i]["s"], valid[i]["f"])
                    ):
                        note = valid[index]
                        first, _last, before = local_info[index]
                        is_tie = before or part_start > first
                        beat.notes.append(gp.Note(
                            beat,
                            value=0 if is_tie else note["f"],
                            string=7 - note["s"],
                            type=gp.NoteType.tie if is_tie else gp.NoteType.normal,
                        ))
                        total_notes += 1
                else:
                    beat.status = gp.BeatStatus.rest
                voice.beats.append(beat)
                total_beats += 1
                part_start += part_slots

        for slot, bpm in tempo_by_slot.items():
            beat = beat_by_slot.get(slot)
            if beat is None:
                continue
            beat.effect.mixTableChange = gp.MixTableChange(
                tempo=gp.MixTableItem(
                    value=int(round(bpm)), duration=0, allTracks=True
                ),
                hideTempo=True,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gp.write(song, str(out_path), version=(5, 1, 0))

    end_tick = plan[-1][0] + plan[-1][1]
    return {
        "measures": len(plan),
        "beats": total_beats,
        "notes": total_notes,
        "overlap_trimmed": overlap_trimmed,
        "skipped": skipped,
        "capo": capo,
        "title": song.title,
        "tempo": initial_tempo,
        "duration": round(_seconds_between(tempi, 0, end_tick), 3),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Export final.json to GP5")
    parser.add_argument("final_json")
    parser.add_argument("timing_midi")
    parser.add_argument("output")
    parser.add_argument("--capo", type=int, default=0)
    parser.add_argument("--title", default="Uploaded MIDI")
    args = parser.parse_args()
    records = json.loads(Path(args.final_json).read_text(encoding="utf-8"))
    info = build_gp5(
        records,
        args.timing_midi,
        args.output,
        capo=args.capo,
        title=args.title,
    )
    print(json.dumps(info, indent=2))
