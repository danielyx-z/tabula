"""Export final tablature plus dead notes/articulations to Guitar Pro 5.

This is a feature-aware copy of ``webapp/gp5.py``.  The webapp exporter is left
untouched; this module shares its measure/tempo logic but accepts extra dead
notes and per-note effects.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

import guitarpro as gp
import mido

PPQ = 960
GP_START = 960
GRID_MAX = 120

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

_TUNING_BY_CANONICAL_STRING = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}
_GP_STRINGS = [
    gp.GuitarString(number, value)
    for number, value in ((1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40))
]

_SLIDES = {
    "intoFromAbove": gp.SlideType.intoFromAbove,
    "intoFromBelow": gp.SlideType.intoFromBelow,
    "shiftSlideTo": gp.SlideType.shiftSlideTo,
    "legatoSlideTo": gp.SlideType.legatoSlideTo,
    "outDownwards": gp.SlideType.outDownwards,
    "outUpwards": gp.SlideType.outUpwards,
}


def _clean_title(value: str) -> str:
    value = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in value)
    value = " ".join(value.split()).strip()
    return (value or "Uploaded MIDI")[:80]


def _scale_tick(tick: int, source_ppq: int) -> int:
    return int(round(tick * PPQ / source_ppq))


def read_timing(
    midi_path: str | Path,
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
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
    # Playback stays at one tempo: MuScriptor's expressive drifts are dropped
    # so the score does not drift against itself.
    retained: dict[int, float] = {}
    last_bpm: float | None = None
    for tick, bpm in sorted(tempi.items()):
        if last_bpm is None or abs(float(bpm) - last_bpm) >= 10.0:
            retained[int(tick)] = float(bpm)
            last_bpm = float(bpm)
    return retained, dict(sorted(signatures.items()))


def _measure_plan(max_tick: int, signatures: dict[int, tuple[int, int]]):
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
    if grid < 20:
        grid = math.gcd(length, 20) or 20
    return grid


def _rebuild_rests(voice, grid: int) -> None:
    """Merge adjacent rest beats, then split them into plain values.

    Runs are split independently, so two 32nd rests can end up side by side
    where one 16th belongs; merging the span first and re-splitting greedily
    gives the fewest, largest rests.  Dots are dropped because a hidden rest
    would otherwise leave its augmentation dot on the staff.
    """
    beats = list(voice.beats)
    if not beats:
        return
    rebuilt = []
    index = 0
    while index < len(beats):
        beat = beats[index]
        if beat.status != gp.BeatStatus.rest:
            rebuilt.append(beat)
            index += 1
            continue
        total = 0
        cursor = beat.start
        while index < len(beats) and beats[index].status == gp.BeatStatus.rest:
            total += int(beats[index].duration.time)
            index += 1
        slots = total // grid
        if slots <= 0:
            continue
        for duration, part_slots in _split_durations(slots, grid, dotted=False):
            fresh = gp.Beat(voice, duration=duration, start=cursor)
            fresh.status = gp.BeatStatus.rest
            rebuilt.append(fresh)
            cursor += part_slots * grid
    voice.beats = rebuilt


def _split_durations(
    slots: int, grid: int, *, dotted: bool = True
) -> list[tuple[gp.Duration, int]]:
    """Split a span greedily from the longest value.

    ``dotted=False`` keeps silent rests dot-free: a dotted rest still renders
    its augmentation dot even though the rest itself is hidden.
    """
    remaining = slots * grid
    result: list[tuple[gp.Duration, int]] = []
    for ticks, denominator, is_dotted in _DURATIONS:
        if is_dotted and not dotted:
            continue
        if ticks % grid != 0:
            continue
        while ticks <= remaining:
            result.append(
                (gp.Duration(denominator, isDotted=is_dotted), ticks // grid)
            )
            remaining -= ticks
    if remaining:
        raise ValueError(
            f"cannot represent {remaining} ticks on a {grid}-tick grid"
        )
    return result


def _resolve_string_overlaps(records: list[dict[str, Any]]) -> int:
    """Keep at most one sounding note per string at any time.

    Tabula can emit a sustained note and a later attack on the same string.
    Guitar Pro cannot represent both in one voice, so end the earlier note at
    the attack (or drop an exact duplicate) rather than writing an invalid file.
    """
    by_string: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_string.setdefault(int(record["s"]), []).append(record)

    kept: list[dict[str, Any]] = []
    adjusted = 0
    for sequence in by_string.values():
        sequence.sort(
            key=lambda r: (
                int(r["t"]),
                0 if r["kind"] == "pitched" else 1,
                int(r["f"]),
            )
        )
        resolved: list[dict[str, Any]] = []
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

    records[:] = sorted(kept, key=lambda n: (n["t"], n["s"], n["kind"]))
    return adjusted


def _seconds_between(
    tempi: dict[int, float], start_tick: int, end_tick: int
) -> float:
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


def _note_value(note: dict[str, Any], capo: int) -> int | None:
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


def _slide_type(value: Any) -> gp.SlideType:
    if isinstance(value, gp.SlideType):
        return value
    key = str(value)
    if key not in _SLIDES:
        raise ValueError(f"unknown slide type {value!r}")
    return _SLIDES[key]


def _bend_effect(spec: Any) -> gp.BendEffect:
    if isinstance(spec, dict):
        points = spec.get("points", [])
        bend_type = str(spec.get("type", "bend"))
    else:
        points = spec
        bend_type = "bend"
    effect = gp.BendEffect()
    effect.type = getattr(gp.BendType, bend_type, gp.BendType.bend)
    effect.points = []
    for point in points:
        position, value = int(point[0]), int(point[1])
        effect.points.append(
            gp.BendPoint(
                position=max(0, min(12, position)),
                value=max(0, min(12, value)),
            )
        )
    return effect


def _apply_features(note: gp.Note, feature: dict[str, Any] | None) -> None:
    if not feature:
        return
    if feature.get("hammer") or feature.get("hp"):
        note.effect.hammer = True
    if feature.get("vibrato"):
        note.effect.vibrato = True
    slide = feature.get("slide")
    if slide:
        values = slide if isinstance(slide, list) else [slide]
        note.effect.slides = [_slide_type(value) for value in values]
    bend = feature.get("bend")
    if bend:
        note.effect.bend = _bend_effect(bend)


def _dead_records(
    dead_notes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Turn dead-note markers into percussive records (plus a skip count).

    Dead notes keep the source note's length: alphaTab turns that into a thump
    of the same duration instead of a single-tick click.
    """
    records: list[dict[str, Any]] = []
    skipped = 0
    for raw in dead_notes:
        try:
            string = int(raw["s"])
            tick = int(raw["t"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if string not in _TUNING_BY_CANONICAL_STRING or tick < 0:
            skipped += 1
            continue
        records.append(
            {
                "t": tick,
                "dur": max(480, int(raw.get("dur", 480))),
                "pitch": None,
                "s": string,
                "f": 0,
                "kind": "dead",
                "staccato": bool(
                    raw.get("staccato")
                    or raw.get("kind") == "body_percussion"
                ),
            }
        )
    return records, skipped


def _apply_program_changes(track) -> None:
    """Play dead notes with GM program 28 (muted electric guitar).

    The nylon program is restored on the next beat, so the switch stays local
    to the damped chug.
    """
    program = 24
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                wants = (
                    28
                    if any(note.type == gp.NoteType.dead for note in beat.notes)
                    else 24
                )
                if wants == program:
                    continue
                change = beat.effect.mixTableChange or gp.MixTableChange()
                change.instrument = gp.MixTableItem(value=wants, allTracks=True)
                beat.effect.mixTableChange = change
                program = wants


def build_gp5(
    notes: Iterable[dict[str, Any]],
    timing_midi: str | Path,
    out_path: str | Path,
    *,
    capo: int = 0,
    title: str = "Uploaded MIDI",
    dead_notes: Iterable[dict[str, Any]] | None = None,
    features: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a GP5 file and return a small summary."""
    capo = max(0, int(capo))
    features = features or {}

    records: list[dict[str, Any]] = []
    skipped = 0

    for raw in notes:
        if raw.get("_drop"):
            continue
        value = _note_value(raw, capo)
        if value is None:
            skipped += 1
            continue
        records.append(
            {
                "t": int(raw["t"]),
                "dur": max(1, int(raw["dur"])),
                "pitch": int(raw["pitch"]),
                "s": int(raw["s"]),
                "f": value,
                "kind": "pitched",
            }
        )

    dead_records, skipped_dead = _dead_records(dead_notes or [])
    records.extend(dead_records)
    skipped += skipped_dead
    dead_count = len(dead_records)

    records.sort(key=lambda n: (n["t"], n["s"], n["kind"]))
    overlap_trimmed = _resolve_string_overlaps(records)

    tempi, signatures = read_timing(timing_midi)
    max_tick = max((n["t"] + n["dur"] for n in records), default=0)
    plan = _measure_plan(max_tick, signatures)
    end_tick = plan[-1][0] + plan[-1][1]

    # Drop leading measures that contain no performance at all.  This removes
    # blank bars caused by an intro/silence before the first transcribed note.
    lead_index = 0
    for index, (start, length, _numerator, _denominator) in enumerate(plan):
        end = start + length
        if any(
            note["t"] < end and note["t"] + note["dur"] > start
            for note in records
        ):
            lead_index = index
            break
    lead_start = plan[lead_index][0]
    if lead_start:
        plan = [
            (start - lead_start, length, numerator, denominator)
            for start, length, numerator, denominator in plan[lead_index:]
        ]

    def tempo_at(tick: int) -> float:
        value = 120.0
        for change_tick, bpm in tempi.items():
            if change_tick <= tick:
                value = float(bpm)
            else:
                break
        return value

    initial_tempo = int(round(tempo_at(lead_start)))
    song = gp.Song(
        title=_clean_title(title), artist="Tabula", tempo=initial_tempo
    )
    song.measureHeaders = []
    song.tracks = []

    track = gp.Track(song, name="Guitar", offset=capo, fretCount=24)
    track.strings = [gp.GuitarString(s.number, s.value) for s in _GP_STRINGS]
    track.channel.instrument = 24
    song.tracks.append(track)

    # A note that starts within one grid step of a bar line and keeps sounding
    # past it is moved onto the bar.  Left in place it becomes a stray 32nd
    # (or a 32nd plus a re-strike), which reads and sounds like a note that is
    # too fast; as an attack on the bar it is clean and still in time.
    snap_grid = _grid_for(max(entry[1] for entry in plan))
    bar_starts = [entry[0] for entry in plan]
    bar_starts.append(plan[-1][0] + plan[-1][1])
    adjusted = []
    for record in records:
        onset = int(record["t"])
        length = int(record["dur"])
        for bar in bar_starts:
            if bar - snap_grid < onset < bar and onset + length > bar:
                shift = bar - onset
                record = dict(record, t=bar, dur=max(snap_grid, length - shift))
                break
        adjusted.append(record)
    records = adjusted

    total_beats = 0
    total_notes = 0
    for number, (start, length, numerator, denominator) in enumerate(plan, 1):
        song.newMeasure()
        header = song.measureHeaders[-1]
        header.number = number
        header.start = GP_START + start
        header.timeSignature = gp.TimeSignature(
            numerator, gp.Duration(denominator)
        )
        measure = track.measures[-1]
        voice = measure.voices[0]
        absolute_start = lead_start + start
        end = start + length
        absolute_end = absolute_start + length
        grid = _grid_for(length)
        slots = length // grid

        local: list[tuple[int, int, int, bool]] = []
        local_info: dict[int, tuple[int, int, bool]] = {}
        for index, note in enumerate(records):
            note_end = note["t"] + note["dur"]
            if note["t"] >= absolute_end or note_end <= absolute_start:
                continue
            first = int(
                round((max(note["t"], absolute_start) - absolute_start) / grid)
            )
            last = int(
                round((min(note_end, absolute_end) - absolute_start) / grid)
            )
            # A note in the last half-grid of a measure rounds up to one slot
            # past the bar; keep it in the last slot rather than dropping it.
            first = max(0, min(max(0, slots - 1), first))
            last = max(first + 1, min(slots, last))
            info = (first, last, note["t"] < absolute_start)
            local.append((index, *info))
            local_info[index] = info

        tempo_by_slot: dict[int, float] = {}
        for change_tick, bpm in tempi.items():
            if change_tick == 0:
                # The opening tempo is the song header; repeating it as an
                # automation on bar 1 is redundant.
                continue
            if not (absolute_start <= change_tick < absolute_end):
                continue
            slot = int(round((change_tick - absolute_start) / grid))
            if 0 <= slot < slots:
                tempo_by_slot[slot] = bpm
        tempo_slots = set(tempo_by_slot)

        def slot_priority(item: int) -> tuple[int, int, int]:
            record = records[item]
            return (
                int(record["t"]),
                1 if record["kind"] == "pitched" else 0,
                -int(record["f"]),
            )

        slot_active: list[tuple[int, ...]] = []
        for slot in range(slots):
            # A note can end just after a measure boundary while the next
            # note on the same string starts a few ticks later.  Both round
            # into the same grid slot; writing both would put two notes on
            # one string in the same beat, which corrupts the GP5 stream.
            current: dict[int, int] = {}
            for index, first, last, _before in local:
                if not (first <= slot < last):
                    continue
                string = records[index]["s"]
                previous = current.get(string)
                if previous is None:
                    current[string] = index
                    continue
                # Prefer the later attack; on a tie prefer a pitched note.
                if slot_priority(index) > slot_priority(previous):
                    current[string] = index
            slot_active.append(tuple(sorted(current.values())))

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

        for run_start, run_end, active in runs:
            part_start = run_start
            for duration, part_slots in _split_durations(
                run_end - run_start, grid
            ):
                beat = gp.Beat(
                    voice,
                    duration=duration,
                    start=GP_START + start + part_start * grid,
                )
                if active:
                    beat.status = gp.BeatStatus.normal
                    for index in sorted(
                        active,
                        key=lambda i: (
                            records[i]["s"],
                            records[i]["kind"],
                            records[i]["f"],
                        ),
                    ):
                        record = records[index]
                        first, _last, before = local_info[index]
                        is_tie = before or part_start > first

                        if record["kind"] == "dead":
                            if is_tie:
                                continue
                            gp_note = gp.Note(
                                beat,
                                value=0,
                                string=7 - record["s"],
                                type=gp.NoteType.dead,
                            )
                            if record.get("staccato"):
                                gp_note.effect.staccato = True
                            beat.notes.append(gp_note)
                            total_notes += 1
                            continue

                        if is_tie:
                            # One attack per MIDI note.  A continuation (the
                            # note still sounding from an earlier beat or bar)
                            # is never re-struck: a re-strike at a bar line is
                            # a note the performance does not have.
                            continue

                        gp_note = gp.Note(
                            beat,
                            value=record["f"],
                            string=7 - record["s"],
                            type=gp.NoteType.normal,
                        )
                        if part_start == first:
                            _apply_features(
                                gp_note,
                                features.get((record["t"], record["s"])),
                            )
                        beat.notes.append(gp_note)
                        total_notes += 1
                    if active and not beat.notes:
                        # Every active note was a continuation: keep the slot
                        # silent rather than an empty "normal" beat.
                        beat.status = gp.BeatStatus.rest
                else:
                    beat.status = gp.BeatStatus.rest
                voice.beats.append(beat)
                total_beats += 1
                part_start += part_slots

        _rebuild_rests(voice, grid)

        for slot, bpm in tempo_by_slot.items():
            tick = GP_START + start + slot * grid
            beat = next(
                (
                    candidate
                    for candidate in voice.beats
                    if candidate.start <= tick < candidate.start + candidate.duration.time
                ),
                None,
            )
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
    _apply_program_changes(track)

    gp.write(song, str(out_path), version=(5, 1, 0))

    return {
        "measures": len(plan),
        "beats": total_beats,
        "notes": total_notes,
        "dead_notes": dead_count,
        "overlap_trimmed": overlap_trimmed,
        "skipped": skipped,
        "capo": capo,
        "title": song.title,
        "tempo": initial_tempo,
        "duration": round(_seconds_between(tempi, lead_start, end_tick), 3),
        "trimmed_lead_ticks": lead_start,
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
