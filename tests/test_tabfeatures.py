"""Tests for the audio-to-tab stitching pipeline.

These tests avoid the Tabula model so they run quickly.  The GP5 test parses the
written file back, and the audio test uses a synthetic tone + clicks so it does
not depend on any external recording.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.audio_events import detect_audio_events  # noqa: E402
from tabfeatures.dead import insert_dead_notes, trim_notes_for_dead  # noqa: E402
from tabfeatures.gp5 import build_gp5  # noqa: E402
from tabfeatures.hp import infer_hp  # noqa: E402
from tabfeatures.overlay import apply_overlay  # noqa: E402
from tabfeatures.reconcile import align_events, read_midi_notes  # noqa: E402
from tabfeatures.render import render_tab_with_features  # noqa: E402
from tabfeatures.ticks import (  # noqa: E402
    parse_grid,
    seconds_to_tick,
    snap_tick,
)


def _write_midi(path: Path, starts: list[float], pitch: int = 55) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=24)
    for start in starts:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=float(start),
                end=float(start) + 0.4,
            )
        )
    pm.instruments.append(instrument)
    pm.write(str(path))


def test_grid_parsing_and_snapping() -> None:
    assert parse_grid("1/16") == 240
    assert parse_grid("1/16T") == 160
    assert parse_grid("16") == 240
    assert parse_grid(120) == 120
    assert snap_tick(250, 240) == 240
    assert snap_tick(400, 240) == 480
    assert snap_tick(-10, 240) == 0
    with pytest.raises(ValueError):
        parse_grid("nope")


def test_seconds_to_tick_uses_tempo_map(tmp_path: Path) -> None:
    midi = tmp_path / "timing.mid"
    _write_midi(midi, [0.0, 0.5])
    pm, _ = read_midi_notes(midi)
    assert seconds_to_tick(pm, 0.0) == 0
    # 120 BPM: one quarter note is 0.5 s = 960 ticks at PPQ 960.
    assert seconds_to_tick(pm, 0.5) == 960


def test_dead_notes_use_next_free_low_string() -> None:
    notes = [
        {"t": 0, "dur": 240, "pitch": 40, "s": 1, "f": 0},
        {"t": 0, "dur": 240, "pitch": 45, "s": 2, "f": 0},
    ]
    events = [
        {"time": 0.0, "tick": 0, "matched": False, "class": "dead", "confidence": 0.9},
        {"time": 0.125, "tick": 240, "matched": False, "class": "dead", "confidence": 0.8},
        {"time": 0.125, "tick": 240, "matched": False, "class": "dead", "confidence": 0.7},
        {"time": 0.25, "tick": 480, "matched": True, "class": "note_attack", "confidence": 0.0},
        {"time": 0.375, "tick": 720, "matched": False, "class": "dead", "confidence": 0.1},
    ]
    dead, stats = insert_dead_notes(
        notes, events, strings="lowest", grid_ticks=240, min_confidence=0.5
    )
    assert {(d["t"], d["s"]) for d in dead} == {(0, 3), (240, 1)}
    assert stats["inserted"] == 2
    assert stats["filtered"]["matched_note_attack"] == 1
    assert stats["filtered"]["below_confidence"] == 1
    assert stats["duplicate_ticks_collapsed"] == 1


def test_dead_notes_all_strings_and_nearest() -> None:
    notes = [
        {"t": 0, "dur": 240, "pitch": 40, "s": 1, "f": 0},
        {"t": 0, "dur": 240, "pitch": 45, "s": 2, "f": 0},
    ]
    events = [
        {"time": 0.0, "tick": 0, "matched": False, "class": "dead", "confidence": 0.9}
    ]
    dead_all, _ = insert_dead_notes(notes, events, strings="all", grid_ticks=240)
    assert {d["s"] for d in dead_all} == {3, 4, 5, 6}

    notes_nearest = [
        {"t": 480, "dur": 240, "pitch": 62, "s": 4, "f": 7},
    ]
    dead_nearest, _ = insert_dead_notes(
        notes_nearest, events, strings="nearest", grid_ticks=240
    )
    assert dead_nearest[0]["s"] == 4


def test_dead_note_collision_uses_quantized_slot() -> None:
    # final.json can sit on a 20-tick grid; tick 260 rounds to the same 1/16
    # slot as dead note tick 240.
    notes = [{"t": 260, "dur": 240, "pitch": 40, "s": 1, "f": 0}]
    events = [
        {"time": 0.0, "tick": 240, "matched": False, "class": "dead", "confidence": 0.9}
    ]
    dead, _ = insert_dead_notes(notes, events, strings="lowest", grid_ticks=240)
    assert dead[0]["s"] == 2


def test_dead_note_trims_ringing_note() -> None:
    notes = [
        {"t": 0, "dur": 960, "pitch": 40, "s": 1, "f": 0},
        {"t": 0, "dur": 960, "pitch": 45, "s": 2, "f": 0},
    ]
    dead = [{"t": 240, "dur": 240, "s": 1, "f": 0}]
    trim_notes_for_dead(notes, dead)
    assert notes[0]["dur"] == 240
    assert notes[1]["dur"] == 960


def test_overlay_applies_features_and_can_clear_hp() -> None:
    notes = [
        {"t": 0, "dur": 240, "pitch": 55, "s": 3, "f": 5},
        {"t": 240, "dur": 240, "pitch": 57, "s": 3, "f": 7},
    ]
    features = {(240, 3): {"hammer": True, "hp": "h"}}
    overlay = {
        "version": 1,
        "notes": [
            {"t": 0, "s": 3, "f": 5, "dead": True},
            {
                "t": 240,
                "s": 3,
                "f": 7,
                "slide": "legatoSlideTo",
                "vibrato": True,
                "bend": [[0, 0], [12, 2]],
            },
        ],
    }
    notes, dead, features, errors = apply_overlay(notes, [], features, overlay)
    assert errors == []
    assert notes[0].get("_drop") is True
    assert {(d["t"], d["s"]) for d in dead} == {(0, 3)}
    assert features[(240, 3)]["hp"] == "h"
    assert features[(240, 3)]["slide"] == "legatoSlideTo"
    assert features[(240, 3)]["vibrato"] is True
    assert len(features[(240, 3)]["bend"]["points"]) == 2

    clear_features = {(240, 3): {"hammer": True, "hp": "h"}}
    _, _, cleared, errors = apply_overlay(
        [{"t": 240, "dur": 240, "pitch": 57, "s": 3, "f": 7}],
        [],
        clear_features,
        {"notes": [{"t": 240, "s": 3, "hammer": False}]},
    )
    assert errors == []
    assert (240, 3) not in cleared


def test_hp_inference_hammer_and_pull() -> None:
    hammer_notes = [
        {"t": 0, "dur": 480, "pitch": 55, "s": 3, "f": 5},
        {"t": 240, "dur": 480, "pitch": 57, "s": 3, "f": 7},
    ]
    features, review = infer_hp(hammer_notes)
    assert features[(240, 3)]["hp"] == "h"
    assert features[(240, 3)]["hammer"] is True
    assert len(review) == 1

    pull_notes = [
        {"t": 0, "dur": 480, "pitch": 57, "s": 3, "f": 7},
        {"t": 240, "dur": 480, "pitch": 55, "s": 3, "f": 5},
    ]
    features, _ = infer_hp(pull_notes)
    assert features[(240, 3)]["hp"] == "p"

    chord_notes = [
        {"t": 0, "dur": 480, "pitch": 55, "s": 3, "f": 5},
        {"t": 0, "dur": 480, "pitch": 59, "s": 4, "f": 4},
        {"t": 240, "dur": 480, "pitch": 57, "s": 3, "f": 7},
    ]
    assert infer_hp(chord_notes)[0] == {}

    open_notes = [
        {"t": 0, "dur": 480, "pitch": 50, "s": 3, "f": 0},
        {"t": 240, "dur": 480, "pitch": 52, "s": 3, "f": 2},
    ]
    assert infer_hp(open_notes)[0] == {}

    far_notes = [
        {"t": 0, "dur": 480, "pitch": 55, "s": 3, "f": 5},
        {"t": 960, "dur": 480, "pitch": 57, "s": 3, "f": 7},
    ]
    assert infer_hp(far_notes)[0] == {}

    same_slot_notes = [
        {"t": 0, "dur": 480, "pitch": 55, "s": 3, "f": 5},
        {"t": 100, "dur": 480, "pitch": 57, "s": 3, "f": 7},
    ]
    assert infer_hp(same_slot_notes, grid_ticks=240)[0] == {}


def test_ascii_render_marks_dead_and_hp() -> None:
    notes = [
        {"t": 0, "dur": 240, "pitch": 55, "s": 3, "f": 5},
        {"t": 240, "dur": 240, "pitch": 57, "s": 3, "f": 7},
    ]
    dead = [{"t": 480, "dur": 240, "s": 1, "f": 0}]
    features = {(240, 3): {"hp": "h", "hammer": True}}
    text = render_tab_with_features(notes, dead, features, capo=0)
    assert "5h" in text
    assert "x" in text
    assert "E2|" in text
    assert "hammer-on" in text


def test_reconcile_audio_onsets_to_midi(tmp_path: Path) -> None:
    midi = tmp_path / "notes.mid"
    _write_midi(midi, [0.0, 0.5, 1.0])
    pm, midi_onsets = read_midi_notes(midi)

    events = [
        {"time": 0.0, "class": "dead", "confidence": 0.9},
        {"time": 0.35, "class": "dead", "confidence": 0.8},
        {"time": 0.5, "class": "dead", "confidence": 0.9},
        {"time": 1.0, "class": "dead", "confidence": 0.9},
    ]
    aligned, alignment = align_events(
        events, pm, midi_onsets, grid_ticks=240, auto_offset=False
    )
    assert [event["matched"] for event in aligned] == [True, False, True, True]
    assert alignment["matched_events"] == 3
    assert alignment["unmatched_events"] == 1
    assert alignment["applied_offset_sec"] == 0.0

    shifted = [
        {"time": 0.1, "class": "dead", "confidence": 0.9},
        {"time": 0.6, "class": "dead", "confidence": 0.9},
        {"time": 1.1, "class": "dead", "confidence": 0.9},
    ]
    aligned, alignment = align_events(
        shifted, pm, midi_onsets, grid_ticks=240, auto_offset=True
    )
    assert alignment["applied_offset_sec"] == pytest.approx(-0.1, abs=0.002)
    assert all(event["matched"] for event in aligned)


def test_gp5_roundtrip_dead_and_articulations(tmp_path: Path) -> None:
    import guitarpro as gp

    midi = tmp_path / "timing.mid"
    _write_midi(midi, [0.0, 0.5, 1.0])

    notes = [
        {"t": 0, "dur": 480, "pitch": 55, "s": 3, "f": 5},
        {"t": 480, "dur": 480, "pitch": 57, "s": 3, "f": 7},
        {"t": 960, "dur": 480, "pitch": 59, "s": 3, "f": 9},
    ]
    dead = [{"t": 480, "dur": 240, "s": 1, "f": 0}]
    features = {
        (480, 3): {"hammer": True, "hp": "h"},
        (960, 3): {
            "vibrato": True,
            "slide": "shiftSlideTo",
            "bend": {"type": "bend", "points": [[0, 0], [12, 2]]},
        },
    }
    output = tmp_path / "score.gp5"
    info = build_gp5(
        notes,
        midi,
        output,
        capo=0,
        title="Test",
        dead_notes=dead,
        features=features,
    )
    assert info["dead_notes"] == 1
    assert info["skipped"] == 0

    song = gp.parse(str(output))
    parsed = [
        note
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        for note in beat.notes
    ]
    dead_parsed = [note for note in parsed if note.type == gp.NoteType.dead]
    assert len(dead_parsed) == 1
    assert dead_parsed[0].string == 6
    assert dead_parsed[0].value == 0

    hammered = [note for note in parsed if note.effect.hammer]
    assert len(hammered) == 1
    assert hammered[0].string == 4

    vibrato = [note for note in parsed if note.effect.vibrato]
    assert len(vibrato) == 1
    assert gp.SlideType.shiftSlideTo in (vibrato[0].effect.slides or [])
    assert vibrato[0].effect.bend is not None
    assert len(vibrato[0].effect.bend.points) == 2


def test_audio_event_detection_finds_click(tmp_path: Path) -> None:
    sr = 22050
    duration = 2.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    t = np.arange(len(y)) / sr
    tone_window = ((t >= 0.15) & (t <= 0.75)).astype(np.float32)
    y += 0.25 * np.sin(2 * math.pi * 440.0 * t).astype(np.float32) * tone_window

    rng = np.random.default_rng(0)
    for click_time in (1.0, 1.5):
        start = int(click_time * sr)
        length = int(0.03 * sr)
        decay = np.exp(-np.arange(length) / (0.004 * sr))
        y[start : start + length] += (
            0.9 * rng.standard_normal(length) * decay
        ).astype(np.float32)

    audio = tmp_path / "clicks.wav"
    sf.write(audio, y, sr, subtype="PCM_16")

    result = detect_audio_events(audio, sr=sr, delta=0.03, wait=2)
    assert result["events"]
    click_events = [
        event
        for event in result["events"]
        if abs(float(event["time"]) - 1.0) <= 0.06
    ]
    assert click_events
    assert click_events[0]["class"] in {"dead", "percussive"}
