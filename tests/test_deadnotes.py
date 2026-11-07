"""Tests for MIDI-anchored dead-note labelling and classifier plumbing."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.deadnotes.features import (  # noqa: E402
    FEATURE_NAMES,
    AudioContext,
    FeatureConfig,
    extract_group_rows,
)
from tabfeatures.deadnotes.inference import apply_predictions  # noqa: E402
from tabfeatures.midi_groups import load_note_groups  # noqa: E402


def _write_chord_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=90.0)
    inst = pretty_midi.Instrument(program=24)
    for pitch in (40, 47, 52):
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=0.0, end=0.6)
        )
    inst.notes.append(
        pretty_midi.Note(velocity=80, pitch=55, start=0.75, end=1.0)
    )
    pm.instruments.append(inst)
    pm.write(str(path))


def _write_tone(path: Path, *, sr: int = 22050) -> None:
    duration = 2.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    t = np.arange(len(y)) / sr
    for start, pitch in ((0.0, 40.0), (0.0, 47.0), (0.0, 52.0), (0.75, 55.0)):
        f0 = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
        mask = (t >= start) & (t < start + 0.5)
        y[mask] += 0.12 * np.sin(2 * math.pi * f0 * (t[mask] - start))
    sf.write(path, y, sr, subtype="PCM_16")


def test_midi_groups_are_chord_members_not_audio_onsets(tmp_path: Path) -> None:
    midi = tmp_path / "take.mid"
    _write_chord_midi(midi)
    payload = load_note_groups(midi)
    assert len(payload["groups"]) == 2
    chord = payload["groups"][0]
    assert chord["chord_size"] == 3
    assert [note["pitch"] for note in chord["notes"]] == [40, 47, 52]
    assert len({note["note_id"] for note in chord["notes"]}) == 3
    assert chord["notes"][1]["interval_to_lower"] == 7
    assert payload["duration"] >= 0.99


def test_feature_rows_are_one_per_chord_member(tmp_path: Path) -> None:
    midi = tmp_path / "take.mid"
    audio = tmp_path / "take.wav"
    _write_chord_midi(midi)
    _write_tone(audio)
    groups = load_note_groups(midi)["groups"]
    rows = extract_group_rows(audio, groups, config=FeatureConfig())
    assert len(rows) == 4
    assert [row["note"]["pitch"] for row in rows] == [40, 47, 52, 55]
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    first = rows[0]["features"]
    assert "n_presence" in first
    assert "g_harmonic_ratio" in first
    assert len(first) == len(FEATURE_NAMES)


def test_apply_predictions_relabels_one_chord_member() -> None:
    final_notes = [
        {"t": 0, "dur": 240, "pitch": 40, "s": 1, "f": 0},
        {"t": 0, "dur": 240, "pitch": 47, "s": 2, "f": 2},
        {"t": 0, "dur": 240, "pitch": 52, "s": 3, "f": 2},
    ]
    input_notes = [
        {"note_id": "g000000:n0", "tick": 0, "pitch": 40},
        {"note_id": "g000000:n1", "tick": 0, "pitch": 47},
        {"note_id": "g000000:n2", "tick": 0, "pitch": 52},
    ]
    predictions = [
        {"note_id": "g000000:n0", "score": 0.2, "label": "pitched"},
        {"note_id": "g000000:n1", "score": 0.91, "label": "dead"},
        {"note_id": "g000000:n2", "score": 0.1, "label": "pitched"},
    ]
    result, dead, decisions, stats = apply_predictions(
        final_notes, predictions, input_notes
    )
    assert len(dead) == 1
    assert (dead[0]["t"], dead[0]["s"]) == (0, 2)
    assert result[1].get("_drop") is True
    assert not result[0].get("_drop")
    assert not result[2].get("_drop")
    assert stats["dead"] == 1
    assert all(item["matched"] for item in decisions)


def test_apply_predictions_drops_noise_without_dead_note() -> None:
    final_notes = [{"t": 0, "dur": 240, "pitch": 55, "s": 3, "f": 5}]
    input_notes = [{"note_id": "g000000:n0", "tick": 0, "pitch": 55}]
    result, dead, _decisions, stats = apply_predictions(
        final_notes,
        [{"note_id": "g000000:n0", "score": 0.9, "label": "noise"}],
        input_notes,
    )
    assert result[0].get("_drop") is True
    assert dead == []
    assert stats["noise"] == 1


def test_gp5_renders_classifier_dead_note(tmp_path: Path) -> None:
    import guitarpro as gp

    from tabfeatures.gp5 import build_gp5

    midi = tmp_path / "timing.mid"
    _write_chord_midi(midi)
    gp5 = tmp_path / "score.gp5"
    notes = [
        {"t": 0, "dur": 240, "pitch": 40, "s": 1, "f": 0},
        {"t": 0, "dur": 240, "pitch": 52, "s": 3, "f": 2},
    ]
    dead = [{"t": 0, "dur": 1, "s": 2, "f": 0, "kind": "dead"}]
    build_gp5(notes, midi, gp5, dead_notes=dead)
    song = gp.parse(str(gp5))
    types = [
        note.type
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        for note in beat.notes
    ]
    assert gp.NoteType.dead in types
    assert types.count(gp.NoteType.dead) == 1


def test_gp5_rounds_boundary_notes_without_duplicate_strings(
    tmp_path: Path,
) -> None:
    import guitarpro as gp

    from tabfeatures.gp5 import build_gp5

    timing = tmp_path / "timing.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(
        pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=3.0)
    )
    pm.instruments.append(inst)
    pm.write(str(timing))

    # The first note ends 10 ticks after measure 2 starts; the next note on
    # the same string starts 10 ticks later.  Both round into the first grid
    # slot, and writing both used to corrupt the GP5 beat stream.
    notes = [
        {"t": 3820, "dur": 30, "pitch": 58, "s": 4, "f": 3},
        {"t": 3860, "dur": 500, "pitch": 59, "s": 4, "f": 4},
    ]
    output = tmp_path / "score.gp5"
    build_gp5(notes, timing, output, capo=0, title="boundary")

    song = gp.parse(str(output))
    assert len(song.measureHeaders) == 2
    for measure in song.tracks[0].measures:
        for voice in measure.voices:
            for beat in voice.beats:
                strings = [note.string for note in beat.notes]
                assert len(strings) == len(set(strings))


def test_gp5_never_writes_ties_across_beats(tmp_path: Path) -> None:
    import guitarpro as gp

    from tabfeatures.gp5 import build_gp5

    timing = tmp_path / "timing.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(
        pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=3.0)
    )
    pm.instruments.append(inst)
    pm.write(str(timing))

    short = tmp_path / "short.gp5"
    build_gp5(
        [{"t": 3800, "dur": 460, "pitch": 45, "s": 2, "f": 0}],
        timing,
        short,
        capo=0,
        title="short",
    )
    short_song = gp.parse(str(short))
    assert not any(
        note.type == gp.NoteType.tie
        for measure in short_song.tracks[0].measures
        for voice in measure.voices
        for beat in voice.beats
        for note in beat.notes
    )

    long = tmp_path / "long.gp5"
    build_gp5(
        [{"t": 0, "dur": 5000, "pitch": 47, "s": 2, "f": 2}],
        timing,
        long,
        capo=0,
        title="long",
    )
    long_song = gp.parse(str(long))
    assert not any(
        note.type == gp.NoteType.tie
        for measure in long_song.tracks[0].measures
        for voice in measure.voices
        for beat in voice.beats
        for note in beat.notes
    )


def test_gp5_suppresses_small_tempo_drifts(tmp_path: Path) -> None:
    import mido
    import guitarpro as gp

    from tabfeatures.gp5 import build_gp5

    midi = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    previous = 0
    for tick, bpm in ((0, 120), (960, 125), (1920, 135), (2880, 140), (3840, 150)):
        track.append(
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(bpm),
                time=tick - previous,
            )
        )
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    timing = tmp_path / "tempo.mid"
    midi.save(str(timing))

    output = tmp_path / "score.gp5"
    build_gp5(
        [{"t": 0, "dur": 4000, "pitch": 47, "s": 2, "f": 2}],
        timing,
        output,
        capo=0,
        title="tempo",
    )
    song = gp.parse(str(output))
    tempos = {
        beat.effect.mixTableChange.tempo.value
        for measure in song.tracks[0].measures
        for voice in measure.voices
        for beat in voice.beats
        if beat.effect.mixTableChange is not None
        and beat.effect.mixTableChange.tempo is not None
    }
    assert 125 not in tempos
    assert 140 not in tempos
    assert {120, 135, 150}.issubset(tempos)


def test_gp5_trims_leading_empty_measures(tmp_path: Path) -> None:
    import guitarpro as gp

    from tabfeatures.gp5 import build_gp5

    timing = tmp_path / "timing.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(
        pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=6.0)
    )
    pm.instruments.append(inst)
    pm.write(str(timing))

    output = tmp_path / "trimmed.gp5"
    info = build_gp5(
        [{"t": 8000, "dur": 480, "pitch": 47, "s": 2, "f": 2}],
        timing,
        output,
        capo=0,
        title="trimmed",
    )
    assert info["trimmed_lead_ticks"] == 7680
    song = gp.parse(str(output))
    assert song.measureHeaders[0].start == 960
    assert any(
        beat.notes
        for beat in song.tracks[0].measures[0].voices[0].beats
    )


def test_apply_predictions_skips_unmatched_model_drop() -> None:
    final_notes = [{"t": 480, "dur": 240, "pitch": 55, "s": 3, "f": 5}]
    input_notes = [{"note_id": "g000000:n0", "tick": 0, "pitch": 55}]
    result, dead, _decisions, stats = apply_predictions(
        final_notes,
        [{"note_id": "g000000:n0", "score": 0.99, "label": "dead"}],
        input_notes,
    )
    assert dead == []
    assert stats["unmatched"] == 1
    assert not result[0].get("_drop")


def test_load_note_groups_subtracts_muscriptor_bar_offset(tmp_path: Path) -> None:
    import mido

    midi = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(
        mido.MetaMessage("marker", text="muscriptor:bar_offset=0.2000", time=0)
    )
    start_tick = int(mido.second2tick(1.2, 960, tempo))
    track.append(mido.Message("program_change", program=24, channel=0, time=0))
    track.append(mido.Message("note_on", note=52, velocity=90, time=start_tick))
    track.append(mido.Message("note_off", note=52, velocity=0, time=240))
    path = tmp_path / "offset.mid"
    midi.save(str(path))

    payload = load_note_groups(path)
    assert payload["bar_offset_s"] == 0.2
    note = payload["groups"][0]["notes"][0]
    assert note["tick"] == start_tick
    assert note["time_s"] == 1.0
    assert payload["groups"][0]["time_s"] == 1.0
