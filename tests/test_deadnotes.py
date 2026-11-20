"""Tests for MIDI-anchored dead-note labelling and classifier plumbing."""

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

from tabfeatures.deadnotes.features import (  # noqa: E402
    FEATURE_NAMES,
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


def test_gp5_does_not_restrike_at_a_bar_line(tmp_path: Path) -> None:
    """No ties and no invented attacks: the note simply ends at the bar."""
    import guitarpro as gp

    written = _gp5_notes(
        tmp_path,
        [{"t": 3000, "dur": 2600, "pitch": 47, "s": 1, "f": 7}],
        "sustain",
    )
    assert len(written) == 1, written
    assert written[0].type == gp.NoteType.normal
    assert written[0].realValue == 47


def test_gp5_does_not_restrike_a_split_note(tmp_path: Path) -> None:
    """A run split inside one bar (e.g. 600 ticks) is not re-struck."""
    import guitarpro as gp

    written = _gp5_notes(
        tmp_path,
        [{"t": 0, "dur": 600, "pitch": 52, "s": 3, "f": 2}],
        "split",
    )
    assert len(written) == 1, written
    assert written[0].type == gp.NoteType.normal


def test_gp5_quantises_to_thirty_seconds(tmp_path: Path) -> None:
    """Onsets snap to the 32nd grid, which keeps the groove closer to the MIDI."""
    import guitarpro as gp

    written = _gp5_notes(
        tmp_path,
        [
            {"t": 1240, "dur": 480, "pitch": 47, "s": 1, "f": 7},
            {"t": 1760, "dur": 480, "pitch": 52, "s": 3, "f": 2},
        ],
        "offsets",
    )
    ticks = sorted(note.beat.start - 960 for note in written)
    assert ticks == [1200, 1800], ticks
    assert all(note.type == gp.NoteType.normal for note in written)


def test_gp5_uses_thirty_seconds_only_to_keep_two_attacks(tmp_path: Path) -> None:
    """Two attacks a 32nd apart stay apart instead of merging into one slot."""
    written = _gp5_notes(
        tmp_path,
        [
            {"t": 1200, "dur": 120, "pitch": 47, "s": 1, "f": 7},
            {"t": 1280, "dur": 120, "pitch": 52, "s": 3, "f": 2},
        ],
        "collision",
    )
    ticks = sorted(note.beat.start - 960 for note in written)
    assert ticks == [1200, 1320], ticks


def test_gp5_plays_at_one_tempo(tmp_path: Path) -> None:
    """MuScriptor's tempo drifts are dropped so the score stays flat."""
    import mido

    from tabfeatures.gp5 import read_timing

    midi = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    previous = 0
    for tick, bpm in ((0, 163), (960, 157), (1920, 167), (2880, 175)):
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

    tempi, _signatures = read_timing(timing)
    # 157 and 167 are within 10 BPM of 163 and are dropped; 175 is kept.
    assert [round(v) for v in tempi.values()] == [163, 175]


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


def test_normalise_bar_offset_folds_shift_into_notes(tmp_path: Path) -> None:
    """The tab must share the recording's timeline, not MuScriptor's bar grid."""
    import mido

    from tabfeatures.midi_groups import normalise_bar_offset, read_bar_offset

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

    shift = normalise_bar_offset(path)
    assert shift == pytest.approx(0.2, abs=1e-4)
    assert read_bar_offset(path) == 0.0

    notes = [
        note
        for instrument in pretty_midi.PrettyMIDI(str(path)).instruments
        for note in instrument.notes
    ]
    assert len(notes) == 1
    assert float(notes[0].start) == pytest.approx(1.0, abs=1e-3)
    assert float(notes[0].end) == pytest.approx(1.125, abs=1e-3)

    # Idempotent: a second pass has nothing left to remove.
    assert normalise_bar_offset(path) == 0.0


def test_normalise_bar_offset_ignores_plain_midi(tmp_path: Path) -> None:
    from tabfeatures.midi_groups import normalise_bar_offset

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=24)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=52, start=0.5, end=0.8))
    pm.instruments.append(instrument)
    path = tmp_path / "plain.mid"
    pm.write(str(path))

    before = path.read_bytes()
    assert normalise_bar_offset(path) == 0.0
    assert path.read_bytes() == before


def _gp5_notes(tmp_path: Path, notes: list[dict], name: str) -> list:
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

    output = tmp_path / f"{name}.gp5"
    build_gp5(notes, timing, output, capo=0, title=name)
    song = gp.parse(str(output))
    return [
        note
        for measure in song.tracks[0].measures
        for voice in measure.voices
        for beat in voice.beats
        for note in beat.notes
    ]


def test_gp5_keeps_a_note_in_the_last_grid_slot(tmp_path: Path) -> None:
    """A note that rounds past the bar is written on the bar, not dropped."""
    import guitarpro as gp

    written = _gp5_notes(
        tmp_path,
        [{"t": 3820, "dur": 40, "pitch": 52, "s": 3, "f": 2}],
        "bar-end",
    )
    attacks = [n for n in written if n.type == gp.NoteType.normal]
    assert len(attacks) == 1, written
    assert attacks[0].value == 2
    assert attacks[0].realValue == 52
    assert attacks[0].beat.start - 960 == 3840
