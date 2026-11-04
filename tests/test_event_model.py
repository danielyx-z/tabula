"""Tests for event features, additive predictions, and GP5 body percussion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.events.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_event_features,
)
from tabfeatures.events.inference import apply_event_predictions  # noqa: E402
from tabfeatures.dead import merge_dead_notes  # noqa: E402
from tabfeatures.events.config import load_events_config  # noqa: E402
from tabfeatures.events.schema import label_heads  # noqa: E402
from tabfeatures.gp5 import build_gp5  # noqa: E402


def _tone(sr: int = 22050) -> tuple[np.ndarray, int]:
    time = np.arange(int(sr * 0.6)) / sr
    y = (0.4 * np.sin(2 * np.pi * 110 * time) * np.exp(-time * 8)).astype(
        np.float32
    )
    return y, sr


def test_event_config_loads_defaults() -> None:
    config = load_events_config()
    assert config["thresholds"]["body_percussion"] == 0.60
    assert config["feature_window"]["after_s"] == 0.35


def test_label_heads_are_additive_for_flicks() -> None:
    assert label_heads("pitched") == {
        "pitched": 1,
        "dead_string": 0,
        "bass_flick": 0,
        "body_percussion": 0,
    }
    assert label_heads("bass_flick") == {
        "pitched": 0,
        "dead_string": 1,
        "bass_flick": 1,
        "body_percussion": 0,
    }


def test_features_have_stable_schema_and_pitch_context() -> None:
    y, sr = _tone()
    features = extract_event_features(y, sr, 0.05, pitch=45, string=1)
    assert set(features) == set(FEATURE_NAMES)
    assert features["low_ratio"] > 0.5
    assert features["f0_hz"] == pytest.approx(110.0, rel=0.03)
    assert features["pitch"] == 45
    assert features["string"] == 1


def _prediction(heads: dict[str, float], **extra: object) -> dict[str, object]:
    return {
        "tick": 0,
        "time": 0.0,
        "matched": True,
        "matched_note": {"pitch": 64},
        "string": 3,
        "heads": heads,
        **extra,
    }


def test_bass_flick_adds_x_without_dropping_melody() -> None:
    notes = [{"t": 0, "dur": 480, "pitch": 64, "s": 3, "f": 0}]
    result, dead, review, stats = apply_event_predictions(
        notes,
        [
            _prediction(
                {
                    "pitched": 0.2,
                    "dead_string": 0.1,
                    "bass_flick": 0.9,
                    "body_percussion": 0.1,
                }
            )
        ],
        grid_ticks=240,
    )
    assert not any(note.get("_drop") for note in result)
    assert len(dead) == 1
    assert dead[0]["kind"] == "dead"
    assert dead[0]["s"] == 1
    assert dead[0]["staccato"] is False
    assert stats["additive"] == 1
    assert review == []


def test_body_percussion_uses_lowest_free_string_and_staccato() -> None:
    notes = [
        {"t": 0, "dur": 480, "pitch": 64, "s": 3, "f": 0},
        {"t": 0, "dur": 480, "pitch": 59, "s": 1, "f": 0},
    ]
    result, dead, review, stats = apply_event_predictions(
        notes,
        [
            _prediction(
                {
                    "pitched": 0.6,
                    "dead_string": 0.1,
                    "bass_flick": 0.1,
                    "body_percussion": 0.9,
                }
            )
        ],
        grid_ticks=240,
    )
    assert len(result) == 2
    assert len(dead) == 1
    assert dead[0]["kind"] == "body_percussion"
    assert dead[0]["staccato"] is True
    assert dead[0]["s"] == 2
    assert stats["additive"] == 1


def test_body_percussion_with_no_free_string_goes_to_review() -> None:
    notes = [
        {"t": 0, "dur": 480, "pitch": pitch, "s": string, "f": 0}
        for string, pitch in enumerate((40, 45, 50, 55, 59, 64), start=1)
    ]
    _result, dead, review, stats = apply_event_predictions(
        notes,
        [
            _prediction(
                {
                    "pitched": 0.1,
                    "dead_string": 0.1,
                    "bass_flick": 0.1,
                    "body_percussion": 0.9,
                }
            )
        ],
        grid_ticks=240,
    )
    assert dead == []
    assert review[0]["reason"] == "no_free_string"
    assert review[0]["kind"] == "body_percussion"
    assert stats["skipped"] == 1


def test_dead_string_can_replace_matched_note() -> None:
    notes = [{"t": 0, "dur": 480, "pitch": 64, "s": 3, "f": 0}]
    result, dead, _review, stats = apply_event_predictions(
        notes,
        [
            _prediction(
                {
                    "pitched": 0.1,
                    "dead_string": 0.9,
                    "bass_flick": 0.1,
                    "body_percussion": 0.1,
                }
            )
        ],
        grid_ticks=240,
    )
    assert result[0]["_drop"] is True
    assert len(dead) == 1
    assert dead[0]["s"] == 3
    assert stats["converted"] == 1


def test_merge_keeps_body_percussion_marker() -> None:
    body = {
        "t": 0,
        "s": 1,
        "kind": "body_percussion",
        "staccato": True,
        "confidence": 0.6,
    }
    plain = {"t": 0, "s": 1, "kind": "dead", "confidence": 0.99}
    assert merge_dead_notes([plain], [body]) == [body]


def test_body_percussion_gp5_round_trip(tmp_path: Path) -> None:
    guitarpro = pytest.importorskip("guitarpro")
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(pretty_midi.Instrument(program=24))
    midi_path = tmp_path / "empty.mid"
    midi.write(str(midi_path))
    out_path = tmp_path / "body.gp5"
    info = build_gp5(
        [],
        midi_path,
        out_path,
        dead_notes=[
            {
                "t": 0,
                "dur": 1,
                "s": 1,
                "f": 0,
                "kind": "body_percussion",
                "staccato": True,
            }
        ],
    )
    assert info["dead_notes"] == 1
    song = guitarpro.parse(str(out_path))
    notes = [
        note
        for track in song.tracks
        for measure in track.measures
        for voice in measure.voices
        for beat in voice.beats
        for note in beat.notes
    ]
    assert len(notes) == 1
    assert notes[0].type == guitarpro.NoteType.dead
    assert notes[0].effect.staccato is True


def test_tiny_model_trains_and_predicts(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    from tabfeatures.events.model import load_event_model, train_event_model

    rng = np.random.default_rng(4)
    rows = []
    for index in range(48):
        label = ["pitched", "dead_string", "bass_flick", "body_percussion"][
            index % 4
        ]
        features = {
            name: float(rng.normal()) for name in FEATURE_NAMES
        }
        features["low_ratio"] = 0.9 if label == "bass_flick" else 0.2
        features["percussive_ratio"] = 0.8 if label != "pitched" else 0.2
        rows.append({"label": label, "features": features})

    payload = train_event_model(
        rows,
        out_dir=tmp_path / "model",
        num_boost_round=4,
    )
    assert set(payload["heads"]) == {
        "pitched",
        "dead_string",
        "bass_flick",
        "body_percussion",
    }
    model = load_event_model(tmp_path / "model")
    assert model is not None
    assert model.feature_window == {"before_s": 0.03, "after_s": 0.35}
    scores = model.predict(rows[0]["features"])
    assert set(scores) == {
        "pitched",
        "dead_string",
        "bass_flick",
        "body_percussion",
    }
    assert all(0.0 <= value <= 1.0 for value in scores.values())
