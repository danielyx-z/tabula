"""Tests for the browser-labeller storage, segmentation, and Flask API."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.labeling import candidates, store  # noqa: E402


def _click_wav(path: Path, times: list[float], *, sr: int = 22050) -> None:
    y = np.zeros(int(sr * (max(times) + 0.8)), dtype=np.float32)
    pulse = np.hanning(300).astype(np.float32) * 0.8
    for time_s in times:
        start = int(time_s * sr)
        y[start : start + len(pulse)] += pulse
    sf.write(path, y, sr)


def test_candidates_are_split_and_bounded(tmp_path: Path) -> None:
    audio = tmp_path / "clicks.wav"
    _click_wav(audio, [0.2, 0.7, 1.2])
    result = candidates.detect_candidates(audio)
    rows = result["candidates"]
    assert 3 <= len(rows) <= 4
    for row in rows:
        assert row["start_s"] <= row["onset_s"] <= row["end_s"]
        assert row["end_s"] <= result["duration"] + 1e-6
    assert all(
        rows[i]["onset_s"] < rows[i + 1]["onset_s"]
        for i in range(len(rows) - 1)
    )


def test_store_append_undo_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path / "labels")
    session_id = store.new_session_id()
    store.ensure_session(session_id)
    store.write_candidates(
        session_id,
        {
            "sr": 22050,
            "duration": 1.0,
            "candidates": [
                {"index": 0, "onset_s": 0.1, "start_s": 0.0, "end_s": 0.4},
                {"index": 1, "onset_s": 0.5, "start_s": 0.4, "end_s": 0.8},
            ],
        },
    )
    store.write_meta(session_id, {"original_name": "take.wav"})
    store.append_label(session_id, index=0, label="bass_flick")
    store.append_label(session_id, index=1, label="body_percussion")
    latest = store.latest_labels(session_id)
    assert latest[0]["label"] == "bass_flick"
    assert latest[1]["label"] == "body_percussion"
    removed = store.undo_label(session_id)
    assert removed is not None and removed["index"] == 1
    assert store.state(session_id)["labeled"] == 1
    assert "bass_flick" in store.export_labels(session_id)
    store.remove_session(session_id)


def test_flask_label_session_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path / "labels")
    from webapp.server import app

    audio = tmp_path / "click.wav"
    _click_wav(audio, [0.2, 0.7])
    buffer = io.BytesIO()
    buffer.write(audio.read_bytes())
    buffer.seek(0)

    client = app.test_client()
    response = client.post(
        "/api/label/sessions",
        data={"audio": (buffer, "click.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    session_id = payload["session_id"]
    assert payload["total"] >= 1
    assert payload["audio_url"].endswith("/audio")
    sessions = client.get("/api/label/sessions").get_json()["sessions"]
    assert any(item["session_id"] == session_id for item in sessions)

    label_response = client.post(
        f"/api/label/sessions/{session_id}/labels",
        json={"index": 0, "label": "dead_string"},
    )
    assert label_response.status_code == 200
    assert label_response.get_json()["labeled"] == 1

    audio_response = client.get(f"/api/label/sessions/{session_id}/audio")
    assert audio_response.status_code == 200
    assert audio_response.mimetype == "audio/wav"

    undo_response = client.post(f"/api/label/sessions/{session_id}/undo", json={})
    assert undo_response.status_code == 200
    assert undo_response.get_json()["labeled"] == 0

    invalid = client.post(
        f"/api/label/sessions/{session_id}/labels",
        json={"index": 0, "label": "not_a_label"},
    )
    assert invalid.status_code == 400

    resume = client.get(f"/api/label/sessions/{session_id}")
    assert resume.status_code == 200
    assert resume.get_json()["total"] == payload["total"]

    store.remove_session(session_id)
