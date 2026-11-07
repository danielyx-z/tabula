"""Tests for the MIDI-anchored label store."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.labeling import store  # noqa: E402


def _candidates() -> dict:
    return {
        "duration": 2.0,
        "groups": [
            {
                "group_id": "g000000",
                "time_s": 0.0,
                "chord_size": 2,
                "notes": [
                    {
                        "note_id": "g000000:n0",
                        "group_id": "g000000",
                        "tick": 0,
                        "time_s": 0.0,
                        "pitch": 40,
                        "velocity": 90,
                        "chord_rank": 0,
                        "interval_to_lower": 0,
                    },
                    {
                        "note_id": "g000000:n1",
                        "group_id": "g000000",
                        "tick": 0,
                        "time_s": 0.0,
                        "pitch": 47,
                        "velocity": 88,
                        "chord_rank": 1,
                        "interval_to_lower": 7,
                    },
                ],
            }
        ],
    }


def test_label_state_and_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path)
    session_id = "0" * 32
    store.create_session(session_id)
    store.write_candidates(session_id, _candidates())
    store.write_meta(session_id, {"reviewed": True, "original_name": "take.wav"})
    store.append_labels(
        session_id,
        [{"note_id": "g000000:n1", "label": "dead"}],
        op_id="op1",
    )

    state = store.label_state(session_id)
    assert state["summary"] == {
        "notes": 2,
        "groups": 1,
        "dead": 1,
        "noise": 0,
        "ignore": 0,
        "reviewed": True,
    }
    labels = {
        note["note_id"]: note["label"]
        for group in state["groups"]
        for note in group["notes"]
    }
    assert labels == {"g000000:n0": "pitched", "g000000:n1": "dead"}

    exported = [
        json.loads(line)
        for line in store.export_labels(session_id).splitlines()
        if line.strip()
    ]
    assert len(exported) == 2
    assert {row["label"] for row in exported} == {"pitched", "dead"}
    assert all(row["reviewed"] for row in exported)


def test_latest_label_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path)
    session_id = "1" * 32
    store.create_session(session_id)
    store.write_candidates(session_id, _candidates())
    store.append_labels(session_id, [{"note_id": "g000000:n0", "label": "dead"}])
    store.append_labels(
        session_id, [{"note_id": "g000000:n0", "label": "pitched"}]
    )
    assert store.latest_labels(session_id)["g000000:n0"] == "pitched"


def test_noise_label_is_stored_and_exported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path)
    session_id = "2" * 32
    store.create_session(session_id)
    store.write_candidates(session_id, _candidates())
    store.append_labels(
        session_id,
        [
            {"note_id": "g000000:n0", "label": "noise"},
            {"note_id": "g000000:n1", "label": "dead"},
        ],
    )
    state = store.label_state(session_id)
    assert state["summary"]["noise"] == 1
    assert state["summary"]["dead"] == 1
    labels = {
        note["note_id"]: note["label"]
        for group in state["groups"]
        for note in group["notes"]
    }
    assert labels["g000000:n0"] == "noise"


def test_ignore_label_is_stored_but_excluded_from_training(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "LABEL_ROOT", tmp_path)
    session_id = "3" * 32
    store.create_session(session_id)
    store.write_candidates(session_id, _candidates())
    (tmp_path / session_id / "audio.wav").write_bytes(b"")
    store.write_meta(session_id, {"reviewed": True})
    store.append_labels(
        session_id,
        [
            {"note_id": "g000000:n0", "label": "ignore"},
            {"note_id": "g000000:n1", "label": "dead"},
        ],
    )

    state = store.label_state(session_id)
    assert state["summary"]["ignore"] == 1
    assert state["summary"]["dead"] == 1

    exported = [
        json.loads(line)
        for line in store.export_labels(session_id).splitlines()
        if line.strip()
    ]
    assert {row["note_id"]: row["label"] for row in exported} == {
        "g000000:n0": "ignore",
        "g000000:n1": "dead",
    }

    from tabfeatures.deadnotes import dataset as dead_dataset

    rows = [
        {"note": {"note_id": "g000000:n0"}},
        {"note": {"note_id": "g000000:n1"}},
        {"note": {"note_id": "g000000:n2"}},
    ]
    monkeypatch.setattr(
        dead_dataset,
        "extract_group_rows",
        lambda *args, **kwargs: rows,
    )
    labelled = dead_dataset.session_rows(session_id, label_root=tmp_path)
    assert [row["label"] for row in labelled] == ["dead", "pitched"]
