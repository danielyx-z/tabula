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
