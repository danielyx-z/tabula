"""Build classifier training rows from MIDI-anchored label sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabfeatures.labeling.store import (
    LABEL_ROOT,
    latest_labels,
    list_sessions,
    read_candidates,
    read_meta,
    session_dir,
)

from .features import FeatureConfig, extract_group_rows


def session_rows(
    session_id: str,
    *,
    label_root: str | Path = LABEL_ROOT,
    config: FeatureConfig | None = None,
    include_unreviewed: bool = False,
) -> list[dict[str, Any]]:
    directory = Path(label_root).expanduser() / session_id
    audio_path = directory / "audio.wav"
    candidates = read_candidates(session_id, root=label_root)
    groups = candidates.get("groups", [])
    if not audio_path.is_file() or not groups:
        return []

    meta = read_meta(session_id, root=label_root)
    reviewed = bool(meta.get("reviewed"))
    labels = latest_labels(session_id, root=label_root)
    rows = extract_group_rows(audio_path, groups, config=config)
    labelled: list[dict[str, Any]] = []
    for row in rows:
        note = row.get("note") or {}
        note_id = str(note.get("note_id") or "")
        if not note_id:
            continue
        if note_id in labels:
            label = labels[note_id]
        elif reviewed or include_unreviewed:
            label = "pitched"
        else:
            continue
        labelled.append(
            {
                **row,
                "label": label,
                "session_id": session_id,
                "reviewed": reviewed,
                "source": meta.get("source"),
            }
        )
    return labelled


def load_rows(
    sessions: list[str] | None = None,
    *,
    label_root: str | Path = LABEL_ROOT,
    config: FeatureConfig | None = None,
    include_unreviewed: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    ids = sessions or [item["id"] for item in list_sessions()]
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    for session_id in ids:
        session = session_rows(
            session_id,
            label_root=label_root,
            config=config,
            include_unreviewed=include_unreviewed,
        )
        if session:
            rows.extend(session)
            used.append(session_id)
    return rows, used
