"""Persistent browser-labeller sessions.

Sessions live under ``data/labels`` (already gitignored), not ``webapp/jobs``:
Flask cleans old job directories automatically and labels must survive a
restart.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import LABELS, normalize_label

REPO = Path(__file__).resolve().parents[3]
LABEL_ROOT = REPO / "data" / "labels"
_SESSION_RE = re.compile(r"^[0-9a-f]{32}$")

META_NAME = "meta.json"
CANDIDATES_NAME = "candidates.json"
LABELS_NAME = "labels.jsonl"
AUDIO_NAME = "audio.wav"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def new_session_id() -> str:
    return uuid.uuid4().hex


def is_session_id(value: str) -> bool:
    return bool(_SESSION_RE.fullmatch(str(value or "")))


def session_dir(session_id: str) -> Path:
    if not is_session_id(session_id):
        raise ValueError("invalid label session id")
    return LABEL_ROOT / session_id


def list_sessions(root: str | Path | None = None) -> list[str]:
    base = Path(root).expanduser() if root is not None else LABEL_ROOT
    if not base.exists():
        return []
    return sorted(
        path.name
        for path in base.iterdir()
        if path.is_dir()
        and is_session_id(path.name)
        and (path / META_NAME).exists()
        and (path / CANDIDATES_NAME).exists()
    )


def ensure_session(session_id: str) -> Path:
    path = session_dir(session_id)
    path.mkdir(parents=True, exist_ok=False)
    return path


def remove_session(session_id: str) -> None:
    shutil.rmtree(session_dir(session_id), ignore_errors=True)


def write_meta(session_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    payload = dict(meta)
    payload.setdefault("session_id", session_id)
    payload.setdefault("created_at", _utc_now())
    payload["labels"] = list(LABELS)
    _atomic_write(
        session_dir(session_id) / META_NAME,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    return payload


def read_meta(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id) / META_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("label session meta must be an object")
    return payload


def write_candidates(session_id: str, payload: dict[str, Any]) -> None:
    _atomic_write(
        session_dir(session_id) / CANDIDATES_NAME,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def read_candidates(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id) / CANDIDATES_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate payload must be an object")
    return payload


def _read_label_lines(session_id: str) -> list[dict[str, Any]]:
    path = session_dir(session_id) / LABELS_NAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def latest_labels(session_id: str) -> dict[int, dict[str, Any]]:
    """Return the most recent label for each candidate index."""
    latest: dict[int, dict[str, Any]] = {}
    for record in _read_label_lines(session_id):
        try:
            index = int(record["index"])
        except (KeyError, TypeError, ValueError):
            continue
        latest[index] = record
    return latest


def append_labels(
    session_id: str,
    *,
    indices: list[int],
    label: str,
) -> list[dict[str, Any]]:
    """Append one label record per index as a single undoable group."""
    candidates = read_candidates(session_id).get("candidates", [])
    canonical = normalize_label(label)
    ordered = [int(index) for index in indices]
    if not ordered:
        raise ValueError("at least one label index is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError("label indices must be unique")
    for index in ordered:
        if not 0 <= index < len(candidates):
            raise ValueError("label index is outside the candidate range")

    group_id = uuid.uuid4().hex
    updated_at = _utc_now()
    records = [
        {
            "session_id": session_id,
            "index": index,
            "onset_s": float(candidates[index].get("onset_s", 0.0)),
            "label": canonical,
            "group_id": group_id,
            "updated_at": updated_at,
        }
        for index in ordered
    ]
    path = session_dir(session_id) / LABELS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def append_label(
    session_id: str,
    *,
    index: int,
    label: str,
) -> dict[str, Any]:
    return append_labels(session_id, indices=[index], label=label)[0]


def append_pitched_run(
    session_id: str,
    *,
    start_index: int,
    max_count: int = 8,
    max_gap_s: float = 0.25,
) -> dict[str, Any]:
    """Label a short contiguous run of clearly normal notes as ``pitched``.

    The run stops at a gap, at an already labelled candidate, or at
    ``max_count``.  It is stored as one group so Undo removes the whole run.
    """
    candidates = read_candidates(session_id).get("candidates", [])
    start = int(start_index)
    if not 0 <= start < len(candidates):
        raise ValueError("label index is outside the candidate range")
    limit = max(1, min(16, int(max_count)))
    gap_limit = max(0.0, float(max_gap_s))
    latest = latest_labels(session_id)

    indices: list[int] = []
    previous_onset: float | None = None
    for index in range(start, min(len(candidates), start + limit)):
        if index != start and index in latest:
            break
        onset_s = float(candidates[index].get("onset_s", 0.0))
        if (
            previous_onset is not None
            and onset_s - previous_onset > gap_limit
        ):
            break
        indices.append(index)
        previous_onset = onset_s

    records = append_labels(session_id, indices=indices, label="pitched")
    return {
        "group_id": records[0]["group_id"] if records else None,
        "records": records,
        "indices": indices,
    }


def undo_label(session_id: str) -> dict[str, Any] | None:
    """Remove and return the last saved label group, if any."""
    path = session_dir(session_id) / LABELS_NAME
    records = _read_label_lines(session_id)
    if not records:
        return None
    group_id = records[-1].get("group_id")
    removed: list[dict[str, Any]] = []
    while records and (
        records[-1].get("group_id") == group_id
        if group_id is not None
        else len(removed) == 0
    ):
        removed.append(records.pop())
    removed.reverse()
    remaining = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    _atomic_write(path, remaining)
    indices = [int(record["index"]) for record in removed if "index" in record]
    return {
        "records": removed,
        "indices": indices,
        "count": len(removed),
        "index": indices[0] if indices else None,
        "group_id": group_id,
    }


def export_labels(session_id: str) -> str:
    """Export one current record per candidate, not the edit history."""
    latest = latest_labels(session_id)
    return "".join(
        json.dumps(latest[index], ensure_ascii=False) + "\n"
        for index in sorted(latest)
    )


def state(session_id: str) -> dict[str, Any]:
    meta = read_meta(session_id)
    candidates = read_candidates(session_id)
    latest = latest_labels(session_id)
    rows = candidates.get("candidates", [])
    return {
        "session_id": session_id,
        "meta": meta,
        "duration": candidates.get("duration", meta.get("duration", 0.0)),
        "candidates": rows,
        "labels": [latest[index] for index in sorted(latest)],
        "labeled": len(latest),
        "total": len(rows),
        "audio_url": f"/api/label/sessions/{session_id}/audio",
    }


def recent_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Summarise existing sessions for the labeller's resume list."""
    rows: list[dict[str, Any]] = []
    for session_id in list_sessions():
        try:
            meta = read_meta(session_id)
            candidates = read_candidates(session_id)
            latest = latest_labels(session_id)
        except (OSError, ValueError):
            continue
        rows.append(
            {
                "session_id": session_id,
                "original_name": meta.get("original_name"),
                "created_at": meta.get("created_at"),
                "duration": candidates.get(
                    "duration", meta.get("duration", 0.0)
                ),
                "labeled": len(latest),
                "total": len(candidates.get("candidates", [])),
            }
        )
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows[: max(1, int(limit))]
