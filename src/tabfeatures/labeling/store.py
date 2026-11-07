"""Persist dead-note label sessions under ``data/labels_dead``."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
LABEL_ROOT = Path(
    os.environ.get("TABULA_LABEL_ROOT", str(REPO / "data" / "labels_dead"))
).expanduser()
_SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
_META = "meta.json"
_CANDIDATES = "candidates.json"
_LABELS = "labels.jsonl"
NOTE_LABELS = frozenset({"pitched", "dead", "noise", "ignore"})
TRAINING_LABELS = frozenset({"pitched", "dead", "noise"})


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_session_id() -> str:
    seed = f"{datetime.now(timezone.utc).isoformat()}:{os.getpid()}"
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=16).hexdigest()


def is_session_id(value: str) -> bool:
    return bool(_SESSION_RE.fullmatch(str(value or "")))


def session_dir(session_id: str, root: str | Path | None = None) -> Path:
    if not is_session_id(session_id):
        raise ValueError("invalid label session id")
    return Path(root).expanduser() / session_id if root is not None else LABEL_ROOT / session_id


def ensure_session(session_id: str, root: str | Path | None = None) -> Path:
    directory = session_dir(session_id, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def remove_session(session_id: str) -> None:
    directory = session_dir(session_id)
    shutil.rmtree(directory, ignore_errors=True)


def create_session(session_id: str | None = None) -> tuple[str, Path]:
    identifier = session_id or new_session_id()
    directory = ensure_session(identifier)
    return identifier, directory


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_candidates(session_id: str, payload: dict[str, Any]) -> None:
    _write_json(ensure_session(session_id) / _CANDIDATES, payload)


def read_candidates(
    session_id: str, root: str | Path | None = None
) -> dict[str, Any]:
    payload = _read_json(session_dir(session_id, root=root) / _CANDIDATES, {})
    return payload if isinstance(payload, dict) else {}


def write_meta(session_id: str, meta: dict[str, Any]) -> None:
    _write_json(ensure_session(session_id) / _META, meta)


def read_meta(session_id: str, root: str | Path | None = None) -> dict[str, Any]:
    payload = _read_json(session_dir(session_id, root=root) / _META, {})
    return payload if isinstance(payload, dict) else {}


def append_labels(
    session_id: str,
    labels: list[dict[str, Any]],
    *,
    op_id: str | None = None,
) -> list[dict[str, Any]]:
    """Append note labels.  Every record carries an operation id for undo."""
    directory = ensure_session(session_id)
    path = directory / _LABELS
    written: list[dict[str, Any]] = []
    with path.open("a", encoding="utf-8") as handle:
        for item in labels:
            note_id = str(item.get("note_id") or "")
            label = str(item.get("label") or "")
            if not note_id or label not in NOTE_LABELS:
                continue
            record = {
                "note_id": note_id,
                "label": label,
                "op_id": str(op_id or ""),
                "updated_at": now_iso(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.append(record)
    return written


def latest_labels(
    session_id: str, root: str | Path | None = None
) -> dict[str, str]:
    path = session_dir(session_id, root=root) / _LABELS
    latest: dict[str, str] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        note_id = str(record.get("note_id") or "")
        label = str(record.get("label") or "")
        if note_id and label in NOTE_LABELS:
            latest[note_id] = label
    return latest


def label_state(session_id: str) -> dict[str, Any]:
    candidates = read_candidates(session_id)
    meta = read_meta(session_id)
    labels = latest_labels(session_id)
    groups = candidates.get("groups", [])
    dead = 0
    noise = 0
    ignore = 0
    for group in groups:
        for note in group.get("notes", []):
            note_id = str(note.get("note_id") or "")
            note["label"] = labels.get(note_id, "pitched")
            if note["label"] == "dead":
                dead += 1
            elif note["label"] == "noise":
                noise += 1
            elif note["label"] == "ignore":
                ignore += 1
    return {
        "id": session_id,
        "meta": meta,
        "duration": candidates.get("duration"),
        "beats": candidates.get("beats", []),
        "downbeats": candidates.get("downbeats", []),
        "tempo_changes": candidates.get("tempo_changes", []),
        "time_signatures": candidates.get("time_signatures", []),
        "groups": groups,
        "summary": {
            "notes": sum(len(group.get("notes", [])) for group in groups),
            "groups": len(groups),
            "dead": dead,
            "noise": noise,
            "ignore": ignore,
            "reviewed": bool(meta.get("reviewed")),
        },
    }


def mark_reviewed(session_id: str, reviewed: bool = True) -> dict[str, Any]:
    meta = read_meta(session_id)
    meta["reviewed"] = bool(reviewed)
    meta["reviewed_at"] = now_iso() if reviewed else None
    write_meta(session_id, meta)
    return meta


def export_labels(session_id: str) -> str:
    """Export a complete per-note label file for training/inspection."""
    state = label_state(session_id)
    meta = state.get("meta") or {}
    rows: list[dict[str, Any]] = []
    for group in state["groups"]:
        for note in group.get("notes", []):
            rows.append(
                {
                    "session_id": session_id,
                    "note_id": note["note_id"],
                    "group_id": note["group_id"],
                    "tick": note.get("tick"),
                    "time_s": note.get("time_s"),
                    "pitch": note.get("pitch"),
                    "velocity": note.get("velocity"),
                    "chord_size": group.get("chord_size"),
                    "chord_rank": note.get("chord_rank"),
                    "interval_to_lower": note.get("interval_to_lower"),
                    "label": note.get("label", "pitched"),
                    "reviewed": bool(meta.get("reviewed")),
                    "source": meta.get("source"),
                }
            )
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def list_sessions(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = Path(root).expanduser() if root is not None else LABEL_ROOT
    if not base.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in base.iterdir():
        if not path.is_dir() or not is_session_id(path.name):
            continue
        meta = read_meta(path.name, root=base)
        if not meta and not (path / _CANDIDATES).is_file():
            continue
        sessions.append(
            {
                "id": path.name,
                "name": meta.get("original_name") or path.name,
                "created_at": meta.get("created_at"),
                "reviewed": bool(meta.get("reviewed")),
                "notes": int(meta.get("note_count") or 0),
                "dead": int(meta.get("dead_count") or 0),
                "noise": int(meta.get("noise_count") or 0),
                "ignore": int(meta.get("ignore_count") or 0),
            }
        )
    sessions.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return sessions
