"""Read the event-model configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs" / "events.yaml"


def load_events_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the ``event_model`` section, or an empty mapping.

    The config is a convenience for choosing training defaults.  A trained
    model always carries its own copy of the thresholds and feature window in
    ``metadata.json``, so inference does not depend on the YAML file.
    """
    target = Path(path).expanduser() if path else DEFAULT_CONFIG
    if not target.exists():
        return {}
    import yaml

    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"event config must be a mapping: {target}")
    section = payload.get("event_model", payload)
    if not isinstance(section, dict):
        raise ValueError(f"event_model config must be a mapping: {target}")
    return section


def resolve_config_path(value: str | Path | None) -> Path:
    path = Path(value).expanduser() if value else DEFAULT_CONFIG
    return path if path.is_absolute() else REPO / path
