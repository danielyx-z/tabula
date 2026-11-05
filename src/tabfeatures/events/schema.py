"""Shared event labels and model heads."""

from __future__ import annotations

from ..labeling.schema import LABELS, label_heads, normalize_label

HEADS: tuple[str, ...] = (
    "pitched",
    "dead_note",
    "body_percussion",
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "dead_note": 0.60,
    "body_percussion": 0.60,
    "pitched": 0.50,
}

__all__ = [
    "LABELS",
    "HEADS",
    "DEFAULT_THRESHOLDS",
    "normalize_label",
    "label_heads",
]
