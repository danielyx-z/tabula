"""Human labels used by the browser labeller and the event model."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "pitched",
    "dead_string",
    "bass_flick",
    "body_percussion",
)

KEY_TO_LABEL = {
    "1": "pitched",
    "2": "dead_string",
    "3": "bass_flick",
    "4": "body_percussion",
}

LABEL_TO_KEY = {label: key for key, label in KEY_TO_LABEL.items()}


def normalize_label(value: object) -> str:
    """Return a canonical label or raise ``ValueError``."""
    label = str(value or "").strip().lower()
    if label not in LABELS:
        raise ValueError(
            f"unknown label {value!r}; expected one of {', '.join(LABELS)}"
        )
    return label


def label_heads(label: str) -> dict[str, int]:
    """Expand one human label into the binary heads used by the model.

    A bass flick is also a dead-string event, so it is positive for both
    ``dead_string`` and ``bass_flick``.  A body tap is not a string event.
    """
    label = normalize_label(label)
    return {
        "pitched": int(label == "pitched"),
        "dead_string": int(label in {"dead_string", "bass_flick"}),
        "bass_flick": int(label == "bass_flick"),
        "body_percussion": int(label == "body_percussion"),
    }
