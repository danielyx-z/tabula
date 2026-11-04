"""Tiny event model for dead notes, bass flicks, and body percussion."""

from .config import load_events_config
from .features import FEATURE_NAMES, extract_event_features, feature_vector
from .inference import apply_event_predictions, predict_events
from .model import EventModel, load_event_model, train_event_model
from .schema import DEFAULT_THRESHOLDS, HEADS, LABELS, label_heads

__all__ = [
    "LABELS",
    "HEADS",
    "DEFAULT_THRESHOLDS",
    "label_heads",
    "FEATURE_NAMES",
    "load_events_config",
    "extract_event_features",
    "feature_vector",
    "EventModel",
    "load_event_model",
    "train_event_model",
    "predict_events",
    "apply_event_predictions",
]
