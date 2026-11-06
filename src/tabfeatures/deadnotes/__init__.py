"""Per-note pitched-vs-dead classification.

The classifier sits after the tab model.  It consumes the original audio, the
MIDI note events, and the model's string assignment, then relabels individual
notes as dead notes.  The renderer consumes the resulting ``dead_notes`` list.
"""

from .features import FEATURE_NAMES, AudioContext, extract_group_rows
from .inference import apply_dead_classifier, apply_predictions
from .model import DEFAULT_MODEL_DIR, DeadNoteModel, load_model, train_model

__all__ = [
    "FEATURE_NAMES",
    "AudioContext",
    "extract_group_rows",
    "apply_dead_classifier",
    "apply_predictions",
    "DEFAULT_MODEL_DIR",
    "DeadNoteModel",
    "load_model",
    "train_model",
]
