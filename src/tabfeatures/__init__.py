"""Audio-side dead-note detection and tab stitching.

This package deliberately lives outside ``src/tablature`` so the Tabula
training/inference path is untouched.  It consumes:

* the original audio file, for events Standard MIDI cannot represent, and
* a MIDI file produced elsewhere (e.g. MuScriptor), already quantised.

and produces a tab with ``x`` dead notes plus optional inferred hammer-ons /
pull-offs and manual articulation overlays.
"""

from .audio_events import detect_audio_events, write_audio_events
from .dead import insert_dead_notes, trim_notes_for_dead
from .hp import infer_hp
from .overlay import apply_overlay, load_overlay
from .reconcile import align_events, read_midi_notes
from .render import render_tab_with_features
from .ticks import PPQ, parse_grid, seconds_to_tick, snap_tick

__all__ = [
    "PPQ",
    "parse_grid",
    "seconds_to_tick",
    "snap_tick",
    "detect_audio_events",
    "write_audio_events",
    "read_midi_notes",
    "align_events",
    "insert_dead_notes",
    "trim_notes_for_dead",
    "infer_hp",
    "load_overlay",
    "apply_overlay",
    "render_tab_with_features",
]
