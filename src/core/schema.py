"""Core data types shared by the parser, filters and writers."""

from __future__ import annotations

from dataclasses import dataclass, field

TICKS_PER_QUARTER = 960


@dataclass(slots=True)
class Note:
    t: int          # onset in ticks (PPQ 960)
    dur: int        # duration in ticks
    pitch: int      # MIDI pitch
    s: int          # string, 1 = thickest
    f: int          # fret


@dataclass(slots=True)
class Track:
    """One guitar track parsed out of a GuitarPro file."""

    source: str                     # archive member / file path
    song: str                       # song key used for splits and dedupe
    index: int                      # track index within the file
    tuning: tuple[int, ...]
    frets: int
    instrument: int
    tempo: int
    notes: list[Note] = field(default_factory=list)
    bars: list[float] = field(default_factory=list)   # bar durations in ticks
    flags: set[str] = field(default_factory=set)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def end(self) -> int:
        return max((n.t + n.dur for n in self.notes), default=0)


