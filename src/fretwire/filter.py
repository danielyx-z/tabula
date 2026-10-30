"""Track admission rules.

Everything here is deliberately conservative: the MVP trains on a clean,
homogeneous slice (6-string guitar, standard tuning, no capo) and the rejects
are logged with a reason so the decision is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fretboard import STANDARD
from .parse.alphatex import GUITAR_PROGRAMS
from .schema import Track


@dataclass(frozen=True)
class FilterConfig:
    tuning: tuple[int, ...] = STANDARD
    min_notes: int = 50
    min_bars: int = 4
    max_harmonic_ratio: float = 0.20
    require_guitar_program: bool = False
    max_fret: int = 24


def evaluate(track: Track, cfg: FilterConfig = FilterConfig()) -> tuple[str, str] | None:
    """Return ``(reason, detail)`` if the track must be dropped, else ``None``."""
    if len(track.tuning) != 6:
        return "not_six_string", str(track.tuning)
    if tuple(track.tuning) != cfg.tuning:
        return "non_standard_tuning", " ".join(map(str, track.tuning))
    if cfg.require_guitar_program and track.instrument not in GUITAR_PROGRAMS:
        return "not_guitar_program", str(track.instrument)
    if track.n_notes < cfg.min_notes:
        return "too_few_notes", str(track.n_notes)
    if len(track.bars) < cfg.min_bars:
        return "too_few_bars", str(len(track.bars))
    if "bad_string" in track.flags or "bad_fret" in track.flags:
        return "invalid_fret_or_string", ",".join(sorted(track.flags))
    if "instrument_change" in track.flags:
        return "instrument_change", ""
    skipped = sorted(f for f in track.flags if f.startswith("skipped:"))
    if skipped:
        return "unparsed_characters", ",".join(skipped)
    harmonics = track.stats.get("harmonic", 0)
    if harmonics and harmonics / max(1, track.n_notes + harmonics) > cfg.max_harmonic_ratio:
        return "harmonic_heavy", f"{harmonics} harmonic notes"
    for note in track.notes:
        if note.pitch < 0 or note.pitch > 127:
            return "pitch_out_of_range", str(note.pitch)
        if note.f > cfg.max_fret:
            return "fret_out_of_mvp_range", str(note.f)
    return None
