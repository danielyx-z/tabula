"""Tick/grid helpers shared by the stitching stages."""

from __future__ import annotations

from typing import Any

PPQ = 960

#: Grid tokens at PPQ 960.  These mirror the precision choices exposed by
#: MuScriptor's ``--quantize`` option so the sidecar can snap to the same grid
#: the MIDI was quantised to.
GRID_PRESETS: dict[str, int] = {
    "1/4": PPQ,
    "1/8": PPQ // 2,
    "1/8T": PPQ // 3,
    "1/16": PPQ // 4,
    "1/16T": PPQ // 6,
    "1/32": PPQ // 8,
    "1/32T": PPQ // 12,
}


def parse_grid(value: str | int | float | None) -> int:
    """Return a grid size in ticks.

    Accepts ``"1/16"``, ``"1/16T"``, ``"16"`` (meaning 1/16), or a raw tick
    count (``240``).
    """
    if value is None:
        return GRID_PRESETS["1/16"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        grid = int(value)
        if grid <= 0:
            raise ValueError(f"grid must be positive, got {value!r}")
        return grid

    key = str(value).strip().upper().replace(" ", "")
    if key in GRID_PRESETS:
        return GRID_PRESETS[key]
    if "/" not in key and key.isdigit():
        key = f"1/{key}"
        if key in GRID_PRESETS:
            return GRID_PRESETS[key]
    raise ValueError(
        f"unknown grid {value!r}; use one of {', '.join(GRID_PRESETS)} "
        "or a tick count"
    )


def seconds_to_tick(pm: Any, seconds: float) -> int:
    """Convert seconds to PPQ-960 ticks using the MIDI file's tempo map."""
    raw = pm.time_to_tick(float(seconds)) * PPQ / pm.resolution
    return int(round(raw))


def tick_to_seconds(pm: Any, tick: int) -> float:
    """Convert PPQ-960 ticks back to seconds using the MIDI tempo map."""
    file_tick = float(tick) * pm.resolution / PPQ
    return float(pm.tick_to_time(file_tick))


def snap_tick(tick: int | float, grid: int) -> int:
    """Snap ``tick`` to the nearest multiple of ``grid``."""
    if grid <= 0:
        raise ValueError("grid must be positive")
    return int(round(float(tick) / grid) * grid)


def bar_beat_label(tick: int, *, ppq: int = PPQ, beats_per_bar: int = 4) -> str:
    """Human-readable ``bar:beat`` label, using a plain 4/4 assumption."""
    bar_ticks = ppq * beats_per_bar
    bar = int(tick) // bar_ticks + 1
    beat = (int(tick) % bar_ticks) / ppq + 1.0
    return f"{bar}:{beat:.2f}"
