"""Quantisation: 960 PPQ, 20-tick grid (1/48 note). Single source of truth.

A 20-tick grid represents triplets (320), dotted values (720) and 32nd notes
(120) exactly. TIME_SHIFT is compositional: a small set of durations covers
every multiple of the grid, so the vocabulary does not need one token per
possible shift value.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schema import Note

PPQ = 960
GRID = 20  # 1/48 note
MIN_DUR = GRID  # minimum duration

# Every multiple of GRID is a sum of these.  The set is deliberately small:
# a shift of 960 costs one token, while an odd value like 340 costs three
# (240 + 60 + 40) instead of being unrepresentable.
SHIFT_TOKENS: tuple[int, ...] = (20, 40, 60, 120, 180, 240, 480, 720, 960, 1920, 3840)


@dataclass(frozen=True)
class QuantConfig:
    ppq: int = PPQ
    grid: int = GRID
    min_dur: int = MIN_DUR
    max_shift: int = 3840  # one 4/4 bar
    shift_tokens: tuple[int, ...] = SHIFT_TOKENS


CFG = QuantConfig()


def quantise_tick(t: int, grid: int = GRID) -> int:
    """Round tick to nearest grid multiple."""
    return round(t / grid) * grid


def quantise_dur(dur: int, grid: int = GRID, min_dur: int = MIN_DUR) -> int:
    """Quantise duration, clamp to minimum."""
    q = quantise_tick(dur, grid)
    return max(min_dur, q)


def decompose_shift(
    shift: int,
    grid: int = GRID,
    max_shift: int = 3840,
    shift_tokens: tuple[int, ...] = SHIFT_TOKENS,
) -> list[int]:
    """Decompose a time shift into valid TIME_SHIFT token values.

    Greedy largest-first.  Because the token set contains 20, 40 and 60, every
    multiple of the 20-tick grid is representable exactly; a raw shift is first
    rounded to that grid.
    """
    if shift <= 0:
        return []

    remaining = quantise_tick(shift, grid)
    if remaining <= 0:
        return []

    tokens = sorted(shift_tokens, reverse=True)
    out: list[int] = []
    while remaining > 0:
        token = next((t for t in tokens if t <= remaining), None)
        if token is None:  # sub-grid remainder; quantisation drops it
            break
        out.append(token)
        remaining -= token
    return out


def build_shift_vocab(cfg: QuantConfig = CFG) -> list[int]:
    return list(cfg.shift_tokens)


SHIFT_VOCAB = build_shift_vocab()
SHIFT_TO_ID = {v: i for i, v in enumerate(SHIFT_VOCAB)}
ID_TO_SHIFT = {i: v for i, v in enumerate(SHIFT_VOCAB)}


def shift_to_token(shift: int) -> str:
    return f"TIME_SHIFT<{shift}>"


def token_to_shift(tok: str) -> int:
    assert tok.startswith("TIME_SHIFT<") and tok.endswith(">")
    return int(tok[11:-1])


def encode_shifts(shifts: list[int]) -> list[str]:
    """Encode raw tick shifts into TIME_SHIFT tokens."""
    tokens: list[str] = []
    for s in shifts:
        for part in decompose_shift(s):
            tokens.append(shift_to_token(part))
    return tokens


def decode_shifts(tokens: list[str]) -> list[int]:
    """Decode TIME_SHIFT tokens back to tick shifts."""
    return [token_to_shift(t) for t in tokens if t.startswith("TIME_SHIFT<")]


def shifts_roundtrip_error(original_shifts: list[int]) -> tuple[int, int]:
    """Check round-trip fidelity. Returns (exact_matches, total_shifts)."""
    exact = 0
    for shift in original_shifts:
        if sum(decompose_shift(shift)) == quantise_tick(shift):
            exact += 1
    return exact, len(original_shifts)


def quantise_notes(
    notes: list[Note],
    grid: int = GRID,
    chord_window: int = 40,
) -> list[Note]:
    """
    Quantise note onsets and durations to the grid, and collapse strums within
    ``chord_window`` ticks into a single onset group (matching training preprocessing).

    Returns new Note objects; does not modify input.
    """
    if not notes:
        return []

    # Quantise onsets and durations.
    quantised = []
    for n in notes:
        t = quantise_tick(n.t, grid)
        dur = quantise_dur(n.dur, grid)
        quantised.append(Note(t=t, dur=dur, pitch=n.pitch, s=n.s, f=n.f))

    # Sort by onset, then string (matching preprocess_guitarset).
    quantised.sort(key=lambda n: (n.t, n.s, n.pitch, n.f))

    # Group onsets within chord_window ticks: collapse to the earliest tick.
    if chord_window > 0:
        i = 0
        while i < len(quantised):
            j = i + 1
            while j < len(quantised) and quantised[j].t - quantised[i].t <= chord_window:
                j += 1
            if j > i + 1:
                # Collapse onsets i..j-1 to the first onset.
                new_t = quantised[i].t
                for k in range(i, j):
                    quantised[k] = Note(
                        t=new_t,
                        dur=quantised[k].dur,
                        pitch=quantised[k].pitch,
                        s=quantised[k].s,
                        f=quantised[k].f,
                    )
            i = j

    # Deduplicate same string at same tick (keep first).
    seen: set[tuple[int, int]] = set()
    unique = []
    for n in quantised:
        key = (n.s, n.t)
        if key in seen:
            continue
        seen.add(key)
        unique.append(n)

    return unique