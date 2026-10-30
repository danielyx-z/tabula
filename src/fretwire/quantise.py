"""Quantisation: 960 PPQ, 60-tick grid (1/16 note). Single source of truth."""

from __future__ import annotations

from dataclasses import dataclass

PPQ = 960
GRID = 60  # 1/16 note
MIN_DUR = GRID  # minimum duration


@dataclass(frozen=True)
class QuantConfig:
    ppq: int = PPQ
    grid: int = GRID
    min_dur: int = MIN_DUR
    max_shift: int = 15360  # 2 bars at 1/16
    coarse_fallbacks: tuple[int, ...] = (11520, 15360)  # 1.5 bars, 2 bars


CFG = QuantConfig()


def quantise_tick(t: int, grid: int = GRID) -> int:
    """Round tick to nearest grid multiple."""
    return round(t / grid) * grid


def quantise_dur(dur: int, grid: int = GRID, min_dur: int = MIN_DUR) -> int:
    """Quantise duration, clamp to minimum."""
    q = round(dur / grid) * grid
    return max(min_dur, q)


def decompose_shift(shift: int, grid: int = GRID, max_shift: int = 15360,
                    coarse_fallbacks: tuple[int, ...] = (11520, 15360)) -> list[int]:
    """
    Decompose a time shift into available TIME_SHIFT tokens.
    Uses fine grid shifts (multiples of grid up to max_shift) plus coarse fallbacks.
    """
    if shift <= 0:
        return []
    out = []
    remaining = shift
    # Use coarse fallbacks first for large shifts
    for coarse in sorted(coarse_fallbacks, reverse=True):
        while remaining >= coarse:
            out.append(coarse)
            remaining -= coarse
    # Then use fine grid
    while remaining >= max_shift:
        out.append(max_shift)
        remaining -= max_shift
    if remaining > 0:
        # Round to the nearest grid step.  Do not try to "absorb" a sub-grid
        # remainder into the previous token: that can produce a value that is
        # not in SHIFT_VOCAB (e.g. 15360 + 20 -> 15380), which would encode as
        # <unk>.  Dropping a remainder below half a grid step is the same
        # quantisation the rest of the pipeline uses.
        q = quantise_tick(remaining, grid)
        if q > 0:
            out.append(q)
    return out


def build_shift_vocab(cfg: QuantConfig = CFG) -> list[int]:
    """All valid TIME_SHIFT values: multiples of grid up to max_shift + coarse fallbacks."""
    shifts = list(range(cfg.grid, cfg.max_shift + 1, cfg.grid))
    for c in cfg.coarse_fallbacks:
        if c not in shifts:
            shifts.append(c)
    return sorted(set(shifts))


SHIFT_VOCAB = build_shift_vocab()
SHIFT_TO_ID = {v: i for i, v in enumerate(SHIFT_VOCAB)}
ID_TO_SHIFT = {i: v for i, v in enumerate(SHIFT_VOCAB)}


def shift_to_token(shift: int) -> str:
    return f"TIME_SHIFT<{shift}>"


def token_to_shift(tok: str) -> int:
    assert tok.startswith("TIME_SHIFT<") and tok.endswith(">")
    return int(tok[11:-1])


def encode_shifts(shifts: list[int]) -> list[str]:
    """Encode a list of raw shifts (ticks) into TIME_SHIFT tokens using decomposition."""
    tokens = []
    for s in shifts:
        for part in decompose_shift(s):
            tokens.append(shift_to_token(part))
    return tokens


def decode_shifts(tokens: list[str]) -> list[int]:
    """Decode TIME_SHIFT tokens back to tick shifts."""
    return [token_to_shift(t) for t in tokens if t.startswith("TIME_SHIFT<")]


def shifts_roundtrip_error(original_shifts: list[int]) -> tuple[int, int]:
    """
    Check round-trip fidelity. Returns (exact_matches, total_shifts).
    A shift matches if the sum of decoded tokens equals the original.
    """
    tokens = encode_shifts(original_shifts)
    decoded = decode_shifts(tokens)
    # Reconstruct by summing consecutive tokens
    reconstructed = []
    i = 0
    for orig in original_shifts:
        s = 0
        while i < len(decoded) and s < orig:
            s += decoded[i]
            i += 1
        reconstructed.append(s)
    exact = sum(1 for a, b in zip(original_shifts, reconstructed) if a == b)
    return exact, len(original_shifts)