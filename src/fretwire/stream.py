"""Decode a model TAB+TIME_SHIFT stream into timed notes.

The decoder emits only ``TAB<s,f>`` and ``TIME_SHIFT<n>`` tokens; it never
emits note-offs.  Onsets therefore come from the TIME_SHIFT clock, and a
note's duration is inferred from the next onset on the same string (the way a
tab is read), capped so a note cannot ring forever.

Everything here is derived from the model's own tokens.  No ground-truth
timing or pitch is consulted.
"""

from __future__ import annotations

from .fretboard import STANDARD, pitch_of
from .quantise import token_to_shift
from .tokenizer import EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, parse_tab


def decode_tab_stream(
    tokens: list[str],
    base_t: int = 0,
    tuning: tuple[int, ...] = STANDARD,
    default_dur: int = 60,
    min_dur: int = 30,
    max_dur: int = 960,
) -> list[dict]:
    """Return ``[{t, dur, pitch, s, f}, ...]`` from decoder tokens."""
    t = int(base_t)
    notes: list[dict] = []

    for tok in tokens:
        if tok in (EOS_TOKEN, PAD_TOKEN):
            break
        if tok == UNK_TOKEN:
            continue
        if tok.startswith("TIME_SHIFT<"):
            t += token_to_shift(tok)
        elif tok.startswith("TAB<"):
            s, f = parse_tab(tok)
            notes.append(
                {
                    "t": t,
                    "s": int(s),
                    "f": int(f),
                    "pitch": int(pitch_of(s, f, tuning)),
                }
            )

    for i, n in enumerate(notes):
        end = None
        for m in notes[i + 1 :]:
            if m["s"] == n["s"] and m["t"] > n["t"]:
                end = m["t"]
                break
        n["dur"] = default_dur if end is None else max(min_dur, min(max_dur, end - n["t"]))

    return notes
