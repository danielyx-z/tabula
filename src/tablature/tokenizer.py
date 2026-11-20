"""Tokenizer: vocab build, encode/decode, min-freq pruning, freeze + hash."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from core.quantise import (
    GRID,
    SHIFT_VOCAB,
    encode_shifts,
    quantise_tick,
    shift_to_token,
)
from core.schema import Note

# Special tokens (T5 conventions)
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# Pitch range for guitar (standard tuning, 24 frets): 40 (E2) to 88 (E6)
# But we'll cover a bit wider for safety: 36..96 (61 pitches)
MIN_PITCH = 36
MAX_PITCH = 96
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1  # 61

# Strings 1..6 (1 = thickest/low E), frets 0..24
NUM_STRINGS = 6
MAX_FRET = 24


def note_on_token(pitch: int) -> str:
    return f"NOTE_ON<{pitch}>"


def note_off_token(pitch: int) -> str:
    return f"NOTE_OFF<{pitch}>"


def tab_token(s: int, f: int) -> str:
    return f"TAB<{s},{f}>"


def parse_note_on(tok: str) -> int:
    assert tok.startswith("NOTE_ON<") and tok.endswith(">")
    return int(tok[8:-1])



def parse_tab(tok: str) -> tuple[int, int]:
    assert tok.startswith("TAB<") and tok.endswith(">")
    s, f = tok[4:-1].split(",")
    return int(s), int(f)


@dataclass(frozen=True)
class Vocab:
    """Frozen vocabulary with string<->id mappings."""
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    hash: str  # blake2b of the token list

    @property
    def size(self) -> int:
        return len(self.tokens)

    def __getitem__(self, tok: str) -> int:
        return self.token_to_id.get(tok, self.token_to_id[UNK_TOKEN])

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self[tok] for tok in tokens]

    def to_json(self) -> dict:
        return {
            "tokens": list(self.tokens),
            "hash": self.hash,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Vocab":
        tokens = tuple(data["tokens"])
        token_to_id = {t: i for i, t in enumerate(tokens)}
        id_to_token = {i: t for i, t in enumerate(tokens)}
        return cls(tokens, token_to_id, id_to_token, data["hash"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls.from_json(json.loads(Path(path).read_text()))


def build_base_vocab() -> list[str]:
    """Base vocabulary before frequency pruning."""
    tokens = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
    # NOTE_ON / NOTE_OFF for pitches 36..96
    for p in range(MIN_PITCH, MAX_PITCH + 1):
        tokens.append(note_on_token(p))
        tokens.append(note_off_token(p))
    # TIME_SHIFT tokens
    tokens.extend(shift_to_token(s) for s in SHIFT_VOCAB)
    # TAB tokens: 6 strings x 25 frets = 150
    for s in range(1, NUM_STRINGS + 1):
        for f in range(0, MAX_FRET + 1):
            tokens.append(tab_token(s, f))
    return tokens


def count_token_frequencies(tracks_iter: Iterator[tuple[dict, any]],
                            max_tracks: int = 0) -> Counter[str]:
    """Count token frequencies across tracks for pruning."""
    from core.schema import Note
    freq = Counter()
    for i, (meta, notes_arr) in enumerate(tracks_iter):
        if max_tracks and i >= max_tracks:
            break
        # Convert structured array to Note objects
        notes = []
        for row in notes_arr:
            notes.append(Note(
                t=int(row['t']),
                dur=int(row['dur']),
                pitch=int(row['pitch']),
                s=int(row['s']),
                f=int(row['f']),
            ))
        # Encoder side: NOTE_ON, NOTE_OFF, TIME_SHIFT
        events = notes_to_events(notes)
        for tok in events:
            freq[tok] += 1
        # Decoder side: TAB, TIME_SHIFT
        target = notes_to_target(notes)
        for tok in target:
            freq[tok] += 1
    return freq


def _event_stream(notes: list[Note]) -> list[tuple[int, int, Note]]:
    """Chronological ON/OFF events as ``(time, kind, note)``.

    ``kind`` is 1 for NOTE_ON and 0 for NOTE_OFF.  At the same tick note-offs
    sort before note-ons, so a retrigger on the same pitch is unambiguous.
    """
    events: list[tuple[int, int, Note]] = []
    for n in notes:
        events.append((n.t, 1, n))
        events.append((n.t + n.dur, 0, n))
    events.sort(key=lambda e: (e[0], e[1], e[2].pitch, e[2].s, e[2].f))
    return events


def notes_to_events(notes: list[Note], base_t: int = 0,
                    snap_shifts: bool = False) -> list[str]:
    """Convert notes to the v3 encoder stream (NOTE_ON/OFF + TIME_SHIFT).

    Events are interleaved chronologically.  ``base_t`` shifts the clock so a
    window taken from the middle of a track starts at zero instead of emitting
    one enormous leading TIME_SHIFT token.

    If ``snap_shifts`` is true, each inter-onset time delta is snapped to the
    nearest valid TIME_SHIFT token value before encoding.  This is useful when
    the input comes from an audio-to-MIDI step (e.g. MuScriptor) whose
    timings don't align to the training grid.
    """
    if not notes:
        return []
    tokens: list[str] = []
    last_t = base_t
    for time, kind, n in _event_stream(notes):
        delta = time - last_t
        if delta > 0:
            if snap_shifts:
                # Snap to the encoder grid.  TIME_SHIFT is compositional, so
                # any grid multiple decomposes exactly; snapping to a single
                # token value would be far coarser.
                delta = quantise_tick(delta, GRID)
            tokens.extend(encode_shifts([delta]))
            last_t = time
        tokens.append(note_on_token(n.pitch) if kind else note_off_token(n.pitch))
    return tokens


def notes_to_target(notes: list[Note], base_t: int = 0) -> list[str]:
    """Convert notes to the v3 decoder stream (TAB + TIME_SHIFT).

    The target follows the same event timeline as the encoder.  TIME_SHIFT is
    emitted whenever the clock advances, including at note-off times; TAB is
    emitted only for note-ons.  This is the paper's v3 example::

        input   NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55>
        target  TAB<3,0>    TIME_SHIFT<120>
    """
    if not notes:
        return []
    tokens: list[str] = []
    last_t = base_t
    for time, kind, n in _event_stream(notes):
        delta = time - last_t
        if delta > 0:
            tokens.extend(encode_shifts([delta]))
            last_t = time
        if kind:
            tokens.append(tab_token(n.s, n.f))
    return tokens


def build_vocab_from_tracks(tracks_iter: Iterator[tuple[dict, list[Note]]],
                            min_freq: int = 1,
                            max_tracks: int = 0) -> Vocab:
    """Build vocabulary from tracks, pruning tokens absent from the corpus."""
    base = build_base_vocab()
    freq = count_token_frequencies(tracks_iter, max_tracks)

    # Always keep special tokens and all TAB tokens (they're the target)
    # Keep the specials, all TAB tokens and the full (small) TIME_SHIFT set.
    # Only NOTE_ON / NOTE_OFF are pruned.  Keep every pitch that occurs at
    # least once: pruning a rare pitch creates an <unk> in its own training
    # data.  An unseen duration is different; TIME_SHIFT is compositional.
    keep = set()
    for tok in base:
        if tok in (PAD_TOKEN, EOS_TOKEN, UNK_TOKEN):
            keep.add(tok)
        elif tok.startswith("TAB<") or tok.startswith("TIME_SHIFT<"):
            keep.add(tok)
        elif freq.get(tok, 0) >= min_freq:
            keep.add(tok)

    # Rebuild in deterministic order: specials, NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB
    ordered = []
    for tok in base:
        if tok in keep:
            ordered.append(tok)

    tokens = tuple(ordered)
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}
    # Hash for reproducibility
    h = hashlib.blake2b()
    for t in tokens:
        h.update(t.encode())
    return Vocab(tokens, token_to_id, id_to_token, h.hexdigest()[:16])


# Default vocab (unpruned) for testing / initial runs
DEFAULT_VOCAB = Vocab(
    tokens=tuple(build_base_vocab()),
    token_to_id={t: i for i, t in enumerate(build_base_vocab())},
    id_to_token={i: t for i, t in enumerate(build_base_vocab())},
    hash="unpruned",
)

# Pad/EOS/UNK IDs for convenience
PAD_ID = DEFAULT_VOCAB[PAD_TOKEN]
EOS_ID = DEFAULT_VOCAB[EOS_TOKEN]
UNK_ID = DEFAULT_VOCAB[UNK_TOKEN]