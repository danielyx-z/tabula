"""Minimal Standard MIDI File writer (one track, tempo + note events)."""

from __future__ import annotations

import struct
from pathlib import Path

from core.quantise import PPQ


def _vlq(value: int) -> bytes:
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(out)


def write_smf(
    path: str | Path,
    notes,
    bpm: float = 120.0,
    ppq: int = PPQ,
    program: int = 25,
) -> None:
    """Write ``notes`` (t, dur, pitch, ...) as a type-0 MIDI file."""
    events: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC0, program]))]
    for n in notes:
        t = int(n["t"])
        events.append((t, 1, bytes([0x90, int(n["pitch"]), 90])))
        events.append((t + int(n["dur"]), 0, bytes([0x80, int(n["pitch"]), 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    data = bytearray()
    previous = 0
    micros = int(round(60_000_000 / max(1.0, bpm)))
    data += _vlq(0) + b"\xFF\x51\x03" + micros.to_bytes(3, "big")
    for tick, _, payload in events:
        data += _vlq(tick - previous) + payload
        previous = tick
    data += _vlq(0) + b"\xFF\x2F\x00"

    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq))
        fh.write(b"MTrk" + struct.pack(">I", len(data)) + bytes(data))
