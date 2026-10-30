from pathlib import Path

p = Path("/home/daniel/Projects/tabula/src/fretwire/tokenizer.py")
s = p.read_text()

# Add snap_shifts parameter to notes_to_events
old = """def notes_to_events(notes: list[Note], base_t: int = 0) -> list[str]:
    \"\"\"Convert notes to the v3 encoder stream (NOTE_ON/OFF + TIME_SHIFT).

    Events are interleaved chronologically.  ``base_t`` shifts the clock so a
    window taken from the middle of a track starts at zero instead of emitting
    one enormous leading TIME_SHIFT token.
    \"\"\"
    if not notes:
        return []
    tokens: list[str] = []
    last_t = base_t
    for time, kind, n in _event_stream(notes):
        delta = time - last_t
        if delta > 0:
            tokens.extend(encode_shifts([delta]))
            last_t = time
        tokens.append(note_on_token(n.pitch) if kind else note_off_token(n.pitch))
    return tokens"""

new = """def notes_to_events(notes: list[Note], base_t: int = 0,
                    snap_shifts: bool = False) -> list[str]:
    \"\"\"Convert notes to the v3 encoder stream (NOTE_ON/OFF + TIME_SHIFT).

    Events are interleaved chronologically.  ``base_t`` shifts the clock so a
    window taken from the middle of a track starts at zero instead of emitting
    one enormous leading TIME_SHIFT token.

    If ``snap_shifts`` is true, each inter-onset time delta is snapped to the
    nearest valid TIME_SHIFT token value before encoding.  This is useful when
    the input comes from an audio-to-MIDI step (e.g. basic-pitch) whose
    timings don't align to the training grid.
    \"\"\"
    if not notes:
        return []
    tokens: list[str] = []
    last_t = base_t
    for time, kind, n in _event_stream(notes):
        delta = time - last_t
        if delta > 0:
            if snap_shifts:
                # Snap to nearest valid TIME_SHIFT token so the encoder sees
                # the same token distribution it was trained on.
                delta = min(SHIFT_VOCAB, key=lambda v: abs(v - delta))
            tokens.extend(encode_shifts([delta]))
            last_t = time
        tokens.append(note_on_token(n.pitch) if kind else note_off_token(n.pitch))
    return tokens"""

assert s.count(old) == 1, "notes_to_events block not found"
s = s.replace(old, new)

p.write_text(s)
print("tokenizer.py updated: notes_to_events now accepts snap_shifts")

PYEOF
