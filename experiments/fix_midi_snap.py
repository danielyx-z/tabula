from pathlib import Path

p = Path("/home/daniel/Projects/tabula/src/fretwire/parse/midi.py")
s = p.read_text()

old = """    from ..tokenizer import notes_to_events
    # notes_to_events expects Note objects with .t, .dur, .pitch, .s, .f
    # It ignores s/f for encoder.  We just need to ensure the Note objects
    # have the right fields.
    event_tokens = notes_to_events(notes, base_t=base_t)
    return vocab.encode(event_tokens)"""
new = """    from ..tokenizer import notes_to_events
    # notes_to_events expects Note objects with .t, .dur, .pitch, .s, .f
    # It ignores s/f for encoder.  We just need to ensure the Note objects
    # have the right fields.
    event_tokens = notes_to_events(notes, base_t=base_t, snap_shifts=snap_shifts)
    return vocab.encode(event_tokens)"""
assert s.count(old) == 1, f"midi.py: count={s.count(old)}"
s = s.replace(old, new)

# Add snap_shifts parameter to function signature
old_sig = """def midi_to_encoder_tokens(
    notes: list[Note],
    vocab,
    base_t: int = 0,
) -> list[int]:"""
new_sig = """def midi_to_encoder_tokens(
    notes: list[Note],
    vocab,
    base_t: int = 0,
    snap_shifts: bool = False,
) -> list[int]:"""
assert s.count(old_sig) == 1, f"midi.py sig: count={s.count(old_sig)}"
s = s.replace(old_sig, new_sig)

p.write_text(s)
print("parse/midi.py updated")

# Fix cli.py
p = Path("/home/daniel/Projects/tabula/src/fretwire/cli.py")
s = p.read_text()
old = "enc_ids = midi_to_encoder_tokens(midi.notes, vocab)"
new = "enc_ids = midi_to_encoder_tokens(midi.notes, vocab, snap_shifts=True)"
assert s.count(old) == 1, f"cli: count={s.count(old)}"
s = s.replace(old, new)
p.write_text(s)
print("src/fretwire/cli.py updated")
