from pathlib import Path

# Fix capture_artifacts.py
p = Path("/home/daniel/Projects/tabula/capture_artifacts.py")
s = p.read_text()
old = 'enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0))'
new = 'enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0, snap_shifts=True))'
assert s.count(old) == 1, f"capture_artifacts: count={s.count(old)}"
s = s.replace(old, new)
p.write_text(s)
print("capture_artifacts.py updated")

# Fix scripts/audio2tab.py
p = Path("/home/daniel/Projects/tabula/scripts/audio2tab.py")
s = p.read_text()
old = 'enc_ids = vocab.encode(notes_to_events(note_objs, base_t=0))'
new = 'enc_ids = vocab.encode(notes_to_events(note_objs, base_t=0, snap_shifts=True))'
assert s.count(old) == 1, f"audio2tab: count={s.count(old)}"
s = s.replace(old, new)
p.write_text(s)
print("scripts/audio2tab.py updated")

# Fix src/fretwire/cli.py
p = Path("/home/daniel/Projects/tabula/src/fretwire/cli.py")
s = p.read_text()
old = 'enc_ids = vocab.encode(notes_to_events(note_objs, base_t=0))'
new = 'enc_ids = vocab.encode(notes_to_events(note_objs, base_t=0, snap_shifts=True))'
assert s.count(old) == 1, f"cli: count={s.count(old)}"
s = s.replace(old, new)
p.write_text(s)
print("src/fretwire/cli.py updated")
