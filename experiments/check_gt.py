import sys
sys.path.insert(0, "src")
from fretwire.dataset import iter_tracks
from fretwire.schema import Note
from fretwire.tokenizer import Vocab
from fretwire.fretboard import pitch_of, STANDARD

vocab = Vocab.load("vocab_v1_full.json")

gt_notes = []
for meta, notes_arr in iter_tracks("data/processed", split="train"):
    if int(meta.get("id", -1)) == 6:
        gt_notes = [Note(t=int(r["t"]), dur=int(r["dur"]), pitch=int(r["pitch"]), s=int(r["s"]), f=int(r["f"])) for r in notes_arr]
        break

gt_notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
print(f"Total GT notes: {len(gt_notes)}")
print(f"First 20 GT notes:")
mismatches = 0
for i, n in enumerate(gt_notes[:20]):
    pitch = pitch_of(n.s, n.f, STANDARD)
    match = "" if pitch == n.pitch else f"  <-- MISMATCH: s={n.s} f={n.f} -> pitch_of={pitch} != {n.pitch}"
    if pitch != n.pitch:
        mismatches += 1
    print(f"  {i}: t={n.t} dur={n.dur} pitch={n.pitch} s={n.s} f={n.f} -> pitch_of={pitch}{match}")
print(f"\nPitch/s/f mismatches in first 20: {mismatches}")
