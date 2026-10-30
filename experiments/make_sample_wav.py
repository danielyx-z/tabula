import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")
from fretwire.dataset import iter_tracks
from fretwire.schema import Note
from preview import synthesize, write_wav
import numpy as np

for meta, notes_arr in iter_tracks("data/processed", split="train"):
    if int(meta.get("id", -1)) == 4:
        notes = [Note(t=int(r["t"]), dur=int(r["dur"]), pitch=int(r["pitch"]), s=int(r["s"]), f=int(r["f"])) for r in notes_arr]
        break

arr = np.empty(len(notes), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(notes):
    arr[i] = (n.t, n.dur, n.pitch, n.s, n.f)

samples = synthesize(arr, 120.0)
write_wav("/tmp/tabula_sample.wav", samples)
print(f"Saved /tmp/tabula_sample.wav, {len(samples)/44100:.1f}s")
