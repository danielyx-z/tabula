import sys
sys.path.insert(0, "src")
from fretwire.dataset import iter_tracks
from fretwire.schema import Note
sys.path.insert(0, "/home/daniel/Projects/tabula")
from preview import write_smf
import numpy as np

count = 0
for meta, notes_arr in iter_tracks("data/processed", split="train"):
    notes = []
    for r in notes_arr:
        notes.append(Note(t=int(r["t"]), dur=int(r["dur"]), pitch=int(r["pitch"]), s=int(r["s"]), f=int(r["f"])))
    if len(notes) > 50:
        print(f"Track {meta.get('id')}: {meta.get('song')}, {len(notes)} notes")
        arr = np.empty(len(notes), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
        for i, n in enumerate(notes):
            arr[i] = (n.t, n.dur, n.pitch, n.s, n.f)
        write_smf("/tmp/tabula_sample.mid", arr)
        print("Saved /tmp/tabula_sample.mid")
        break
    count += 1
    if count > 100:
        break
