import sys
sys.path.insert(0, "src")
from fretwire.dataset import iter_tracks
count = 0
for meta, notes in iter_tracks("data/processed", split="train"):
    print(f"id={meta.get('id')} song={meta.get('song')} notes={len(notes)} tuning={meta.get('tuning_name')}")
    count += 1
    if count >= 5:
        break
