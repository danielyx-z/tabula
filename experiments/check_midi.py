import sys
sys.path.insert(0, "src")
import pretty_midi

pm = pretty_midi.PrettyMIDI("/tmp/tabula_guitar_demo/basic_pitch_midi.mid")
print(f"Resolution (PPQ): {pm.resolution}")
print(f"Estimated tempo: {pm.estimate_tempo():.1f} BPM")
for inst in pm.instruments:
    if inst.is_drum:
        continue
    notes = sorted(inst.notes, key=lambda n: n.start)
    print(f"Track: {inst.name}, Program: {inst.program}, Notes: {len(notes)}")
    if notes:
        print(f"First note: start={notes[0].start:.3f}s, pitch={notes[0].pitch}")
        print(f"Last note: start={notes[-1].start:.3f}s, end={notes[-1].end:.3f}s")
        print(f"Audio duration: {notes[-1].end:.2f}s")
    break
