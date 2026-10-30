import sys
sys.path.insert(0, "src")
from fretwire.parse.midi import parse_midi

midi = parse_midi("test_scale.mid")
print(f"Track: {midi.name}")
print(f"Program: {midi.program}")
print(f"Notes: {len(midi.notes)}")
tempo = midi.tempo_changes[0][1] if midi.tempo_changes else "N/A"
print(f"Tempo: {tempo} BPM")
for i, n in enumerate(midi.notes[:10]):
    print(f"  {i}: t={n.t} dur={n.dur} pitch={n.pitch} s={n.s} f={n.f}")
print("...")