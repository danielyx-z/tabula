import sys
sys.path.insert(0, "src")
from basic_pitch.inference import predict

model_output, midi_data, note_events = predict(
    "/tmp/audio2tab_demo/laufey_valentine_15s.wav",
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=127.70,
    melodia_trick=True,
    midi_tempo=120.0,
)

print("=== BASIC-PITCH INPUT NOTES (first 15 seconds) ===\n")
print(f"Total notes: {sum(len(inst.notes) for inst in midi_data.instruments if not inst.is_drum)}\n")
print(f"{'start(s)':>8}  {'end(s)':>8}  {'dur(s)':>7}  {'pitch':>6}  {'note':>5}  {'vel':>4}")
print("-" * 40)

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

for inst in midi_data.instruments:
    if inst.is_drum:
        continue
    notes = sorted(inst.notes, key=lambda n: n.start)
    for n in notes:
        name = note_names[n.pitch % 12] + str(n.pitch // 12)
        print(
            f"{n.start:>8.3f}  {n.end:>8.3f}  {n.end-n.start:>7.3f}  "
            f"{n.pitch:>6}  {name:>5}  {n.velocity:>4}"
        )
