import sys
sys.path.insert(0, "src")
from basic_pitch.inference import predict

model_output, midi_data, note_events = predict(
    "/tmp/audio2tab_demo/laufey_valentine.wav",
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=127.70,
    melodia_trick=True,
    midi_tempo=120.0,
)

# Print first 12 notes with timing in seconds
for inst in midi_data.instruments:
    if inst.is_drum:
        continue
    notes = sorted(inst.notes, key=lambda n: n.start)
    for n in notes[:12]:
        print(
            f"start={n.start:.3f}s  end={n.end:.3f}s  dur={n.end-n.start:.3f}s  "
            f"pitch={n.pitch}  vel={n.velocity}"
        )
    break

print(f"Total notes (non-drum): {sum(len(inst.notes) for inst in midi_data.instruments if not inst.is_drum)}")
print(f"Estimated tempo: {midi_data.estimate_tempo():.1f} BPM")
