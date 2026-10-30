#!/usr/bin/env python3
"""Convert GuitarSet JAMS annotation to MIDI using jams + pretty_midi."""
import sys
import glob
import jams
import pretty_midi

GUITAR_PROGRAM = pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)')


def jam_to_midi(jams_path, output_path):
    jam = jams.load(jams_path)
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=GUITAR_PROGRAM)

    note_annotations = jam.search(namespace='note_midi')
    if not note_annotations:
        print(f"No note_midi annotations in {jams_path}")
        return

    count = 0
    for annotation in note_annotations:
        for data in annotation.data:
            start = data.time
            end = start + data.duration
            pitch = int(round(data.value))
            confidence = getattr(data, 'confidence', 1.0)
            velocity = int(confidence * 127) if confidence else 100
            if 0 <= pitch <= 127:
                instrument.notes.append(pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start, end=end
                ))
                count += 1

    pm.instruments.append(instrument)
    pm.write(output_path)
    print(f"{jams_path} -> {output_path} ({count} notes)")


if __name__ == "__main__":
    import os
    jams_dir = "/tmp/guitarSet_annotation"
    out_dir = "/tmp/guitarset_midi"
    os.makedirs(out_dir, exist_ok=True)
    for jf in sorted(glob.glob(f"{jams_dir}/*.jams")):
        name = jf.split("/")[-1].replace(".jams", ".mid")
        jam_to_midi(jf, f"{out_dir}/{name}")
