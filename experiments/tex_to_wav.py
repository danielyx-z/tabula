#!/usr/bin/env python3
"""Convert alphaTex .tex files to WAV using fretwire's synthesizer."""
import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

from fretwire.parse.alphatex import parse_file
from preview import synthesize, write_wav
import numpy as np
import os

TEX_FILES = [
    "/tmp/guitar_tab_data/Blues_Acoustic_Guitar/Blues_Acoustic_Guitar_-_Blues_Correction,_Blues_Chords.tex",
    "/tmp/guitar_tab_data/Babylon_Circus/Babylon_Circus_-_Paul.tex",
    "/tmp/guitar_tab_data/Jag_Panzer/Jag_Panzer_-_Art_Of_Chris_Broderick.tex",
    "/tmp/guitar_tab_data/Deadwyler,_Jason/Deadwyler,_Jason_-_Pyrochylia.tex",
    "/tmp/guitar_tab_data/U.D.O/U.D.O._-_We-re_History.tex",
]

OUT_DIR = "/tmp/guitar_tab_wav"
os.makedirs(OUT_DIR, exist_ok=True)

for tex_path in TEX_FILES:
    name = tex_path.split("/")[-1].replace(".tex", "")
    print(f"Processing: {name}")
    try:
        tracks = parse_file(open(tex_path).read(), tex_path, name)
        if not tracks:
            print(f"  No tracks parsed, skipping")
            continue
        t = tracks[0]
        notes_arr = np.empty(len(t.notes), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
        for i, n in enumerate(t.notes):
            notes_arr[i] = (n.t, n.dur, n.pitch, n.s, n.f)
        samples = synthesize(notes_arr, t.tempo)
        out_path = f"{OUT_DIR}/{name}.wav"
        write_wav(out_path, samples)
        print(f"  Saved: {out_path} ({len(samples)/44100:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
