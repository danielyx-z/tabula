#!/usr/bin/env python3
"""Test T5 model on a simple C major scale."""
import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search, greedy_decode
from fretwire.metrics import decode_predictions
from fretwire.postproc import postprocess
from fretwire.render import render_tab
from fretwire.schema import Note
from preview import write_smf, synthesize, write_wav
from fretwire.fretboard import pitch_of, STANDARD

OUT_DIR = "/tmp/scale_test"
import os
os.makedirs(OUT_DIR, exist_ok=True)

vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to("cuda").eval()
model.to(torch.bfloat16)

# C major scale: C4 D4 E4 F4 G4 A4 B4 C5
scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]

# Create input MIDI notes
notes = []
for i, pitch in enumerate(scale_notes):
    notes.append(Note(t=i*480, dur=480, pitch=pitch, s=0, f=0))

print(f"Input scale: {[n.pitch for n in notes]}")

# Build encoder tokens
enc_ids = vocab.encode(notes_to_events(notes, base_t=0, snap_shifts=False))
print(f"Encoder tokens: {len(enc_ids)}")

# Inference chunks
chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(notes, vocab, "scale_test", chunk_cfg)
print(f"Chunks: {len(chunks)}")

pad_id = vocab[PAD_TOKEN]
eos_id = vocab[EOS_TOKEN]

all_pred_tabs = []
for chunk_idx, chunk in enumerate(chunks):
    enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device="cuda")
    attn = (enc != pad_id).long()
    prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device="cuda")
    max_new = min(512, max(16, 6 * chunk.meta["num_notes"] + 16))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        gen = constrained_beam_search(
            model, enc, vocab, num_beams=4, max_length=512,
            use_pitch_constraint=True, decoder_input_ids=prompt,
            attention_mask=attn, max_new_tokens=max_new,
            prompt_tabs=0,
        )
    gen_ids = gen[0, prompt.size(1):].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]
    pred_tabs = decode_predictions(gen_ids, vocab)
    all_pred_tabs.extend(pred_tabs)
    print(f"Chunk {chunk_idx+1}: {pred_tabs}")

print(f"\nGenerated tabs: {all_pred_tabs}")

# Show what pitch each predicted tab produces
for i, (s, f) in enumerate(all_pred_tabs):
    pred_pitch = pitch_of(s, f, STANDARD)
    gt_pitch = scale_notes[i] if i < len(scale_notes) else "N/A"
    match = "✓" if pred_pitch == gt_pitch else "✗"
    print(f"  Note {i}: GT pitch={gt_pitch} -> Pred ({s},{f}) = pitch {pred_pitch} {match}")

# Save input MIDI
pred_note_dicts = []
for i, n in enumerate(notes):
    if i < len(all_pred_tabs):
        s, f = all_pred_tabs[i]
    else:
        s, f = 1, 0
    pred_note_dicts.append({"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": s, "f": f})

write_smf(f"{OUT_DIR}/input_scale.mid", pred_note_dicts)
arr_in = np.empty(len(pred_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(pred_note_dicts):
    arr_in[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_in = synthesize(arr_in, 120.0)
write_wav(f"{OUT_DIR}/input_scale.wav", samples_in)
print(f"\nSaved input WAV: {OUT_DIR}/input_scale.wav ({len(samples_in)/44100:.1f}s)")

# Save output tab
proc_notes = postprocess(all_pred_tabs, notes, mode="full")
proc_tab_dicts = []
for i, n in enumerate(notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    proc_tab_dicts.append({"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": s, "f": f})

write_smf(f"{OUT_DIR}/output_tab.mid", proc_tab_dicts)
arr_out = np.empty(len(proc_tab_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(proc_tab_dicts):
    arr_out[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_out = synthesize(arr_out, 120.0)
write_wav(f"{OUT_DIR}/output_tab.wav", samples_out)
print(f"Saved output WAV: {OUT_DIR}/output_tab.wav ({len(samples_out)/44100:.1f}s)")

# Render tabs
print("\nInput tab:")
print(render_tab(notes, width=100))
print("\nOutput tab:")
print(render_tab(proc_notes, width=100))
