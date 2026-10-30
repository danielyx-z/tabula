#!/usr/bin/env python3
"""Run MIDI through T5 model -> raw tab -> post-processed tab."""
import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search
from fretwire.metrics import decode_predictions
from fretwire.postproc import postprocess
from fretwire.render import render_tab
from fretwire.schema import Note
from fretwire.fretboard import pitch_of, STANDARD
from fretwire.parse.midi import parse_midi
from preview import write_smf, synthesize, write_wav

OUT_DIR = "/tmp/lover_final"
import os
os.makedirs(OUT_DIR, exist_ok=True)

vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to("cuda").eval()
model.to(torch.bfloat16)

# Load MIDI input
midi_data = parse_midi("/tmp/lover_girl.mid")
all_notes = midi_data.notes
all_notes.sort(key=lambda n: (n.t, n.pitch))
print(f"Input notes: {len(all_notes)}")

enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0, snap_shifts=False))
print(f"Encoder tokens: {len(enc_ids)}")

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(all_notes, vocab, "lover_girl", chunk_cfg)
print(f"Inference chunks: {len(chunks)}")

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
    print(f"  Chunk {chunk_idx+1}: {pred_tabs}")

print(f"\nTotal predicted TAB events: {len(all_pred_tabs)}")

proc_notes = postprocess(all_pred_tabs, all_notes, mode="full")
proc_tabs = [(n.s, n.f) for n in proc_notes]

print(f"\n=== MODEL PREDICTIONS ===")
for i, (s, f) in enumerate(proc_tabs):
    pred_pitch = pitch_of(s, f, STANDARD)
    gt_pitch = all_notes[i].pitch if i < len(all_notes) else "MISSING"
    match = "✓" if pred_pitch == gt_pitch else "✗"
    print(f"  {i:3} pred=({s},{f}) pitch={pred_pitch} gt={gt_pitch} {match}")

print(f"\n=== TAB OUTPUT ===")
print(render_tab(proc_notes, width=100))

# Save input MIDI
input_note_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in all_notes]
write_smf(f"{OUT_DIR}/input.mid", input_note_dicts)
print(f"\nSaved input MIDI: {OUT_DIR}/input.mid")

# Save raw tab MIDI
raw_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(all_pred_tabs):
        s, f = all_pred_tabs[i]
    else:
        s, f = 1, 0
    raw_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": pitch_of(s, f, STANDARD), "s": s, "f": f})

raw_midi_path = f"{OUT_DIR}/raw.mid"
write_smf(raw_midi_path, raw_note_dicts)
print(f"Saved raw MIDI: {raw_midi_path}")

# Save post-processed MIDI
proc_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    proc_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": pitch_of(s, f, STANDARD), "s": s, "f": f})

pred_midi_path = f"{OUT_DIR}/final.mid"
write_smf(pred_midi_path, proc_note_dicts)
print(f"Saved final MIDI: {pred_midi_path}")

# Synthesize WAVs
arr_input = np.empty(len(input_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(input_note_dicts):
    arr_input[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_in = synthesize(arr_input, 120.0)
write_wav(f"{OUT_DIR}/input.wav", samples_in)
print(f"Saved input WAV: {OUT_DIR}/input.wav ({len(samples_in)/44100:.1f}s)")

arr_raw = np.empty(len(raw_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(raw_note_dicts):
    arr_raw[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_raw = synthesize(arr_raw, 120.0)
write_wav(f"{OUT_DIR}/raw.wav", samples_raw)
print(f"Saved raw WAV: {OUT_DIR}/raw.wav ({len(samples_raw)/44100:.1f}s)")

arr_proc = np.empty(len(proc_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(proc_note_dicts):
    arr_proc[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_proc = synthesize(arr_proc, 120.0)
write_wav(f"{OUT_DIR}/final.wav", samples_proc)
print(f"Saved final WAV: {OUT_DIR}/final.wav ({len(samples_proc)/44100:.1f}s)")

# Tab text
tab_text = render_tab(proc_notes, width=100)
tab_path = f"{OUT_DIR}/final_tab.txt"
with open(tab_path, "w") as f:
    f.write("=== PREDICTED TABLATURE ===\n")
    f.write(tab_text)
    f.write("\n")
print(f"\nSaved tab text: {tab_path}")
print(tab_text)

# Stats
pitch_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if pitch_of(all_pred_tabs[i][0], all_pred_tabs[i][1], STANDARD) == all_notes[i].pitch)
tab_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if all_pred_tabs[i] == (all_notes[i].s, all_notes[i].f))
print(f"\n=== STATS ===")
print(f"Pitch accuracy: {pitch_matches}/{len(all_notes)} = {pitch_matches/len(all_notes)*100:.1f}%")
print(f"Tab accuracy: {tab_matches}/{len(all_notes)} = {tab_matches/len(all_notes)*100:.1f}%")
