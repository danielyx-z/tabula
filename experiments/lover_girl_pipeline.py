#!/usr/bin/env python3
"""Run MIDI through T5 model -> tab output."""
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
from fretwire.fretboard import pitch_of, STANDARD
from fretwire.parse.midi import parse_midi
from preview import write_smf, synthesize, write_wav

OUT_DIR = "/tmp/lover_girl_out"
import os
os.makedirs(OUT_DIR, exist_ok=True)

vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to("cuda").eval()
model.to(torch.bfloat16)

# Load MIDI using fretwire's parser (handles tempo correctly)
midi_track = parse_midi("/tmp/lover_girl.mid")

all_notes = midi_track.notes
all_notes.sort(key=lambda n: (n.t, n.pitch))
print(f"MIDI notes: {len(all_notes)}")
print(f"First 30 pitches: {[n.pitch for n in all_notes[:30]]}")

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

# Save output MIDI
proc_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    proc_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": pitch_of(s, f, STANDARD), "s": s, "f": f})

pred_midi_path = f"{OUT_DIR}/predicted_tab.mid"
write_smf(pred_midi_path, proc_note_dicts)
print(f"\nSaved predicted MIDI: {pred_midi_path}")

# Save WAV
arr = np.empty(len(proc_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(proc_note_dicts):
    arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples = synthesize(arr, 120.0)
pred_wav_path = f"{OUT_DIR}/predicted_tab.wav"
write_wav(pred_wav_path, samples)
print(f"Saved predicted WAV: {pred_wav_path} ({len(samples)/44100:.1f}s)")

# Save input MIDI
input_note_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in all_notes]
input_midi_path = f"{OUT_DIR}/input_midi.mid"
write_smf(input_midi_path, input_note_dicts)
print(f"Saved input MIDI: {input_midi_path}")

print(f"\n=== STATS ===")
pitch_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if pitch_of(all_pred_tabs[i][0], all_pred_tabs[i][1], STANDARD) == all_notes[i].pitch)
tab_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if all_pred_tabs[i] == (all_notes[i].s, all_notes[i].f))
print(f"Pitch accuracy: {pitch_matches}/{len(all_notes)} = {pitch_matches/len(all_notes)*100:.1f}%")
print(f"Tab accuracy: {tab_matches}/{len(all_notes)} = {tab_matches/len(all_notes)*100:.1f}%")
