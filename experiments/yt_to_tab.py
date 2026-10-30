#!/usr/bin/env python3
"""Run YouTube audio through basic-pitch -> T5 model -> tab output."""
import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

from basic_pitch.inference import predict
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search
from fretwire.metrics import decode_predictions
from fretwire.postproc import postprocess
from fretwire.render import render_tab
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.schema import Note
from fretwire.fretboard import pitch_of, STANDARD
import numpy as np
import torch
from transformers import T5ForConditionalGeneration

vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to("cuda").eval()
model.to(torch.bfloat16)

model_output, midi_data, note_events = predict(
    "/tmp/yt_10s.wav",
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=127.70,
    melodia_trick=True,
    midi_tempo=120.0,
)

all_notes = []
for inst in midi_data.instruments:
    if inst.is_drum:
        continue
    for n in inst.notes:
        onset_tick = int(round(midi_data.time_to_tick(n.start) * 960 / 220))
        end_tick = int(round(midi_data.time_to_tick(n.end) * 960 / 220))
        dur = max(1, end_tick - onset_tick)
        all_notes.append(Note(t=onset_tick, dur=dur, pitch=n.pitch, s=0, f=0))

all_notes.sort(key=lambda n: (n.t, n.pitch))
print(f"basic-pitch found {len(all_notes)} notes")
print(f"First 30 pitches: {[n.pitch for n in all_notes[:30]]}")

enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0, snap_shifts=True))
print(f"Encoder tokens: {len(enc_ids)}")

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(all_notes, vocab, "yt_input", chunk_cfg)
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
    print(f"Chunk {chunk_idx+1}: {pred_tabs}")

print(f"\nTotal predicted tabs: {len(all_pred_tabs)}")
print(f"\nPer-note comparison:")
print(f"{'#':>3} {'GT pitch':>8} {'Pred (s,f)':>12} {'Pred pitch':>9} {'Match':>6}")
for i, gt in enumerate(all_notes):
    if i < len(all_pred_tabs):
        s, f = all_pred_tabs[i]
        pred_pitch = pitch_of(s, f, STANDARD)
        match = "✓" if pred_pitch == gt.pitch else "✗"
        print(f"  {i:3} {gt.pitch:>8} ({s},{f})   {pred_pitch:>9} {match}")
    else:
        print(f"  {i:3} {gt.pitch:>8}   NONE   N/A ✗")

proc_notes = postprocess(all_pred_tabs, all_notes, mode="full")

print(f"\nTab output:\n")
print(render_tab(proc_notes, width=100))
