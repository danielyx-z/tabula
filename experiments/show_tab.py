#!/usr/bin/env python3
"""Render predicted tab from the 10s YouTube pipeline."""
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

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(all_notes, vocab, "yt_input", chunk_cfg)

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

proc_notes = postprocess(all_pred_tabs, all_notes, mode="full")

print("=== GROUND TRUTH NOTES (from basic-pitch) ===")
for i, gt in enumerate(all_notes):
    print(f"  {i:3} pitch={gt.pitch:>3} t={gt.t:>6} dur={gt.dur:>4}")

print(f"\n=== MODEL PREDICTIONS ({len(all_pred_tabs)} tabs) ===")
for i, (s, f) in enumerate(all_pred_tabs):
    pred_pitch = pitch_of(s, f, STANDARD)
    gt_pitch = all_notes[i].pitch if i < len(all_notes) else "MISSING"
    match = "✓" if pred_pitch == gt_pitch else "✗"
    print(f"  {i:3} pred=({s},{f}) pitch={pred_pitch} gt={gt_pitch} {match}")

print(f"\n=== MISSING NOTES ({len(all_notes) - len(all_pred_tabs)} missing) ===")
for i in range(len(all_pred_tabs), len(all_notes)):
    gt = all_notes[i]
    print(f"  {i:3} pitch={gt.pitch} t={gt.t} dur={gt.dur} - NO PREDICTION")

print(f"\n=== PREDICTED TABLATURE ===")
print(render_tab(proc_notes, width=100))

print(f"\n=== STATS ===")
pitch_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if pitch_of(all_pred_tabs[i][0], all_pred_tabs[i][1], STANDARD) == all_notes[i].pitch)
tab_matches = sum(1 for i in range(min(len(all_pred_tabs), len(all_notes))) if all_pred_tabs[i] == (all_notes[i].s, all_notes[i].f))
print(f"Pitch accuracy: {pitch_matches}/{len(all_notes)} = {pitch_matches/len(all_notes)*100:.1f}%")
print(f"Tab accuracy: {tab_matches}/{len(all_notes)} = {tab_matches/len(all_notes)*100:.1f}%")
print(f"Missing notes: {len(all_notes) - len(all_pred_tabs)}/{len(all_notes)}")
