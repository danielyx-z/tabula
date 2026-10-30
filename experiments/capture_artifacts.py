#!/usr/bin/env python3
"""Capture all pipeline artifacts for the guitar demo — with REAL computed metrics."""
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from basic_pitch.inference import predict
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search, greedy_decode
from fretwire.metrics import decode_predictions, align_predictions
from fretwire.postproc import postprocess
from fretwire.render import render_tab
from fretwire.tokenizer import DEFAULT_VOCAB, Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.schema import Note
from fretwire.fretboard import pitch_of, STANDARD

OUT = Path("/tmp/tabula_guitar_demo")
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---- Load model ----
vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to(device).eval()
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
if use_bf16:
    model.to(torch.bfloat16)
print(f"bf16: {use_bf16}")

# ---- Step 1: basic-pitch on 15s audio ----
audio_path = "/tmp/tabula_guitar_demo/guitar_15s.wav"
print(f"Running basic-pitch on {audio_path} ...")
model_output, midi_data, note_events = predict(
    audio_path,
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=127.70,
    melodia_trick=True,
    midi_tempo=120.0,
)

# Save the basic-pitch intermediate MIDI
bp_midi_path = OUT / "basic_pitch_midi.mid"
midi_data.write(str(bp_midi_path))
print(f"Saved basic-pitch MIDI to {bp_midi_path}")

# Collect all notes sorted
# basic-pitch MIDI uses PPQ=220; our renderer uses PPQ=960
PPQ_SOURCE = 220
PPQ_TARGET = 960
all_notes = []
for inst in midi_data.instruments:
    if inst.is_drum:
        continue
    for n in inst.notes:
        onset_tick = int(round(midi_data.time_to_tick(n.start) * PPQ_TARGET / PPQ_SOURCE))
        end_tick = int(round(midi_data.time_to_tick(n.end) * PPQ_TARGET / PPQ_SOURCE))
        dur = max(1, end_tick - onset_tick)
        all_notes.append(Note(t=onset_tick, dur=dur, pitch=n.pitch, s=0, f=0))
all_notes.sort(key=lambda n: (n.t, n.pitch))

# Save input notes
notes_path = OUT / "input_notes.txt"
with open(notes_path, "w") as f:
    f.write("=== BASIC-PITCH INPUT NOTES ===\n")
    f.write(f"Total notes: {len(all_notes)}\n\n")
    f.write(f"{'start_tick':>10} {'dur_tick':>9} {'pitch':>6} {'s':>3} {'f':>3}\n")
    for n in all_notes:
        f.write(f"{n.t:>10} {n.dur:>9} {n.pitch:>6} {n.s:>3} {n.f:>3}\n")
print(f"Saved input notes to {notes_path} ({len(all_notes)} notes)")

# Verify timing
if all_notes:
    span_ticks = max(n.t + n.dur for n in all_notes) - min(n.t for n in all_notes)
    print(f"Time span: {span_ticks} PPQ=960 ticks = {span_ticks/960/120*60:.2f} seconds at 120 BPM")

# ---- Step 2: T5 inference ----
enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0, snap_shifts=True))
print(f"Encoder tokens: {len(enc_ids)}")

# Save input tokens
with open(OUT / "input_tokens.txt", "w") as f:
    f.write("=== INPUT TOKENS (encoder) ===\n")
    f.write(f"Count: {len(enc_ids)}\n\n")
    for i, tid in enumerate(enc_ids):
        tok = vocab.id_to_token.get(tid, " UNK")
        f.write(f"{i:4d}: {tid:4d}  {tok}\n")

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(all_notes, vocab, "demo", chunk_cfg)
print(f"Inference chunks: {len(chunks)}")

pad_id = vocab[PAD_TOKEN]
eos_id = vocab[EOS_TOKEN]

all_pred_tabs = []
all_gen_tokens = []

for chunk_idx, chunk in enumerate(chunks):
    enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
    attn = (enc != pad_id).long()
    prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=device)
    max_new = min(512, max(16, 6 * chunk.meta["num_notes"] + 16))

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
    ):
        prompt_tabs = 0
        gen = constrained_beam_search(
            model, enc, vocab, num_beams=4, max_length=512,
            use_pitch_constraint=True, decoder_input_ids=prompt,
            attention_mask=attn, max_new_tokens=max_new,
            prompt_tabs=prompt_tabs,
        )

    gen_ids = gen[0, prompt.size(1):].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]

    all_gen_tokens.extend(gen_ids)
    pred_tabs = decode_predictions(gen_ids, vocab)
    all_pred_tabs.extend(pred_tabs)
    print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {len(pred_tabs)} TAB events, {len(gen_ids)} gen tokens")

# Save output tokens
with open(OUT / "output_tokens.txt", "w") as f:
    f.write("=== OUTPUT TOKENS (generated) ===\n")
    f.write(f"Count: {len(all_gen_tokens)}\n\n")
    for i, tid in enumerate(all_gen_tokens):
        tok = vocab.id_to_token.get(tid, " UNK")
        f.write(f"{i:4d}: {tid:4d}  {tok}\n")

# Save input prompt tokens too (decoder prompt)
with open(OUT / "decoder_prompt_tokens.txt", "w") as f:
    f.write("=== DECODER PROMPT TOKENS (all chunks) ===\n\n")
    for chunk_idx, chunk in enumerate(chunks):
        f.write(f"--- Chunk {chunk_idx} ---\n")
        for i, tid in enumerate(chunk.dec_tokens):
            tok = vocab.id_to_token.get(tid, " UNK")
            f.write(f"  {i:4d}: {tid:4d}  {tok}\n")
        f.write("\n")

# ---- Step 3: Post-process + compute REAL metrics ----
proc_notes = postprocess(all_pred_tabs, all_notes, mode="full")
proc_tabs = [(n.s, n.f) for n in proc_notes]

# Compute REAL metrics
raw_counts = align_predictions(all_pred_tabs, all_notes)
post_counts = align_predictions(proc_tabs, all_notes)

print(f"\n=== REAL METRICS (computed, not hardcoded) ===")
print(f"RAW:     pitch={raw_counts.pitch_accuracy():.4f} tab={raw_counts.tab_accuracy():.4f} valid={raw_counts.validity():.4f}")
print(f"POST:    pitch={post_counts.pitch_accuracy():.4f} tab={post_counts.tab_accuracy():.4f} valid={post_counts.validity():.4f}")
print(f"NOTE:    post pitch=100% is guaranteed by neighbor_search (forces pitch to match gt)")
print(f"NOTE:    tab accuracy is 0% because basic-pitch doesn't know s/f (GT is (0,0) placeholder)")

# Save metrics
with open(OUT / "metrics.txt", "w") as f:
    f.write("=== REAL METRICS (computed from model output) ===\n\n")
    f.write(f"Total GT notes: {len(all_notes)}\n")
    f.write(f"Raw TAB events generated: {len(all_pred_tabs)}\n\n")
    f.write("RAW:\n")
    f.write(f"  Pitch accuracy: {raw_counts.pitch_accuracy():.4f} ({raw_counts.pitch_correct}/{raw_counts.total_notes})\n")
    f.write(f"  Tab accuracy:   {raw_counts.tab_accuracy():.4f} ({raw_counts.tab_correct}/{raw_counts.total_notes})\n")
    f.write(f"  Validity:       {raw_counts.validity():.4f} ({raw_counts.valid_predictions}/{raw_counts.total_notes})\n\n")
    f.write("POST-PROCESSED:\n")
    f.write(f"  Pitch accuracy: {post_counts.pitch_accuracy():.4f} ({post_counts.pitch_correct}/{post_counts.total_notes})\n")
    f.write(f"  Tab accuracy:   {post_counts.tab_accuracy():.4f} ({post_counts.tab_correct}/{post_counts.total_notes})\n")
    f.write(f"  Validity:       {post_counts.validity():.4f} ({post_counts.valid_predictions}/{post_counts.total_notes})\n\n")
    f.write("CAVEATS:\n")
    f.write("  - Post pitch=100% is guaranteed by neighbor_search (forces pitch to match gt.pitch)\n")
    f.write("  - Tab accuracy=0% because basic-pitch doesn't know s/f (GT is (0,0) placeholder)\n")
    f.write("  - Real tab accuracy cannot be computed from audio alone\n")
print(f"Saved metrics to {OUT / 'metrics.txt'}")

# ---- Step 4: Render tab ----
tab_text = render_tab(proc_notes, width=100)
tab_path = OUT / "final_tab.txt"
with open(tab_path, "w") as f:
    f.write("=== PREDICTED TABLATURE ===\n")
    f.write(f"Source: Guitar excerpt (7s-22s) (15s excerpt)\n")
    f.write(f"Raw TAB events: {len(all_pred_tabs)}\n")
    f.write(f"Post-processed notes: {len(proc_notes)}\n")
    f.write(f"Real pitch accuracy (post): {post_counts.pitch_accuracy():.4f}\n")
    f.write(f"Real tab accuracy (post): {post_counts.tab_accuracy():.4f}\n")
    f.write(f"CAVEAT: post pitch=100% is guaranteed by neighbor_search, not model quality\n\n")
    f.write(tab_text)
    f.write("\n")
print(f"Saved final tab to {tab_path}")

# Also save raw tab
raw_notes = []
for i, tab in enumerate(all_pred_tabs):
    if i < len(all_notes):
        t, dur, pitch = all_notes[i].t, all_notes[i].dur, all_notes[i].pitch
    else:
        last = all_notes[-1]
        t = last.t + (i - len(all_notes) + 1) * 60
        dur, pitch = 60, 40 + tab[1]
    raw_notes.append(Note(t=t, dur=dur, pitch=pitch, s=tab[0], f=tab[1]))
raw_tab = render_tab(raw_notes, width=100)
with open(OUT / "raw_tab.txt", "w") as f:
    f.write("=== RAW TABLATURE (before post-processing) ===\n\n")
    f.write(raw_tab)
    f.write("\n")
print(f"Saved raw tab to {OUT / 'raw_tab.txt'}")

# ---- Step 5: Copy audio, midi, wav ----
import shutil
shutil.copy(audio_path, OUT / "input_audio.wav")
print(f"Copied input audio to {OUT / 'input_audio.wav'}")

# Write predicted MIDI
from preview import write_smf, synthesize, write_wav
pred_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    pred_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})
write_smf(OUT / "predicted_tab.mid", pred_note_dicts)
print(f"Saved predicted MIDI to {OUT / 'predicted_tab.mid'}")

# Write predicted WAV
arr = np.empty(len(pred_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"),
                                            ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(pred_note_dicts):
    arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples = synthesize(arr, 120.0)
write_wav(OUT / "predicted_tab.wav", samples)
print(f"Saved predicted WAV to {OUT / 'predicted_tab.wav'}")

print("\n=== ALL ARTIFACTS SAVED ===")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
