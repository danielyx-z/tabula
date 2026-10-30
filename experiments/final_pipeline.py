#!/usr/bin/env python3
"""Full pipeline: audio -> basic-pitch MIDI -> T5 model -> tab -> MIDI + WAV."""
import sys
sys.path.insert(0, "/home/daniel/Projects/tabula")
sys.path.insert(0, "/home/daniel/Projects/tabula/src")

from basic_pitch.inference import predict
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search, greedy_decode
from fretwire.metrics import decode_predictions
from fretwire.postproc import postprocess
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.schema import Note
from fretwire.fretboard import pitch_of, STANDARD
import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from preview import write_smf, synthesize, write_wav
from pathlib import Path

OUT_DIR = "/tmp/yt_final"
import os
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_PATH = "/tmp/yt_10s.wav"

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

# ---- Step 1: basic-pitch audio -> MIDI ----
print(f"Running basic-pitch on {AUDIO_PATH} ...")
model_output, midi_data, note_events = predict(
    AUDIO_PATH,
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=127.70,
    melodia_trick=True,
    midi_tempo=120.0,
)

# Save basic-pitch MIDI
bp_midi_path = f"{OUT_DIR}/basic_pitch_midi.mid"
midi_data.write(bp_midi_path)
print(f"Saved basic-pitch MIDI: {bp_midi_path}")

# Collect notes
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
print(f"basic-pitch found {len(all_notes)} notes")
print(f"Pitches: {[n.pitch for n in all_notes[:30]]}")

# Save input MIDI (basic-pitch output)
input_midi_path = f"{OUT_DIR}/input_midi.mid"
input_note_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": 0, "f": 0} for n in all_notes]
write_smf(input_midi_path, input_note_dicts)
print(f"Saved input MIDI: {input_midi_path}")

# ---- Step 2: T5 inference ----
enc_ids = vocab.encode(notes_to_events(all_notes, base_t=0, snap_shifts=True))
print(f"Encoder tokens: {len(enc_ids)}")

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(all_notes, vocab, "yt_input", chunk_cfg)
print(f"Inference chunks: {len(chunks)}")

pad_id = vocab[PAD_TOKEN]
eos_id = vocab[EOS_TOKEN]

all_pred_tabs = []
for chunk_idx, chunk in enumerate(chunks):
    enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
    attn = (enc != pad_id).long()
    prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=device)
    max_new = min(512, max(16, 6 * chunk.meta["num_notes"] + 16))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
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

# ---- Step 3: Post-process ----
proc_notes = postprocess(all_pred_tabs, all_notes, mode="full")
proc_tabs = [(n.s, n.f) for n in proc_notes]
print(f"Post-processed notes: {len(proc_notes)}")

# Pitch match analysis
print(f"\nPer-note comparison:")
pitch_matches = 0
tab_matches = 0
for i, gt in enumerate(all_notes):
    if i < len(all_pred_tabs):
        s, f = all_pred_tabs[i]
        pred_pitch = pitch_of(s, f, STANDARD)
        match = "✓" if pred_pitch == gt.pitch else "✗"
        if pred_pitch == gt.pitch:
            pitch_matches += 1
        if (s, f) == (gt.s, gt.f):
            tab_matches += 1
        print(f"  {i:3} GT:{gt.pitch:>3} -> Pred ({s},{f})={pred_pitch} {match}")
    else:
        print(f"  {i:3} GT:{gt.pitch:>3} -> MISSING ✗")

print(f"\nPitch accuracy: {pitch_matches}/{len(all_notes)} = {pitch_matches/len(all_notes)*100:.1f}%")
print(f"Tab accuracy: {tab_matches}/{len(all_notes)} = {tab_matches/len(all_notes)*100:.1f}%")

# ---- Step 4: Save outputs ----
# Predicted tab MIDI
proc_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    proc_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})

pred_midi_path = f"{OUT_DIR}/predicted_tab.mid"
write_smf(pred_midi_path, proc_note_dicts)
print(f"\nSaved predicted MIDI: {pred_midi_path}")

# Predicted WAV
arr = np.empty(len(proc_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(proc_note_dicts):
    arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples = synthesize(arr, 120.0)
pred_wav_path = f"{OUT_DIR}/predicted_tab.wav"
write_wav(pred_wav_path, samples)
print(f"Saved predicted WAV: {pred_wav_path} ({len(samples)/44100:.1f}s)")

# Also save raw tab
raw_note_dicts = []
for i, gt in enumerate(all_notes):
    if i < len(all_pred_tabs):
        s, f = all_pred_tabs[i]
    else:
        s, f = 1, 0
    raw_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})

raw_midi_path = f"{OUT_DIR}/raw_tab.mid"
write_smf(raw_midi_path, raw_note_dicts)
print(f"Saved raw MIDI: {raw_midi_path}")

arr_raw = np.empty(len(raw_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"), ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(raw_note_dicts):
    arr_raw[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_raw = synthesize(arr_raw, 120.0)
raw_wav_path = f"{OUT_DIR}/raw_tab.wav"
write_wav(raw_wav_path, samples_raw)
print(f"Saved raw WAV: {raw_wav_path} ({len(samples_raw)/44100:.1f}s)")

# Copy input audio to output dir
import shutil
shutil.copy(AUDIO_PATH, f"{OUT_DIR}/input_audio.wav")
print(f"Copied input audio: {OUT_DIR}/input_audio.wav")

# Tab text
tab_text = render_tab(proc_notes, width=100)
tab_path = f"{OUT_DIR}/final_tab.txt"
with open(tab_path, "w") as f:
    f.write("=== PREDICTED TABLATURE ===\n")
    f.write(f"Source: YouTube 10s clip\n")
    f.write(f"Raw TAB events: {len(all_pred_tabs)}\n")
    f.write(f"Post-processed notes: {len(proc_notes)}\n")
    f.write(f"Pitch accuracy: {pitch_matches}/{len(all_notes)} = {pitch_matches/len(all_notes)*100:.1f}%\n")
    f.write(f"Tab accuracy: {tab_matches}/{len(all_notes)} = {tab_matches/len(all_notes)*100:.1f}%\n\n")
    f.write(tab_text)
    f.write("\n")
print(f"Saved tab text: {tab_path}")
print("\n" + tab_text)
print(f"\n=== ALL OUTPUTS IN {OUT_DIR} ===")
for p in sorted(Path(OUT_DIR).iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
