#!/usr/bin/env python3
"""Evaluate T5 model on training data where ground truth (string/fret) is known.

The model should achieve near-perfect tab accuracy on training-distribution data
because it was trained on alphaTex/GuitarPro symbolic data.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from fretwire.dataset import iter_tracks
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search, greedy_decode
from fretwire.metrics import decode_predictions, align_predictions
from fretwire.postproc import postprocess
from fretwire.render import render_tab
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN, notes_to_events
from fretwire.schema import Note
from fretwire.fretboard import STANDARD

OUT = Path("/tmp/tabula_train_demo")
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

# ---- Pick a small training track ----
track_id = 6  # ~234 notes, clean training data
gt_notes = []
track_meta = None
for meta, notes_arr in iter_tracks("data/processed", split="train"):
    if int(meta.get("id", -1)) == track_id:
        track_meta = meta
        gt_notes = [
            Note(
                t=int(r["t"]), dur=int(r["dur"]), pitch=int(r["pitch"]),
                s=int(r["s"]), f=int(r["f"]),
            )
            for r in notes_arr
        ]
        break

if not gt_notes:
    raise SystemExit(f"Track {track_id} not found")

gt_notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))
print(f"Track: {track_meta.get('id')} - {track_meta.get('song')}")
print(f"Tuning: {track_meta.get('tuning_name')}")
print(f"GT notes: {len(gt_notes)}")
print(f"Time span: {max(n.t + n.dur for n in gt_notes) - min(n.t for n in gt_notes)} ticks")

# Save GT notes
with open(OUT / "gt_notes.txt", "w") as f:
    f.write("=== GROUND TRUTH NOTES (training data) ===\n")
    f.write(f"Track: {track_meta.get('id')} - {track_meta.get('song')}\n")
    f.write(f"Total notes: {len(gt_notes)}\n\n")
    f.write(f"{'start_tick':>10} {'dur_tick':>9} {'pitch':>6} {'s':>3} {'f':>3}\n")
    for n in gt_notes:
        f.write(f"{n.t:>10} {n.dur:>9} {n.pitch:>6} {n.s:>3} {n.f:>3}\n")

# ---- Build encoder tokens ----
enc_ids = vocab.encode(notes_to_events(gt_notes, base_t=0, snap_shifts=False))
print(f"Encoder tokens: {len(enc_ids)}")

with open(OUT / "encoder_tokens.txt", "w") as f:
    f.write("=== ENCODER TOKENS ===\n")
    f.write(f"Count: {len(enc_ids)}\n\n")
    for i, tid in enumerate(enc_ids):
        tok = vocab.id_to_token.get(tid, " UNK")
        f.write(f"{i:4d}: {tid:4d}  {tok}\n")

# ---- Inference chunks ----
chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(gt_notes, vocab, f"train_{track_id}", chunk_cfg)
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
    prompt_tabs = 0

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
    ):
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
    print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {len(pred_tabs)} TAB events")

# Save output tokens
with open(OUT / "output_tokens.txt", "w") as f:
    f.write("=== OUTPUT TOKENS (generated) ===\n")
    f.write(f"Count: {len(all_gen_tokens)}\n\n")
    for i, tid in enumerate(all_gen_tokens):
        tok = vocab.id_to_token.get(tid, " UNK")
        f.write(f"{i:4d}: {tid:4d}  {tok}\n")

# Save decoder prompts
with open(OUT / "decoder_prompt_tokens.txt", "w") as f:
    f.write("=== DECODER PROMPT TOKENS (all chunks) ===\n\n")
    for chunk_idx, chunk in enumerate(chunks):
        f.write(f"--- Chunk {chunk_idx} ---\n")
        for i, tid in enumerate(chunk.dec_tokens):
            tok = vocab.id_to_token.get(tid, " UNK")
            f.write(f"  {i:4d}: {tid:4d}  {tok}\n")
        f.write("\n")

# ---- Compute REAL metrics against ground truth ----
raw_counts = align_predictions(all_pred_tabs, gt_notes)

proc_notes = postprocess(all_pred_tabs, gt_notes, mode="full")
proc_tabs = [(n.s, n.f) for n in proc_notes]
post_counts = align_predictions(proc_tabs, gt_notes)

print(f"\n=== REAL METRICS (training data) ===")
print(f"Total GT notes: {len(gt_notes)}")
print(f"Raw TAB events generated: {len(all_pred_tabs)}")
print()
print(f"RAW:")
print(f"  Pitch accuracy: {raw_counts.pitch_accuracy():.4f} ({raw_counts.pitch_correct}/{raw_counts.total_notes})")
print(f"  Tab accuracy:   {raw_counts.tab_accuracy():.4f} ({raw_counts.tab_correct}/{raw_counts.total_notes})")
print(f"  Validity:       {raw_counts.validity():.4f} ({raw_counts.valid_predictions}/{raw_counts.total_notes})")
print()
print(f"POST-PROCESSED:")
print(f"  Pitch accuracy: {post_counts.pitch_accuracy():.4f} ({post_counts.pitch_correct}/{post_counts.total_notes})")
print(f"  Tab accuracy:   {post_counts.tab_accuracy():.4f} ({post_counts.tab_correct}/{post_counts.total_notes})")
print(f"  Validity:       {post_counts.validity():.4f} ({post_counts.valid_predictions}/{post_counts.total_notes})")

# Save metrics
with open(OUT / "metrics.txt", "w") as f:
    f.write("=== REAL METRICS (training data) ===\n\n")
    f.write(f"Track: {track_meta.get('id')} - {track_meta.get('song')}\n")
    f.write(f"Total GT notes: {len(gt_notes)}\n")
    f.write(f"Raw TAB events generated: {len(all_pred_tabs)}\n\n")
    f.write("RAW:\n")
    f.write(f"  Pitch accuracy: {raw_counts.pitch_accuracy():.4f} ({raw_counts.pitch_correct}/{raw_counts.total_notes})\n")
    f.write(f"  Tab accuracy:   {raw_counts.tab_accuracy():.4f} ({raw_counts.tab_correct}/{raw_counts.total_notes})\n")
    f.write(f"  Validity:       {raw_counts.validity():.4f} ({raw_counts.valid_predictions}/{raw_counts.total_notes})\n\n")
    f.write("POST-PROCESSED:\n")
    f.write(f"  Pitch accuracy: {post_counts.pitch_accuracy():.4f} ({post_counts.pitch_correct}/{post_counts.total_notes})\n")
    f.write(f"  Tab accuracy:   {post_counts.tab_accuracy():.4f} ({post_counts.tab_correct}/{post_counts.total_notes})\n")
    f.write(f"  Validity:       {post_counts.validity():.4f} ({post_counts.valid_predictions}/{post_counts.total_notes})\n")

# ---- Render tabs ----
tab_text = render_tab(proc_notes, width=100)
with open(OUT / "final_tab.txt", "w") as f:
    f.write("=== PREDICTED TABLATURE (training data) ===\n")
    f.write(f"Track: {track_meta.get('id')} - {track_meta.get('song')}\n")
    f.write(f"Raw TAB events: {len(all_pred_tabs)}\n")
    f.write(f"Post-processed notes: {len(proc_notes)}\n")
    f.write(f"Raw pitch accuracy: {raw_counts.pitch_accuracy():.4f}\n")
    f.write(f"Raw tab accuracy:   {raw_counts.tab_accuracy():.4f}\n")
    f.write(f"Post pitch accuracy: {post_counts.pitch_accuracy():.4f}\n")
    f.write(f"Post tab accuracy:   {post_counts.tab_accuracy():.4f}\n\n")
    f.write(tab_text)
    f.write("\n")

# Render GT tab for comparison
gt_tab = render_tab(gt_notes, width=100)
with open(OUT / "gt_tab.txt", "w") as f:
    f.write("=== GROUND TRUTH TABLATURE ===\n")
    f.write(f"Track: {track_meta.get('id')} - {track_meta.get('song')}\n")
    f.write(f"Total notes: {len(gt_notes)}\n\n")
    f.write(gt_tab)
    f.write("\n")

# Render raw tab
raw_notes = []
for i, tab in enumerate(all_pred_tabs):
    if i < len(gt_notes):
        t, dur, pitch = gt_notes[i].t, gt_notes[i].dur, gt_notes[i].pitch
    else:
        last = gt_notes[-1]
        t = last.t + (i - len(gt_notes) + 1) * 60
        dur, pitch = 60, 40 + tab[1]
    from fretwire.schema import Note as Note2
    raw_notes.append(Note2(t=t, dur=dur, pitch=pitch, s=tab[0], f=tab[1]))
raw_tab = render_tab(raw_notes, width=100)
with open(OUT / "raw_tab.txt", "w") as f:
    f.write("=== RAW TABLATURE (before post-processing) ===\n\n")
    f.write(raw_tab)
    f.write("\n")

# ---- Save MIDI and WAV ----
from preview import write_smf, synthesize, write_wav

# Predicted MIDI
pred_note_dicts = []
for i, gt in enumerate(gt_notes):
    if i < len(proc_notes):
        s, f = proc_notes[i].s, proc_notes[i].f
    else:
        s, f = 1, 0
    pred_note_dicts.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})
write_smf(OUT / "predicted_tab.mid", pred_note_dicts)

# Ground truth MIDI
gt_note_dicts = [{"t": n.t, "dur": n.dur, "pitch": n.pitch, "s": n.s, "f": n.f} for n in gt_notes]
write_smf(OUT / "ground_truth.mid", gt_note_dicts)

# Predicted WAV
arr = np.empty(len(pred_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"),
                                            ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(pred_note_dicts):
    arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples = synthesize(arr, 120.0)
write_wav(OUT / "predicted_tab.wav", samples)

# GT WAV
arr_gt = np.empty(len(gt_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"),
                                             ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
for i, n in enumerate(gt_note_dicts):
    arr_gt[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
samples_gt = synthesize(arr_gt, 120.0)
write_wav(OUT / "ground_truth.wav", samples_gt)

print(f"\n=== ALL ARTIFACTS SAVED ===")
for pth in sorted(OUT.iterdir()):
    print(f"  {pth.name}  ({pth.stat().st_size / 1024:.1f} KB)")
