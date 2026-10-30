#!/usr/bin/env python3
"""Compute real pitch/tab accuracy from captured artifacts."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/daniel/Projects/tabula/src")

import numpy as np
from fretwire.schema import Note
from fretwire.metrics import align_predictions, decode_predictions
from fretwire.fretboard import pitch_of, STANDARD

OUT = Path("/tmp/tabula_guitar_demo")

# Load input notes (ground truth: pitch known, s/f=0 placeholder)
gt_notes = []
with open(OUT / "input_notes.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].isdigit():
            t, dur, pitch, s, f = map(int, parts[:5])
            gt_notes.append(Note(t=t, dur=dur, pitch=pitch, s=s, f=f))

# Load raw predictions from output_tokens.txt — decode TAB tokens
from fretwire.tokenizer import Vocab
vocab = Vocab.load("vocab_v1_full.json")

gen_ids = []
with open(OUT / "output_tokens.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            tid = int(parts[1])
            gen_ids.append(tid)

pred_tabs = decode_predictions(gen_ids, vocab)

# Post-process
from fretwire.postproc import postprocess
proc_notes = postprocess(pred_tabs, gt_notes, mode="full")
proc_tabs = [(n.s, n.f) for n in proc_notes]

print(f"GT notes: {len(gt_notes)}")
print(f"Raw pred tabs: {len(pred_tabs)}")
print(f"Post-processed notes: {len(proc_notes)}")
print()

# Raw accuracy
raw_counts = align_predictions(pred_tabs, gt_notes)
print("=== RAW (before post-processing) ===")
print(f"Pitch accuracy: {raw_counts.pitch_accuracy():.4f} ({raw_counts.pitch_correct}/{raw_counts.total_notes})")
print(f"Tab accuracy:   {raw_counts.tab_accuracy():.4f} ({raw_counts.tab_correct}/{raw_counts.total_notes})")
print(f"Validity:       {raw_counts.validity():.4f} ({raw_counts.valid_predictions}/{raw_counts.total_notes})")
print()

# Post-processed accuracy
post_counts = align_predictions(proc_tabs, gt_notes)
print("=== POST-PROCESSED ===")
print(f"Pitch accuracy: {post_counts.pitch_accuracy():.4f} ({post_counts.pitch_correct}/{post_counts.total_notes})")
print(f"Tab accuracy:   {post_counts.tab_accuracy():.4f} ({post_counts.tab_correct}/{post_counts.total_notes})")
print(f"Validity:       {post_counts.validity():.4f} ({post_counts.valid_predictions}/{post_counts.total_notes})")
print()

# Per-note breakdown
print("=== Per-note comparison ===")
print(f"{'#':>3} {'t':>7} {'pitch':>6} {'GT(s,f)':>9} {'Raw(s,f)':>10} {'Post(s,f)':>11} {'Raw OK?':>7} {'Post OK?':>8}")
mismatches = 0
for i, gt in enumerate(gt_notes):
    raw_sf = pred_tabs[i] if i < len(pred_tabs) else None
    post_sf = proc_tabs[i] if i < len(proc_notes) else None
    raw_ok = raw_sf == (gt.s, gt.f) if raw_sf else False
    post_ok = post_sf == (gt.s, gt.f) if post_sf else False
    if not raw_ok or not post_ok:
        mismatches += 1
    raw_pitch_ok = raw_sf and pitch_of(raw_sf[0], raw_sf[1], STANDARD) == gt.pitch if raw_sf else False
    post_pitch_ok = post_sf and pitch_of(post_sf[0], post_sf[1], STANDARD) == gt.pitch if post_sf else False
    if not raw_ok or not post_ok or not raw_pitch_ok or not post_pitch_ok:
        print(f"{i:>3} {gt.t:>7} {gt.pitch:>6} {(gt.s,gt.f)!s:>9} {str(raw_sf):>10} {str(post_sf):>11} {str(raw_ok):>7} {str(post_ok):>8}  [raw_pitch={raw_pitch_ok} post_pitch={post_pitch_ok}]")

print(f"\nMismatches (GT vs pred): {mismatches}/{len(gt_notes)}")
