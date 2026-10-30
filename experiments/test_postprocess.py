#!/usr/bin/env python3
"""Test post-processing on trained model with training dataset."""

import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fretwire import *
from fretwire.tokenizer import Vocab, notes_to_events, notes_to_target, parse_tab, tab_token, EOS_TOKEN, PAD_TOKEN
from fretwire.decode import make_logits_processor, PitchValidLogitsProcessorV2, greedy_decode
from fretwire.dataset import iter_tracks
from fretwire.schema import Note
from fretwire.fretboard import STANDARD, pitch_name, pitch_of
from fretwire.postproc import repair_overlaps, neighbor_search, postprocess, DecodedNote
from fretwire.model import make_model
from transformers import T5ForConditionalGeneration


def test_postprocessing():
    """Test post-processing on trained model with training dataset."""
    vocab = Vocab.load('vocab_v1_full.json')
    model = T5ForConditionalGeneration.from_pretrained('runs/pretrain_gpu_v2/final')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Get a few tracks from training set
    for meta, notes_arr in iter_tracks('data/processed', split='train'):
        notes = []
        for row in notes_arr:
            notes.append(Note(
                t=int(row['t']),
                dur=int(row['dur']),
                pitch=int(row['pitch']),
                s=int(row['s']),
                f=int(row['f'])
            ))
        if len(notes) >= 50:
            break
    
    print(f"Track: {meta['song']} (track {meta['track']})")
    print(f"GT notes: {len(notes)}")
    
    # Encode
    enc = notes_to_events(notes)
    enc_ids = vocab.encode(enc)
    if len(enc_ids) > 512:
        enc_ids = enc_ids[:512]
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=device)
    
    # Ground truth
    gt = sorted(notes, key=lambda n: (n.t, n.pitch))
    gt_tabs = [(n.s, n.f) for n in gt]
    print(f"GT decoder tokens: {len(notes_to_target(notes))}")
    
    # Generate with greedy + min_length
    processor = make_logits_processor(vocab, enc_tensor)
    
    with torch.no_grad():
        gen = model.generate(
            enc_tensor,
            max_length=512,
            min_length=100,  # Force longer generation
            num_beams=1,
            logits_processor=[processor],
            eos_token_id=vocab[EOS_TOKEN],
            pad_token_id=vocab[PAD_TOKEN],
        )
    
    tokens = vocab.decode(gen[0].tolist())
    pred_tabs = [(s, f) for tok in tokens if tok.startswith('TAB<') for s, f in [parse_tab(tok)]]
    
    print(f"\nGenerated {len(gen[0])} tokens, {len(pred_tabs)} TABs")
    print(f"First 20 pred TABs: {pred_tabs[:20]}")
    print(f"First 20 GT TABs: {gt_tabs[:20]}")
    
    # Align and score (no post-processing)
    n = min(len(pred_tabs), len(gt))
    from fretwire.metrics import align_predictions, print_metrics
    counts = align_predictions(pred_tabs[:n], gt[:n])
    print_metrics(counts, 'Raw (no post) ')
    
    # Now apply post-processing
    # Convert pred_tabs to DecodedNote with timing from GT (since we don't have timing from model)
    pred_notes = []
    for i, (s, f) in enumerate(pred_tabs[:len(gt)]):
        gt_note = gt[i]
        pred_notes.append(DecodedNote(
            t=gt_note.t,
            dur=gt_note.dur,
            pitch=pitch_of(s, f, STANDARD),
            s=s,
            f=f,
        ))
    
    print("\n=== Applying Post-Processing ===")
    processed = postprocess(pred_notes, gt)
    
    # Convert back to tabs
    processed_tabs = [(n.s, n.f) for n in processed[:len(gt)]]
    
    print(f"Processed {len(processed_tabs)} TABs")
    print(f"First 20 processed: {processed_tabs[:20]}")
    print(f"First 20 GT: {gt_tabs[:20]}")
    
    # Align and score with post-processing
    n2 = min(len(processed_tabs), len(gt))
    counts2 = align_predictions(processed_tabs[:n2], gt[:n2])
    print_metrics(counts2, 'Post-processed ')


def test_beam_search():
    """Test beam search with fixed logits processor."""
    vocab = Vocab.load('vocab_v1_full.json')
    model = T5ForConditionalGeneration.from_pretrained('runs/pretrain_gpu_v2/final')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    for meta, notes_arr in iter_tracks('data/processed', split='train'):
        notes = []
        for row in notes_arr:
            notes.append(Note(
                t=int(row['t']),
                dur=int(row['dur']),
                pitch=int(row['pitch']),
                s=int(row['s']),
                f=int(row['f'])
            ))
        if len(notes) >= 50:
            break
    
    enc = notes_to_events(notes)
    enc_ids = vocab.encode(enc)
    if len(enc_ids) > 512:
        enc_ids = enc_ids[:512]
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=device)
    
    # Test beam search with constrained_beam_search
    print(f"\n=== Beam Search (num_beams=4) ===")
    with torch.no_grad():
        from fretwire.decode import constrained_beam_search
        gen = constrained_beam_search(model, enc_tensor, vocab, num_beams=4, max_length=512)
    
    tokens = vocab.decode(gen[0].tolist())
    pred_tabs = [(s, f) for tok in tokens if tok.startswith('TAB<') for s, f in [parse_tab(tok)]]
    
    gt = sorted(notes, key=lambda n: (n.t, n.pitch))
    gt_tabs = [(n.s, n.f) for n in gt]
    
    print(f"Generated {len(gen[0])} tokens, {len(pred_tabs)} TABs")
    print(f"First 20 pred: {pred_tabs[:20]}")
    print(f"First 20 GT: {gt_tabs[:20]}")


def show_sample_tab():
    """Show a sample MIDI input and TAB output."""
    vocab = Vocab.load('vocab_v1_full.json')
    model = T5ForConditionalGeneration.from_pretrained('runs/pretrain_gpu_v2/final')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    for meta, notes_arr in iter_tracks('data/processed', split='train'):
        notes = []
        for row in notes_arr:
            notes.append(Note(
                t=int(row['t']),
                dur=int(row['dur']),
                pitch=int(row['pitch']),
                s=int(row['s']),
                f=int(row['f'])
            ))
        if 20 <= len(notes) <= 50:
            break
    
    print(f"\n{'='*60}")
    print(f"Sample: {meta['song']} (track {meta['track']})")
    print(f"{'='*60}")
    
    enc = notes_to_events(notes)
    enc_ids = vocab.encode(enc)
    if len(enc_ids) > 512:
        enc_ids = enc_ids[:512]
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=device)
    
    # Show MIDI input
    print(f"\nMIDI Input ({len(notes)} notes):")
    print(f"{'Idx':>4} {'Time':>8} {'Pitch':>5} {'Dur':>6} {'String':>6} {'Fret':>4}")
    print("-" * 40)
    for i, n in enumerate(sorted(notes, key=lambda n: n.t)[:20]):
        print(f"{i:4} {n.t:8} {n.pitch:5} {n.dur:6} {n.s:6} {n.f:4}")
    
    # Generate with min_length
    processor = make_logits_processor(vocab, torch.tensor([enc_ids], dtype=torch.long, device=device))
    
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([enc_ids], dtype=torch.long, device=device),
            max_length=512,
            min_length=100,
            num_beams=1,
            logits_processor=[processor],
            eos_token_id=vocab[EOS_TOKEN],
            pad_token_id=vocab[PAD_TOKEN],
        )
    
    tokens = vocab.decode(gen[0].tolist())
    pred_tabs = [(s, f) for tok in tokens if tok.startswith('TAB<') for s, f in [parse_tab(tok)]]
    
    print(f"\nGenerated {len(gen[0])} tokens, {len(pred_tabs)} TABs")
    
    # Show TAB output
    gt = sorted(notes, key=lambda n: n.t)
    gt_tabs = [(n.s, n.f) for n in gt]
    pred_tabs = pred_tabs[:len(gt)]
    
    print(f"\n=== PREDICTED TAB (first 30 notes) ===")
    print(f"{'Idx':>4} {'Pitch':>5} {'Pred (s,f)':>10} {'GT (s,f)':>10} {'Match'}")
    print("-" * 50)
    for i in range(min(30, len(pred_tabs), len(gt_tabs))):
        ps, pf = pred_tabs[i]
        gs, gf = gt_tabs[i]
        pred_pitch = pitch_of(ps, pf, STANDARD)
        match = '✓' if (ps == gs and pf == gf) else '✗'
        print(f"{i:4} {pred_pitch:5} ({ps},{pf:2d})         ({gs},{gf:2d})         {match}")
    
    # Show ASCII tab
    print("\n=== ASCII TAB (Predicted) ===")
    print_ascii_tab(pred_tabs, notes)
    print("\n=== ASCII TAB (Ground Truth) ===")
    print_ascii_tab(gt_tabs, notes)


def print_ascii_tab(tabs, notes, max_cols=80):
    """Print ASCII tab."""
    if not tabs:
        return
    
    # Group by time
    from collections import defaultdict
    tab_by_time = defaultdict(list)
    for (s, f), n in zip(tabs, notes):
        tab_by_time[n.t].append((s, f))
    
    times = sorted(tab_by_time.keys())
    if not times:
        return
    
    strings = {s: [] for s in range(1, 7)}
    
    for t in times:
        tab_at_t = dict(tab_by_time[t])
        for s in range(1, 7):
            if s in tab_at_t:
                f = tab_at_t[s]
                strings[s].append(str(f).rjust(2))
            else:
                strings[s].append('-'.rjust(2))
        for s in range(1, 7):
            strings[s].append('  ')
    
    for s in range(6, 0, -1):
        line = f'{pitch_name(STANDARD[s-1]):>3} |' + ''.join(strings[s])
        print(line)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING POST-PROCESSING ON TRAINED MODEL")
    print("=" * 60)
    
    test_postprocessing()
    test_beam_search()
    show_sample_tab()