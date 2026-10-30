#!/usr/bin/env python3
"""Minimal training script without accelerate for memory debugging."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fretwire import *
from fretwire.chunk import make_chunks_from_notes
from fretwire.train import ChunkIterableDataset, collate_fn
from fretwire.dataset import iter_tracks
from fretwire.model import make_model
from fretwire.tokenizer import Vocab
from fretwire.schema import Note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--vocab-path", default="vocab_v1_full.json")
    ap.add_argument("--output-dir", default="runs/pretrain_manual")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tracks", type=int, default=5000)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    vocab = Vocab.load(args.vocab_path)
    print(f"Vocab size: {vocab.size}", flush=True)

    model = make_model(vocab.size, PAD_ID, EOS_ID)
    model.gradient_checkpointing_enable()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    train_ds = ChunkIterableDataset(args.data_dir, "train", vocab, max_tracks=args.max_tracks)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=lambda b: collate_fn(b, max_len=512))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dl):
            if batch_idx >= args.steps_per_epoch:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=batch["decoder_attention_mask"].to(device),
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if step % 50 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {step} | Loss: {loss.item() * args.grad_accum:.4f} | LR: {lr:.2e}", flush=True)

        print(f"Epoch {epoch} avg loss: {epoch_loss / args.steps_per_epoch:.4f}", flush=True)

        # Save checkpoint
        if epoch % 2 == 0:
            model.save_pretrained(output_dir / f"epoch-{epoch}")
            vocab.save(output_dir / f"epoch-{epoch}" / "vocab.json")

    model.save_pretrained(output_dir / "final")
    vocab.save(output_dir / "final" / "vocab.json")
    print("Done!", flush=True)


if __name__ == "__main__":
    main()