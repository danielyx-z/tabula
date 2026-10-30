#!/usr/bin/env python3
"""Evaluate a saved checkpoint on full tracks.

Example (run on a GPU allocation):

    PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
        --checkpoint runs/pretrain_t5_v5/best --tracks 1 --postproc full

Useful for comparing post-processing changes without starting a training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from accelerate import Accelerator
from transformers import T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fretwire.chunk import MAX_TOKENS  # noqa: E402
from fretwire.metrics import print_metrics  # noqa: E402
from fretwire.tokenizer import Vocab  # noqa: E402
from fretwire.train import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed"))
    ap.add_argument("--vocab", default=str(ROOT / "vocab/guitarset.json"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--tracks", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=MAX_TOKENS)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--postproc", choices=["none", "overlap", "full"], default="full")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    if model.config.vocab_size != vocab.size:
        raise SystemExit(
            f"checkpoint vocab {model.config.vocab_size} != vocab {vocab.size}"
        )

    accelerator = Accelerator(mixed_precision="no")
    model = accelerator.prepare(model)
    post, raw = evaluate(
        model,
        args.data_dir,
        args.split,
        vocab,
        accelerator,
        max_batches=args.tracks,
        max_len=args.max_len,
        beam=args.beam,
        postproc_mode=args.postproc,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print_metrics(raw, "Raw ")
    print_metrics(post, "Post ")


if __name__ == "__main__":
    main()
