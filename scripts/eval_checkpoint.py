#!/usr/bin/env python3
"""Evaluate a saved checkpoint on full tracks.

Example (run on a GPU allocation):

    PYTHONPATH=src .venv/bin/python3 scripts/eval_checkpoint.py \
        --checkpoint runs/finetune_t5_guitarset/best --tracks 4 --postproc full

Use ``--no-pitch-constraint`` for the paper's unconstrained baseline, and
``--max-fret``/``--fret-penalty`` to tune the low-position prior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from accelerate import Accelerator
from transformers import T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.chunk import MAX_TOKENS  # noqa: E402
from tablature.metrics import print_metrics, print_rhythm_metrics  # noqa: E402
from tablature.tokenizer import Vocab  # noqa: E402
from model.train import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed_guitarset"))
    ap.add_argument("--vocab", default=str(ROOT / "vocab/canonical.json"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--tracks", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=MAX_TOKENS)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--postproc", choices=["none", "overlap", "full"], default="full")
    ap.add_argument("--pitch-constraint", dest="pitch_constraint",
                    action="store_true", default=True,
                    help="Mask TAB tokens to positions that match the input pitch")
    ap.add_argument("--no-pitch-constraint", dest="pitch_constraint",
                    action="store_false",
                    help="Paper baseline: no pitch or low-position mask")
    ap.add_argument("--max-fret", type=int, default=12,
                    help="Prefer positions at or below this fret when possible; "
                         "-1 disables the low-position rule")
    ap.add_argument("--fret-penalty", type=float, default=0.0,
                    help="Soft penalty (-lambda*fret) among allowed positions")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-mode", choices=["pred", "gt"], default="pred",
                    help="Decoder context at chunk boundaries: pred=model's own, gt=ground truth")
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    if model.config.vocab_size != vocab.size:
        raise SystemExit(
            f"checkpoint vocab {model.config.vocab_size} != vocab {vocab.size}"
        )

    accelerator = Accelerator(mixed_precision="no")
    model = accelerator.prepare(model)
    post, raw, post_rhythm, raw_rhythm = evaluate(
        model,
        args.data_dir,
        args.split,
        vocab,
        accelerator,
        max_batches=args.tracks,
        max_len=args.max_len,
        beam=args.beam,
        postproc_mode=args.postproc,
        pitch_constraint=args.pitch_constraint,
        max_fret=None if args.max_fret < 0 else args.max_fret,
        fret_penalty=args.fret_penalty,
        shuffle=args.shuffle,
        seed=args.seed,
        context_mode=args.context_mode,
    )
    print_metrics(raw, "Raw ")
    print_metrics(post, "Post ")
    print_rhythm_metrics(raw_rhythm, "Raw Rhythm ")


if __name__ == "__main__":
    main()