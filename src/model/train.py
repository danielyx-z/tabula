"""Training and evaluation for the Fretting-Transformer T5 baseline.

Paper-faithful choices:

* reduced T5 (see :mod:`model.model`)
* Adafactor with a self-adaptive learning rate (default; pass ``--lr`` for a
  fixed-rate cosine schedule)
* standard cross-entropy supplied by ``T5ForConditionalGeneration``
* 512-token training chunks, 20-note inference chunks with one note of
  encoder/decoder context
* post-processing (overlap correction + neighbor search) before metrics

There is no alignment loss, note-count head or pitch-constrained training loss:
those are not part of the Fretting-Transformer paper.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers import T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

from model.chunk import (
    MAX_TOKENS,
    ChunkConfig,
    make_inference_chunks,
    make_training_chunks,
)
from corpus.dataset import iter_tracks
from tablature.metrics import (
    EvalCounts,
    RhythmCounts,
    aggregate_counts,
    aggregate_rhythm,
    align_predictions,
    check_playability,
    compute_rhythm_metrics,
    decode_predictions,
    difficulty_terms,
    print_metrics,
    print_rhythm_metrics,
)
from model.model import count_parameters, make_model
from tablature.postproc import postprocess
from core.schema import Note
from tablature.tokenizer import DEFAULT_VOCAB, EOS_TOKEN, PAD_TOKEN, Vocab


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/processed_guitarset"
    vocab_path: str = "vocab/guitarset.json"
    max_tracks: int = 0  # 0 = all tracks
    # Optimization
    lr: Optional[float] = None  # None = Adafactor self-adaptive
    weight_decay: float = 0.0
    warmup_ratio: float = 0.02
    max_epochs: int = 0  # 0 = unlimited; training stops at --max-steps
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    # Early stopping on validation loss (paper-style); full tab metrics are
    # expensive because they decode whole tracks.
    patience: int = 10
    min_delta: float = 1e-4
    # Logging / evaluation. eval_every, metrics_every and save_every are in
    # optimizer steps (micro-batches / grad_accum), not micro-batches.
    log_every: int = 100
    eval_every: int = 500
    eval_loss_batches: int = 0  # 0 = all validation chunks
    eval_batches: int = 10
    metrics_every: int = 2500
    eval_beam: int = 1
    postproc_mode: str = "full"
    save_every: int = 2500
    output_dir: str = "runs/pretrain_t5_guitarset"
    seed: int = 42
    # DataLoader / sequence length
    batch_size: int = 8
    max_steps: int = 30000  # optimizer steps; 0 = unlimited (stop via max_epochs)
    max_len: int = MAX_TOKENS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def records_to_notes(notes_arr) -> list[Note]:
    return [
        Note(
            t=int(row["t"]),
            dur=int(row["dur"]),
            pitch=int(row["pitch"]),
            s=int(row["s"]),
            f=int(row["f"]),
        )
        for row in notes_arr
    ]


class ChunkIterableDataset(IterableDataset):
    """Yield token-capped training chunks from the sharded dataset."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        vocab: Vocab,
        max_tracks: int = 0,
        max_len: int = MAX_TOKENS,
        shuffle: bool = True,
        seed: int = 42,
        max_chunks_per_track: int = 0,
        shuffle_buffer: int = 512,
    ):
        self.data_dir = data_dir
        self.split = split
        self.vocab = vocab
        self.max_tracks = max_tracks
        self.max_chunks_per_track = max_chunks_per_track
        self.cfg = ChunkConfig(max_tokens=max_len)
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = max(0, shuffle_buffer)
        self._epoch = 0

    @staticmethod
    def _item(chunk) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(chunk.enc_tokens, dtype=torch.long),
            "decoder_input_ids": torch.tensor(chunk.dec_tokens, dtype=torch.long),
            "labels": torch.tensor(chunk.labels, dtype=torch.long),
        }

    def __iter__(self):
        seed = self.seed + self._epoch
        self._epoch += 1
        rng = random.Random(seed)
        buffer: list[dict[str, torch.Tensor]] = []
        buffered = self.shuffle and self.shuffle_buffer > 0

        for track_count, (meta, notes_arr) in enumerate(
            iter_tracks(
                self.data_dir,
                split=self.split,
                shuffle=self.shuffle,
                seed=seed,
            )
        ):
            if self.max_tracks and track_count >= self.max_tracks:
                break
            notes = records_to_notes(notes_arr)
            chunks = make_training_chunks(
                notes,
                self.vocab,
                meta.get("id", meta.get("song", track_count)),
                self.cfg,
            )
            if self.max_chunks_per_track:
                chunks = chunks[: self.max_chunks_per_track]
            for chunk in chunks:
                if not chunk.enc_tokens or not chunk.labels:
                    continue
                if not buffered:
                    yield self._item(chunk)
                    continue
                buffer.append(self._item(chunk))
                if len(buffer) >= self.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()

        if buffered:
            rng.shuffle(buffer)
        yield from buffer


def _infinite_chunk_iterator(
    data_dir: str,
    split: str,
    vocab: Vocab,
    max_tracks: int,
    max_len: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
    batch_size: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever, re-iterating the dataset each pass."""
    while True:
        ds = ChunkIterableDataset(
            data_dir,
            split,
            vocab,
            max_tracks=max_tracks,
            max_len=max_len,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=lambda b: collate_fn(b, pad_id=vocab[PAD_TOKEN], max_len=max_len),
            num_workers=0,
        )
        for batch in dl:
            yield batch


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
    max_len: int = MAX_TOKENS,
) -> dict[str, torch.Tensor]:
    """Pad encoder/decoder sequences and build T5 attention masks."""
    if not batch:
        raise ValueError("empty batch")
    max_enc = min(max_len, max(len(x["input_ids"]) for x in batch))
    max_dec = min(max_len, max(len(x["decoder_input_ids"]) for x in batch))

    enc = torch.full((len(batch), max_enc), pad_id, dtype=torch.long)
    dec = torch.full((len(batch), max_dec), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_dec), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        enc_len = min(len(item["input_ids"]), max_enc)
        dec_len = min(len(item["decoder_input_ids"]), max_dec)
        label_len = min(len(item["labels"]), max_dec)
        enc[i, :enc_len] = item["input_ids"][:enc_len]
        dec[i, :dec_len] = item["decoder_input_ids"][:dec_len]
        labels[i, :label_len] = item["labels"][:label_len]

    return {
        "input_ids": enc,
        "attention_mask": (enc != pad_id).long(),
        "decoder_input_ids": dec,
        "decoder_attention_mask": (dec != pad_id).long(),
        "labels": labels,
    }


@torch.no_grad()
def evaluate(
    model,
    data_dir: str,
    split: str,
    vocab: Vocab,
    accelerator: Accelerator,
    max_batches: int = 10,
    max_len: int = MAX_TOKENS,
    beam: int = 1,
    postproc_mode: str = "full",
    max_chunks: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    context_mode: str = "pred",  # "pred" | "gt"
) -> tuple[EvalCounts, EvalCounts, RhythmCounts, RhythmCounts]:
    """Evaluate full tracks and return ``(postprocessed, raw, post_rhythm, raw_rhythm)`` counts.

    ``max_batches`` is a track budget, so with ``shuffle=True`` this evaluates a
    deterministic random sample of tracks rather than the first shard.

    ``context_mode`` controls the decoder prompt at each chunk boundary:
    - "gt": use the ground-truth previous TAB (oracle upper bound)
    - "pred": use the model's own previous prediction (deployment realism)
    """
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    had_checkpointing = bool(getattr(unwrapped, "is_gradient_checkpointing", False))

    if had_checkpointing:
        unwrapped.gradient_checkpointing_disable()
    unwrapped.config.use_cache = True
    unwrapped.eval()

    device = accelerator.device
    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]
    chunk_cfg = ChunkConfig(max_tokens=max_len)

    processed_counts: list[EvalCounts] = []
    raw_counts: list[EvalCounts] = []
    processed_rhythm: list[RhythmCounts] = []
    raw_rhythm: list[RhythmCounts] = []
    difficulty_sum = 0.0
    difficulty_n = 0
    raw_difficulty_sum = 0.0
    raw_difficulty_n = 0

    try:
        for track_idx, (meta, notes_arr) in enumerate(
            iter_tracks(
                data_dir,
                split=split,
                shuffle=shuffle,
                seed=seed,
            )
        ):
            if track_idx >= max_batches:
                break
            notes = records_to_notes(notes_arr)
            if not notes:
                continue

            # Sort GT to match the chunker's sort order: (t, s, pitch, f)
            gt = sorted(notes, key=lambda n: (n.t, n.s, n.pitch, n.f))
            chunks = make_inference_chunks(
                gt, vocab, meta.get("id", track_idx), chunk_cfg
            )
            if max_chunks is not None:
                chunks = chunks[:max_chunks]
            if not chunks:
                continue

            chunk_preds: list[list[tuple[int, int]]] = []
            chunk_pred_tokens: list[list[str]] = []
            # For "pred" context mode, carry the model's own previous TAB forward.
            prev_pred_tab: tuple[int, int] | None = None

            for chunk_idx, chunk in enumerate(chunks):
                enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
                attention_mask = (enc != pad_id).long()

                if context_mode == "gt" or chunk_idx == 0:
                    # First chunk has no previous prediction, or gt mode uses GT.
                    context_ids = (
                        [vocab[tab_token(chunk.meta.get("context_s", 0), chunk.meta.get("context_f", 0))]]
                        if chunk.meta.get("has_context") and context_mode == "gt"
                        else []
                    )
                else:
                    # Use the model's own previous TAB as context.
                    assert prev_pred_tab is not None
                    context_ids = [vocab[tab_token(prev_pred_tab[0], prev_pred_tab[1])]]

                dec_tokens = [eos_id] + context_ids
                prompt = torch.tensor([dec_tokens], dtype=torch.long, device=device)
                max_new = min(max_len, max(16, 6 * chunk.meta["num_notes"] + 16))

                with accelerator.autocast():
                    generated = unwrapped.generate(
                        input_ids=enc,
                        attention_mask=attention_mask,
                        decoder_input_ids=prompt,
                        max_new_tokens=max_new,
                        num_beams=beam,
                        do_sample=False,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )
                generated = generated[:, prompt.size(1):]
                gen_tokens = vocab.decode(generated[0].tolist())
                pred_tabs = decode_predictions(generated[0].tolist(), vocab)
                chunk_preds.append(pred_tabs)
                chunk_pred_tokens.append(gen_tokens)

                if context_mode == "pred" and pred_tabs:
                    prev_pred_tab = pred_tabs[-1]

            flat_preds = [tab for preds in chunk_preds for tab in preds]
            flat_pred_tokens = [tok for tokens in chunk_pred_tokens for tok in tokens]
            gt_eval = gt[: int(chunks[-1].meta["note_end"])]

            # Rhythm metrics on raw predictions (not post-processed).
            raw_rhythm.append(compute_rhythm_metrics(flat_preds, flat_pred_tokens, gt_eval, vocab))

            # Post-process each chunk against its own ground truth.
            processed_chunks = []
            raw_terms = []
            for chunk, preds in zip(chunks, chunk_preds):
                start = int(chunk.meta["note_start"])
                end = int(chunk.meta["note_end"])
                gt_chunk = gt_eval[start:end]
                if not gt_chunk:
                    processed_chunks.append([])
                    continue
                processed_chunks.append(
                    postprocess(preds, gt_chunk, mode=postproc_mode)
                )
                raw_terms.extend(
                    difficulty_terms(
                        postprocess(preds, gt_chunk, mode="none")
                    )
                )

            processed = [note for chunk in processed_chunks for note in chunk]
            processed_tabs = [(n.s, n.f) for n in processed]
            if postproc_mode != "none" and len(processed_tabs) != len(gt_eval):
                raise RuntimeError(
                    f"post-processing returned {len(processed_tabs)} notes for "
                    f"{len(gt_eval)} ground-truth notes"
                )

            terms = difficulty_terms(processed)
            difficulty_sum += float(sum(terms))
            difficulty_n += len(terms)
            raw_difficulty_sum += float(sum(raw_terms))
            raw_difficulty_n += len(raw_terms)

            # Score each chunk against its own ground-truth notes.
            for chunk_idx, (chunk, preds) in enumerate(
                zip(chunks, chunk_preds)
            ):
                start = int(chunk.meta["note_start"])
                end = int(chunk.meta["note_end"])
                gt_chunk = gt_eval[start:end]
                if not gt_chunk:
                    continue
                if len(gt_chunk) == 1:
                    boundary = [True]
                else:
                    boundary = [True] + [False] * (len(gt_chunk) - 2) + [True]

                raw = align_predictions(preds, gt_chunk, boundary)
                for name, value in zip(
                    ("same_string_overlap", "chord_span_gt_5", "simult_gt_4"),
                    check_playability(preds, gt_chunk),
                ):
                    setattr(raw, name, getattr(raw, name) + value)
                raw_counts.append(raw)

                proc_chunk = (
                    preds
                    if postproc_mode == "none"
                    else [
                        (note.s, note.f)
                        for note in processed_chunks[chunk_idx]
                    ]
                )
                proc = align_predictions(proc_chunk, gt_chunk, boundary)
                for name, value in zip(
                    ("same_string_overlap", "chord_span_gt_5", "simult_gt_4"),
                    check_playability(proc_chunk, gt_chunk),
                ):
                    setattr(proc, name, getattr(proc, name) + value)
                processed_counts.append(proc)

            # Rhythm metrics on post-processed (uses GT timing via postprocess).
            # Note: this will be trivially perfect since postprocess assigns GT onsets.
            # We compute it for completeness but raw is the meaningful one.
            processed_rhythm.append(RhythmCounts(matched=len(gt_eval), missing=0, extra=0))
    finally:
        unwrapped.train(was_training)
        if had_checkpointing:
            unwrapped.gradient_checkpointing_enable()
        unwrapped.config.use_cache = False

    processed_total = aggregate_counts(processed_counts)
    processed_total.difficulty_sum = difficulty_sum
    processed_total.difficulty_n = difficulty_n
    raw_total = aggregate_counts(raw_counts)
    raw_total.difficulty_sum = raw_difficulty_sum
    raw_total.difficulty_n = raw_difficulty_n
    return processed_total, raw_total, aggregate_rhythm(processed_rhythm), aggregate_rhythm(raw_rhythm)


@torch.no_grad()
def evaluate_loss(
    model,
    data_dir: str,
    split: str,
    vocab: Vocab,
    accelerator: Accelerator,
    max_batches: int = 0,  # 0 = all validation chunks
    max_len: int = MAX_TOKENS,
    batch_size: int = 8,
) -> float:
    """Fast teacher-forced validation loss on token-capped chunks.

    This mirrors the paper's validation objective and is much cheaper than
    decoding whole tracks. It is used for early stopping; full tab metrics are
    reported separately on a smaller cadence.
    """
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    had_checkpointing = bool(getattr(unwrapped, "is_gradient_checkpointing", False))

    if had_checkpointing:
        unwrapped.gradient_checkpointing_disable()
    unwrapped.config.use_cache = False
    unwrapped.eval()

    pad_id = vocab[PAD_TOKEN]
    ds = ChunkIterableDataset(
        data_dir,
        split,
        vocab,
        max_tracks=0,
        max_len=max_len,
        shuffle=True,
        seed=0,
        max_chunks_per_track=0,  # use all chunks
        shuffle_buffer=max(1, batch_size * 2),
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id, max_len=max_len),
        num_workers=0,
    )

    total_loss = 0.0
    total_tokens = 0
    try:
        for batch_idx, batch in enumerate(dl):
            if max_batches and batch_idx >= max_batches:
                break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = unwrapped(**batch)
            n_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(outputs.loss.item()) * n_tokens
            total_tokens += n_tokens
    finally:
        unwrapped.train(was_training)
        if had_checkpointing:
            unwrapped.gradient_checkpointing_enable()
        unwrapped.config.use_cache = False

    return total_loss / total_tokens if total_tokens else float("inf")


def _save_checkpoint(
    model,
    vocab: Vocab,
    accelerator: Accelerator,
    directory: Path,
    optimizer=None,
    scheduler=None,
    state: dict | None = None,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    unwrapped.save_pretrained(directory)
    vocab.save(directory / "vocab.json")

    if optimizer is not None:
        payload = {"optimizer": optimizer.state_dict()}
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if state:
            payload.update(state)
        torch.save(payload, directory / "trainer_state.pt")


def _save_final(
    model,
    vocab: Vocab,
    accelerator: Accelerator,
    output_dir: Path,
    optimizer=None,
    scheduler=None,
    state: dict | None = None,
) -> None:
    _save_checkpoint(
        model,
        vocab,
        accelerator,
        output_dir / "final",
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
    )


def train(cfg: TrainConfig, resume_from: str = "") -> None:
    if cfg.batch_size <= 0 or cfg.grad_accum <= 0:
        raise ValueError("--batch-size and --grad-accum must be positive")
    if cfg.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if cfg.log_every <= 0 or cfg.eval_every <= 0 or cfg.save_every <= 0:
        raise ValueError(
            "--log-every, --eval-every and --save-every must be positive "
            "(--metrics-every may be 0 to disable full decoding)"
        )
    if cfg.metrics_every < 0:
        raise ValueError("--metrics-every must be >= 0")
    if cfg.eval_beam < 1:
        raise ValueError("--eval-beam must be >= 1")
    if cfg.max_len <= 0:
        raise ValueError("--max-len must be positive")

    use_bf16 = (
        cfg.bf16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    if cfg.bf16 and torch.cuda.is_available() and not use_bf16:
        print(
            "warning: bf16 was requested but this GPU does not support it; "
            "falling back to fp32",
            flush=True,
        )
    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else "no",
        gradient_accumulation_steps=cfg.grad_accum,
    )
    set_seed(cfg.seed)

    vocab = (
        Vocab.load(cfg.vocab_path)
        if Path(cfg.vocab_path).exists()
        else DEFAULT_VOCAB
    )
    if vocab is DEFAULT_VOCAB:
        accelerator.print(
            f"Vocab not found at {cfg.vocab_path}; using unpruned default vocab"
        )

    if resume_from:
        model = T5ForConditionalGeneration.from_pretrained(resume_from)
        if model.config.vocab_size != vocab.size:
            ckpt_vocab = Path(resume_from) / "vocab.json"
            if ckpt_vocab.exists():
                vocab = Vocab.load(ckpt_vocab)
                accelerator.print(f"Loaded vocab from checkpoint: {ckpt_vocab}")
            else:
                raise ValueError(
                    f"Checkpoint vocab size ({model.config.vocab_size}) does not "
                    f"match vocab file ({vocab.size}) and no vocab.json exists in "
                    f"{resume_from}"
                )
    else:
        model = make_model(vocab.size, vocab[PAD_TOKEN], vocab[EOS_TOKEN])

    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    if cfg.max_len > model.config.max_position_embeddings:
        raise ValueError(
            f"--max-len ({cfg.max_len}) exceeds model max_position_embeddings "
            f"({model.config.max_position_embeddings})"
        )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    total, trainable = count_parameters(model)
    accelerator.print(
        f"Model: T5, {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)"
    )
    accelerator.print(f"Vocab size: {vocab.size}")
    accelerator.print(f"Device: {accelerator.device} | bf16: {use_bf16}")

    if cfg.lr is None:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
            weight_decay=cfg.weight_decay,
        )
        scheduler = None
        accelerator.print(
            "Optimizer: Adafactor "
            f"(self-adaptive learning rate, weight_decay={cfg.weight_decay:g})"
        )
    else:
        optimizer = Adafactor(
            model.parameters(),
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
        num_training_steps = cfg.max_steps if cfg.max_steps else 30000
        num_warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        accelerator.print(
            f"Optimizer: Adafactor (fixed lr={cfg.lr:g}, cosine, "
            f"weight_decay={cfg.weight_decay:g})"
        )

    if scheduler is None:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2)
    )

    best_val_loss = float("inf")
    patience_counter = 0
    update_step = 0
    last_metrics_step = 0
    start_epoch = 0

    if resume_from:
        state_path = Path(resume_from) / "trainer_state.pt"
        if state_path.exists():
            try:
                state = torch.load(
                    state_path, map_location="cpu", weights_only=False
                )
                optimizer.load_state_dict(state["optimizer"])
                if scheduler is not None and state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                update_step = int(state.get("update_step", 0))
                best_val_loss = float(state.get("best_val_loss", best_val_loss))
                patience_counter = int(state.get("patience_counter", 0))
                start_epoch = int(state.get("epoch", 0))
                last_metrics_step = update_step
                accelerator.print(
                    f"Resumed trainer state from {state_path} "
                    f"(step {update_step}, epoch {start_epoch})"
                )
            except Exception as exc:
                accelerator.print(
                    f"warning: could not restore trainer state from {state_path}: "
                    f"{exc}; continuing with weights/vocab only"
                )
        else:
            accelerator.print(
                f"No trainer_state.pt in {resume_from}; "
                "resuming weights/vocab only"
            )

    # Build the infinite training iterator.
    batch_iter = _infinite_chunk_iterator(
        cfg.data_dir,
        "train",
        vocab,
        max_tracks=cfg.max_tracks,
        max_len=cfg.max_len,
        shuffle=True,
        seed=cfg.seed + start_epoch,
        shuffle_buffer=64,
        batch_size=cfg.batch_size,
    )

    accelerator.print("Starting training...")
    epoch = start_epoch
    while True:
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for _ in range(cfg.batch_size * cfg.grad_accum):  # one optimizer step
            batch = next(batch_iter)
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_batches += 1

            if not accelerator.sync_gradients:
                continue

            update_step += 1

            if update_step % cfg.log_every == 0:
                if scheduler is not None:
                    lr_str = f"{scheduler.get_last_lr()[0]:.2e}"
                else:
                    lr = optimizer.param_groups[0].get("lr")
                    lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) and lr else "self-adaptive"
                accelerator.print(
                    f"Step {update_step} | Loss: {loss.item():.4f} | LR: {lr_str}"
                )

            if update_step % cfg.eval_every == 0:
                val_loss = evaluate_loss(
                    model,
                    cfg.data_dir,
                    "valid",
                    vocab,
                    accelerator,
                    max_batches=cfg.eval_loss_batches,
                    max_len=cfg.max_len,
                    batch_size=cfg.batch_size,
                )
                accelerator.print(
                    f"\n--- Validation loss @ step {update_step}: {val_loss:.4f} ---"
                )

                state = {
                    "update_step": update_step,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "epoch": epoch,
                }
                if val_loss < best_val_loss - cfg.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    state["best_val_loss"] = best_val_loss
                    state["patience_counter"] = 0
                    _save_checkpoint(
                        model,
                        vocab,
                        accelerator,
                        output_dir / "best",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                    )
                    accelerator.print(f"New best validation loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    state["patience_counter"] = patience_counter
                    if patience_counter >= cfg.patience:
                        accelerator.print(f"Early stopping at step {update_step}")
                        _save_final(
                            model,
                            vocab,
                            accelerator,
                            output_dir,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            state=state,
                        )
                        return

                # Whole-track decoding is expensive; only do it occasionally.
                if (
                    cfg.metrics_every
                    and update_step - last_metrics_step >= cfg.metrics_every
                ):
                    last_metrics_step = update_step
                    val_counts, raw_counts, val_rhythm, raw_rhythm = evaluate(
                        model,
                        cfg.data_dir,
                        "valid",
                        vocab,
                        accelerator,
                        max_batches=cfg.eval_batches,
                        max_len=cfg.max_len,
                        beam=cfg.eval_beam,
                        postproc_mode=cfg.postproc_mode,
                        shuffle=True,
                        seed=cfg.seed + 1,
                        context_mode="pred",
                    )
                    accelerator.print(f"--- Tab metrics @ step {update_step} ---")
                    print_metrics(raw_counts, "Raw ")
                    print_metrics(val_counts, "Post ")
                    accelerator.print(f"--- Rhythm metrics (raw) @ step {update_step} ---")
                    print_rhythm_metrics(raw_rhythm, "Raw ")

            if update_step % cfg.save_every == 0:
                _save_checkpoint(
                    model,
                    vocab,
                    accelerator,
                    output_dir / f"checkpoint-{update_step}",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state={
                        "update_step": update_step,
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                        "epoch": epoch,
                    },
                )

            # Stop conditions
            if cfg.max_steps and update_step >= cfg.max_steps:
                accelerator.print(f"Reached max-steps={cfg.max_steps}, stopping.")
                _save_final(
                    model,
                    vocab,
                    accelerator,
                    output_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state={
                        "update_step": update_step,
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                        "epoch": epoch,
                    },
                )
                return

        avg_loss = epoch_loss / max(1, epoch_batches)
        accelerator.print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        epoch += 1

        if cfg.max_epochs and epoch >= cfg.max_epochs:
            _save_final(
                model,
                vocab,
                accelerator,
                output_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                state={
                    "update_step": update_step,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "epoch": epoch,
                },
            )
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed_guitarset")
    ap.add_argument("--vocab-path", default="vocab/guitarset.json")
    ap.add_argument("--output-dir", default="runs/pretrain_t5_guitarset")
    ap.add_argument("--max-epochs", type=int, default=0,
                    help="Max epochs; 0 = unlimited (stop via --max-steps)")
    ap.add_argument("--max-steps", type=int, default=30000,
                    help="Max optimizer steps; 0 = unlimited (stop via --max-epochs)")
    ap.add_argument("--lr", type=float, default=None,
                    help="Fixed learning rate; omit for Adafactor self-adaptive")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--grad-checkpointing", dest="gradient_checkpointing",
                    action="store_true", help="Trade speed for memory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="Validation-loss cadence in optimizer steps")
    ap.add_argument("--eval-loss-batches", type=int, default=0,
                    help="Validation chunks; 0 = all")
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--metrics-every", type=int, default=2500,
                    help="Whole-track tab metrics cadence in optimizer steps; 0 disables")
    ap.add_argument("--eval-beam", type=int, default=1)
    ap.add_argument("--postproc", choices=["none", "overlap", "full"], default="full")
    ap.add_argument("--save-every", type=int, default=2500,
                    help="Checkpoint cadence in optimizer steps")
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adafactor weight decay (default 0, as in T5)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=MAX_TOKENS)
    ap.add_argument("--resume-from", type=str, default="",
                    help="Checkpoint directory: restores weights, vocabulary, "
                         "optimizer/scheduler and early-stopping state")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        grad_accum=args.grad_accum,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_loss_batches=args.eval_loss_batches,
        eval_batches=args.eval_batches,
        metrics_every=args.metrics_every,
        eval_beam=args.eval_beam,
        postproc_mode=args.postproc,
        save_every=args.save_every,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        max_tracks=args.max_tracks,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()