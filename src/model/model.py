"""Model: paper-faithful T5 encoder-decoder.

The Fretting-Transformer paper (Hamberger et al., 2025) uses a reduced T5
trained from scratch:

    d_model = 128, d_ff = 1024, 3 encoder/3 decoder layers, 4 heads

This module is deliberately thin: it only builds ``T5Config`` /
``T5ForConditionalGeneration`` with those values.  Everything else (loss,
beam search, save/load) comes from Hugging Face, which is what the paper uses.
"""

from __future__ import annotations

from transformers import T5Config, T5ForConditionalGeneration

# Explicit paper values.  ``d_kv`` is kept at the T5 default (64); the paper's
# published model checkpoint used the same value.
PAPER_CONFIG = {
    "d_model": 128,
    "d_ff": 1024,
    "d_kv": 64,
    "num_layers": 3,
    "num_decoder_layers": 3,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "feed_forward_proj": "relu",
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}


def make_config(
    vocab_size: int,
    pad_id: int = 0,
    eos_id: int = 1,
    **overrides,
) -> T5Config:
    """Return the paper's T5 configuration for a frozen vocabulary."""
    values = dict(PAPER_CONFIG)
    values.update(overrides)
    return T5Config(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        decoder_start_token_id=eos_id,
        **values,
    )


def make_model(
    vocab_size: int,
    pad_id: int = 0,
    eos_id: int = 1,
    **overrides,
) -> T5ForConditionalGeneration:
    """Build a from-scratch T5 model with the paper's reduced configuration."""
    return T5ForConditionalGeneration(make_config(vocab_size, pad_id, eos_id, **overrides))


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model) -> float:
    total, _ = count_parameters(model)
    return total * 4 / (1024 ** 2)
