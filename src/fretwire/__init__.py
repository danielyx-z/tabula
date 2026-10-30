"""Fretwire: MIDI -> guitar tab transcription (paper-faithful MVP).

Layers, bottom up and deliberately isolated from each other:

    fretboard   pitch <-> (string, fret); no rhythm, no notation
    parse       alphaTex -> canonical Note events
    filter      which tracks are admitted into the corpus
    dataset     sharded memmap storage
    quantise    960 PPQ / 20-tick grid, compositional TIME_SHIFT (single source of truth)
    tokenizer   v3 event encoding, vocab build/freeze
    chunk       512-token training windows, 20-note inference chunks
    model       reduced T5 encoder-decoder (~2.8M params)
    train       Adafactor + standard cross-entropy, early stop on val loss
    decode      T5 greedy/beam decoding; optional pitch mask (off by default)
    postproc    overlap correction + +/-5-note neighbor search
    metrics     pitch/tab accuracy, difficulty, error decomposition
    render      ASCII tab output
    cli         midi2tab entry point
"""

from .fretboard import STANDARD, pitch_name, pitch_of, positions
from .quantise import PPQ, GRID, quantise_tick, quantise_dur, decompose_shift, SHIFT_VOCAB
from .tokenizer import (Vocab, DEFAULT_VOCAB, build_vocab_from_tracks,
                        notes_to_events, notes_to_target, PAD_ID, EOS_ID, UNK_ID)
from .chunk import (Chunk, ChunkConfig, make_chunks_from_notes,
                    make_training_chunks, make_inference_chunks, K, L, MAX_TOKENS)
from .model import make_model, make_config, count_parameters
from .decode import (PitchValidLogitsProcessorV2, make_logits_processor,
                     constrained_beam_search, greedy_decode)
from .postproc import (DecodedNote, repair_overlaps, neighbor_search, postprocess)
from .metrics import (EvalCounts, align_predictions, compute_difficulty,
                      difficulty_terms, check_playability, aggregate_counts,
                      print_metrics)
from .render import render_tab, notes_to_tab
from .cli import main as cli_main

__all__ = [
    # fretboard
    "STANDARD", "pitch_name", "pitch_of", "positions",
    # quantise
    "PPQ", "GRID", "quantise_tick", "quantise_dur", "decompose_shift", "SHIFT_VOCAB",
    # tokenizer
    "Vocab", "DEFAULT_VOCAB", "build_vocab_from_tracks",
    "notes_to_events", "notes_to_target", "PAD_ID", "EOS_ID", "UNK_ID",
    # chunk
    "Chunk", "ChunkConfig", "make_chunks_from_notes", "make_training_chunks",
    "make_inference_chunks", "K", "L", "MAX_TOKENS",
    # model
    "make_model", "make_config", "count_parameters",
    # decode
    "PitchValidLogitsProcessorV2", "make_logits_processor",
    "constrained_beam_search", "greedy_decode",
    # postproc
    "DecodedNote", "repair_overlaps", "neighbor_search", "postprocess",
    # metrics
    "EvalCounts", "align_predictions", "compute_difficulty", "difficulty_terms",
    "check_playability", "aggregate_counts", "print_metrics",
    # render
    "render_tab", "notes_to_tab",
    # cli
    "cli_main",
]
