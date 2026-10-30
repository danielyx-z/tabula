"""CLI: midi2tab entry point.

Usage:

    PYTHONPATH=src .venv/bin/python3 -m fretwire.cli song.mid \
        --checkpoint runs/pretrain_t5_v5/best --beam 1 --width 100

Parses a MIDI file, runs the trained T5 model over 20-note inference chunks,
post-processes the output (overlap correction + ±5-note neighbor search),
and prints ASCII tablature. Optionally writes a MIDI file and/or WAV audio
of the predicted tab.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from .chunk import ChunkConfig, make_inference_chunks
from .decode import constrained_beam_search, greedy_decode
from .metrics import decode_predictions
from .parse.midi import parse_midi, midi_to_encoder_tokens
from .postproc import postprocess
from .render import render_tab
from .tokenizer import DEFAULT_VOCAB, Vocab, PAD_TOKEN, EOS_TOKEN


def load_model(checkpoint_dir: str, vocab_path: str, device: str = "cuda"):
    """Load a T5 checkpoint and its frozen vocabulary."""
    vocab = Vocab.load(vocab_path) if Path(vocab_path).exists() else DEFAULT_VOCAB
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    if model.config.vocab_size != vocab.size:
        raise ValueError(
            f"Checkpoint vocab size ({model.config.vocab_size}) does not match "
            f"vocab file ({vocab.size})"
        )
    model.to(device)
    model.eval()
    return model, vocab


def _make_pred_notes(midi_notes, proc_notes):
    """Build dict-like note records with predicted s/f but original timing/pitch."""
    pred_notes = []
    for i, gt in enumerate(midi_notes):
        if i < len(proc_notes):
            s, f = proc_notes[i].s, proc_notes[i].f
        else:
            s, f = 1, 0
        pred_notes.append({"t": gt.t, "dur": gt.dur, "pitch": gt.pitch, "s": s, "f": f})
    return pred_notes


def _synthesize_wav(pred_notes, bpm: float) -> tuple[np.ndarray, int]:
    """Synthesize WAV audio from predicted notes using fretwire.preview."""
    # Import here to avoid circular import with fretwire.__init__
    from preview import synthesize

    arr = np.empty(len(pred_notes), dtype=[("t", "<i4"), ("dur", "<i4"),
                                           ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
    for i, n in enumerate(pred_notes):
        arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])
    samples = synthesize(arr, bpm)
    return samples, 44100


def _write_midi(path: Path, pred_notes):
    """Write a Standard MIDI File from predicted notes."""
    from preview import write_smf
    write_smf(path, pred_notes)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 44100):
    """Write WAV file."""
    from preview import write_wav
    write_wav(path, samples)


def _play_wav(samples: np.ndarray, sr: int = 44100):
    """Play WAV audio through the system player."""
    from preview import play
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = Path(tf.name)
    from preview import write_wav
    write_wav(tmp_path, samples)
    proc = play(tmp_path)
    if proc:
        proc.wait()
    tmp_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="midi2tab")
    ap.add_argument("input", help="Input MIDI file (.mid/.midi)")
    ap.add_argument("--checkpoint", default="runs/pretrain_t5_v5/best", help="Model checkpoint dir")
    ap.add_argument("--vocab", default="vocab_v1_full.json", help="Vocab file")
    ap.add_argument("--beam", type=int, default=4, help="Beam size (1=greedy)")
    ap.add_argument("--max-len", type=int, default=512, help="Max decode length")
    ap.add_argument("--no-play", action="store_true", help="Don't play audio")
    ap.add_argument("--mid", help="Output MIDI file")
    ap.add_argument("--wav", help="Output WAV file")
    ap.add_argument("--width", type=int, default=80, help="Terminal width")
    ap.add_argument("--program", type=int, default=None, help="GM program to extract (0-127), default=first non-drum")
    ap.add_argument("--no-merge", action="store_true", help="Don't merge multi-track MIDI")
    ap.add_argument("--postproc", choices=["none", "overlap", "full"], default="full", help="Post-processing mode")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, vocab = load_model(args.checkpoint, args.vocab, device)

    # Parse MIDI
    print(f"Parsing {args.input}...")
    midi = parse_midi(
        args.input,
        target_program=args.program,
        merge_tracks=not args.no_merge,
    )
    print(f"  Track: {midi.name}  program={midi.program}  notes={len(midi.notes)}")
    if not midi.notes:
        raise SystemExit("No notes in selected track(s)")

    # Build encoder tokens
    enc_ids = midi_to_encoder_tokens(midi.notes, vocab, snap_shifts=True)
    print(f"Encoder tokens: {len(enc_ids)}")

    # Inference chunks (20 notes + context carry-over)
    chunk_cfg = ChunkConfig(max_tokens=args.max_len)
    chunks = make_inference_chunks(midi.notes, vocab, "midi_input", chunk_cfg)
    print(f"Inference chunks: {len(chunks)}")

    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    all_pred_tabs: list[tuple[int, int]] = []

    for chunk_idx, chunk in enumerate(chunks):
        enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
        attn = (enc != pad_id).long()
        prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=device)
        max_new = min(args.max_len, max(16, 6 * chunk.meta["num_notes"] + 16))

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            # prompt_tabs=0: the context TAB in the decoder prompt is counted
            # in tab_count, which correctly shifts the index to the first
            # target note's pitch.  prompt_tabs>0 would under-index.
            prompt_tabs = 0
            if args.beam > 1:
                gen = constrained_beam_search(
                    model, enc, vocab, num_beams=args.beam, max_length=args.max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )
            else:
                gen = greedy_decode(
                    model, enc, vocab, max_length=args.max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )

        gen_ids = gen[0, prompt.size(1):].tolist()
        # Strip trailing EOS/pad
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]

        pred_tabs = decode_predictions(gen_ids, vocab)
        all_pred_tabs.extend(pred_tabs)
        print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {len(pred_tabs)} TAB events")

    # Post-process against the full note list
    proc_notes = postprocess(all_pred_tabs, midi.notes, mode=args.postproc)

    # Render ASCII tab
    print("\n" + "=" * 60)
    print("PREDICTED TABLATURE")
    print("=" * 60)
    print(render_tab(proc_notes, width=args.width))

    # Build note-like objects with predicted s/f for output files
    pred_note_objs = _make_pred_notes(midi.notes, proc_notes)

    # Optional: write MIDI
    if args.mid:
        _write_midi(Path(args.mid), pred_note_objs)
        print(f"\nMIDI written to {args.mid}")

    # Optional: write / play WAV
    bpm = midi.tempo_changes[0][1] if midi.tempo_changes else 120.0
    samples, sr = _synthesize_wav(pred_note_objs, bpm)

    if args.wav:
        _write_wav(Path(args.wav), samples, sr)
        print(f"WAV written to {args.wav}")

    if not args.no_play:
        _play_wav(samples, sr)


if __name__ == "__main__":
    main()
