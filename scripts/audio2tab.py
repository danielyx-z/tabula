#!/usr/bin/env python3
"""Audio → tablature pipeline using Spotify basic-pitch + Tabula T5.

Full chain:

    YouTube URL (or local audio file)
        → yt-dlp download (audio only)
        → basic-pitch (audio-to-MIDI, ONNX)
        → pretty_midi note events
        → Tabula T5 (MIDI-to-tab, 20-note chunks + post-processing)
        → ASCII tablature

Usage:

    # From a YouTube link
    PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py "https://youtu.be/..." \
        --checkpoint runs/pretrain_t5_guitarset/best --beam 1 --width 100 --no-play

    # From a local audio file
    PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py song.wav \
        --checkpoint runs/pretrain_t5_guitarset/best --beam 1 --width 100 --no-play

    # Also write MIDI + WAV of the predicted tab
    PYTHONPATH=src .venv/bin/python3 scripts/audio2tab.py song.wav \
        --checkpoint runs/pretrain_t5_guitarset/best --mid out.mid --wav out.wav --no-play
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from basic_pitch.inference import predict  # noqa: E402
from model.chunk import ChunkConfig, make_inference_chunks  # noqa: E402
from model.decode import constrained_beam_search, greedy_decode  # noqa: E402
from tablature.metrics import decode_predictions  # noqa: E402
from tablature.postproc import postprocess  # noqa: E402
from tablature.render import render_tab  # noqa: E402
from tablature.tokenizer import DEFAULT_VOCAB, Vocab, PAD_TOKEN, EOS_TOKEN  # noqa: E402

# Import audio/MIDI functions lazily to avoid circular imports
from audio.midi import write_smf  # noqa: E402
from audio.synth import synthesize  # noqa: E402
from audio.wav import play, write_wav  # noqa: E402

PPQ = 960

YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)"
)


def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_URL_RE.match(text.strip()))


def download_youtube_audio(url: str, out_dir: str | Path) -> Path:
    """Download best audio from a YouTube URL using yt-dlp.

    Returns the path to the downloaded audio file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": {
            "FFmpegExtractAudio": ["-ar", "22050", "-ac", "1"],
        },
        "quiet": False,
        "no_warnings": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Find the downloaded wav file
    video_id = info.get("id", "unknown")
    wav_path = out_dir / f"{video_id}.wav"
    if not wav_path.exists():
        # Fallback: find any audio file in out_dir
        for f in out_dir.iterdir():
            if f.suffix.lower() in (".wav", ".m4a", ".webm", ".mp3", ".opus"):
                return f
        raise FileNotFoundError(f"yt-dlp did not produce an audio file in {out_dir}")

    return wav_path


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


def audio_to_midi_notes(audio_path: str | Path):
    """
    Run basic-pitch on an audio file and return (pretty_midi, note_dicts).

    note_dicts are in Tabula's internal format:
        {"t": onset_tick, "dur": dur_tick, "pitch": midi_pitch, "s": 0, "f": 0}
    """
    print(f"Running basic-pitch on {audio_path} ...")
    model_output, midi_data, note_events = predict(
        str(audio_path),
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=127.70,
        melodia_trick=True,
        midi_tempo=120.0,
    )

    # Convert pretty_midi notes to our internal Note format
    notes = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            # basic-pitch MIDI uses PPQ=220; scale to our PPQ=960
            onset_tick = int(round(midi_data.time_to_tick(note.start) * 960 / 220))
            end_tick = int(round(midi_data.time_to_tick(note.end) * 960 / 220))
            dur = max(1, end_tick - onset_tick)
            notes.append(
                {
                    "t": onset_tick,
                    "dur": dur,
                    "pitch": note.pitch,
                    "s": 0,  # placeholder; T5 predicts this
                    "f": 0,
                }
            )

    notes.sort(key=lambda n: (n["t"], n["pitch"]))
    print(f"  basic-pitch found {len(notes)} notes")
    return midi_data, notes


def run_t5_inference(
    model,
    vocab,
    notes: list[dict],
    device: str,
    beam: int = 1,
    max_len: int = 512,
    postproc_mode: str = "full",
):
    """Run Tabula T5 on note dicts and return post-processed tab notes."""
    from core.schema import Note
    from tablature.tokenizer import notes_to_events

    # Convert dicts to Note objects
    note_objs = [
        Note(t=n["t"], dur=n["dur"], pitch=n["pitch"], s=n["s"], f=n["f"])
        for n in notes
    ]

    # Build encoder tokens
    enc_ids = vocab.encode(notes_to_events(note_objs, base_t=0, snap_shifts=True))
    print(f"Encoder tokens: {len(enc_ids)}")

    # Inference chunks (20 notes + context)
    chunk_cfg = ChunkConfig(max_tokens=max_len)
    chunks = make_inference_chunks(note_objs, vocab, "audio_input", chunk_cfg)
    print(f"Inference chunks: {len(chunks)}")

    pad_id = vocab[PAD_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    all_pred_tabs: list[tuple[int, int]] = []

    for chunk_idx, chunk in enumerate(chunks):
        enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device=device)
        attn = (enc != pad_id).long()
        prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device=device)
        max_new = min(max_len, max(16, 6 * chunk.meta["num_notes"] + 16))

        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            # prompt_tabs=0: the context TAB in the decoder prompt is counted
            # in tab_count, which correctly shifts the index to the first
            # target note's pitch.  prompt_tabs>0 would under-index.
            prompt_tabs = 0
            if beam > 1:
                gen = constrained_beam_search(
                    model, enc, vocab, num_beams=beam, max_length=max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )
            else:
                gen = greedy_decode(
                    model, enc, vocab, max_length=max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )

        gen_ids = gen[0, prompt.size(1):].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]

        pred_tabs = decode_predictions(gen_ids, vocab)
        all_pred_tabs.extend(pred_tabs)
        print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {len(pred_tabs)} TAB events")

    # Post-process against the full note list
    proc_notes = postprocess(all_pred_tabs, note_objs, mode=postproc_mode)
    return proc_notes


def main() -> None:
    ap = argparse.ArgumentParser(prog="audio2tab")
    ap.add_argument("input", help="YouTube URL or local audio file (.wav/.mp3/.flac/...)")
    ap.add_argument("--checkpoint", default="runs/pretrain_t5_guitarset/best", help="Model checkpoint dir")
    ap.add_argument("--vocab", default="vocab/guitarset.json", help="Vocab file")
    ap.add_argument("--beam", type=int, default=4, help="Beam size (1=greedy)")
    ap.add_argument("--max-len", type=int, default=512, help="Max decode length")
    ap.add_argument("--no-play", action="store_true", help="Don't play audio")
    ap.add_argument("--mid", help="Output MIDI file")
    ap.add_argument("--wav", help="Output WAV file")
    ap.add_argument("--width", type=int, default=80, help="Terminal width")
    ap.add_argument("--postproc", choices=["none", "overlap", "full"], default="full", help="Post-processing mode")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, vocab = load_model(args.checkpoint, args.vocab, device)

    # Step 0: If input is a YouTube URL, download audio first
    audio_path = args.input
    tmp_dir: tempfile.TemporaryDirectory | None = None
    if is_youtube_url(args.input):
        tmp_dir = tempfile.TemporaryDirectory(prefix="audio2tab_yt_")
        print(f"Downloading audio from YouTube: {args.input}")
        audio_path = download_youtube_audio(args.input, tmp_dir.name)
        print(f"  Downloaded to: {audio_path}")

    try:
        # Step 1: Audio → MIDI (basic-pitch)
        midi_data, notes = audio_to_midi_notes(audio_path)
        if not notes:
            raise SystemExit("No notes detected in audio")

        # Step 2: MIDI → tab (Tabula T5)
        proc_notes = run_t5_inference(
            model, vocab, notes, device,
            beam=args.beam, max_len=args.max_len, postproc_mode=args.postproc,
        )

        # Render ASCII tab
        print("\n" + "=" * 60)
        print("PREDICTED TABLATURE")
        print("=" * 60)
        print(render_tab(proc_notes, width=args.width))

        # Build note-like dicts for output files
        pred_note_dicts = []
        for i, gt in enumerate(notes):
            if i < len(proc_notes):
                s, f = proc_notes[i].s, proc_notes[i].f
            else:
                s, f = 1, 0
            pred_note_dicts.append(
                {"t": gt["t"], "dur": gt["dur"], "pitch": gt["pitch"], "s": s, "f": f}
            )

        # Optional: write MIDI
        if args.mid:
            write_smf(Path(args.mid), pred_note_dicts)
            print(f"\nMIDI written to {args.mid}")

        # Optional: write / play WAV
        arr = np.empty(len(pred_note_dicts), dtype=[("t", "<i4"), ("dur", "<i4"),
                                                    ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
        for i, n in enumerate(pred_note_dicts):
            arr[i] = (n["t"], n["dur"], n["pitch"], n["s"], n["f"])

        bpm = 120.0  # basic-pitch default tempo
        samples = synthesize(arr, bpm)

        if args.wav:
            write_wav(Path(args.wav), samples)
            print(f"WAV written to {args.wav}")

        if not args.no_play:
            # Write samples to a temp wav file for playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_wav = Path(tf.name)
            write_wav(tmp_wav, samples)
            proc = play(tmp_wav)
            if proc:
                proc.wait()
            tmp_wav.unlink(missing_ok=True)

    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
