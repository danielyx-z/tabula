#!/usr/bin/env python3
"""Stitch audio-only performance features onto the existing Tabula tab.

Pipeline:

    1. audio  -- HPSS + percussive onset detection + MIDI reconciliation
    2. tab    -- existing ``scripts/midi2tab.py`` (unchanged)
    3. stitch -- non-note validation, dead notes, audio-gated H/P, manual
       overlays, ASCII + GP5

Example::

    python scripts/stitch_tab.py \\
        --audio song.wav \\
        --midi transcribed.mid \\
        --out-dir out/song

The stage can be resumed with ``--stage audio|tab|stitch|all``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures import (  # noqa: E402
    PPQ,
    align_events,
    apply_overlay,
    detect_audio_events,
    infer_hp,
    insert_dead_notes,
    load_overlay,
    parse_grid,
    read_midi_notes,
    render_tab_with_features,
    trim_notes_for_dead,
    write_audio_events,
)
from tabfeatures.audio_events import write_stems  # noqa: E402
from tabfeatures.dead import merge_dead_notes  # noqa: E402
from tabfeatures.evidence import AudioAnalysis  # noqa: E402
from tabfeatures.gp5 import build_gp5  # noqa: E402
from tabfeatures.non_note import detect_non_notes  # noqa: E402
from tabfeatures.overlay import empty_overlay  # noqa: E402
from tabfeatures.ticks import tick_to_seconds  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    """Resolve checkpoint/vocab relative to the cwd, then the repo root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def _user_path(value: str | Path) -> Path:
    """Resolve a user-supplied path against the current working directory."""
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_stage_audio(args: argparse.Namespace, out: Path, grid_ticks: int) -> dict[str, Any]:
    print("[audio] detecting percussive onsets ...", flush=True)
    result = detect_audio_events(
        args.audio,
        sr=args.sr,
        delta=args.onset_delta,
        wait=args.onset_wait,
        before=args.window_before,
        after=args.window_after,
        return_stems=args.keep_intermediate,
    )
    write_audio_events(out / "audio_events.json", result)
    if args.keep_intermediate:
        write_stems(out, result)
        print(f"[audio] wrote HPSS stems to {out}", flush=True)

    pm, midi_onsets = read_midi_notes(args.midi)
    events, alignment = align_events(
        result["events"],
        pm,
        midi_onsets,
        grid_ticks=grid_ticks,
        tolerance_ticks=args.tolerance_ticks or None,
        auto_offset=not args.no_auto_offset,
    )
    payload = {
        "grid_ticks": grid_ticks,
        "alignment": alignment,
        "events": events,
    }
    _write_json(out / "percussion_events.json", payload)
    _write_json(out / "alignment_report.json", alignment)

    print(
        f"[audio] {alignment['total_events']} onsets, "
        f"{alignment['matched_events']} matched to MIDI, "
        f"offset {alignment['applied_offset_sec']:+.3f}s",
        flush=True,
    )
    for warning in alignment.get("warnings", []):
        print(f"[audio] warning: {warning}", flush=True)
    return payload


def run_stage_tab(args: argparse.Namespace, out: Path) -> Path:
    tab_dir = out / "tabula"
    final_json = tab_dir / "final.json"
    if final_json.exists() and not args.force:
        print(f"[tab] reusing {final_json}", flush=True)
        return final_json

    tab_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _repo_path(args.checkpoint)
    vocab = _repo_path(args.vocab)
    if not checkpoint.exists():
        raise SystemExit(
            f"[tab] checkpoint not found: {checkpoint}\n"
            "Pass --checkpoint with the Tabula checkpoint directory."
        )
    if not vocab.exists():
        raise SystemExit(
            f"[tab] vocab not found: {vocab}\nPass --vocab with the vocab JSON."
        )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "midi2tab.py"),
        str(_user_path(args.midi)),
        "--out-dir",
        str(tab_dir),
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(vocab),
        "--no-audio",
        "--beam",
        str(args.beam),
    ]
    log_path = tab_dir / "run.log"
    print(f"[tab] running midi2tab (log: {log_path})", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise SystemExit(
            f"[tab] midi2tab failed with exit code {process.returncode}\n{tail}"
        )
    if not final_json.exists():
        raise SystemExit(
            f"[tab] midi2tab finished but {final_json} is missing; see {log_path}"
        )
    print(f"[tab] wrote {final_json}", flush=True)
    return final_json


def _ensure_overlay(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, empty_overlay())
        print(f"[stitch] created overlay template {path}", flush=True)
    return path


def run_stage_stitch(args: argparse.Namespace, out: Path, grid_ticks: int) -> dict[str, Any]:
    tab_dir = out / "tabula"
    final_json = tab_dir / "final.json"
    if not final_json.exists():
        raise SystemExit(
            f"[stitch] missing {final_json}; run --stage tab or --stage all first"
        )

    events_path = out / "percussion_events.json"
    if not events_path.exists():
        raise SystemExit(
            f"[stitch] missing {events_path}; run --stage audio or --stage all first"
        )
    events_payload = _load_json(events_path)
    if isinstance(events_payload, list):
        events = events_payload
        alignment = {}
        stored_grid = None
    else:
        events = events_payload.get("events", [])
        alignment = events_payload.get("alignment", {})
        stored_grid = events_payload.get("grid_ticks")
    if stored_grid is not None and int(stored_grid) != int(grid_ticks):
        print(
            f"[stitch] using grid {int(stored_grid)} ticks from "
            f"percussion_events.json (CLI asked for {grid_ticks})",
            flush=True,
        )
        grid_ticks = int(stored_grid)

    # Older percussion_events.json files predate audio_class preservation.
    # Backfill from the raw event list when resuming an old job, so the
    # non-note validator keeps the original timbre prior.
    raw_events_path = out / "audio_events.json"
    if raw_events_path.exists():
        raw_payload = _load_json(raw_events_path)
        raw_events = (
            raw_payload.get("events", [])
            if isinstance(raw_payload, dict)
            else raw_payload
        )
        if isinstance(raw_events, list):
            for event in events:
                if event.get("audio_class"):
                    continue
                index = event.get("index")
                if not isinstance(index, int) or not 0 <= index < len(raw_events):
                    continue
                raw_event = raw_events[index]
                if not isinstance(raw_event, dict):
                    continue
                event["audio_class"] = str(raw_event.get("class") or "")
                event["audio_confidence"] = float(
                    raw_event.get("confidence", 0.0) or 0.0
                )
                event["audio_strength"] = float(
                    raw_event.get("strength", 0.0) or 0.0
                )

    if alignment:
        total = int(alignment.get("total_events", 0) or 0)
        match_rate = (
            float(alignment.get("match_rate", 0.0) or 0.0) if total else 1.0
        )
    else:
        total = len(events)
        matched = sum(1 for event in events if event.get("matched"))
        match_rate = (matched / total) if total else 1.0
    if match_rate < args.min_match_rate and not args.allow_low_match_rate:
        raise SystemExit(
            f"[stitch] audio/MIDI match rate is {match_rate:.1%}, below "
            f"--min-match-rate {args.min_match_rate:.0%}. Refusing to insert "
            "dead notes from a poorly aligned pair. Check that both files "
            "cover the same performance, or pass --allow-low-match-rate."
        )

    final_notes = _load_json(final_json)
    if isinstance(final_notes, dict):
        final_notes = final_notes.get("notes", [])
    if not isinstance(final_notes, list):
        raise SystemExit("[stitch] final.json must be a list of notes")

    report_path = tab_dir / "report.json"
    report = _load_json(report_path) if report_path.exists() else {}
    capo = int(report.get("capo", 0) or 0)

    # final.mid preserves the source tempo map and is also used to convert
    # H/P ticks to audio time.  Resolve it before the audio-aware stages.
    timing_midi = tab_dir / "final.mid"
    if not timing_midi.exists():
        timing_midi = _user_path(args.midi)

    analysis: AudioAnalysis | None = None
    if args.non_note or (args.auto_hp and args.hp_audio):
        print("[stitch] measuring audio note evidence ...", flush=True)
        analysis = AudioAnalysis.load(args.audio, sr=args.sr)

    non_note_dead: list[dict[str, Any]] = []
    non_note_review: list[dict[str, Any]] = []
    non_note_stats: dict[str, Any] = {"enabled": False}
    if args.non_note:
        final_notes, non_note_dead, non_note_review, non_note_stats = detect_non_notes(
            final_notes,
            events,
            analysis=analysis,
            grid_ticks=grid_ticks,
            min_confidence=args.non_note_min_confidence,
            unmatched_min_confidence=args.non_note_unmatched_min_confidence,
            unmatched_max_harmonic_ratio=args.non_note_unmatched_max_harmonic_ratio,
            review_confidence=args.non_note_review_confidence,
            slap_min_pitch_support=args.slap_min_pitch_support,
        )
        print(
            f"[stitch] non-note evidence: {non_note_stats.get('converted', 0)} replaced, "
            f"{non_note_stats.get('slap', 0)} slap markers, "
            f"{non_note_stats.get('reviewed', 0)} in review",
            flush=True,
        )

    dead_notes, dead_stats = insert_dead_notes(
        final_notes,
        events,
        strings=args.dead_strings,
        grid_ticks=grid_ticks,
        min_confidence=args.min_confidence,
        skip_ticks=set(non_note_stats.get("used_event_ticks", [])),
    )
    dead_notes = merge_dead_notes(non_note_dead, dead_notes)
    final_notes = trim_notes_for_dead(final_notes, dead_notes)

    tick_seconds = None
    if analysis is not None and args.auto_hp:
        pm, _ = read_midi_notes(timing_midi)
        tick_seconds = lambda tick: tick_to_seconds(pm, tick)

    if args.auto_hp:
        features, hp_review = infer_hp(
            final_notes,
            max_gap_ticks=int(round(args.hp_max_gap_beats * PPQ)),
            max_interval=args.hp_max_interval,
            ring_tolerance=args.ring_tolerance,
            skip_chords=not args.allow_hp_in_chords,
            grid_ticks=grid_ticks,
            events=events if args.hp_audio else None,
            analysis=analysis if args.hp_audio else None,
            offset_sec=float(alignment.get("applied_offset_sec", 0.0) or 0.0),
            tick_seconds=tick_seconds if args.hp_audio else None,
            min_confidence=args.hp_min_confidence if args.hp_audio else 0.0,
        )
    else:
        features, hp_review = {}, []

    overlay_path = (
        _user_path(args.articulations)
        if args.articulations
        else out / "articulations.json"
    )
    overlay = load_overlay(_ensure_overlay(overlay_path))
    final_notes, dead_notes, features, overlay_errors = apply_overlay(
        final_notes, dead_notes, features, overlay
    )
    final_notes = [note for note in final_notes if not note.get("_drop")]
    final_notes = trim_notes_for_dead(final_notes, dead_notes)

    slots_per_bar = max(1, int(round(PPQ * 4 / grid_ticks)))
    tab_text = render_tab_with_features(
        final_notes,
        dead_notes,
        features,
        capo=capo,
        width=args.width,
        slot_ticks=grid_ticks,
        slots_per_bar=slots_per_bar,
    )
    tab_path = out / "tab_with_features.txt"
    tab_path.write_text(tab_text, encoding="utf-8")

    _write_json(out / "dead_notes.json", dead_notes)
    _write_json(out / "hp_review.json", hp_review)
    _write_json(out / "non_note_review.json", non_note_review)

    gp5_info: dict[str, Any] | None = None
    gp5_path = out / "song.gp5"
    if not args.no_gp5:
        gp5_info = build_gp5(
            final_notes,
            timing_midi,
            gp5_path,
            capo=capo,
            title=args.title,
            dead_notes=dead_notes,
            features=features,
        )

    hp_applied = sum(
        1 for feature in features.values() if feature.get("hp")
    )
    hp_inferred = sum(1 for item in hp_review if item.get("accepted"))
    pipeline_report = {
        "audio": str(_user_path(args.audio)),
        "midi": str(_user_path(args.midi)),
        "grid_ticks": grid_ticks,
        "capo": capo,
        "notes": len(final_notes),
        "dead_notes": len(dead_notes),
        "dead_stats": dead_stats,
        "non_note": non_note_stats,
        "hp_candidates": len(hp_review),
        "hp_inferred": hp_inferred,
        "hp_applied": hp_applied,
        "overlay": str(overlay_path),
        "overlay_errors": overlay_errors,
        "alignment": alignment,
        "gp5_timing_midi": str(timing_midi),
        "gp5": gp5_info,
        "outputs": {
            "tab": str(tab_path),
            "gp5": str(gp5_path) if gp5_info else None,
            "dead_notes": str(out / "dead_notes.json"),
            "hp_review": str(out / "hp_review.json"),
            "non_note_review": str(out / "non_note_review.json"),
        },
    }
    _write_json(out / "pipeline_report.json", pipeline_report)

    print(
        f"[stitch] {len(final_notes)} notes, {len(dead_notes)} dead notes, "
        f"{hp_inferred}/{len(hp_review)} H/P accepted",
        flush=True,
    )
    for error in overlay_errors:
        print(f"[stitch] overlay error: {error}", flush=True)
    print(f"[stitch] wrote {tab_path}", flush=True)
    if gp5_info:
        print(f"[stitch] wrote {gp5_path}", flush=True)
    return pipeline_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add audio-detected dead notes and articulations to a Tabula MIDI-to-tab job."
        )
    )
    parser.add_argument("--audio", required=True, help="source audio file")
    parser.add_argument(
        "--midi",
        required=True,
        help="quantised MIDI from MuScriptor (or another transcriber)",
    )
    parser.add_argument("--out-dir", required=True, help="job output directory")
    parser.add_argument(
        "--stage",
        choices=("audio", "tab", "stitch", "all"),
        default="all",
        help="run one stage or the whole pipeline (default: all)",
    )
    parser.add_argument(
        "--grid",
        default="1/16",
        help="quantisation grid used to snap audio events (default: 1/16)",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/finetune_t5_guitarset/best",
        help="Tabula checkpoint directory",
    )
    parser.add_argument(
        "--vocab",
        default="vocab/canonical.json",
        help="Tabula vocab JSON",
    )
    parser.add_argument("--beam", type=int, default=4, help="midi2tab beam size")

    parser.add_argument(
        "--dead-strings",
        choices=("lowest", "nearest", "all"),
        default="lowest",
        help="string placement for dead notes (default: lowest)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="minimum dead-note confidence (default: 0.6)",
    )
    parser.add_argument(
        "--no-non-note",
        dest="non_note",
        action="store_false",
        default=True,
        help="disable automatic conversion of matched non-note hits to dead notes",
    )
    parser.add_argument(
        "--non-note-min-confidence",
        type=float,
        default=0.60,
        help="minimum score to convert a matched non-note to X (default: 0.60)",
    )
    parser.add_argument(
        "--non-note-unmatched-min-confidence",
        type=float,
        default=0.50,
        help=(
            "minimum score to convert a note using a nearby unmatched "
            "percussive onset (default: 0.50)"
        ),
    )
    parser.add_argument(
        "--non-note-unmatched-max-harmonic-ratio",
        type=float,
        default=0.70,
        help=(
            "ignore unmatched onsets above this harmonic ratio as likely "
            "real notes (default: 0.70)"
        ),
    )
    parser.add_argument(
        "--non-note-review-confidence",
        type=float,
        default=0.45,
        help="minimum score to record a non-note candidate in review (default: 0.45)",
    )
    parser.add_argument(
        "--slap-min-pitch-support",
        type=float,
        default=0.12,
        help=(
            "minimum chord pitch support to keep a percussive chord and add "
            "an x beneath it (default: 0.12)"
        ),
    )
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=0.5,
        help="refuse to stitch if fewer than this fraction of audio onsets match MIDI (default: 0.5)",
    )
    parser.add_argument(
        "--allow-low-match-rate",
        action="store_true",
        help="insert dead notes even when audio and MIDI align poorly",
    )
    parser.add_argument(
        "--tolerance-ticks",
        type=int,
        default=0,
        help="audio/MIDI match tolerance; 0 = half a grid cell",
    )
    parser.add_argument(
        "--no-auto-offset",
        action="store_true",
        help="do not estimate and apply a global audio/MIDI offset",
    )

    parser.add_argument(
        "--auto-hp",
        dest="auto_hp",
        action="store_true",
        default=True,
        help="infer hammer-ons/pull-offs (default: on)",
    )
    parser.add_argument(
        "--no-auto-hp",
        dest="auto_hp",
        action="store_false",
        help="disable hammer-on/pull-off inference",
    )
    parser.add_argument(
        "--no-hp-audio",
        dest="hp_audio",
        action="store_false",
        default=True,
        help="use model-only H/P rules instead of audio-gated inference",
    )
    parser.add_argument(
        "--hp-min-confidence",
        type=float,
        default=0.70,
        help="minimum audio-gated H/P confidence (default: 0.70)",
    )
    parser.add_argument(
        "--hp-max-gap-beats",
        type=float,
        default=0.5,
        help="maximum H/P gap in quarter notes (default: 0.5)",
    )
    parser.add_argument(
        "--hp-max-interval",
        type=int,
        default=7,
        help="maximum H/P interval in semitones (default: 7)",
    )
    parser.add_argument(
        "--ring-tolerance",
        type=int,
        default=60,
        help="extra ticks allowed for the first note to still ring (default: 60)",
    )
    parser.add_argument(
        "--allow-hp-in-chords",
        action="store_true",
        help="allow H/P inference at chord onsets (off by default)",
    )
    parser.add_argument(
        "--articulations",
        default=None,
        help="manual overlay JSON (default: <out-dir>/articulations.json)",
    )

    parser.add_argument("--sr", type=int, default=22050, help="audio analysis sample rate")
    parser.add_argument(
        "--onset-delta",
        type=float,
        default=0.05,
        help="onset detector threshold (default: 0.05)",
    )
    parser.add_argument(
        "--onset-wait",
        type=int,
        default=2,
        help="onset detector wait frames (default: 2)",
    )
    parser.add_argument(
        "--window-before",
        type=float,
        default=0.03,
        help="seconds of context before an onset (default: 0.03)",
    )
    parser.add_argument(
        "--window-after",
        type=float,
        default=0.18,
        help="seconds of context after an onset (default: 0.18)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="write harmonic.wav and percussive.wav",
    )

    parser.add_argument("--width", type=int, default=120, help="max ASCII tab width")
    parser.add_argument(
        "--title", default="Uploaded MIDI", help="GP5 song title"
    )
    parser.add_argument(
        "--no-gp5",
        action="store_true",
        help="skip GP5 export (ASCII and JSON are still written)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run midi2tab even if final.json exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = _user_path(args.audio)
    midi = _user_path(args.midi)
    out = _user_path(args.out_dir)
    if not audio.exists():
        raise SystemExit(f"audio not found: {audio}")
    if not midi.exists():
        raise SystemExit(f"MIDI not found: {midi}")
    out.mkdir(parents=True, exist_ok=True)
    args.audio = str(audio)
    args.midi = str(midi)
    args.out_dir = str(out)

    grid_ticks = parse_grid(args.grid)
    if args.stage in {"audio", "all"}:
        run_stage_audio(args, out, grid_ticks)
    if args.stage in {"tab", "all"}:
        run_stage_tab(args, out)
    if args.stage in {"stitch", "all"}:
        run_stage_stitch(args, out, grid_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
