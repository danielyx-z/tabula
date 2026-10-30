#!/usr/bin/env python3
"""Render a parsed track and play it back, with the tab highlighting live.

    python scripts/preview.py --index 0                    # first track in the dataset
    python scripts/preview.py --song wasteland --bars 8
    python scripts/preview.py --tex some/song.tex          # bypass the dataset
    python scripts/preview.py --index 0 --mid out.mid --no-play

Audio is synthesised here (additive pluck) and fed to ``aplay``; no soundfont,
no PortAudio, no MIDI synth needed. The tab highlight is driven by wall clock
from the moment ``aplay`` starts, so what you see is what you hear.

The tab is laid out in systems ("lines" of bars) that fit the terminal width,
the way printed tab does it, so long excerpts wrap at bar lines instead of
mid-bar.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dataset import iter_tracks  # noqa: E402
from core.fretboard import format_tuning  # noqa: E402
from audio.midi import write_smf  # noqa: E402
from audio.synth import synthesize  # noqa: E402
from audio.wav import play, write_wav  # noqa: E402
from corpus.parse.alphatex import parse_file  # noqa: E402

PPQ = 960
BAR = PPQ * 4
LABEL_W = 2
STRING_LABELS = {1: "E", 2: "A", 3: "D", 4: "G", 5: "B", 6: "e"}
HIGHLIGHT = "\033[7m"
RESET = "\033[0m"


# --------------------------------------------------------------------------- tab


@dataclass
class Column:
    onset: int
    width: int
    cells: dict[int, int] = field(default_factory=dict)   # string -> fret
    bar_after: bool = False


def build_columns(notes, start: int, end: int) -> list[Column]:
    """One column per onset, width proportional to the gap that follows it."""
    sel = notes[(notes["t"] >= start) & (notes["t"] < end)]
    onsets = sorted({int(t) for t in sel["t"]})
    if not onsets:
        return []

    at: dict[int, dict[int, int]] = {t: {} for t in onsets}
    for n in sel:
        at[int(n["t"])][int(n["s"])] = int(n["f"])

    columns = []
    for i, t in enumerate(onsets):
        gap = (onsets[i + 1] if i + 1 < len(onsets) else end) - t
        width = max(3, min(12, round(gap / 240 * 2.5)))
        bar_after = i == len(onsets) - 1 or (onsets[i + 1] // BAR != t // BAR)
        columns.append(Column(t, width, at[t], bar_after))
    return columns


def build_systems(columns: list[Column], max_width: int) -> list[list[Column]]:
    """Greedily pack whole bars into lines that fit ``max_width``."""
    avail = max(24, max_width - LABEL_W)
    systems: list[list[Column]] = []
    current: list[Column] = []
    used = 0

    i = 0
    while i < len(columns):
        bar = columns[i].onset // BAR
        j = i
        while j < len(columns) and columns[j].onset // BAR == bar:
            j += 1
        group = columns[i:j]

        width = sum(c.width for c in group) + sum(1 for c in group if c.bar_after)
        if current and used + width > avail:
            systems.append(current)
            current, used = [], 0

        if not current and width > avail:            # one bar wider than the screen
            seps = sum(1 for c in group if c.bar_after)
            room = max(3 * len(group) + seps, avail)
            shrink = max(3, (room - seps) // len(group))
            for c in group:
                c.width = min(c.width, shrink)
            width = sum(c.width for c in group) + seps

        current.extend(group)
        used += width
        i = j

    if current:
        systems.append(current)
    return systems


def render_system(columns: list[Column], tuning, highlight: int | None = None) -> list[str]:
    lines: list[str] = []

    if highlight is not None:
        ruler = [" "] * (LABEL_W + sum(c.width + (1 if c.bar_after else 0) for c in columns))
        offset = LABEL_W
        for c in columns:
            if c.onset == highlight:
                ruler[min(offset + c.width // 2, len(ruler) - 1)] = "^"
            offset += c.width + (1 if c.bar_after else 0)
        lines.append("".join(ruler).rstrip())

    for string in range(len(tuning), 0, -1):
        out = [f"{STRING_LABELS.get(string, str(string))}|"]
        for c in columns:
            cell = str(c.cells.get(string, ""))
            pad = "-" * (c.width - len(cell))
            if highlight is not None and c.onset == highlight and cell:
                out.append(pad + HIGHLIGHT + cell + RESET)
            else:
                out.append(pad + cell)
            if c.bar_after:
                out.append("|")
        lines.append("".join(out))
    return lines


def render_all(columns: list[Column], tuning, max_width: int) -> list[str]:
    systems = build_systems(columns, max_width)
    lines: list[str] = []
    for k, system in enumerate(systems):
        if k:
            lines.append("")
        lines.extend(render_system(system, tuning))
    return lines or ["(no notes in range)"]


def column_times(notes, start: int, end: int) -> list[int]:
    sel = notes[(notes["t"] >= start) & (notes["t"] < end)]
    return sorted({int(t) for t in sel["t"]})


# ---------------------------------------------------------------------- preview


def animate(frame_fn, times, seconds_of, total: float, proc) -> None:
    sys.stdout.write("\033[2J\033[H")
    start = time.monotonic()
    last = None
    while True:
        elapsed = time.monotonic() - start
        current = -1
        for t in times:
            if seconds_of(t) <= elapsed:
                current = t
        alive = proc is not None and proc.poll() is None
        if elapsed > total and not alive:
            break
        if current != last or not alive:
            sys.stdout.write("\033[H" + frame_fn(current) + "\n\033[J")
            sys.stdout.flush()
            last = current
        if elapsed > total + 2:
            break
        time.sleep(1 / 40)
    sys.stdout.write("\n")


# -------------------------------------------------------------------------- main


def load_track(args):
    if args.tex:
        path = Path(args.tex)
        tracks = parse_file(path.read_text(encoding="utf-8", errors="replace"), str(path), path.stem)
        if not tracks:
            raise SystemExit("no tracks in file")
        t = tracks[args.track or 0]
        notes = np.empty(len(t.notes), dtype=[("t", "<i4"), ("dur", "<i4"),
                                              ("pitch", "<i1"), ("s", "<i1"), ("f", "<i1")])
        for i, n in enumerate(t.notes):
            notes[i] = (n.t, n.dur, n.pitch, n.s, n.f)
        return {"source": str(path), "song": path.stem, "tempo": t.tempo,
                "tuning": list(t.tuning), "tuning_name": format_tuning(t.tuning)}, notes

    if args.index is None and not args.song:
        for meta, notes in iter_tracks(args.dataset):
            return meta, notes
        raise SystemExit("dataset is empty")

    for meta, notes in iter_tracks(args.dataset):
        if args.index is not None and meta["id"] == args.index:
            return meta, notes
        if args.song and args.song.lower() in meta["song"].lower():
            return meta, notes
    raise SystemExit("track not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data/processed_guitarset"))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--song", default=None)
    ap.add_argument("--tex", default=None)
    ap.add_argument("--track", type=int, default=0, help="track index inside --tex")
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--start-bar", type=int, default=0)
    ap.add_argument("--bpm", type=float, default=None, help="override tempo")
    ap.add_argument("--width", type=int, default=None, help="tab width in columns")
    ap.add_argument("--mid", default=None)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    meta, notes = load_track(args)
    bpm = args.bpm or meta.get("tempo") or 120.0
    start = args.start_bar * BAR
    end = start + args.bars * BAR
    width = args.width or max(60, shutil.get_terminal_size((100, 24)).columns)

    sel = notes[(notes["t"] >= start) & (notes["t"] < end)]
    print(f"{meta['song']}  track {meta.get('track', 0)}  "
          f"{meta.get('tuning_name', '')}  {bpm:g} bpm  {len(sel)} notes  "
          f"bars {args.start_bar + 1}-{args.start_bar + args.bars}")
    print()

    columns = build_columns(notes, start, end)
    systems = build_systems(columns, width)
    print("\n".join(render_all(columns, meta["tuning"], width)))

    wav = Path(args.wav) if args.wav else Path("/tmp/tabula-preview.wav")
    samples = synthesize(sel, bpm)
    write_wav(wav, samples)
    print(f"\naudio: {wav}  {len(samples) / 44100:.2f}s  peak {np.max(np.abs(samples)):.2f}")

    if args.mid:
        write_smf(Path(args.mid), sel, bpm)
        print(f"midi:  {args.mid}")

    if args.no_play:
        return 0

    times = column_times(notes, start, end)
    tps = PPQ * bpm / 60.0
    t0 = int(sel["t"].min()) if len(sel) else 0
    seconds_of = lambda tick: (tick - t0) / tps  # noqa: E731
    total = len(samples) / 44100

    proc = play(wav)
    if proc is None:
        print("no audio player found; keeping the wav", file=sys.stderr)
        return 1

    where = {c.onset: si for si, system in enumerate(systems) for c in system}

    def frame(onset: int) -> str:
        si = where.get(onset, 0)
        system = systems[si] if systems else []
        first, last = (system[0].onset // BAR, system[-1].onset // BAR) if system else (0, 0)
        head = f"{meta['song']}  {bpm:g} bpm  bars {first + 1}-{last + 1}"
        body = render_system(system, meta["tuning"], highlight=onset if onset >= 0 else None)
        return head + "\n\n" + "\n".join(body)

    if args.no_live or not sys.stdout.isatty():
        proc.wait()
        return 0

    animate(frame, times, seconds_of, total, proc)
    proc.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
