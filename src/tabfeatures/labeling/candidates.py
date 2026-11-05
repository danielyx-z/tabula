"""Lightweight onset segmentation for the browser labeller.

The labeller does not need the full pitch/timbre feature pass used by the
stitching pipeline.  It only needs stable, non-overlapping event candidates
around the percussive attacks that a guitarist can hear and label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..audio_events import load_mono, split_hpss


def _group_onsets(times: list[float], *, merge_s: float) -> list[float]:
    groups: list[float] = []
    for time_s in sorted(float(value) for value in times):
        if groups and time_s - groups[-1] <= float(merge_s):
            continue
        groups.append(time_s)
    return groups


def _basic_pitch_onsets(
    audio_path: str | Path,
    *,
    merge_s: float = 0.04,
) -> list[float]:
    """Use basic-pitch note starts as high-confidence pitched attacks.

    The labeler previously built every candidate from a percussive onset
    envelope, which can split one chord into several candidates or shift a
    candidate away from the actual note.  basic-pitch gives note-aligned starts;
    the percussive pass below still supplies non-pitched attacks.
    """
    try:
        from basic_pitch.inference import predict
    except Exception:
        return []
    try:
        _model_output, midi_data, _note_events = predict(
            str(audio_path),
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=50.0,
            melodia_trick=True,
            midi_tempo=120.0,
        )
    except Exception:
        return []
    times = [
        float(note.start)
        for instrument in midi_data.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    return _group_onsets(times, merge_s=merge_s)


def _merge_times(
    times: list[float],
    strengths: list[float],
    *,
    min_gap_s: float,
) -> list[float]:
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    merged: list[tuple[float, float]] = []
    for index in order:
        time_s = float(times[index])
        strength = float(strengths[index])
        if merged and time_s - merged[-1][0] < float(min_gap_s):
            # Keep the stronger of two detections in the same transient.
            if strength > merged[-1][1]:
                merged[-1] = (time_s, strength)
            continue
        merged.append((time_s, strength))
    return [time_s for time_s, _ in merged]


def build_candidates(
    onsets: list[float],
    duration_s: float,
    *,
    before: float = 0.08,
    after: float = 0.30,
) -> list[dict[str, Any]]:
    """Build sorted playback windows from onset times."""
    merged = sorted(float(value) for value in onsets)
    candidates: list[dict[str, Any]] = []
    for index, onset_s in enumerate(merged):
        next_onset = merged[index + 1] if index + 1 < len(merged) else duration_s
        end_s = min(duration_s, onset_s + float(after), next_onset - 0.005)
        start_s = max(0.0, onset_s - float(before))
        if end_s <= start_s:
            end_s = min(duration_s, start_s + 0.1)
        candidates.append(
            {
                "index": index,
                "onset_s": round(float(onset_s), 6),
                "start_s": round(float(start_s), 6),
                "end_s": round(float(end_s), 6),
                "duration_s": round(float(end_s - start_s), 6),
            }
        )
    return candidates


def detect_candidates(
    audio_path: str | Path,
    *,
    sr: int = 22050,
    hop_length: int = 256,
    delta: float = 0.05,
    wait: int = 2,
    before: float = 0.08,
    after: float = 0.30,
    min_gap_s: float = 0.06,
) -> dict[str, Any]:
    """Return onset candidates with playback windows.

    Pitched notes come from basic-pitch note starts.  A percussive onset pass
    adds non-pitched attacks (dead strings, flicks, body knocks) that do not
    overlap a detected note.  Every candidate still represents one attack.
    """
    y, actual_sr = load_mono(audio_path, sr=sr)
    duration_s = float(len(y) / actual_sr) if len(y) else 0.0
    if len(y) == 0:
        return {"sr": actual_sr, "duration": duration_s, "candidates": []}

    _, percussive = split_hpss(y, kernel_size=31, margin=1.0)
    onset_env = librosa.onset.onset_strength(
        y=percussive,
        sr=actual_sr,
        hop_length=hop_length,
    )
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=actual_sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        delta=float(delta),
        wait=int(wait),
    )
    times = [
        float(librosa.frames_to_time(int(frame), sr=actual_sr, hop_length=hop_length))
        for frame in frames
    ]
    strengths = [
        float(onset_env[int(frame)]) if 0 <= int(frame) < len(onset_env) else 0.0
        for frame in frames
    ]
    percussive_times = _merge_times(times, strengths, min_gap_s=min_gap_s)
    pitched_times = _basic_pitch_onsets(audio_path)
    if pitched_times:
        extras = [
            time_s
            for time_s in percussive_times
            if not any(abs(time_s - note_s) <= 0.05 for note_s in pitched_times)
        ]
        merged = sorted([*pitched_times, *extras])
        source = "basic_pitch+percussive"
    else:
        merged = percussive_times
        source = "percussive"

    candidates = build_candidates(
        merged,
        duration_s,
        before=float(before),
        after=float(after),
    )
    return {
        "version": 2,
        "detector": source,
        "sr": int(actual_sr),
        "duration": round(duration_s, 6),
        "candidates": candidates,
    }
