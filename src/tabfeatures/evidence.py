"""Shared audio evidence for note validation and articulation inference.

The stitcher needs to answer two questions from the same audio window:

* Is this MIDI note backed by a stable pitched sound, or is it a knock / muted
  attack / other non-note event?
* If two notes are adjacent on one string, did the second note get its own
  pluck attack, or is it a hammer-on / pull-off?

This module keeps the signal processing in one place.  It deliberately measures
evidence relative to the expected MIDI pitch, so it is not tied to particular
strings, frets, songs, or a "low G" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np


@dataclass
class AudioAnalysis:
    """A compact STFT view of the audio used for per-onset measurements."""

    y: np.ndarray
    sr: int
    hop_length: int
    stft: np.ndarray
    freqs: np.ndarray

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 256,
    ) -> "AudioAnalysis":
        y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
        y = np.asarray(y, dtype=np.float32)
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=actual_sr, n_fft=n_fft)
        return cls(
            y=y,
            sr=int(actual_sr),
            hop_length=int(hop_length),
            stft=stft,
            freqs=np.asarray(freqs, dtype=float),
        )

    def pitch_support(
        self,
        time_s: float,
        midi_pitch: int,
        *,
        start: float = 0.01,
        end: float = 0.22,
        harmonics: int = 6,
        fmin: float = 50.0,
        fmax: float = 5000.0,
    ) -> float:
        """Return how much harmonic energy matches ``midi_pitch``.

        The value is the sum of the strongest bins around the expected
        fundamental and its first harmonics, divided by the total energy in the
        analysis band.  It is intentionally a broad measure: a plucked note has
        a stable harmonic series, while a knock or muted hit does not.
        """
        if self.stft.size == 0:
            return 0.0
        start_frame = max(0, int(round((float(time_s) + float(start)) * self.sr / self.hop_length)))
        end_frame = min(
            self.stft.shape[1],
            int(round((float(time_s) + float(end)) * self.sr / self.hop_length)),
        )
        if end_frame <= start_frame:
            return 0.0

        window = self.stft[:, start_frame:end_frame]
        band = (self.freqs >= fmin) & (self.freqs <= fmax)
        total = float(np.mean(np.sum(window[band, :], axis=0)))
        if total <= 1e-9:
            return 0.0

        f0 = 440.0 * (2.0 ** ((float(midi_pitch) - 69.0) / 12.0))
        peaks: list[float] = []
        for harmonic in range(1, int(harmonics) + 1):
            center = f0 * harmonic
            if center > fmax:
                break
            low = center * (2.0 ** (-1.0 / 12.0))
            high = center * (2.0 ** (1.0 / 12.0))
            bins = (self.freqs >= low) & (self.freqs <= high)
            if not np.any(bins):
                continue
            peaks.append(float(np.mean(np.max(window[bins, :], axis=0))))
        return float(sum(peaks) / (total + 1e-9))


def event_audio_class(event: dict[str, Any]) -> str:
    """Return the pre-reconciliation audio class when available."""
    return str(event.get("audio_class") or event.get("class") or "")


def best_event_for_note(
    note: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    grid_ticks: int = 240,
) -> dict[str, Any] | None:
    """Find the audio onset that best represents a MIDI note.

    Multiple onsets can be matched to the same MIDI note during alignment
    (especially when a transient produces several detections).  Prefer the
    onset closest to the MIDI tick, then the strongest one.
    """
    target_tick = int(note.get("t", 0))
    target_pitch = int(note.get("pitch", -1))
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for event in events:
        if not event.get("matched"):
            continue
        matched = event.get("matched_note") or {}
        if int(matched.get("pitch", -1)) != target_pitch:
            continue
        tick = int(matched.get("tick", event.get("tick", -1)))
        distance = abs(tick - target_tick)
        if distance > max(1, int(grid_ticks)):
            continue
        candidates.append((distance, -float(event.get("strength", 0.0) or 0.0), event))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def nearest_unmatched_event_for_note(
    note: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    grid_ticks: int = 240,
) -> dict[str, Any] | None:
    """Find a nearby percussive onset that did not match a MIDI note.

    Alignment can miss a transient when the audio and MIDI clocks drift by a
    fraction of a grid cell.  The note still needs validation, so an unmatched
    dead/percussive onset within one grid cell is used as evidence.
    """
    target_tick = int(note.get("t", 0))
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for event in events:
        if event.get("matched"):
            continue
        if event_audio_class(event) not in {"dead", "percussive"}:
            continue
        tick = int(event.get("tick", -1))
        if tick < 0:
            continue
        distance = abs(tick - target_tick)
        if distance > max(1, int(grid_ticks)):
            continue
        candidates.append(
            (distance, -float(event.get("strength", 0.0) or 0.0), event)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def local_median_strength(
    events: list[dict[str, Any]],
    *,
    around_time: float | None = None,
    window_sec: float = 4.0,
    fallback: float = 1.0,
) -> float:
    """Median strength of matched onsets, optionally around one time."""
    values = [
        float(event.get("strength", 0.0) or 0.0)
        for event in events
        if event.get("matched") and float(event.get("strength", 0.0) or 0.0) > 0.0
        and (
            around_time is None
            or abs(float(event.get("time", 0.0)) - float(around_time)) <= float(window_sec)
        )
    ]
    return float(np.median(values)) if values else float(fallback)


def non_note_score(event: dict[str, Any], pitch_support: float) -> float:
    """Estimate whether an onset lacks evidence of a real pitched note.

    The score combines harmonicity, periodicity, and expected-pitch support.
    The raw audio class is a small prior, not the decision.  A value near 1 is
    a non-note; a value near 0 is a normal plucked note.
    """
    harmonic = max(0.0, min(1.0, float(event.get("harmonic_ratio", 0.0) or 0.0)))
    voiced = max(0.0, min(1.0, float(event.get("voiced_prob", 0.0) or 0.0)))
    support = max(0.0, min(1.0, float(pitch_support)))

    score = (
        0.55 * (1.0 - min(1.0, harmonic / 0.60))
        + 0.30 * (1.0 - min(1.0, voiced / 0.60))
        + 0.15 * (1.0 - min(1.0, support / 0.20))
    )
    audio_class = event_audio_class(event)
    if audio_class in {"dead", "percussive"}:
        score += 0.08
    elif audio_class == "pitched":
        score -= 0.10
    return float(max(0.0, min(1.0, score)))
