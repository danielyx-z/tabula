"""Note-presence evidence for polyphonic guitar audio.

The heuristic in :mod:`tabfeatures.non_note` asks whether a note's expected
pitch has energy.  In a chord that is fragile: another note, a harmonic, or a
reverb tail can supply the energy.  This module asks a stronger question:
*can the observed spectrum be explained without this note?*

Each note contributes a harmonic template (fundamental plus overtones).  A
non-negative least-squares fit explains the spectrum with all chord templates
plus a broadband noise template.  A note whose coefficient collapses when the
other notes are present is likely dead; a note that keeps a strong coefficient
is likely pitched.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import librosa
import numpy as np
from scipy.optimize import nnls

from .evidence import AudioAnalysis


def _midi_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))


def _harmonic_template(
    freqs: np.ndarray,
    pitch: int,
    *,
    harmonics: int = 8,
    width_semitones: float = 0.38,
    fmax: float = 5000.0,
) -> np.ndarray:
    """Return a smooth harmonic-series template for one MIDI pitch."""
    f0 = _midi_hz(pitch)
    template = np.zeros_like(freqs, dtype=float)
    for harmonic in range(1, int(harmonics) + 1):
        center = f0 * harmonic
        if center > fmax:
            break
        distance = 12.0 * np.log2(np.maximum(freqs, 1e-9) / center)
        weight = 1.0 / float(harmonic)
        template += weight * np.exp(-0.5 * (distance / width_semitones) ** 2)
    norm = float(np.linalg.norm(template))
    if norm > 0.0:
        template /= norm
    return template


@dataclass
class NotePresenceAnalyzer:
    """Fit chord spectra with note-wise harmonic templates."""

    analysis: AudioAnalysis
    n_fft: int = 4096
    hop_length: int = 256
    harmonics: int = 8
    width_semitones: float = 0.38
    fmin: float = 55.0
    fmax: float = 5000.0
    stft: np.ndarray = field(init=False)
    freqs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.stft = np.abs(
            librosa.stft(
                self.analysis.y,
                n_fft=int(self.n_fft),
                hop_length=int(self.hop_length),
            )
        )
        self.freqs = librosa.fft_frequencies(
            sr=self.analysis.sr, n_fft=int(self.n_fft)
        )

    def _spectrum(
        self,
        time_s: float,
        *,
        start: float = 0.07,
        end: float = 0.30,
    ) -> np.ndarray | None:
        start_frame = max(
            0, int(round((float(time_s) + float(start)) * self.analysis.sr / self.hop_length))
        )
        end_frame = min(
            self.stft.shape[1],
            int(round((float(time_s) + float(end)) * self.analysis.sr / self.hop_length)),
        )
        if end_frame <= start_frame:
            return None
        # Median over the sustain is less sensitive to the attack click and to
        # a single frame of vibrato/reverb.
        return np.median(self.stft[:, start_frame:end_frame], axis=1)

    def group_scores(
        self,
        time_s: float,
        notes: list[dict[str, Any]],
        *,
        next_time_s: float | None = None,
        start: float = 0.07,
        end: float = 0.30,
    ) -> dict[tuple[int, int], dict[str, float]]:
        """Return per-note presence evidence for one chord/onset.

        The score is ``coefficient / (coefficient + noise)``.  It is high when
        the note's harmonic series explains part of the spectrum that the other
        chord notes and the noise template do not explain.
        """
        if not notes:
            return {}
        if next_time_s is not None:
            end = max(0.12, min(float(end), float(next_time_s) - float(time_s) - 0.03))
        spectrum = self._spectrum(time_s, start=start, end=end)
        if spectrum is None:
            return {}

        band = (self.freqs >= self.fmin) & (self.freqs <= self.fmax)
        if not np.any(band):
            return {}
        x = np.maximum(spectrum[band], 0.0)
        norm = float(np.linalg.norm(x))
        if norm <= 1e-9:
            return {
                (int(note["t"]), int(note["s"])): {
                    "presence": 0.0,
                    "coefficient": 0.0,
                    "noise": 0.0,
                    "explained": 0.0,
                }
                for note in notes
            }
        x = x / norm

        unique_pitches = sorted({int(note["pitch"]) for note in notes})
        columns = [
            _harmonic_template(
                self.freqs[band],
                pitch,
                harmonics=self.harmonics,
                width_semitones=self.width_semitones,
                fmax=self.fmax,
            )
            for pitch in unique_pitches
        ]
        noise = np.ones(int(np.sum(band)), dtype=float)
        noise /= max(float(np.linalg.norm(noise)), 1e-9)
        matrix = np.column_stack([*columns, noise])
        try:
            coefficients, residual = nnls(matrix, x)
        except Exception:
            return {}
        explained = max(
            0.0,
            min(1.0, 1.0 - (float(residual) ** 2) / max(1.0, 1.0)),
        )

        by_pitch = {
            pitch: float(coefficients[index])
            for index, pitch in enumerate(unique_pitches)
        }
        noise_coefficient = float(coefficients[-1])
        result: dict[tuple[int, int], dict[str, float]] = {}
        for note in notes:
            pitch = int(note["pitch"])
            coefficient = by_pitch.get(pitch, 0.0)
            presence = coefficient / (
                coefficient + noise_coefficient + 1e-9
            )
            result[(int(note["t"]), int(note["s"]))] = {
                "presence": float(max(0.0, min(1.0, presence))),
                "coefficient": coefficient,
                "noise": noise_coefficient,
                "explained": float(explained),
            }
        return result
