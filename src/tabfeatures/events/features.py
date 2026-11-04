"""Audio features shared by event-model training and inference.

The feature set is deliberately small and interpretable.  XGBoost consumes
these tabular measurements; it never sees the raw waveform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

FEATURE_NAMES: tuple[str, ...] = (
    "rms_db",
    "peak",
    "crest",
    "flatness",
    "centroid_hz",
    "bandwidth_hz",
    "rolloff_hz",
    "zcr",
    "low_ratio",
    "mid_ratio",
    "high_ratio",
    "harmonic_ratio",
    "percussive_ratio",
    "low_harmonic_ratio",
    "low_percussive_ratio",
    "f0_hz",
    "voiced_prob",
    "f0_std",
    "pitch_support",
    "decay_db",
    "attack_strength",
    "spectral_flux",
    "pitch",
    "string",
)


def load_audio(path: str | Path, *, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(actual_sr)


def _safe(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _window(
    y: np.ndarray,
    sr: int,
    onset_s: float,
    *,
    before: float = 0.03,
    after: float = 0.35,
) -> np.ndarray:
    start = max(0, int(round((float(onset_s) - float(before)) * sr)))
    end = min(len(y), int(round((float(onset_s) + float(after)) * sr)))
    if end - start < 128:
        return np.asarray([], dtype=np.float32)
    return np.asarray(y[start:end], dtype=np.float32)


def _band_ratios(seg: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    n_fft = 2048
    if len(seg) < n_fft:
        n_fft = max(256, 2 ** int(np.floor(np.log2(max(256, len(seg))))))
    stft = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=max(64, n_fft // 4)))
    power = np.asarray(stft, dtype=np.float64) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = float(np.sum(power)) + 1e-12
    low = float(np.sum(power[(freqs >= 60.0) & (freqs <= 250.0)]))
    mid = float(np.sum(power[(freqs > 250.0) & (freqs <= 1200.0)]))
    high = float(np.sum(power[(freqs > 1200.0) & (freqs <= 5000.0)]))
    flux = float(np.mean(np.abs(np.diff(stft, axis=1)))) if stft.shape[1] > 1 else 0.0
    return low / total, mid / total, high / total, flux


def _hpss_ratio(seg: np.ndarray) -> tuple[float, float]:
    if len(seg) < 512:
        return float("nan"), float("nan")
    try:
        harmonic, percussive = librosa.effects.hpss(seg, kernel_size=31)
        eh = float(np.sum(np.asarray(harmonic, dtype=np.float64) ** 2))
        ep = float(np.sum(np.asarray(percussive, dtype=np.float64) ** 2))
        total = eh + ep + 1e-12
        return eh / total, ep / total
    except Exception:
        return float("nan"), float("nan")


def _low_hpss_ratio(seg: np.ndarray, sr: int) -> tuple[float, float]:
    if len(seg) < 256:
        return float("nan"), float("nan")
    try:
        sos = butter(4, 250.0, btype="lowpass", fs=sr, output="sos")
        low = sosfiltfilt(sos, seg)
        return _hpss_ratio(np.asarray(low, dtype=np.float32))
    except Exception:
        return float("nan"), float("nan")


def _f0_features(seg: np.ndarray, sr: int) -> tuple[float, float, float]:
    if len(seg) < 256:
        return float("nan"), float("nan"), float("nan")
    frame_length = 2048
    while frame_length > len(seg) and frame_length > 256:
        frame_length //= 2
    if frame_length < 256:
        return float("nan"), float("nan"), float("nan")
    try:
        f0, _voiced, voiced_prob = librosa.pyin(
            seg,
            fmin=float(librosa.note_to_hz("E2")),
            fmax=float(librosa.note_to_hz("E6")),
            sr=sr,
            frame_length=frame_length,
            hop_length=max(64, frame_length // 4),
        )
    except Exception:
        return float("nan"), float("nan"), float("nan")
    valid = np.asarray(f0, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    probs = np.asarray(voiced_prob, dtype=float)
    probs = probs[np.isfinite(probs)]
    voiced = float(np.max(probs)) if probs.size else float("nan")
    if not valid.size:
        return float("nan"), voiced, float("nan")
    return float(np.median(valid)), voiced, float(np.std(valid))


def _pitch_support(seg: np.ndarray, sr: int, pitch: int | None) -> float:
    if pitch is None or len(seg) < 256:
        return float("nan")
    n_fft = 2048
    if len(seg) < n_fft:
        n_fft = max(256, 2 ** int(np.floor(np.log2(max(256, len(seg))))))
    stft = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=max(64, n_fft // 4)))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= 50.0) & (freqs <= 5000.0)
    total = float(np.mean(np.sum(stft[band, :], axis=0))) if np.any(band) else 0.0
    if total <= 1e-9:
        return 0.0
    f0 = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
    peaks: list[float] = []
    for harmonic in range(1, 7):
        center = f0 * harmonic
        if center > 5000.0:
            break
        low = center * (2.0 ** (-1.0 / 12.0))
        high = center * (2.0 ** (1.0 / 12.0))
        bins = (freqs >= low) & (freqs <= high)
        if np.any(bins):
            peaks.append(float(np.mean(np.max(stft[bins, :], axis=0))))
    return float(sum(peaks) / (total + 1e-9))


def extract_event_features(
    y: np.ndarray,
    sr: int,
    onset_s: float,
    *,
    pitch: int | None = None,
    string: int | None = None,
    before_s: float = 0.03,
    after_s: float = 0.35,
) -> dict[str, float]:
    """Measure one onset window.

    ``pitch`` and ``string`` are optional.  The browser-labeller data does not
    have them, while the stitcher can pass the matched MIDI note and tab string.
    Missing values become ``NaN``; XGBoost handles them natively.
    """
    seg = _window(y, sr, onset_s, before=before_s, after=after_s)
    if not len(seg):
        return {name: float("nan") for name in FEATURE_NAMES}

    rms = float(np.mean(librosa.feature.rms(y=seg)))
    peak = float(np.max(np.abs(seg)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(seg)))
    low, mid, high, flux = _band_ratios(seg, sr)
    harmonic, percussive = _hpss_ratio(seg)
    low_harmonic, low_percussive = _low_hpss_ratio(seg, sr)
    f0, voiced, f0_std = _f0_features(seg, sr)
    support = _pitch_support(seg, sr, pitch)

    frame = max(1, int(0.02 * sr))
    early_end = min(len(seg), max(frame, int(0.08 * sr)))
    late_start = min(len(seg) - 1, int(0.18 * sr))
    early = float(np.sqrt(np.mean(np.square(seg[:early_end], dtype=np.float64)) + 1e-12))
    late = float(np.sqrt(np.mean(np.square(seg[late_start:], dtype=np.float64)) + 1e-12))
    decay_db = 20.0 * np.log10((late + 1e-9) / (early + 1e-9))
    try:
        onset_strength = float(
            np.max(librosa.onset.onset_strength(y=seg, sr=sr, hop_length=128))
        )
    except Exception:
        onset_strength = float("nan")

    return {
        "rms_db": 20.0 * np.log10(rms + 1e-9),
        "peak": peak,
        "crest": peak / (rms + 1e-9),
        "flatness": flatness,
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff,
        "zcr": zcr,
        "low_ratio": low,
        "mid_ratio": mid,
        "high_ratio": high,
        "harmonic_ratio": harmonic,
        "percussive_ratio": percussive,
        "low_harmonic_ratio": low_harmonic,
        "low_percussive_ratio": low_percussive,
        "f0_hz": f0,
        "voiced_prob": voiced,
        "f0_std": f0_std,
        "pitch_support": support,
        "decay_db": _safe(decay_db),
        "attack_strength": _safe(onset_strength),
        "spectral_flux": _safe(flux),
        "pitch": _safe(pitch),
        "string": _safe(string),
    }


def feature_vector(features: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [_safe(features.get(name)) for name in FEATURE_NAMES],
        dtype=np.float32,
    )


def feature_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([feature_vector(row.get("features", row)) for row in rows])
