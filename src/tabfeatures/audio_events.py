"""Audio event detection for the stitching pipeline.

The goal is intentionally narrow: find percussive onsets, describe them well
enough to tell a dead note / knock from an ordinary plucked note, and leave the
final decision to MIDI reconciliation (a percussive onset that lines up with a
MIDI note is a pluck attack; an unmatched one is a dead-note candidate).

HPSS is only used to make the transient detector less sensitive to sustained
harmonics.  It is not treated as a semantic separator: a plucked guitar note
also has a percussive component, which is exactly why reconciliation is
mandatory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: str | Path, sr: int | None = 22050) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(actual_sr)


def split_hpss(
    y: np.ndarray,
    *,
    kernel_size: int = 31,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(
        y, kernel_size=kernel_size, margin=margin
    )
    return harmonic, percussive


def _segment(
    y: np.ndarray,
    sr: int,
    time_s: float,
    *,
    before: float = 0.03,
    after: float = 0.18,
) -> np.ndarray | None:
    start = max(0, int(round((time_s - before) * sr)))
    end = min(len(y), int(round((time_s + after) * sr)))
    if end - start < 128:
        return None
    return y[start:end]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _f0_confidence(seg: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (median f0, max voiced probability) for a short segment."""
    n = len(seg)
    if n < 256:
        return 0.0, 0.0

    frame_length = 2048
    while frame_length > n and frame_length > 256:
        frame_length //= 2
    if frame_length < 256:
        return 0.0, 0.0

    try:
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            seg,
            fmin=float(librosa.note_to_hz("E2")),
            fmax=float(librosa.note_to_hz("E6")),
            sr=sr,
            frame_length=frame_length,
            hop_length=max(64, frame_length // 4),
        )
    except Exception:
        return 0.0, 0.0

    valid_prob = np.asarray(voiced_prob, dtype=float)
    valid_prob = valid_prob[np.isfinite(valid_prob)]
    max_prob = float(np.max(valid_prob)) if valid_prob.size else 0.0

    valid_f0 = np.asarray(f0, dtype=float)
    valid_f0 = valid_f0[np.isfinite(valid_f0) & (valid_f0 > 0)]
    median_f0 = float(np.median(valid_f0)) if valid_f0.size else 0.0
    return median_f0, max_prob


def _segment_features(seg: np.ndarray, sr: int) -> dict[str, float]:
    """Cheap timbre descriptors for a short onset window."""
    # Spectral flatness: near 0 for a clean tone, higher for noise/clicks.
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))
    )
    bandwidth = float(
        np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))
    )
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(seg)))
    rms = float(np.mean(librosa.feature.rms(y=seg)))

    harmonic_ratio = 1.0 - min(1.0, max(0.0, flatness))
    if len(seg) >= 1024:
        try:
            harmonic, percussive = librosa.effects.hpss(seg, kernel_size=31)
            eh = float(np.sum(np.asarray(harmonic, dtype=np.float64) ** 2))
            ep = float(np.sum(np.asarray(percussive, dtype=np.float64) ** 2))
            harmonic_ratio = eh / (eh + ep + 1e-12)
        except Exception:
            pass

    f0, voiced_prob = _f0_confidence(seg, sr)
    return {
        "flatness": _safe_float(flatness),
        "centroid_hz": _safe_float(centroid),
        "bandwidth_hz": _safe_float(bandwidth),
        "zcr": _safe_float(zcr),
        "rms": _safe_float(rms),
        "harmonic_ratio": _safe_float(harmonic_ratio),
        "f0_hz": _safe_float(f0),
        "voiced_prob": _safe_float(voiced_prob),
    }


def _classify(features: dict[str, float]) -> tuple[str, float]:
    """Classify an onset before MIDI reconciliation.

    ``pitched`` events are deliberately given zero confidence: if the MIDI
    model missed a note we do not want to invent a dead note on top of it.
    """
    flatness = features["flatness"]
    voiced_prob = features["voiced_prob"]
    harmonic_ratio = features["harmonic_ratio"]

    if voiced_prob >= 0.55 and flatness <= 0.18 and harmonic_ratio >= 0.55:
        return "pitched", 0.0

    dead_score = (
        0.45 * min(1.0, flatness / 0.35)
        + 0.35 * (1.0 - min(1.0, voiced_prob))
        + 0.20 * (1.0 - min(1.0, harmonic_ratio))
    )
    dead_score = min(1.0, max(0.0, dead_score))
    if dead_score >= 0.50:
        return "dead", dead_score
    return "percussive", dead_score


def detect_audio_events(
    audio_path: str | Path,
    *,
    sr: int = 22050,
    hop_length: int = 512,
    kernel_size: int = 31,
    margin: float = 1.0,
    delta: float = 0.05,
    wait: int = 2,
    before: float = 0.03,
    after: float = 0.18,
    return_stems: bool = False,
) -> dict[str, Any]:
    """Detect percussive onsets and describe each one.

    Returns a JSON-serialisable dict (plus optional stems when requested).
    """
    y, sr = load_mono(audio_path, sr=sr)
    if y.size == 0:
        return {"sr": sr, "duration": 0.0, "events": []}

    _harmonic, percussive = split_hpss(
        y, kernel_size=kernel_size, margin=margin
    )

    # The percussive stem gives a cleaner transient envelope; timbre features
    # are measured on the full mix so dead-note body resonance is visible.
    onset_env = librosa.onset.onset_strength(
        y=percussive, sr=sr, hop_length=hop_length
    )
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        delta=delta,
        wait=wait,
    )

    events: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        frame_i = int(frame)
        time_s = float(librosa.frames_to_time(frame_i, sr=sr, hop_length=hop_length))
        strength = _safe_float(onset_env[frame_i] if frame_i < len(onset_env) else 0.0)
        seg = _segment(y, sr, time_s, before=before, after=after)
        if seg is None:
            continue
        features = _segment_features(seg, sr)
        klass, confidence = _classify(features)
        events.append(
            {
                "index": index,
                "time": round(time_s, 6),
                "strength": round(strength, 6),
                **{k: round(v, 6) for k, v in features.items()},
                "class": klass,
                "confidence": round(confidence, 6),
            }
        )

    result: dict[str, Any] = {
        "sr": sr,
        "duration": round(float(len(y) / sr), 6),
        "events": events,
    }
    if return_stems:
        result["harmonic"] = _harmonic
        result["percussive"] = percussive
    return result


def write_audio_events(path: str | Path, result: dict[str, Any]) -> None:
    """Write the JSON-serialisable part of :func:`detect_audio_events`."""
    payload = {
        "sr": result.get("sr"),
        "duration": result.get("duration"),
        "events": result.get("events", []),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_stems(out_dir: str | Path, result: dict[str, Any]) -> None:
    """Optionally write HPSS stems for inspection."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = int(result.get("sr") or 22050)
    if "harmonic" in result:
        sf.write(out / "harmonic.wav", result["harmonic"], sr, subtype="PCM_16")
    if "percussive" in result:
        sf.write(out / "percussive.wav", result["percussive"], sr, subtype="PCM_16")
