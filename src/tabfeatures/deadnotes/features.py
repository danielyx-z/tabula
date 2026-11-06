"""Midi-anchored audio features for the dead-note classifier.

Every candidate is a MIDI note.  The audio window starts at the MIDI onset; no
audio onset detector decides which notes exist.  Chord members share the group
features and add per-note harmonic/presence evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from tabfeatures.evidence import AudioAnalysis
from tabfeatures.note_presence import NotePresenceAnalyzer

_SHARED_NAMES = [
    "g_flatness",
    "g_centroid_hz",
    "g_bandwidth_hz",
    "g_rolloff_hz",
    "g_zcr",
    "g_rms",
    "g_rms_std",
    "g_rms_slope",
    "g_attack_db",
    "g_decay_db",
    "g_low_ratio",
    "g_mid_ratio",
    "g_high_ratio",
    "g_harmonic_ratio",
    "g_percussive_ratio",
    "g_spectral_flux",
    "g_f0_hz",
    "g_voiced_prob",
    "g_f0_std",
    "g_chord_size",
    "g_pitch_range",
    "g_bass_pitch",
    "g_local_ioi",
    "g_prev_ioi",
]

_NOTE_NAMES = [
    "n_pitch",
    "n_velocity",
    "n_chord_rank",
    "n_interval_to_lower",
    "n_semitones_above_bass",
    "n_harm_support_attack",
    "n_harm_support_sustain",
    "n_f0_band_attack",
    "n_f0_band_sustain",
    "n_f0_decay_db",
    "n_presence",
    "n_coefficient",
    "n_noise",
    "n_explained",
    "n_presence_relative",
    "n_harm_support_relative",
    "n_f0_band_relative",
    "n_presence_rank",
]


def _expand_mfcc(prefix: str, values: np.ndarray, out: dict[str, float]) -> None:
    for index, value in enumerate(np.asarray(values, dtype=float).reshape(-1)):
        out[f"{prefix}_{index}"] = float(value)


def feature_names() -> list[str]:
    """Full ordered feature vector, including per-MFCC coefficients."""
    names = list(_SHARED_NAMES)
    names += [f"g_mfcc_mean_{i}" for i in range(13)]
    names += [f"g_mfcc_std_{i}" for i in range(13)]
    names += list(_NOTE_NAMES)
    return names


FEATURE_NAMES = tuple(feature_names())


def _safe(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    attack_before_s: float = 0.02
    attack_after_s: float = 0.08
    sustain_start_s: float = 0.03
    sustain_end_s: float = 0.20
    next_guard_s: float = 0.02
    chord_window_ms: float = 30.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sr": int(self.sr),
            "n_fft": int(self.n_fft),
            "hop_length": int(self.hop_length),
            "attack_before_s": float(self.attack_before_s),
            "attack_after_s": float(self.attack_after_s),
            "sustain_start_s": float(self.sustain_start_s),
            "sustain_end_s": float(self.sustain_end_s),
            "next_guard_s": float(self.next_guard_s),
            "chord_window_ms": float(self.chord_window_ms),
        }


class AudioContext:
    """Loaded audio plus an STFT used for all groups in a session."""

    def __init__(self, audio_path: str | Path, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.y, self.sr = librosa.load(
            str(audio_path), sr=int(self.config.sr), mono=True
        )
        self.y = np.asarray(self.y, dtype=np.float32)
        self.stft = np.abs(
            librosa.stft(
                self.y,
                n_fft=int(self.config.n_fft),
                hop_length=int(self.config.hop_length),
            )
        )
        self.freqs = librosa.fft_frequencies(
            sr=int(self.sr), n_fft=int(self.config.n_fft)
        )
        self._analysis = AudioAnalysis(
            y=self.y,
            sr=int(self.sr),
            hop_length=int(self.config.hop_length),
            stft=self.stft,
            freqs=self.freqs,
        )
        self._presence = NotePresenceAnalyzer(self._analysis)

    @property
    def duration(self) -> float:
        return float(len(self.y) / max(1, self.sr))

    def _frames(self, start_s: float, end_s: float) -> tuple[int, int]:
        start = max(
            0,
            int(round(float(start_s) * self.sr / self.config.hop_length)),
        )
        end = min(
            self.stft.shape[1],
            int(round(float(end_s) * self.sr / self.config.hop_length)),
        )
        return start, end

    def _spectrum(self, start_s: float, end_s: float) -> np.ndarray | None:
        start, end = self._frames(start_s, end_s)
        if end <= start:
            return None
        return np.median(self.stft[:, start:end], axis=1)

    def _segment(self, start_s: float, end_s: float) -> np.ndarray | None:
        start = max(0, int(round(float(start_s) * self.sr)))
        end = min(len(self.y), int(round(float(end_s) * self.sr)))
        if end - start < 128:
            return None
        return self.y[start:end]

    def _harmonic_support(self, spectrum: np.ndarray | None, pitch: int) -> float:
        if spectrum is None:
            return float("nan")
        band = (self.freqs >= 50.0) & (self.freqs <= 5000.0)
        total = float(np.sum(spectrum[band])) + 1e-12
        f0 = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
        peaks = 0.0
        for harmonic in range(1, 7):
            center = f0 * harmonic
            if center > 5000.0:
                break
            low = center * (2.0 ** (-1.0 / 12.0))
            high = center * (2.0 ** (1.0 / 12.0))
            bins = (self.freqs >= low) & (self.freqs <= high)
            if np.any(bins):
                peaks += float(np.max(spectrum[bins]))
        return float(peaks / total)

    def _f0_band_ratio(self, spectrum: np.ndarray | None, pitch: int) -> float:
        if spectrum is None:
            return float("nan")
        f0 = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
        low = f0 * (2.0 ** (-1.0 / 3.0))
        high = f0 * (2.0 ** (1.0 / 3.0))
        band = (self.freqs >= 50.0) & (self.freqs <= 5000.0)
        total = float(np.sum(spectrum[band])) + 1e-12
        target = (self.freqs >= low) & (self.freqs <= high)
        return float(np.sum(spectrum[target]) / total)

    def _shared_features(
        self,
        group: dict[str, Any],
        segment: np.ndarray,
        *,
        next_time_s: float | None,
    ) -> dict[str, float]:
        cfg = self.config
        onset = float(group["time_s"])
        start_s = max(0.0, onset - cfg.attack_before_s)
        end_s = min(self.duration, onset + cfg.attack_after_s)
        if next_time_s is not None and next_time_s > onset:
            end_s = min(end_s, float(next_time_s) - cfg.next_guard_s)
        if end_s <= start_s:
            end_s = min(self.duration, start_s + 0.02)

        rms = np.asarray(librosa.feature.rms(y=segment), dtype=float).reshape(-1)
        rms_mean = float(np.mean(rms)) if rms.size else float("nan")
        rms_std = float(np.std(rms)) if rms.size else float("nan")
        if rms.size >= 2:
            x = np.linspace(0.0, 1.0, rms.size)
            slope = float(np.polyfit(x, rms, 1)[0])
        else:
            slope = float("nan")
        first = rms[: max(1, min(rms.size, 3))]
        last = rms[max(0, rms.size - max(1, min(rms.size, 3))) :]
        early = float(np.mean(first)) if first.size else float("nan")
        late = float(np.mean(last)) if last.size else float("nan")
        attack_db = 20.0 * np.log10((early + 1e-9) / (rms_mean + 1e-9))
        decay_db = 20.0 * np.log10((late + 1e-9) / (early + 1e-9))

        mfcc = librosa.feature.mfcc(
            y=segment, sr=self.sr, n_mfcc=13,
            n_fft=min(cfg.n_fft, 2 ** int(np.floor(np.log2(max(256, len(segment)))))),
            hop_length=cfg.hop_length,
        )
        mfcc_mean = np.mean(mfcc, axis=1) if mfcc.size else np.full(13, np.nan)
        mfcc_std = np.std(mfcc, axis=1) if mfcc.size else np.full(13, np.nan)

        flatness = float(np.mean(librosa.feature.spectral_flatness(y=segment)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)))
        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=segment, sr=self.sr, roll_percent=0.85
                )
            )
        )
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(segment)))

        spec = self._spectrum(start_s, end_s)
        if spec is None:
            low_ratio = mid_ratio = high_ratio = flux = float("nan")
        else:
            band = (self.freqs >= 50.0) & (self.freqs <= 5000.0)
            total = float(np.sum(spec[band])) + 1e-12
            low_ratio = float(
                np.sum(spec[(self.freqs >= 60.0) & (self.freqs <= 250.0)]) / total
            )
            mid_ratio = float(
                np.sum(spec[(self.freqs > 250.0) & (self.freqs <= 1200.0)]) / total
            )
            high_ratio = float(
                np.sum(spec[(self.freqs > 1200.0) & (self.freqs <= 5000.0)]) / total
            )
            s0, s1 = self._frames(start_s, end_s)
            window = self.stft[:, s0:s1]
            flux = (
                float(np.mean(np.abs(np.diff(window, axis=1))))
                if window.shape[1] > 1
                else 0.0
            )

        harmonic_ratio = percussive_ratio = float("nan")
        if len(segment) >= 256:
            try:
                harmonic, percussive = librosa.effects.hpss(segment, kernel_size=31)
                eh = float(np.sum(np.asarray(harmonic, dtype=np.float64) ** 2))
                ep = float(np.sum(np.asarray(percussive, dtype=np.float64) ** 2))
                total = eh + ep + 1e-12
                harmonic_ratio = eh / total
                percussive_ratio = ep / total
            except Exception:
                pass

        f0, voiced, f0_std = self._f0(segment)
        notes = group.get("notes", [])
        pitches = [int(note["pitch"]) for note in notes if note.get("pitch") is not None]
        prev_ioi = float("nan")
        if group.get("previous_time_s") is not None:
            prev_ioi = max(0.0, onset - float(group["previous_time_s"]))
        local_ioi = float("nan")
        if next_time_s is not None:
            local_ioi = max(0.0, float(next_time_s) - onset)

        out: dict[str, float] = {
            "g_flatness": _safe(flatness),
            "g_centroid_hz": _safe(centroid),
            "g_bandwidth_hz": _safe(bandwidth),
            "g_rolloff_hz": _safe(rolloff),
            "g_zcr": _safe(zcr),
            "g_rms": _safe(rms_mean),
            "g_rms_std": _safe(rms_std),
            "g_rms_slope": _safe(slope),
            "g_attack_db": _safe(attack_db),
            "g_decay_db": _safe(decay_db),
            "g_low_ratio": _safe(low_ratio),
            "g_mid_ratio": _safe(mid_ratio),
            "g_high_ratio": _safe(high_ratio),
            "g_harmonic_ratio": _safe(harmonic_ratio),
            "g_percussive_ratio": _safe(percussive_ratio),
            "g_spectral_flux": _safe(flux),
            "g_f0_hz": _safe(f0),
            "g_voiced_prob": _safe(voiced),
            "g_f0_std": _safe(f0_std),
            "g_chord_size": float(len(notes)),
            "g_pitch_range": float(max(pitches) - min(pitches)) if pitches else float("nan"),
            "g_bass_pitch": float(min(pitches) if pitches else float("nan")),
            "g_local_ioi": _safe(local_ioi),
            "g_prev_ioi": _safe(prev_ioi),
        }
        _expand_mfcc("g_mfcc_mean", mfcc_mean, out)
        _expand_mfcc("g_mfcc_std", mfcc_std, out)
        return out

    def _f0(self, segment: np.ndarray) -> tuple[float, float, float]:
        if len(segment) < int(0.06 * self.sr):
            return float("nan"), float("nan"), float("nan")
        frame_length = 2048
        while frame_length > len(segment) and frame_length > 256:
            frame_length //= 2
        if frame_length < 256:
            return float("nan"), float("nan"), float("nan")
        try:
            f0, _voiced, voiced_prob = librosa.pyin(
                segment,
                fmin=float(librosa.note_to_hz("E2")),
                fmax=float(librosa.note_to_hz("E6")),
                sr=self.sr,
                frame_length=frame_length,
                hop_length=max(64, frame_length // 4),
            )
        except Exception:
            return float("nan"), float("nan"), float("nan")
        valid = np.asarray(f0, dtype=float)
        valid = valid[np.isfinite(valid) & (valid > 0)]
        probs = np.asarray(voiced_prob, dtype=float)
        probs = probs[np.isfinite(probs)]
        if not valid.size:
            return float("nan"), (float(np.max(probs)) if probs.size else float("nan")), float("nan")
        return (
            float(np.median(valid)),
            float(np.max(probs)) if probs.size else float("nan"),
            float(np.std(valid)),
        )

    def group_rows(
        self,
        group: dict[str, Any],
        *,
        next_time_s: float | None = None,
        previous_time_s: float | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        onset = float(group["time_s"])
        segment = self._segment(
            max(0.0, onset - cfg.attack_before_s),
            min(self.duration, onset + cfg.attack_after_s),
        )
        if segment is None:
            return [
                {"features": {name: float("nan") for name in FEATURE_NAMES}, "note": note}
                for note in group.get("notes", [])
            ]

        group = dict(group)
        group["previous_time_s"] = previous_time_s
        shared = self._shared_features(
            group, segment, next_time_s=next_time_s
        )

        sustain_start = onset + cfg.sustain_start_s
        sustain_end = onset + cfg.sustain_end_s
        if next_time_s is not None:
            sustain_end = min(sustain_end, float(next_time_s) - cfg.next_guard_s)
        sustain_end = max(sustain_start + 0.02, sustain_end)
        attack_spectrum = self._spectrum(
            max(0.0, onset - cfg.attack_before_s),
            min(self.duration, onset + cfg.attack_after_s),
        )
        sustain_spectrum = self._spectrum(sustain_start, sustain_end)

        notes = group.get("notes", [])
        proxy_notes = [
            {"t": int(note.get("tick", 0)), "s": index + 1, "pitch": int(note["pitch"])}
            for index, note in enumerate(notes)
        ]
        try:
            presence = self._presence.group_scores(
                onset,
                proxy_notes,
                next_time_s=next_time_s,
                start=cfg.sustain_start_s,
                end=cfg.sustain_end_s,
            )
        except Exception:
            presence = {}

        bass_pitch = min(int(note["pitch"]) for note in notes) if notes else 0
        raw: list[dict[str, Any]] = []
        for index, note in enumerate(notes):
            pitch = int(note["pitch"])
            attack_support = self._harmonic_support(attack_spectrum, pitch)
            sustain_support = self._harmonic_support(sustain_spectrum, pitch)
            attack_band = self._f0_band_ratio(attack_spectrum, pitch)
            sustain_band = self._f0_band_ratio(sustain_spectrum, pitch)
            if (
                attack_band == attack_band
                and sustain_band == sustain_band
                and attack_band > 0
            ):
                decay_db = 20.0 * np.log10((sustain_band + 1e-9) / (attack_band + 1e-9))
            else:
                decay_db = float("nan")
            presence_values = presence.get(
                (int(note.get("tick", 0)), index + 1), {}
            )
            raw.append(
                {
                    "note": note,
                    "pitch": pitch,
                    "attack_support": _safe(attack_support),
                    "sustain_support": _safe(sustain_support),
                    "attack_band": _safe(attack_band),
                    "sustain_band": _safe(sustain_band),
                    "decay_db": _safe(decay_db),
                    "presence": _safe(presence_values.get("presence")),
                    "coefficient": _safe(presence_values.get("coefficient")),
                    "noise": _safe(presence_values.get("noise")),
                    "explained": _safe(presence_values.get("explained")),
                }
            )

        def _max_positive(values: list[float]) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return max(finite) if finite else float("nan")

        max_presence = _max_positive([item["presence"] for item in raw])
        max_support = _max_positive([item["sustain_support"] for item in raw])
        max_band = _max_positive([item["sustain_band"] for item in raw])
        order = sorted(
            range(len(raw)),
            key=lambda index: (
                raw[index]["presence"]
                if np.isfinite(raw[index]["presence"])
                else -float("inf")
            ),
            reverse=True,
        )
        presence_rank = {
            index: rank / max(1, len(raw) - 1) for rank, index in enumerate(order)
        }

        def _ratio(value: float, peak: float) -> float:
            if not np.isfinite(value) or not np.isfinite(peak) or peak <= 0:
                return float("nan")
            return value / peak

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            note = item["note"]
            pitch = int(item["pitch"])
            features = dict(shared)
            features.update(
                {
                    "n_pitch": float(pitch),
                    "n_velocity": float(note.get("velocity", 0)),
                    "n_chord_rank": float(note.get("chord_rank", index)),
                    "n_interval_to_lower": float(note.get("interval_to_lower", 0)),
                    "n_semitones_above_bass": float(pitch - bass_pitch),
                    "n_harm_support_attack": item["attack_support"],
                    "n_harm_support_sustain": item["sustain_support"],
                    "n_f0_band_attack": item["attack_band"],
                    "n_f0_band_sustain": item["sustain_band"],
                    "n_f0_decay_db": item["decay_db"],
                    "n_presence": item["presence"],
                    "n_coefficient": item["coefficient"],
                    "n_noise": item["noise"],
                    "n_explained": item["explained"],
                    "n_presence_relative": _ratio(item["presence"], max_presence),
                    "n_harm_support_relative": _ratio(
                        item["sustain_support"], max_support
                    ),
                    "n_f0_band_relative": _ratio(item["sustain_band"], max_band),
                    "n_presence_rank": float(presence_rank.get(index, 0.0)),
                }
            )
            rows.append({"features": features, "note": note})
        return rows


def extract_group_rows(
    audio_path: str | Path,
    groups: list[dict[str, Any]],
    *,
    config: FeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract feature rows for every note in every group."""
    context = AudioContext(audio_path, config=config)
    ordered = sorted(groups, key=lambda item: float(item.get("time_s", 0.0)))
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(ordered):
        next_time = (
            float(ordered[index + 1]["time_s"])
            if index + 1 < len(ordered)
            else None
        )
        previous_time = (
            float(ordered[index - 1]["time_s"]) if index > 0 else None
        )
        rows.extend(
            context.group_rows(
                group,
                next_time_s=next_time,
                previous_time_s=previous_time,
            )
        )
    return rows
