"""Plucked-string guitar synthesis.

Each note is a Karplus-Strong string: a filtered noise burst (the pick) is
circulated through a delay line whose length sets the pitch and whose loop
gain sets the decay.  A pluck-position comb, string-dependent brightness and
decay, body-resonance EQ and a gentle limiter make the result read as an
acoustic guitar rather than a sine-wave beep.

Only numpy is required.  The output is deterministic for a given note list.
"""

from __future__ import annotations

import math

import numpy as np

from core.fretboard import midi_to_freq
from core.quantise import PPQ

# T60 (seconds) per string, low E -> high E.  A real guitar's bass rings much
# longer than its treble strings.
_T60 = (5.5, 5.0, 4.2, 3.5, 2.8, 2.2)
# One-pole excitation brightness per string (0 = dark, 1 = raw noise).
_BRIGHT = (0.22, 0.27, 0.32, 0.38, 0.44, 0.50)
# Mix gain per string; bass needs a little less because it carries more energy.
_GAIN = (0.90, 0.86, 0.82, 0.78, 0.75, 0.72)
# Where the string is plucked, as a fraction of its length.
_PLUCK = (0.20, 0.19, 0.18, 0.16, 0.14, 0.12)


def _string_params(pitch: int, s: int) -> tuple[float, float, float, float]:
    """Return (t60, brightness, gain, pluck_ratio) for a note."""
    if 1 <= s <= 6:
        i = s - 1
        return _T60[i], _BRIGHT[i], _GAIN[i], _PLUCK[i]
    # No string information (plain MIDI): interpolate from pitch.
    x = min(1.0, max(0.0, (pitch - 40) / 45.0))
    return (
        5.5 - 3.3 * x,
        0.22 + 0.28 * x,
        0.82 - 0.10 * x,
        0.20 - 0.08 * x,
    )


def _excitation(
    delay: int,
    f0: float,
    sr: int,
    rng: np.random.Generator,
    brightness: float,
    pluck_ratio: float,
) -> np.ndarray:
    """Initial string shape: a plucked harmonic series plus a little pick noise."""
    n = np.arange(delay, dtype=np.float32)
    period = delay + 0.5  # matches the loop's effective delay

    # A real pluck is a smooth displacement, not white noise.  A 1/k^1.6
    # harmonic series guarantees a strong fundamental, and the pluck-position
    # sine notches the harmonics that a pick at that point cannot excite.
    shape = np.zeros(delay, dtype=np.float32)
    max_k = max(1, int(0.45 * sr / f0))
    for k in range(1, max_k + 1):
        amp = (1.0 / (k**1.6)) * abs(math.sin(math.pi * k * pluck_ratio))
        if amp > 1e-4:
            shape += amp * np.sin(2.0 * math.pi * k * n / period)

    # Pick texture: a short, low-passed noise component on top of the shape.
    noise = rng.standard_normal(delay).astype(np.float32)
    noise -= noise.mean()
    a = float(brightness)
    for _ in range(2):
        noise[1:] = a * noise[1:] + (1.0 - a) * noise[:-1]
    noise_peak = float(np.max(np.abs(noise))) or 1.0
    noise /= noise_peak

    excitation = shape + 0.12 * noise
    excitation -= excitation.mean()
    peak = float(np.max(np.abs(excitation)))
    if peak < 1e-6:
        return noise
    return excitation / peak


def _ks_note(
    f0: float,
    n_samples: int,
    sr: int,
    rng: np.random.Generator,
    t60: float,
    brightness: float,
    pluck_ratio: float,
) -> np.ndarray:
    """Synthesize one Karplus-Strong string of ``n_samples`` samples."""
    # The two-point loop filter has an effective delay of N + 0.5 samples.
    delay = max(2, int(round(sr / f0 - 0.5)))
    # Loop gain for the requested T60: amplitude after f0 * t60 periods is -60 dB.
    gain = math.exp(math.log(0.001) / (f0 * t60))
    gain = min(gain, 0.99995)

    excitation = _excitation(delay, f0, sr, rng, brightness, pluck_ratio)

    n = max(int(n_samples), delay + 1)
    out = np.zeros(n, dtype=np.float32)
    out[:delay] = excitation[:delay]

    # The recurrence y[n] = g/2 * (y[n-N] + y[n-N-1]) depends only on earlier
    # blocks of length N, so each block can be evaluated with numpy.
    prev = out[:delay].copy()
    prev2_last = np.float32(0.0)
    start = delay
    while start < n:
        end = min(start + delay, n)
        length = end - start
        ref1 = prev[:length]
        ref2 = np.empty(length, dtype=np.float32)
        ref2[0] = prev2_last
        if length > 1:
            ref2[1:] = prev[: length - 1]
        out[start:end] = (0.5 * gain) * (ref1 + ref2)
        prev2_last = prev[-1]
        prev = out[start:end].copy()
        start = end
    return out


def _body_eq(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply a fixed acoustic-guitar body response in the frequency domain."""
    n = len(samples)
    if n < 1024:
        return samples
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # Low cut below the guitar's range and a gentle top-end roll-off.
    eq = (freqs**2 / (freqs**2 + 55.0**2)) / np.sqrt(1.0 + (freqs / 9000.0) ** 4)
    # Body resonances (Helmholtz air mode and top-plate modes) plus presence.
    for centre, width, db in (
        (100.0, 35.0, 3.5),
        (210.0, 45.0, 2.5),
        (420.0, 60.0, 1.5),
        (1100.0, 350.0, -1.5),
        (2800.0, 700.0, 2.0),
        (6500.0, 1800.0, -2.5),
    ):
        eq *= 1.0 + (10.0 ** (db / 20.0) - 1.0) * np.exp(
            -0.5 * ((freqs - centre) / width) ** 2
        )
    return np.fft.irfft(spectrum * eq, n=n).astype(np.float32)


def synthesize(
    notes,
    bpm: float,
    sr: int = 44100,
    ppq: int = PPQ,
) -> np.ndarray:
    """Render canonical notes to mono float32 audio.

    ``notes`` is a structured array with fields t, dur, pitch, s, f (the same
    layout used everywhere else), or any sequence supporting those keys.
    """
    if not len(notes):
        return np.zeros(sr, dtype=np.float32)

    bpm = max(1.0, float(bpm))
    ticks_per_second = ppq * bpm / 60.0
    rows = [
        (int(n["t"]), max(1, int(n["dur"])), int(n["pitch"]), int(n["s"]))
        for n in notes
    ]
    t0 = min(row[0] for row in rows)

    # A note is muted when the next note on the same string starts.
    damp_ticks: list[int | None] = [None] * len(rows)
    by_string: dict[int, list[tuple[int, int]]] = {}
    for i, (t, _, _, s) in enumerate(rows):
        if 1 <= s <= 6:
            by_string.setdefault(s, []).append((t, i))
    for items in by_string.values():
        items.sort()
        for k in range(len(items) - 1):
            t, i = items[k]
            next_t = items[k + 1][0]
            damp_ticks[i] = max(0, next_t - t)

    rendered: list[tuple[int, np.ndarray]] = []
    for i, (t, dur, pitch, s) in enumerate(rows):
        f0 = midi_to_freq(pitch)
        t60, brightness, gain, pluck = _string_params(pitch, s)

        ring_seconds = dur / ticks_per_second + 3.0
        if damp_ticks[i] is not None:
            ring_seconds = min(ring_seconds, damp_ticks[i] / ticks_per_second)
        ring_seconds = min(6.0, max(0.05, ring_seconds))
        n_samples = int(ring_seconds * sr)
        if n_samples <= 0:
            continue

        rng = np.random.default_rng(
            (pitch << 20) ^ (t << 4) ^ (s & 0xF) ^ 0x5EED
        )
        sig = _ks_note(f0, n_samples, sr, rng, t60, brightness, pluck)

        # Short fade at the end so a damped note does not click.
        fade = min(len(sig), int(0.012 * sr))
        if fade > 1:
            sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        start = int(round((t - t0) / ticks_per_second * sr))
        rendered.append((start, sig * gain))

    if not rendered:
        return np.zeros(sr, dtype=np.float32)

    total = max(start + len(sig) for start, sig in rendered)
    mix = np.zeros(total + int(0.08 * sr), dtype=np.float32)
    for start, sig in rendered:
        mix[start : start + len(sig)] += sig

    mix = _body_eq(mix, sr)

    peak = float(np.max(np.abs(mix)))
    if peak > 1e-6:
        # Soft-knee limit, then normalize to a comfortable level.
        mix = np.tanh(mix / peak * 1.25)
        mix *= 0.89 / max(1e-9, float(np.max(np.abs(mix))))

    fade = min(int(0.005 * sr), len(mix) // 2)
    if fade > 1:
        mix[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mix[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return mix.astype(np.float32, copy=False)
