"""WAV output and playback helpers."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, samples: np.ndarray, sr: int = 44100) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def play(wav: str | Path) -> subprocess.Popen | None:
    """Start the first available system audio player. Returns the process."""
    player = shutil.which("aplay") or shutil.which("paplay") or shutil.which("ffplay")
    if not player:
        return None
    if player.endswith("ffplay"):
        cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav)]
    elif player.endswith("paplay"):
        cmd = [player, str(wav)]
    else:
        cmd = [player, "-q", str(wav)]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError:
        return None
