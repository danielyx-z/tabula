"""Run the local MuScriptor transcription model.

The web app prefers a MIDI upload.  When the user only has audio, MuScriptor
runs locally on the small model and writes the MIDI that feeds the tab model
and the dead-note labeller.

The binary is resolved in this order:

1. ``TABULA_MUSCRIPTOR_BIN`` / ``MUSCRIPTOR_BIN`` (path to the executable)
2. ``muscriptor`` on ``PATH``
3. ``~/Projects/muscriptor/.venv/bin/muscriptor`` (the local development clone)
4. ``uvx muscriptor``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class MuscriptorUnavailable(RuntimeError):
    """Raised when no local MuScriptor installation can be found."""


def _as_executable(value: str | Path) -> Path | None:
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def find_muscriptor() -> list[str]:
    """Return the command prefix used to invoke MuScriptor."""
    for name in ("TABULA_MUSCRIPTOR_BIN", "MUSCRIPTOR_BIN"):
        value = os.environ.get(name)
        if value:
            executable = _as_executable(value)
            if executable is None:
                raise MuscriptorUnavailable(
                    f"{name} points to a missing or non-executable file: {value}"
                )
            return [str(executable)]

    found = shutil.which("muscriptor")
    if found:
        return [found]

    local = Path.home() / "Projects" / "muscriptor" / ".venv" / "bin" / "muscriptor"
    if local.is_file() and os.access(local, os.X_OK):
        return [str(local)]

    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "muscriptor"]

    raise MuscriptorUnavailable(
        "MuScriptor is required to transcribe audio. Install it with "
        "`uvx muscriptor`, or set TABULA_MUSCRIPTOR_BIN to the executable."
    )


def _tail(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text[-limit:] if text else ""


def transcribe_audio(
    audio_path: str | Path,
    output_midi: str | Path,
    *,
    model: str = "small",
    instrument: str = "acoustic_guitar",
    quantize: bool = False,
    grid: int = 8,
    timeout: int = 900,
) -> dict[str, Any]:
    """Transcribe ``audio_path`` to ``output_midi`` with local MuScriptor.

    ``grid`` is MuScriptor's steps-per-beat option: 4 = sixteenth notes,
    8 = thirty-second notes, 16 = sixty-fourth notes.  Quantization is off by
    default so trailing/percussive detail survives; ``midi2tab`` applies its
    own 20-tick grid afterwards.  If you enable it, 8 is a practical middle
    ground; 4 is noticeably harder.
    """
    audio = Path(audio_path).expanduser().resolve()
    output = Path(output_midi).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        *find_muscriptor(),
        "transcribe",
        str(audio),
        "--model",
        str(model),
        "--format",
        "midi",
        "--output",
        str(output),
    ]
    if instrument:
        command.extend(["--instruments", str(instrument)])
    if quantize:
        command.extend(["--quantize", "--grid", str(int(grid))])
    else:
        command.append("--no-quantize")

    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout)),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"MuScriptor transcription timed out after {int(timeout)} seconds."
        ) from exc
    except OSError as exc:
        raise MuscriptorUnavailable(
            f"Could not start MuScriptor: {exc}. Set TABULA_MUSCRIPTOR_BIN."
        ) from exc

    if result.returncode != 0 or not output.is_file():
        detail = _tail(
            (result.stderr or "")
            + ("\n" + result.stdout if result.stdout else "")
        )
        raise RuntimeError(
            "MuScriptor could not transcribe this recording. "
            + (detail or f"exit code {result.returncode}")
        )

    return {
        "command": command,
        "model": str(model),
        "instrument": str(instrument),
        "quantize": bool(quantize),
        "grid": int(grid) if quantize else None,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    """Small CLI used by ``python -m transcribe.muscriptor``."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio with local MuScriptor")
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--model", default="small")
    parser.add_argument("--instrument", default="acoustic_guitar")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args(argv)
    info = transcribe_audio(
        args.audio,
        args.output,
        model=args.model,
        instrument=args.instrument,
        quantize=args.quantize,
        grid=args.grid,
    )
    print(info["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
