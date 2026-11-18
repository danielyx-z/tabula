"""Link handling for the workbench's yt-dlp input."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from webapp.server import _validate_url  # noqa: E402


@pytest.mark.parametrize(
    "value",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://example.com/song.mp3",
        "http://example.com/audio",
    ],
)
def test_public_http_links_are_accepted(value: str) -> None:
    assert _validate_url(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "not a url",
        "file:///etc/passwd",
        "ftp://example.com/song.mp3",
        "http://localhost:8000/song.wav",
        "http://127.0.0.1/song.wav",
        "http://10.0.0.5/song.wav",
        "http://192.168.1.20/song.wav",
        "http://100.111.69.56/song.wav",
        "http://[::1]/song.wav",
        "http://printer.local/song.wav",
    ],
)
def test_local_or_unsupported_links_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_url(value)
