"""Guitar choice for the workbench's MuScriptor input."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import webapp.server as server  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("acoustic", "acoustic"),
        ("Acoustic", "acoustic"),
        (" electric ", "electric"),
        ("ELECTRIC", "electric"),
        ("", "acoustic"),
        ("   ", "acoustic"),
    ],
)
def test_guitar_choice_is_normalised(value: str, expected: str) -> None:
    assert server._parse_guitar(value) == expected


@pytest.mark.parametrize("value", ["classical", "bass", "distorted", "1"])
def test_unknown_guitar_choice_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        server._parse_guitar(value)


@pytest.mark.parametrize(
    ("guitar", "instrument"),
    [
        ("acoustic", "acoustic_guitar"),
        ("electric", "clean_electric_guitar"),
    ],
)
def test_guitar_maps_to_muscriptor_instrument(
    guitar: str, instrument: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "MUSCRIPTOR_INSTRUMENTS", "")
    assert server._guitar_instrument(guitar) == instrument


def test_environment_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "MUSCRIPTOR_INSTRUMENTS", "distorted_electric_guitar")
    assert server._guitar_instrument("acoustic") == "distorted_electric_guitar"
