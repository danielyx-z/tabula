"""MuScriptor model selection and offline availability."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from transcribe import muscriptor  # noqa: E402


def _fake_weights(root: Path, size: str) -> None:
    snapshot = root / f"models--MuScriptor--muscriptor-{size}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"\0" * (2 * 1024 * 1024))


def test_env_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TABULA_MUSCRIPTOR_MODEL", "large")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    assert muscriptor.default_model() == "large"


def test_falls_back_to_small_without_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TABULA_MUSCRIPTOR_MODEL", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert muscriptor.default_model() == "small"


def test_uses_largest_cached_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TABULA_MUSCRIPTOR_MODEL", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    _fake_weights(tmp_path, "small")
    _fake_weights(tmp_path, "medium")
    assert muscriptor.default_model() == "medium"
    _fake_weights(tmp_path, "large")
    assert muscriptor.default_model() == "large"
