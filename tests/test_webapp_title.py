"""Renaming a saved take from the workbench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import guitarpro
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import webapp.server as server  # noqa: E402


def _song(path: Path) -> guitarpro.Song:
    song = guitarpro.Song(title="clip_audio", artist="Tabula", tempo=120)
    header = guitarpro.MeasureHeader(number=1)
    song.measureHeaders = [header]
    track = guitarpro.Track(song=song, number=1)
    track.measures = [guitarpro.Measure(track=track, header=header)]
    song.tracks = [track]
    guitarpro.write(song, str(path), version=(5, 1, 0))
    return song


@pytest.fixture()
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, TestClient]:
    job_id = "a" * 32
    job_dir = tmp_path / job_id
    (job_dir / "out").mkdir(parents=True)
    report = {
        "title": "clip_audio",
        "files": {"audio": "clip_audio.mp3", "midi": "clip_audio.mid"},
        "summary": {"notes": 1},
    }
    (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    _song(job_dir / "score.gp5")
    _song(job_dir / "out" / "song.gp5")
    monkeypatch.setattr(server, "JOBS", tmp_path)
    return job_dir, TestClient(server.app)


def test_rename_updates_report_and_score(job: tuple[Path, TestClient]) -> None:
    job_dir, client = job
    response = client.patch(f"/api/jobs/{job_dir.name}/title", json={"title": "  Lover   Girl "})
    assert response.status_code == 200
    assert response.json() == {"id": job_dir.name, "title": "Lover Girl"}

    report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
    assert report["title"] == "Lover Girl"
    assert report["files"]["title"] == "Lover Girl"

    for name in ("score.gp5", "out/song.gp5"):
        song = guitarpro.parse(str(job_dir / name))
        assert song.title == "Lover Girl"
        assert len(song.measureHeaders) == 1


def test_empty_title_is_rejected(job: tuple[Path, TestClient]) -> None:
    job_dir, client = job
    response = client.patch(f"/api/jobs/{job_dir.name}/title", json={"title": "   "})
    assert response.status_code == 400
    report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
    assert report["title"] == "clip_audio"


def test_unknown_job_is_404(job: tuple[Path, TestClient]) -> None:
    _, client = job
    assert client.patch("/api/jobs/" + "b" * 32 + "/title", json={"title": "x"}).status_code == 404
    assert client.patch("/api/jobs/not-a-job/title", json={"title": "x"}).status_code == 404
