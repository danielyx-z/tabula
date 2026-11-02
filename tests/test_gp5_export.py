"""Contract test for the web app's Guitar Pro 5 exporter.

The reference artifacts under ``out/valentine`` are generated data and are not
tracked. When they are present, this test proves the exported GP5 comes from
those exact files and keeps the source timing, capo, tuning and nylon program.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "out" / "valentine"
MISSING = not (REFERENCE / "final.json").is_file() or not (REFERENCE / "final.mid").is_file()

pytestmark = pytest.mark.skipif(
    MISSING, reason="out/valentine reference artifacts are not present"
)


def test_gp5_export_preserves_score_contract(tmp_path):
    import guitarpro as gp

    from webapp.gp5 import build_gp5

    records = json.loads((REFERENCE / "final.json").read_text(encoding="utf-8"))
    output = tmp_path / "score.gp5"
    info = build_gp5(
        records,
        REFERENCE / "final.mid",
        output,
        capo=1,
        title="Valentine",
    )

    assert info["measures"] == 59
    assert info["beats"] == 427
    assert info["notes"] == 852
    assert info["skipped"] == 0
    assert info["capo"] == 1
    assert info["duration"] == pytest.approx(173.09, abs=0.05)

    song = gp.parse(str(output))
    track = song.tracks[0]
    assert len(song.measureHeaders) == 59
    assert song.measureHeaders[0].start == 960
    assert track.offset == 1
    assert track.channel.instrument == 24
    assert [string.value for string in track.strings] == [64, 59, 55, 50, 45, 40]

    ties = 0
    normal = 0
    tempo_changes = 0
    for measure in track.measures:
        for beat in measure.voices[0].beats:
            change = beat.effect.mixTableChange
            if change is not None and change.tempo is not None:
                tempo_changes += 1
            for note in beat.notes:
                if note.type == gp.NoteType.tie:
                    ties += 1
                else:
                    normal += 1
    assert normal == 729
    assert ties == 123
    assert tempo_changes > 1
