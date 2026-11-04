#!/usr/bin/env python3
"""Evaluate the event model on held-out browser-labeller sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.events.features import extract_event_features, load_audio  # noqa: E402
from tabfeatures.events.model import load_event_model  # noqa: E402
from tabfeatures.events.schema import HEADS  # noqa: E402
from tabfeatures.labeling.store import (  # noqa: E402
    LABEL_ROOT,
    latest_labels,
    list_sessions,
    read_candidates,
)


def _rows(
    session_id: str,
    feature_window: dict[str, float],
) -> list[dict[str, Any]]:
    candidates = read_candidates(session_id).get("candidates", [])
    labels = latest_labels(session_id)
    audio_path = LABEL_ROOT / session_id / "audio.wav"
    if not audio_path.exists():
        return []
    y, sr = load_audio(audio_path)
    rows = []
    for index, record in sorted(labels.items()):
        if not 0 <= int(index) < len(candidates):
            continue
        candidate = candidates[int(index)]
        rows.append(
            {
                "label": str(record.get("label")),
                "features": extract_event_features(
                    y,
                    sr,
                    float(candidate.get("onset_s", 0.0)),
                    before_s=float(feature_window.get("before_s", 0.03)),
                    after_s=float(feature_window.get("after_s", 0.35)),
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--session", action="append", default=[])
    args = parser.parse_args(argv)

    sessions = args.session or list_sessions()
    if not sessions:
        raise SystemExit(f"no label sessions found in {LABEL_ROOT}")
    model = load_event_model(args.model)
    if model is None:
        raise SystemExit(f"could not load event model: {args.model}")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    rows: list[dict[str, Any]] = []
    for session_id in sessions:
        rows.extend(_rows(session_id, model.feature_window))
    if not rows:
        raise SystemExit("no labeled onsets found")

    report: dict[str, Any] = {"sessions": sessions, "rows": len(rows), "heads": {}}
    for head in HEADS:
        booster = model.heads.get(head)
        if booster is None:
            continue
        X = np.asarray(
            [
                [
                    float(row["features"].get(name, np.nan))
                    for name in model.feature_names
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        y = np.asarray(
            [
                1 if (
                    row["label"] == head
                    or (head == "dead_string" and row["label"] == "bass_flick")
                ) else 0
                for row in rows
            ],
            dtype=np.float32,
        )
        probs = np.asarray(
            booster.predict(xgb.DMatrix(X, feature_names=list(model.feature_names)))
        )
        threshold = float(model.thresholds.get(head, 0.6))
        predicted = probs >= threshold
        item: dict[str, Any] = {
            "positives": int(y.sum()),
            "threshold": threshold,
            "confusion": {
                "tp": int(np.sum(predicted & (y == 1))),
                "fp": int(np.sum(predicted & (y == 0))),
                "tn": int(np.sum((~predicted) & (y == 0))),
                "fn": int(np.sum((~predicted) & (y == 1))),
            },
        }
        if 0 < int(y.sum()) < len(y):
            item["auroc"] = float(roc_auc_score(y, probs))
        report["heads"][head] = item
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
