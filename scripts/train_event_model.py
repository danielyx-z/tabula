#!/usr/bin/env python3
"""Train the tiny XGBoost event model from browser-labeller sessions."""

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

from tabfeatures.events.config import (  # noqa: E402
    load_events_config,
    resolve_config_path,
)
from tabfeatures.events.features import (  # noqa: E402
    analysis_after,
    extract_event_features,
    load_audio,
)
from tabfeatures.events.model import MODEL_DIR, train_event_model  # noqa: E402
from tabfeatures.events.schema import HEADS  # noqa: E402
from tabfeatures.labeling.store import (  # noqa: E402
    LABEL_ROOT,
    latest_labels,
    list_sessions,
    read_candidates,
    read_meta,
)


def _rows_for_session(
    session_id: str,
    labels_dir: Path,
    feature_window: dict[str, float],
) -> list[dict[str, Any]]:
    meta = read_meta(session_id)
    candidates = read_candidates(session_id).get("candidates", [])
    labels = latest_labels(session_id)
    audio_path = labels_dir / session_id / "audio.wav"
    if not audio_path.exists():
        return []
    y, sr = load_audio(audio_path)
    rows: list[dict[str, Any]] = []
    for index, record in sorted(labels.items()):
        if not 0 <= int(index) < len(candidates):
            continue
        candidate = candidates[int(index)]
        onset_s = float(candidate.get("onset_s", 0.0))
        next_onset = (
            float(candidates[int(index) + 1].get("onset_s", 0.0))
            if int(index) + 1 < len(candidates)
            else None
        )
        features = extract_event_features(
            y,
            sr,
            onset_s,
            before_s=float(feature_window.get("before_s", 0.02)),
            after_s=analysis_after(
                onset_s,
                next_onset,
                max_after_s=float(feature_window.get("after_s", 0.30)),
            ),
        )
        rows.append(
            {
                "session_id": session_id,
                "index": int(index),
                "label": str(record.get("label")),
                "features": features,
                "original_name": meta.get("original_name"),
            }
        )
    return rows


def _metrics(rows: list[dict[str, Any]], model) -> dict[str, Any]:
    if not rows:
        return {}
    from sklearn.metrics import roc_auc_score

    output: dict[str, Any] = {}
    for head in HEADS:
        import xgboost as xgb

        booster = model.heads.get(head)
        if booster is None:
            continue
        X = np.vstack(
            [
                np.asarray(
                    [
                        float(row["features"].get(name, np.nan))
                        for name in model.feature_names
                    ],
                    dtype=np.float32,
                )
                for row in rows
            ]
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
        result: dict[str, Any] = {"positives": int(y.sum()), "rows": int(len(y))}
        if 0 < int(y.sum()) < len(y):
            result["auroc"] = float(roc_auc_score(y, probs))
        threshold = float(model.thresholds.get(head, 0.6))
        predicted = probs >= threshold
        result["confusion"] = {
            "tp": int(np.sum(predicted & (y == 1))),
            "fp": int(np.sum(predicted & (y == 0))),
            "tn": int(np.sum((~predicted) & (y == 0))),
            "fn": int(np.sum((~predicted) & (y == 1))),
        }
        output[head] = result
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/events.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--holdout-session",
        action="append",
        default=[],
        help="session id to reserve for evaluation (repeatable)",
    )
    args = parser.parse_args(argv)
    config = load_events_config(resolve_config_path(args.config))
    out_dir = args.out_dir or str(
        (ROOT / str(config.get("path", MODEL_DIR))).resolve()
        if not Path(str(config.get("path", MODEL_DIR))).is_absolute()
        else Path(str(config.get("path", MODEL_DIR)))
    )
    feature_window = {
        "before_s": 0.02,
        "after_s": 0.30,
        **(config.get("feature_window") or {}),
    }

    sessions = list_sessions()
    if not sessions:
        raise SystemExit(f"no label sessions found in {LABEL_ROOT}")
    holdout = set(args.holdout_session)
    train_sessions = [sid for sid in sessions if sid not in holdout]
    test_sessions = [sid for sid in sessions if sid in holdout]
    if not train_sessions:
        raise SystemExit("all sessions are held out; nothing to train")

    rows: list[dict[str, Any]] = []
    for session_id in train_sessions:
        rows.extend(_rows_for_session(session_id, LABEL_ROOT, feature_window))
    if not rows:
        raise SystemExit("no labeled onsets found in the training sessions")

    print(f"training rows: {len(rows)} from {len(train_sessions)} session(s)")
    payload = train_event_model(
        rows,
        out_dir=out_dir,
        num_boost_round=args.num_boost_round,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        thresholds=config.get("thresholds"),
        feature_window=feature_window,
        review_margin=float(config.get("review_margin", 0.15)),
    )
    print(json.dumps(payload, indent=2))

    if test_sessions:
        from tabfeatures.events.model import load_event_model

        model = load_event_model(out_dir)
        test_rows: list[dict[str, Any]] = []
        for session_id in test_sessions:
            test_rows.extend(
                _rows_for_session(session_id, LABEL_ROOT, feature_window)
            )
        if model is not None:
            print(json.dumps({"holdout": _metrics(test_rows, model)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
