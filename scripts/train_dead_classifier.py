#!/usr/bin/env python3
"""Train the per-note dead-note classifier from labelled sessions.

Example:

    PYTHONPATH=src .venv/bin/python3 scripts/train_dead_classifier.py \\
        --holdout-session <session_id>

Labels live under ``data/labels_dead``.  A session contributes implicit
"pitched" negatives only after it is marked reviewed; otherwise only its
explicit dead/keep records are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.deadnotes.dataset import load_rows, session_rows  # noqa: E402
from tabfeatures.deadnotes.features import FeatureConfig  # noqa: E402
from tabfeatures.deadnotes.metrics import classifier_metrics  # noqa: E402
from tabfeatures.deadnotes.model import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    load_model,
    train_model,
)
from tabfeatures.labeling.store import LABEL_ROOT, list_sessions  # noqa: E402


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [1 if str(row.get("label")) == "dead" else 0 for row in rows],
        dtype=np.int64,
    )


def _choose_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    min_precision: float,
    fallback: float,
) -> float:
    from sklearn.metrics import precision_recall_curve

    y = _labels(rows)
    if not (0 < int(y.sum()) < len(y)):
        return float(fallback)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # precision/recall are one longer than thresholds; match threshold points.
    best = float(fallback)
    best_recall = -1.0
    for index, threshold in enumerate(thresholds):
        if float(precision[index]) >= float(min_precision) and float(recall[index]) > best_recall:
            best = float(threshold)
            best_recall = float(recall[index])
    return best


def _out_of_fold(
    sessions: list[str],
    *,
    label_root: str | Path,
    config: FeatureConfig,
    include_unreviewed: bool,
    max_depth: int,
    rounds: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Leave-one-session-out scores for threshold selection."""
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for held in sessions:
        train = [
            session_rows(
                session,
                label_root=label_root,
                config=config,
                include_unreviewed=include_unreviewed,
            )
            for session in sessions
            if session != held
        ]
        train_rows = [row for group in train for row in group]
        test_rows = session_rows(
            held,
            label_root=label_root,
            config=config,
            include_unreviewed=include_unreviewed,
        )
        if not train_rows or not test_rows:
            continue
        labels = {str(row.get("label")) for row in train_rows}
        if not {"dead", "pitched"} <= labels:
            continue
        with tempfile.TemporaryDirectory(prefix="tabula-dead-fold-") as fold_dir:
            try:
                train_model(
                    train_rows,
                    out_dir=fold_dir,
                    max_depth=max_depth,
                    num_boost_round=rounds,
                    learning_rate=learning_rate,
                    seed=seed,
                    config=config,
                )
            except ValueError:
                continue
            loaded = load_model(fold_dir)
            if loaded is None:
                continue
            probs = loaded.predict_proba(test_rows)
        rows.extend(test_rows)
        scores.extend(float(value) for value in probs)
    return rows, np.asarray(scores, dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--holdout-session", action="append", default=[])
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--min-dead-precision", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--no-threshold-search",
        action="store_true",
        help="use --threshold or 0.80 instead of leave-one-session-out selection",
    )
    args = parser.parse_args(argv)

    label_root = (
        Path(args.label_root).expanduser()
        if args.label_root
        else LABEL_ROOT
    )
    config = FeatureConfig()
    sessions = [item["id"] for item in list_sessions(label_root)]
    if not sessions:
        raise SystemExit("no label sessions found under data/labels_dead")
    holdout = set(args.holdout_session)
    if not holdout and len(sessions) > 1:
        holdout = {sessions[-1]}
    train_sessions = [session for session in sessions if session not in holdout]
    if not train_sessions:
        raise SystemExit("all sessions are held out; nothing to train")

    kwargs = {
        "label_root": label_root,
        "config": config,
        "include_unreviewed": args.include_unreviewed,
    }
    train_rows, used = load_rows(train_sessions, **kwargs)
    if not train_rows:
        raise SystemExit("no usable labels; mark at least one session reviewed")
    train_labels = {str(row.get("label")) for row in train_rows}
    if not {"dead", "pitched"} <= train_labels:
        raise SystemExit(
            "need both dead and pitched labels "
            f"(found {sorted(train_labels)})"
        )

    threshold = float(args.threshold if args.threshold is not None else 0.80)
    oof_rows: list[dict[str, Any]] = []
    if not args.no_threshold_search and args.threshold is None and len(train_sessions) >= 3:
        oof_rows, oof_scores = _out_of_fold(
            train_sessions,
            label_root=label_root,
            config=config,
            include_unreviewed=args.include_unreviewed,
            max_depth=args.max_depth,
            rounds=args.num_boost_round,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        if len(oof_rows) == len(oof_scores) and oof_rows:
            threshold = _choose_threshold(
                oof_rows,
                oof_scores,
                min_precision=args.min_dead_precision,
                fallback=threshold,
            )

    train_model(
        train_rows,
        out_dir=args.out_dir,
        threshold=threshold,
        num_boost_round=args.num_boost_round,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
        config=config,
        extra_metadata={
            "sessions": used,
            "holdout": sorted(holdout),
            "min_dead_precision": float(args.min_dead_precision),
            "threshold_selection": (
                "leave-one-session-out"
                if oof_rows
                else "manual/default"
            ),
        },
    )

    model = load_model(args.out_dir)
    if model is None:
        raise SystemExit(f"the trained model could not be reloaded from {args.out_dir}")
    train_metrics = classifier_metrics(train_rows, model.predict_proba(train_rows), threshold)
    report: dict[str, Any] = {
        "model": str(Path(args.out_dir)),
        "sessions": used,
        "holdout": sorted(holdout),
        "rows": len(train_rows),
        "threshold": threshold,
        "train": train_metrics,
    }
    if oof_rows:
        report["threshold_oof"] = classifier_metrics(oof_rows, oof_scores, threshold)
    if holdout:
        holdout_rows: list[dict[str, Any]] = []
        for session_id in sorted(holdout):
            holdout_rows.extend(
                session_rows(
                    session_id,
                    label_root=label_root,
                    config=config,
                    include_unreviewed=args.include_unreviewed,
                )
            )
        if holdout_rows:
            report["holdout"] = classifier_metrics(
                holdout_rows,
                model.predict_proba(holdout_rows),
                threshold,
            )
        else:
            report["holdout"] = {"rows": 0, "reason": "no reviewed labels"}

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
