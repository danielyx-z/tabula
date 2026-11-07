#!/usr/bin/env python3
"""Train the per-note pitched/dead/noise classifier from labelled sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.deadnotes.dataset import load_rows, session_rows  # noqa: E402
from tabfeatures.deadnotes.features import FeatureConfig  # noqa: E402
from tabfeatures.deadnotes.metrics import classifier_metrics  # noqa: E402
from tabfeatures.deadnotes.model import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_THRESHOLDS,
    load_model,
    train_model,
)
from tabfeatures.labeling.store import LABEL_ROOT, list_sessions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--holdout-session", action="append", default=[])
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument(
        "--dead-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["dead"],
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["noise"],
    )
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    label_root = Path(args.label_root).expanduser() if args.label_root else LABEL_ROOT
    config = FeatureConfig()
    sessions = [item["id"] for item in list_sessions(label_root)]
    if not sessions:
        raise SystemExit(f"no label sessions found under {label_root}")
    holdout = set(args.holdout_session)
    if not holdout and len(sessions) > 1:
        holdout = {sessions[-1]}
    train_sessions = [session for session in sessions if session not in holdout]
    if not train_sessions:
        raise SystemExit("all sessions are held out; nothing to train")

    train_rows, used = load_rows(
        train_sessions,
        label_root=label_root,
        config=config,
        include_unreviewed=args.include_unreviewed,
    )
    if not train_rows:
        raise SystemExit("no usable labels; mark at least one session reviewed")
    thresholds = {
        "dead": float(args.dead_threshold),
        "noise": float(args.noise_threshold),
    }
    train_model(
        train_rows,
        out_dir=args.out_dir,
        thresholds=thresholds,
        num_boost_round=args.num_boost_round,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
        config=config,
        extra_metadata={
            "sessions": used,
            "holdout": sorted(holdout),
        },
    )
    model = load_model(args.out_dir)
    if model is None:
        raise SystemExit(f"the trained model could not be reloaded from {args.out_dir}")

    report: dict = {
        "model": str(Path(args.out_dir)),
        "sessions": used,
        "holdout": sorted(holdout),
        "rows": len(train_rows),
        "thresholds": model.thresholds,
        "train": classifier_metrics(
            train_rows, model.predict_proba(train_rows), model.thresholds
        ),
    }
    if holdout:
        holdout_rows: list[dict] = []
        for session_id in sorted(holdout):
            holdout_rows.extend(
                session_rows(
                    session_id,
                    label_root=label_root,
                    config=config,
                    include_unreviewed=args.include_unreviewed,
                )
            )
        report["holdout"] = (
            classifier_metrics(
                holdout_rows, model.predict_proba(holdout_rows), model.thresholds
            )
            if holdout_rows
            else {"rows": 0, "reason": "no reviewed labels"}
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
