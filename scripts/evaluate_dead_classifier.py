#!/usr/bin/env python3
"""Evaluate a trained dead-note classifier on labelled sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tabfeatures.deadnotes.dataset import session_rows  # noqa: E402
from tabfeatures.deadnotes.metrics import classifier_metrics  # noqa: E402
from tabfeatures.deadnotes.model import DEFAULT_MODEL_DIR, load_model  # noqa: E402
from tabfeatures.labeling.store import LABEL_ROOT, list_sessions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--label-root", default=str(LABEL_ROOT))
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--include-unreviewed", action="store_true")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    if model is None:
        raise SystemExit(f"dead-note model not found: {args.model}")
    threshold = float(args.threshold if args.threshold is not None else model.threshold)
    sessions = args.session or [item["id"] for item in list_sessions()]
    report = {
        "model": str(args.model),
        "threshold": threshold,
        "sessions": {},
    }
    for session_id in sessions:
        rows = session_rows(
            session_id,
            label_root=args.label_root,
            config=model.config,
            include_unreviewed=args.include_unreviewed,
        )
        if not rows:
            report["sessions"][session_id] = {"rows": 0}
            continue
        scores = model.predict_proba(rows)
        report["sessions"][session_id] = classifier_metrics(
            rows, scores, threshold
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
