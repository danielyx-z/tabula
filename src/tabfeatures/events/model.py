"""Tiny gradient-boosted event model.

One binary XGBoost booster is trained per head.  The heads are intentionally
independent because a pitched note and a body tap / flick can occur together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, feature_vector
from .schema import DEFAULT_THRESHOLDS, HEADS, label_heads

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models" / "events"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EventModel:
    heads: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    feature_window: dict[str, float] = field(
        default_factory=lambda: {"before_s": 0.02, "after_s": 0.30}
    )
    review_margin: float = 0.15
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: dict[str, Any]) -> dict[str, float]:
        import xgboost as xgb

        vector = feature_vector(features)
        data = xgb.DMatrix(
            np.asarray([vector], dtype=np.float32),
            feature_names=list(self.feature_names),
        )
        result: dict[str, float] = {}
        for head in HEADS:
            booster = self.heads.get(head)
            if booster is None:
                result[head] = 0.0
                continue
            result[head] = float(np.asarray(booster.predict(data)).reshape(-1)[0])
        return result

    def save(self, directory: str | Path) -> dict[str, Any]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for head, booster in self.heads.items():
            booster.save_model(str(target / f"{head}.json"))
        payload = {
            "version": 1,
            "trained_at": _utc_now(),
            "feature_names": list(self.feature_names),
            "thresholds": dict(self.thresholds),
            "feature_window": dict(self.feature_window),
            "review_margin": float(self.review_margin),
            "heads": sorted(self.heads),
            **self.metadata,
        }
        (target / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload


def load_event_model(path: str | Path | None) -> EventModel | None:
    if not path:
        return None
    target = Path(path).expanduser()
    if target.is_file():
        metadata_path = target
        target = target.parent
    else:
        metadata_path = target / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"event model metadata not found: {metadata_path}")

    import xgboost as xgb

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("event model metadata must be an object")
    heads: dict[str, Any] = {}
    for head in metadata.get("heads", []):
        model_path = target / f"{head}.json"
        if not model_path.exists():
            continue
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        heads[str(head)] = booster
    feature_names = tuple(metadata.get("feature_names") or FEATURE_NAMES)
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "event model feature schema does not match this code; retrain the model"
        )
    thresholds = {
        **DEFAULT_THRESHOLDS,
        **{
            str(key): float(value)
            for key, value in (metadata.get("thresholds") or {}).items()
        },
    }
    feature_window = {
        "before_s": 0.02,
        "after_s": 0.30,
        **{
            str(key): float(value)
            for key, value in (metadata.get("feature_window") or {}).items()
        },
    }
    return EventModel(
        heads=heads,
        feature_names=feature_names,
        thresholds=thresholds,
        feature_window=feature_window,
        review_margin=float(metadata.get("review_margin", 0.15)),
        metadata=metadata,
    )


def train_event_model(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path = MODEL_DIR,
    num_boost_round: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    seed: int = 17,
    thresholds: dict[str, float] | None = None,
    feature_window: dict[str, float] | None = None,
    review_margin: float = 0.15,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one tiny booster per head from ``{"features", "label"}`` rows."""
    import xgboost as xgb

    if not rows:
        raise ValueError("no label rows supplied")

    X = np.vstack(
        [feature_vector(row.get("features", row)) for row in rows]
    ).astype(np.float32)
    heads: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for head in HEADS:
        y = np.asarray(
            [
                label_heads(str(row["label"]))[head]
                for row in rows
                if row.get("label")
            ],
            dtype=np.float32,
        )
        if len(y) != len(X):
            raise ValueError("every training row must have a label")
        positives = int(np.sum(y))
        negatives = int(len(y) - positives)
        if positives == 0 or negatives == 0:
            summaries[head] = {
                "trained": False,
                "reason": "need at least one positive and one negative",
                "positives": positives,
                "negatives": negatives,
            }
            continue
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": int(max_depth),
            "eta": float(learning_rate),
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1.0,
            "lambda": 1.0,
            "seed": int(seed),
            "nthread": 4,
            "tree_method": "hist",
            "scale_pos_weight": float(negatives / max(1, positives)),
        }
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
        booster = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))
        heads[head] = booster
        summaries[head] = {
            "trained": True,
            "positives": positives,
            "negatives": negatives,
            "rounds": int(num_boost_round),
        }

    model = EventModel(
        heads=heads,
        thresholds={**DEFAULT_THRESHOLDS, **(thresholds or {})},
        feature_window={
            "before_s": 0.02,
            "after_s": 0.30,
            **(feature_window or {}),
        },
        review_margin=float(review_margin),
        metadata={
            "rows": len(rows),
            "heads_summary": summaries,
            "params": {
                "num_boost_round": int(num_boost_round),
                "max_depth": int(max_depth),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
            },
            **(extra_metadata or {}),
        },
    )
    payload = model.save(out_dir)
    payload["heads_summary"] = summaries
    return payload
