"""Small gradient-boosted model for per-note dead-note prediction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, FeatureConfig

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = Path(
    os.environ.get("TABULA_DEAD_MODEL", str(REPO / "data" / "models" / "dead_notes"))
).expanduser()
DEFAULT_THRESHOLD = 0.80


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def feature_vector(features: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [_safe(features.get(name)) for name in FEATURE_NAMES],
        dtype=np.float32,
    )


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([feature_vector(row.get("features", row)) for row in rows])


@dataclass
class DeadNoteModel:
    booster: Any
    threshold: float = DEFAULT_THRESHOLD
    feature_names: tuple[str, ...] = FEATURE_NAMES
    config: FeatureConfig = field(default_factory=FeatureConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, rows: list[dict[str, Any]]) -> np.ndarray:
        import xgboost as xgb

        if not rows:
            return np.zeros(0, dtype=float)
        matrix = feature_matrix(rows)
        data = xgb.DMatrix(
            matrix,
            feature_names=list(self.feature_names),
            missing=np.nan,
        )
        return np.asarray(self.booster.predict(data), dtype=float).reshape(-1)

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scores = self.predict_proba(rows)
        out: list[dict[str, Any]] = []
        for row, score in zip(rows, scores):
            out.append(
                {
                    "note_id": (row.get("note") or {}).get("note_id"),
                    "group_id": (row.get("note") or {}).get("group_id"),
                    "tick": (row.get("note") or {}).get("tick"),
                    "pitch": (row.get("note") or {}).get("pitch"),
                    "score": float(score),
                    "dead": bool(score >= float(self.threshold)),
                }
            )
        return out

    def save(self, directory: str | Path) -> dict[str, Any]:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "model.json"
        self.booster.save_model(str(model_path))
        payload = {
            "version": 1,
            "trained_at": _utc_now(),
            "model": model_path.name,
            "feature_names": list(self.feature_names),
            "threshold": float(self.threshold),
            "feature_config": self.config.to_dict(),
            **self.metadata,
        }
        (target / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload


def load_model(path: str | Path | None = None) -> DeadNoteModel | None:
    target = Path(path).expanduser() if path else DEFAULT_MODEL_DIR
    if target.is_file():
        metadata_path = target
        target = target.parent
    else:
        metadata_path = target / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = str(metadata.get("model") or "model.json")
    model_path = target / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"dead-note model file not found: {model_path}")
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feature_names = tuple(metadata.get("feature_names") or FEATURE_NAMES)
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "dead-note model feature schema does not match this code; retrain it"
        )
    config_data = metadata.get("feature_config") or {}
    config = FeatureConfig(
        **{
            key: value
            for key, value in config_data.items()
            if key in FeatureConfig.__dataclass_fields__
        }
    )
    return DeadNoteModel(
        booster=booster,
        threshold=float(metadata.get("threshold", DEFAULT_THRESHOLD)),
        feature_names=feature_names,
        config=config,
        metadata=metadata,
    )


def train_model(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path = DEFAULT_MODEL_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    num_boost_round: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.04,
    seed: int = 17,
    config: FeatureConfig | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one binary XGBoost booster from labelled feature rows."""
    import xgboost as xgb

    if not rows:
        raise ValueError("no training rows")
    labels = np.asarray(
        [1.0 if str(row.get("label")) == "dead" else 0.0 for row in rows],
        dtype=np.float32,
    )
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"need both classes to train (dead={positives}, pitched={negatives})"
        )
    matrix = feature_matrix(rows)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "seed": int(seed),
        "nthread": 4,
        "tree_method": "hist",
        "scale_pos_weight": float(negatives / max(1, positives)),
    }
    dtrain = xgb.DMatrix(
        matrix,
        label=labels,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )
    booster = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))
    model = DeadNoteModel(
        booster=booster,
        threshold=float(threshold),
        config=config or FeatureConfig(),
        metadata={
            "rows": len(rows),
            "positives": positives,
            "negatives": negatives,
            "params": {
                "num_boost_round": int(num_boost_round),
                "max_depth": int(max_depth),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
            },
            **(extra_metadata or {}),
        },
    )
    return model.save(out_dir)
