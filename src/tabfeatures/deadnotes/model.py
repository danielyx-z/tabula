"""Small gradient-boosted model for per-note pitch/dead/noise prediction."""

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
CLASS_NAMES = ("pitched", "dead", "noise")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DEFAULT_THRESHOLDS = {"dead": 0.80, "noise": 0.80}


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


def label_index(label: Any) -> int:
    return CLASS_TO_INDEX.get(str(label), 0)


@dataclass
class DeadNoteModel:
    booster: Any
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    feature_names: tuple[str, ...] = FEATURE_NAMES
    config: FeatureConfig = field(default_factory=FeatureConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, rows: list[dict[str, Any]]) -> np.ndarray:
        import xgboost as xgb

        if not rows:
            return np.zeros((0, len(CLASS_NAMES)), dtype=float)
        matrix = feature_matrix(rows)
        data = xgb.DMatrix(
            matrix,
            feature_names=list(self.feature_names),
            missing=np.nan,
        )
        scores = np.asarray(self.booster.predict(data), dtype=float)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                f"model returned {scores.shape[1]} classes, expected {len(CLASS_NAMES)}"
            )
        return scores

    def predict(
        self,
        rows: list[dict[str, Any]],
        *,
        thresholds: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        probs = self.predict_proba(rows)
        limits = {**self.thresholds, **(thresholds or {})}
        out: list[dict[str, Any]] = []
        for row, scores in zip(rows, probs):
            pitched, dead, noise = (float(value) for value in scores)
            if (
                noise >= float(limits.get("noise", 0.8))
                and noise >= dead
            ):
                label = "noise"
            elif dead >= float(limits.get("dead", 0.8)):
                label = "dead"
            else:
                label = "pitched"
            out.append(
                {
                    "note_id": (row.get("note") or {}).get("note_id"),
                    "group_id": (row.get("note") or {}).get("group_id"),
                    "tick": (row.get("note") or {}).get("tick"),
                    "pitch": (row.get("note") or {}).get("pitch"),
                    "label": label,
                    "dead": label == "dead",
                    "noise": label == "noise",
                    "score": max(dead, noise),
                    "scores": {
                        "pitched": pitched,
                        "dead": dead,
                        "noise": noise,
                    },
                }
            )
        return out

    def save(self, directory: str | Path) -> dict[str, Any]:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "model.json"
        self.booster.save_model(str(model_path))
        payload = {
            "version": 2,
            "trained_at": _utc_now(),
            "model": model_path.name,
            "classes": list(CLASS_NAMES),
            "feature_names": list(self.feature_names),
            "thresholds": {key: float(value) for key, value in self.thresholds.items()},
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
    if int(metadata.get("version", 1)) < 2:
        raise ValueError("dead-note model is behind the noisy-note schema; retrain it")
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
    thresholds = {
        **DEFAULT_THRESHOLDS,
        **{
            str(key): float(value)
            for key, value in (metadata.get("thresholds") or {}).items()
        },
    }
    return DeadNoteModel(
        booster=booster,
        thresholds=thresholds,
        feature_names=feature_names,
        config=config,
        metadata=metadata,
    )


def train_model(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path = DEFAULT_MODEL_DIR,
    thresholds: dict[str, float] | None = None,
    num_boost_round: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.04,
    seed: int = 17,
    config: FeatureConfig | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a three-class ``pitched``/``dead``/``noise`` XGBoost booster."""
    import xgboost as xgb

    if not rows:
        raise ValueError("no training rows")
    labels = np.asarray(
        [label_index(row.get("label")) for row in rows],
        dtype=np.int64,
    )
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    present = [index for index, count in enumerate(counts) if count > 0]
    if len(present) < 2:
        raise ValueError(
            "need at least two classes to train "
            f"(pitched={counts[0]}, dead={counts[1]}, noise={counts[2]})"
        )
    matrix = feature_matrix(rows)
    # Keep the many pitched rows from overwhelming the rare classes without
    # letting a tiny noise set dominate.
    weights = np.asarray(
        [
            max(0.5, min(8.0, len(labels) / (len(present) * max(1, counts[label]))))
            for label in labels
        ],
        dtype=np.float32,
    )
    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_NAMES),
        "eval_metric": "mlogloss",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "seed": int(seed),
        "nthread": 4,
        "tree_method": "hist",
    }
    dtrain = xgb.DMatrix(
        matrix,
        label=labels,
        weight=weights,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )
    booster = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))
    model = DeadNoteModel(
        booster=booster,
        thresholds={
            **DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        },
        config=config or FeatureConfig(),
        metadata={
            "rows": len(rows),
            "class_counts": {
                name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)
            },
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
