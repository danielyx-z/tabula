"""Three-class classification metrics for the dead/noise model."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import CLASS_NAMES, label_index


def classifier_metrics(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    if len(rows) == 0:
        return {"rows": 0}
    y = np.asarray([label_index(row.get("label")) for row in rows], dtype=np.int64)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have one column per class")
    limits = {"dead": 0.8, "noise": 0.8, **(thresholds or {})}
    predicted = np.zeros(len(probs), dtype=np.int64)
    for index, row in enumerate(probs):
        _pitched, dead, noise = (float(value) for value in row)
        if noise >= float(limits["noise"]) and noise >= dead:
            predicted[index] = 2
        elif dead >= float(limits["dead"]):
            predicted[index] = 1
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    matrix = confusion_matrix(
        y, predicted, labels=list(range(len(CLASS_NAMES)))
    )
    per_class: dict[str, Any] = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "rows": int(len(y)),
        "class_counts": {
            name: int((y == index).sum()) for index, name in enumerate(CLASS_NAMES)
        },
        "thresholds": {
            str(key): float(value) for key, value in (thresholds or {}).items()
        },
        "accuracy": float(np.mean(predicted == y)),
        "per_class": per_class,
        "confusion": {
            "labels": list(CLASS_NAMES),
            "matrix": matrix.astype(int).tolist(),
        },
    }
