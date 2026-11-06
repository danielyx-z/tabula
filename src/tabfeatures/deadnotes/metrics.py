"""Classification metrics for dead-note models."""

from __future__ import annotations

from typing import Any

import numpy as np


def classifier_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(
        [1 if str(row.get("label")) == "dead" else 0 for row in rows],
        dtype=np.int64,
    )
    if len(y) == 0:
        return {"rows": 0}
    predicted = (scores >= float(threshold)).astype(np.int64)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "threshold": float(threshold),
        "precision_dead": float(precision[1]),
        "recall_dead": float(recall[1]),
        "f1_dead": float(f1[1]),
        "precision_pitched": float(precision[0]),
        "recall_pitched": float(recall[0]),
        "confusion": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }
    if 0 < int(y.sum()) < len(y):
        result["roc_auc"] = float(roc_auc_score(y, scores))
        result["pr_auc"] = float(average_precision_score(y, scores))
    return result
