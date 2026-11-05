"""Optional near-duplicate filtering for the abundant pitched rows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import feature_vector


def dedupe_pitched_rows(
    rows: list[dict[str, Any]],
    *,
    distance: float = 1.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop near-identical pitched onsets within each recording.

    Rare-event rows are never removed.  Deduplication is scoped per session so
    repeated phrases in one take are reduced without collapsing the variation
    across different recordings.  Distances are measured after standardizing
    the engineered features.
    """
    pitched = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("label")) == "pitched"
    ]
    summary: dict[str, Any] = {
        "enabled": bool(distance and distance > 0),
        "distance": float(distance),
        "pitched_before": len(pitched),
        "pitched_after": len(pitched),
        "dropped": 0,
    }
    if len(pitched) < 2 or not summary["enabled"]:
        return list(rows), summary

    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(
        [feature_vector(row.get("features", row)) for _, row in pitched],
        dtype=np.float64,
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    standardized = StandardScaler().fit_transform(matrix)

    groups: dict[str, list[int]] = {}
    for position, (_index, row) in enumerate(pitched):
        session_id = str(row.get("session_id") or "__all__")
        groups.setdefault(session_id, []).append(position)

    kept_positions: set[int] = set()
    for positions in groups.values():
        kept_vectors: list[np.ndarray] = []
        for position in positions:
            vector = standardized[position]
            if any(np.linalg.norm(vector - other) < float(distance) for other in kept_vectors):
                continue
            kept_vectors.append(vector)
            kept_positions.add(position)

    keep_indices = {
        pitched[position][0] for position in kept_positions
    }
    result = [
        row
        for index, row in enumerate(rows)
        if str(row.get("label")) != "pitched" or index in keep_indices
    ]
    summary["pitched_after"] = len(keep_indices)
    summary["dropped"] = len(pitched) - len(keep_indices)
    summary["sessions"] = len(groups)
    return result, summary
