"""Shared helpers for ReID embedding aggregation."""
from __future__ import annotations

import numpy as np

REID_EMBEDDING_DIM = 256


def mean_track_embedding(embeddings) -> np.ndarray | None:
    """Mean ReID embedding for a tracklet, ignoring NaN/malformed rows."""
    valid = []
    for e in embeddings:
        if e is None or (isinstance(e, float) and np.isnan(e)):
            continue
        arr = np.asarray(e, dtype=float).reshape(-1)
        if arr.shape == (REID_EMBEDDING_DIM,) and np.isfinite(arr).all():
            valid.append(arr)
    if not valid:
        return None
    return np.mean(valid, axis=0)
