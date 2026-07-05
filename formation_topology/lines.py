import numpy as np


def detect_lines(xs, gap_delta=7.0, max_lines=3):
    """Split depth lines by the largest x-gaps and report centroid distances."""
    xs = np.sort(np.asarray(xs, dtype=float))
    n = len(xs)
    if n == 0:
        return 0, []
    if n == 1:
        return 1, []

    gaps = np.diff(xs)
    candidates = [(gap, idx) for idx, gap in enumerate(gaps) if gap > gap_delta]
    if not candidates:
        return 1, []

    candidates.sort(reverse=True)
    split_count = min(len(candidates), max_lines - 1)
    split_idx = sorted(idx for _, idx in candidates[:split_count])

    groups = []
    start = 0
    for idx in split_idx:
        groups.append(xs[start : idx + 1])
        start = idx + 1
    groups.append(xs[start:])

    centroids = [float(np.mean(group)) for group in groups]
    inter_line_gaps = [
        round(centroids[idx + 1] - centroids[idx], 2)
        for idx in range(len(centroids) - 1)
    ]
    return len(groups), inter_line_gaps
