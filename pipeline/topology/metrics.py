import numpy as np
from scipy.spatial import ConvexHull

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from .pitch import BAND_EDGES_X, LANE_EDGES_Y


def block_height(pts):
    return float(np.mean(pts[:, 0]))


def block_depth(pts):
    x = pts[:, 0]
    return float(np.percentile(x, 90) - np.percentile(x, 10))


def block_width(pts):
    y = pts[:, 1]
    return float(np.percentile(y, 90) - np.percentile(y, 10))


def centroid_y(pts):
    return float(np.mean(pts[:, 1]))


def hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def band_counts(pts):
    lo, hi = BAND_EDGES_X
    counts = [0, 0, 0]
    for xi in pts[:, 0]:
        if xi < lo:
            counts[0] += 1
        elif xi < hi:
            counts[1] += 1
        else:
            counts[2] += 1
    return counts


def lane_counts(pts):
    counts = [0, 0, 0, 0, 0]
    for yi in pts[:, 1]:
        idx = int(np.searchsorted(LANE_EDGES_Y, yi, side="right"))
        counts[idx] += 1
    return counts


def side_overload(pts):
    counts = lane_counts(pts)
    total = sum(counts)
    if total == 0:
        return 0.0
    return float((counts[0] + counts[1] - counts[3] - counts[4]) / total)
