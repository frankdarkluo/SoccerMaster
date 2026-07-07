import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0

BAND_EDGES_X = (-HALF_LENGTH / 3.0, HALF_LENGTH / 3.0)
LANE_EDGES_Y = (-20.16, -7.0, 7.0, 20.16)


def canonicalize(pts, attack_dir):
    """Rotate points into the canonical attacking frame where attack is +x."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if attack_dir < 0:
        return -pts
    return pts.copy()
