"""Infer delivery class and corner landing zone from pitch geometry."""
WIDE_CHANNEL_MIN_Y_M = 20.0
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
CLEARANCE_MIN_SPEED_MPS = 22.0
CLEARANCE_MIN_DISTANCE_M = 30.0
DEFENSIVE_THIRD_MAX_X_M = -17.5
PASS_MIN_FORWARD_M = 3.0
CENTER_BAND_M = 2.0

RESTART_DELIVERY = {
    "corner": "corner_cross",
    "goal_kick": "goal_kick",
    "throw_in": "throw_in",
    "free_kick": "free_kick",
}


def classify_delivery(kick, end_xy, attack_sign, restarts) -> str | None:
    for restart in restarts:
        if abs(restart.t - kick.t) < 0.2 and restart.kind in RESTART_DELIVERY:
            return RESTART_DELIVERY[restart.kind]
    if end_xy is None or attack_sign not in {-1, 1}:
        return None
    x0, y0 = kick.xy
    x1, y1 = end_xy
    distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    forward = (x1 - x0) * attack_sign
    if (kick.speed_mps >= CLEARANCE_MIN_SPEED_MPS
            and distance >= CLEARANCE_MIN_DISTANCE_M
            and x0 * attack_sign <= DEFENSIVE_THIRD_MAX_X_M):
        return "long_clearance"
    in_box = (x1 * attack_sign >= 52.5 - BOX_DEPTH_M
              and abs(y1) <= BOX_HALF_WIDTH_M)
    if abs(y0) >= WIDE_CHANNEL_MIN_Y_M and in_box and abs(y1) < abs(y0):
        return "cross"
    return "open_play_pass" if forward >= PASS_MIN_FORWARD_M else None


def corner_landing_zone(corner_y, landing_xy) -> str | None:
    if landing_xy is None:
        return None
    y = landing_xy[1]
    toward_corner = y * (-1.0 if corner_y < 0 else 1.0)
    if toward_corner >= CENTER_BAND_M:
        return "near_post"
    if toward_corner <= -CENTER_BAND_M:
        return "far_post"
    return "center"
