"""Infer restarts only when a kick follows a tracked dead-ball period."""
from dataclasses import dataclass

PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0
OUT_OF_PLAY_TOLERANCE_M = 3.0
CENTER_SPOT_RADIUS_M = 3.0
GOAL_AREA_DEPTH_M = 7.0
GOAL_AREA_HALF_WIDTH_M = 11.0
CORNER_RADIUS_M = 4.0
THROW_IN_BAND_M = 1.5
DEAD_TO_KICK_MAX_GAP_S = 0.6


@dataclass(frozen=True)
class Restart:
    t: float
    xy: tuple
    kind: str


def classify_restart_location(xy) -> str | None:
    x, y = xy
    ax, ay = abs(x), abs(y)
    if (ax > PITCH_HALF_LENGTH_M + OUT_OF_PLAY_TOLERANCE_M
            or ay > PITCH_HALF_WIDTH_M + OUT_OF_PLAY_TOLERANCE_M):
        return None
    if ax <= CENTER_SPOT_RADIUS_M and ay <= CENTER_SPOT_RADIUS_M:
        return "kickoff"
    if (ax >= PITCH_HALF_LENGTH_M - CORNER_RADIUS_M
            and ay >= PITCH_HALF_WIDTH_M - CORNER_RADIUS_M):
        return "corner"
    if (ax >= PITCH_HALF_LENGTH_M - GOAL_AREA_DEPTH_M
            and ay <= GOAL_AREA_HALF_WIDTH_M):
        return "goal_kick"
    if ay >= PITCH_HALF_WIDTH_M - THROW_IN_BAND_M:
        return "throw_in"
    return "free_kick"


def infer_restarts(kicks, dead_periods) -> list:
    restarts = []
    for kick in kicks:
        preceded = any(
            start <= kick.t <= end + DEAD_TO_KICK_MAX_GAP_S
            for start, end in dead_periods)
        kind = classify_restart_location(kick.xy) if preceded else None
        if kind is not None:
            restarts.append(Restart(kick.t, kick.xy, kind))
    return restarts
