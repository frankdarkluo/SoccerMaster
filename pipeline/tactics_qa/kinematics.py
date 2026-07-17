"""Deterministic ball kinematics; missing track spans remain missing."""
import math
from dataclasses import dataclass

SMOOTH_WINDOW_FRAMES = 5
MAX_STEP_GAP_FRAMES = 3
KICK_MIN_SPEED_MPS = 8.0
KICK_PRE_MAX_SPEED_MPS = 3.0
KICK_LOOKBACK_S = 0.4
DEAD_MAX_SPEED_MPS = 0.8
DEAD_MIN_DURATION_S = 1.0


@dataclass(frozen=True)
class Kick:
    t: float
    xy: tuple
    speed_mps: float


def ball_speeds(ball: dict, fps: float) -> dict:
    raw = {}
    frames = sorted(ball)
    for before, after in zip(frames, frames[1:]):
        gap = after - before
        if gap > MAX_STEP_GAP_FRAMES:
            continue
        x0, y0 = ball[before]
        x1, y1 = ball[after]
        raw[after] = math.hypot(x1 - x0, y1 - y0) * fps / gap
    return {
        frame: sum(raw[f] for f in range(frame - SMOOTH_WINDOW_FRAMES, frame + 1)
                   if f in raw)
        / sum(f in raw for f in range(frame - SMOOTH_WINDOW_FRAMES, frame + 1))
        for frame in raw
    }


def detect_kicks(ball: dict, fps: float) -> list:
    speeds = ball_speeds(ball, fps)
    kicks, last = [], -10**9
    lookback = int(KICK_LOOKBACK_S * fps)
    for frame in sorted(speeds):
        if speeds[frame] < KICK_MIN_SPEED_MPS or frame - last < lookback:
            continue
        prior = [speeds[f] for f in range(frame - lookback, frame) if f in speeds]
        if not prior or min(prior) > KICK_PRE_MAX_SPEED_MPS:
            continue
        xy = ball.get(frame) or ball.get(frame - 1)
        if xy is not None:
            kicks.append(Kick(frame / fps, xy, speeds[frame]))
            last = frame
    return kicks


def dead_ball_periods(ball: dict, fps: float) -> list:
    speeds = ball_speeds(ball, fps)
    if not speeds:
        return []
    periods, start, previous = [], None, None

    def close(end):
        if start is not None and (end - start) / fps >= DEAD_MIN_DURATION_S:
            periods.append((start / fps, end / fps))

    for frame in sorted(speeds):
        if previous is not None and frame - previous > MAX_STEP_GAP_FRAMES:
            close(previous)
            start = None
        if speeds[frame] < DEAD_MAX_SPEED_MPS:
            start = frame if start is None else start
        elif start is not None:
            close(frame)
            start = None
        previous = frame
    close(previous)
    first = min(speeds)
    if periods and periods[0][0] <= (first + SMOOTH_WINDOW_FRAMES + 1) / fps:
        periods[0] = (0.0, periods[0][1])
    return periods
