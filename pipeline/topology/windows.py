from collections import defaultdict
from typing import List, Optional, Set

import numpy as np

from .io_gamestate import Detection


def sliding_windows(
    detections: List[Detection],
    fps: float,
    win_s: float = 3.0,
    stride_s: float = 1.0,
    in_play_frames: Optional[Set[int]] = None,
):
    """Build windows with one component-wise median point per visible track."""
    if not detections:
        return []
    if in_play_frames is not None:
        detections = [d for d in detections if d.frame in in_play_frames]
    if not detections:
        return []

    max_t = max(d.frame for d in detections) / fps
    windows = []
    t = 0.0
    while True:
        t_end = t + win_s
        lo, hi = t * fps, t_end * fps
        in_window = [d for d in detections if lo <= d.frame <= hi]

        by_team_track = defaultdict(lambda: defaultdict(list))
        ball_points = []
        for det in in_window:
            if det.role == "ball":
                ball_points.append((det.x, det.y))
            elif det.role == "player" and det.team in ("left", "right"):
                by_team_track[det.team][det.track_id].append((det.x, det.y))

        teams = {}
        for team, tracks in by_team_track.items():
            points = []
            track_ids = []
            for track_id, xy in tracks.items():
                points.append(np.median(np.asarray(xy, dtype=float), axis=0))
                track_ids.append(track_id)
            teams[team] = {"players": np.asarray(points), "track_ids": track_ids}

        ball = np.median(np.asarray(ball_points, dtype=float), axis=0) if ball_points else None
        windows.append(
            {
                "t_start": round(t, 3),
                "t_end": round(t_end, 3),
                "teams": teams,
                "ball": ball,
            }
        )

        t += stride_s
        if t >= max_t:
            break

    return windows
