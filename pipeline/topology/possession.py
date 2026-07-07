from typing import Optional

import numpy as np


def possession_team(window) -> Optional[str]:
    """Return the team whose window-median player point is nearest the ball."""
    ball = window.get("ball")
    if ball is None:
        return None

    best_team = None
    best_distance = float("inf")
    for team, info in window["teams"].items():
        pts = info["players"]
        if len(pts) == 0:
            continue
        distance = float(np.min(np.linalg.norm(pts - ball, axis=1)))
        if distance < best_distance:
            best_team = team
            best_distance = distance

    return best_team
