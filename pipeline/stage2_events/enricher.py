"""Enrich events with computable tag dimensions."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.types import Event, FrameData
from pipeline.utils.pitch import (
    GOAL_X,
    PENALTY_AREA_LENGTH,
    PENALTY_AREA_WIDTH,
    PITCH_LENGTH,
    SIX_YARD_LENGTH,
    SIX_YARD_WIDTH,
)


def _attack_sign(team: Optional[str]) -> int:
    return 1 if team != "right" else -1


def _goal_x_for_team(team: Optional[str]) -> float:
    return GOAL_X * _attack_sign(team)


def _pitch_zone(ball_x: float, ball_y: float, goal_x: float) -> str:
    dx = abs(abs(ball_x) - goal_x)
    dy = abs(ball_y)
    if dx <= SIX_YARD_LENGTH and dy <= SIX_YARD_WIDTH / 2:
        return "six_yard_box"
    if dx <= PENALTY_AREA_LENGTH and dy <= PENALTY_AREA_WIDTH / 2:
        return "inside_box"
    if abs(ball_x) < PITCH_LENGTH / 4:
        return "halfway_line"
    return "outside_box"


def _shot_distance(ball_x: float, ball_y: float, goal_x: float) -> str:
    dist = math.hypot(abs(ball_x) - goal_x, ball_y)
    if dist <= 8:
        return "close_range"
    if dist <= 16:
        return "mid_range"
    if dist <= 30:
        return "long_range"
    return "half_way"


def _pass_distance(passer_x, passer_y, receiver_x, receiver_y) -> str:
    dist = math.hypot(receiver_x - passer_x, receiver_y - passer_y)
    if dist < 15:
        return "short"
    if dist < 25:
        return "medium"
    return "long"


def _pass_direction(passer_x, passer_y, receiver_x, receiver_y, attack_dir: int) -> str:
    dx = (receiver_x - passer_x) * attack_dir
    dy = receiver_y - passer_y
    if dx < -5:
        return "back_pass"
    if abs(dy) > abs(dx) * 2:
        return "lateral"
    if dx > 5:
        return "forward"
    return "lateral"


def enrich_tags(
    event: Event,
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    goal_x: float = GOAL_X,
    passer_x: Optional[float] = None,
    passer_y: Optional[float] = None,
    receiver_x: Optional[float] = None,
    receiver_y: Optional[float] = None,
    attack_dir: int = 1,
) -> Event:
    """Add computable tags to an event. Returns the same event, mutated."""
    code = event.event_code

    if code in ("football.shoot", "football.goal"):
        event.tags["pitch_zone"] = _pitch_zone(ball_x, ball_y, goal_x)
        event.tags["shot_distance"] = _shot_distance(ball_x, ball_y, goal_x)
        event.tags["pattern_of_play"] = event.tags.get("pattern_of_play", "open_play")

    if code == "football.pass" and passer_x is not None and receiver_x is not None:
        event.tags["pass_distance"] = _pass_distance(passer_x, passer_y, receiver_x, receiver_y)
        event.tags["pass_direction"] = _pass_direction(
            passer_x, passer_y, receiver_x, receiver_y, attack_dir
        )

    return event


def _find_player_xy(frame: FrameData, jersey: Optional[str], team: Optional[str]) -> Optional[Tuple[float, float]]:
    if jersey:
        for player in frame.players:
            if str(player.get("jersey", "")) == str(jersey):
                return player["x"], player["y"]
    if team:
        for player in frame.players:
            if player.get("team") == team:
                return player["x"], player["y"]
    return None


def enrich_events(events: List[Event], frames: List[FrameData]) -> List[Event]:
    """Enrich a list of events using per-frame ball and player positions."""
    frame_by_id: Dict[int, FrameData] = {f.frame_id: f for f in frames}
    for event in events:
        frame = frame_by_id.get(event.frame_id)
        if frame is None:
            continue

        ball_x, ball_y = frame.ball_xy or (0.0, 0.0)
        attack_dir = _attack_sign(event.player_team)
        goal_x = _goal_x_for_team(event.player_team)

        passer_x = passer_y = receiver_x = receiver_y = None
        if event.event_code == "football.pass":
            passer = _find_player_xy(frame, event.player_jersey, event.player_team)
            receiver = _find_player_xy(frame, event.target_jersey, event.target_team)
            if passer:
                passer_x, passer_y = passer
            if receiver:
                receiver_x, receiver_y = receiver

        enrich_tags(
            event,
            ball_x=ball_x,
            ball_y=ball_y,
            goal_x=goal_x,
            passer_x=passer_x,
            passer_y=passer_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            attack_dir=attack_dir,
        )
    return events
