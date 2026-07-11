"""Resolve beam anchor points from predictions.json annotations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X


def load_predictions_index(predictions_json: Path) -> Tuple[Dict[int, str], Dict[str, List[dict]]]:
    with open(predictions_json, encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_frame, frame_to_image_id = frame_index_from_labels(data)
    anns_by_image: Dict[str, List[dict]] = {}
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        anns_by_image.setdefault(image_id, []).append(ann)
    return frame_to_image_id, anns_by_image


def foot_point_from_bbox(bbox_image: dict) -> Optional[Tuple[int, int]]:
    if not isinstance(bbox_image, dict):
        return None
    x_center = bbox_image.get("x_center")
    y = bbox_image.get("y")
    h = bbox_image.get("h")
    if x_center is None or y is None or h is None:
        return None
    return int(round(float(x_center))), int(round(float(y) + float(h)))


def center_point_from_bbox(bbox_image: dict) -> Optional[Tuple[int, int]]:
    if not isinstance(bbox_image, dict):
        return None
    x_center = bbox_image.get("x_center")
    y_center = bbox_image.get("y_center")
    if x_center is None or y_center is None:
        return None
    return int(round(float(x_center))), int(round(float(y_center)))


def _find_ball(annotations: List[dict]) -> Optional[dict]:
    for ann in annotations:
        role = (ann.get("attributes") or {}).get("role")
        if role == "ball":
            return ann
    return None


def _find_player(
    annotations: List[dict],
    jersey: Optional[str],
    team: Optional[str],
) -> Optional[dict]:
    if jersey:
        for ann in annotations:
            attrs = ann.get("attributes") or {}
            if str(attrs.get("jersey", "")) == str(jersey):
                return ann
    if team:
        players = [
            ann for ann in annotations
            if (ann.get("attributes") or {}).get("team") == team
            and (ann.get("attributes") or {}).get("role") in ("player", "goalkeeper")
        ]
        if players:
            return players[0]
    return None


def _nearest_field_player_to_ball(
    annotations: List[dict],
    ball_ann: dict,
    max_dist_m: float = 5.0,
    prefer_outfield: bool = True,
) -> Optional[dict]:
    ball_pitch = ball_ann.get("bbox_pitch") or {}
    bx = ball_pitch.get("x_bottom_middle")
    by = ball_pitch.get("y_bottom_middle")
    if bx is None or by is None:
        return None

    candidates: List[Tuple[float, str, dict]] = []
    for ann in annotations:
        attrs = ann.get("attributes") or {}
        role = attrs.get("role")
        if role not in ("player", "goalkeeper"):
            continue
        bp = ann.get("bbox_pitch") or {}
        px = bp.get("x_bottom_middle")
        py = bp.get("y_bottom_middle")
        if px is None or py is None:
            continue
        dist = math.hypot(float(px) - float(bx), float(py) - float(by))
        candidates.append((dist, role, ann))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    if prefer_outfield:
        outfield = [item for item in candidates if item[1] == "player"]
        if outfield and outfield[0][0] <= max_dist_m:
            return outfield[0][2]

    return candidates[0][2] if candidates[0][0] <= max_dist_m else None


def goal_center_pitch(team: Optional[str]) -> Tuple[float, float]:
    attack_sign = 1.0 if team != "right" else -1.0
    return GOAL_X * attack_sign, 0.0


def resolve_beam_points(
    event: dict,
    annotations: List[dict],
    pitch_to_image_fn=None,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Return (origin_img, target_img) for a light beam.

    Uses bbox_image foot points. Target prefers ball center, then projected goal center.
    """
    ball = _find_ball(annotations)
    player = _find_player(
        annotations,
        event.get("player_jersey"),
        event.get("player_team"),
    )
    if player is None and ball is not None:
        max_dist = 30.0 if event.get("event_code") in ("football.goal", "football.shoot") else 5.0
        player = _nearest_field_player_to_ball(
            annotations,
            ball,
            max_dist_m=max_dist,
            prefer_outfield=event.get("event_code") != "football.save",
        )
    if player is None:
        return None

    origin = foot_point_from_bbox(player.get("bbox_image") or {})
    if origin is None:
        return None

    if event.get("target_jersey"):
        target_player = _find_player(
            annotations,
            event.get("target_jersey"),
            event.get("target_team"),
        )
        if target_player is not None:
            target = foot_point_from_bbox(target_player.get("bbox_image") or {})
            if target is not None:
                return origin, target

    if ball is not None:
        target = center_point_from_bbox(ball.get("bbox_image") or {})
        if target is not None:
            return origin, target

    if pitch_to_image_fn is not None:
        gx, gy = goal_center_pitch(event.get("player_team"))
        projected = pitch_to_image_fn((gx, gy))
        if projected is not None:
            return origin, projected

    # Fallback: extend beam upward in image space toward top of frame.
    tx, ty = origin
    return origin, (tx, max(0, ty - 250))
