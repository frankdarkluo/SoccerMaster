"""Resolve beam anchor points from predictions.json annotations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_predictions_index(predictions_json: Path) -> Tuple[Dict[int, str], Dict[str, List[dict]]]:
    with open(predictions_json, encoding="utf-8") as f:
        data = json.load(f)

    frame_to_image_id = {
        int(Path(image["file_name"]).stem): str(image["image_id"])
        for image in data.get("images", [])
    }
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


