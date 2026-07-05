"""Structural evidence frames for VLM verification.

Draws a dense frame burst around an event: actor red box, ball marker,
receiver box when available, and a homography-projected goal-direction
arrow for shooting events. The output is a small set of annotated JPGs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.stage2_events.types import Event
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X

ACTOR_COLOR = (0, 0, 255)
RECEIVER_COLOR = (0, 200, 255)
BALL_COLOR = (0, 255, 0)
GOAL_ARROW_COLOR = (0, 255, 255)


def load_homography(path: str) -> Dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("frames", data)


def _frame_to_image_id(predictions_json_path: str) -> Dict[int, str]:
    data = json.loads(Path(predictions_json_path).read_text(encoding="utf-8"))
    _, frame_to_image = frame_index_from_labels(data)
    return frame_to_image


def _receiver_bbox_for_frame(
    predictions_json_path: str,
    frame_num: int,
    target_jersey: Optional[str],
    target_team: Optional[str],
) -> Optional[dict]:
    data = json.loads(Path(predictions_json_path).read_text(encoding="utf-8"))
    image_id_to_frame, _ = frame_index_from_labels(data)
    for ann in data.get("annotations", []):
        frame_id = image_id_to_frame.get(str(ann.get("image_id", "")))
        if frame_id != frame_num:
            continue
        attrs = ann.get("attributes") or {}
        if target_jersey is not None and str(attrs.get("jersey", "")) != str(target_jersey):
            continue
        if target_team in ("left", "right") and attrs.get("team") != target_team:
            continue
        bbox = ann.get("bbox_image")
        if isinstance(bbox, dict):
            return bbox
    return None


def project_pitch_to_image(
    homo: Dict[str, dict],
    image_id: str,
    px: float,
    py: float,
) -> Optional[Tuple[float, float]]:
    entry = homo.get(image_id)
    if not entry:
        return None
    matrix = entry.get("H")
    if matrix is None:
        matrix = entry.get("H_inv")
    if matrix is None:
        return None
    h = np.array(matrix, dtype=float)
    v = h @ np.array([px, py, 1.0], dtype=float)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def build_bbox_index(predictions_json_path: str) -> Dict[Tuple[int, int], dict]:
    data = json.loads(Path(predictions_json_path).read_text(encoding="utf-8"))
    image_id_to_frame, _ = frame_index_from_labels(data)
    index: Dict[Tuple[int, int], dict] = {}
    for ann in data.get("annotations", []):
        frame_num = image_id_to_frame.get(str(ann.get("image_id", "")))
        track_id = ann.get("track_id")
        bbox = ann.get("bbox_image")
        if frame_num is None or track_id is None or not isinstance(bbox, dict):
            continue
        index[(frame_num, int(track_id))] = bbox
    return index


def _box(img, bbox, color, thickness=3, expand: int = 0):
    x = int(bbox.get("x", 0)) - expand
    y = int(bbox.get("y", 0)) - expand
    w = int(bbox.get("w", 0)) + 2 * expand
    h = int(bbox.get("h", 0)) + 2 * expand
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def _center(bbox: dict) -> Tuple[int, int]:
    cx = bbox.get("x_center", bbox.get("x", 0))
    cy = bbox.get("y_center", bbox.get("y", 0))
    return int(cx), int(cy)


def build_evidence_frames(
    event: Event,
    frames_dir: Path,
    bbox_index: Dict[Tuple[int, int], dict],
    homo: Dict[str, dict],
    predictions_json_path: str,
    out_dir: Path,
    fps: int = 25,
    window_s: float = 0.5,
    max_frames: int = 12,
) -> List[Path]:
    """Return sorted list of annotated JPG paths."""
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_to_image = _frame_to_image_id(predictions_json_path)

    half = int(round(window_s * fps))
    candidate_fnums = [
        f
        for f in range(max(1, event.frame_id - half), event.frame_id + half + 1)
        if (frames_dir / f"{f:06d}.jpg").exists()
    ]
    if not candidate_fnums:
        return []
    if len(candidate_fnums) > max_frames:
        step = len(candidate_fnums) / max_frames
        candidate_fnums = [candidate_fnums[int(i * step)] for i in range(max_frames)]

    goal_x = GOAL_X if event.player_team != "right" else -GOAL_X
    out_paths: List[Path] = []

    for fnum in candidate_fnums:
        img = cv2.imread(str(frames_dir / f"{fnum:06d}.jpg"))
        if img is None:
            continue

        if event.track_id is not None:
            actor_bbox = bbox_index.get((fnum, int(event.track_id)))
            if actor_bbox:
                _box(img, actor_bbox, ACTOR_COLOR)
        else:
            actor_bbox = None

        ball_bbox = bbox_index.get((fnum, 99))
        if ball_bbox:
            cx, cy = _center(ball_bbox)
            cv2.circle(img, (cx, cy), 8, BALL_COLOR, 2)

        if event.event_code == "football.pass" and event.target_jersey:
            receiver_bbox = _receiver_bbox_for_frame(
                predictions_json_path, fnum, event.target_jersey, event.target_team
            )
            if receiver_bbox:
                _box(img, receiver_bbox, RECEIVER_COLOR, thickness=2, expand=4)

        if event.event_code in ("football.shoot", "football.goal") and actor_bbox:
            image_id = frame_to_image.get(fnum)
            if image_id is not None:
                goal_point = project_pitch_to_image(homo, image_id, goal_x, 0.0)
                if goal_point is not None:
                    ax, ay = _center(actor_bbox)
                    gx, gy = int(goal_point[0]), int(goal_point[1])
                    cv2.arrowedLine(
                        img,
                        (ax, ay),
                        (gx, gy),
                        GOAL_ARROW_COLOR,
                        2,
                        tipLength=0.05,
                    )

        out = out_dir / f"{event.event_id}_{fnum:06d}.jpg"
        cv2.imwrite(str(out), img)
        out_paths.append(out)

    return out_paths
