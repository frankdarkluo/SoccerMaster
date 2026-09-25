"""Tactical topology line rendering on original video frames."""
from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np

from pipeline.stage4_effects.beam_targets import foot_point_from_bbox


def draw_player_marker(
    frame: np.ndarray,
    center: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    radius: int = 6,
    alpha: float = 0.5,
) -> None:
    """Solid dot with soft glow halo."""
    overlay = np.zeros_like(frame)
    cv2.circle(overlay, center, radius * 3, color, -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius * 2)
    mask = overlay.astype(float) / 255.0
    frame[:] = (
        frame.astype(float) * (1 - mask * alpha * 0.4)
        + overlay.astype(float) * alpha * 0.4
    ).clip(0, 255).astype(np.uint8)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def draw_formation_lines(
    frame: np.ndarray,
    positions: List[Tuple[int, int]],
    adjacency: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 150, 50),
    alpha: float = 0.35,
) -> None:
    """Semi-transparent lines with a soft glow underneath."""
    if not adjacency:
        return
    glow_layer = np.zeros_like(frame)
    line_layer = np.zeros_like(frame)
    for i, j in adjacency:
        if i < len(positions) and j < len(positions):
            cv2.line(glow_layer, positions[i], positions[j], color, 4, cv2.LINE_AA)
            cv2.line(line_layer, positions[i], positions[j], color, 1, cv2.LINE_AA)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=6)
    combined = np.maximum(glow_layer, line_layer)
    mask = (combined.astype(float) / 255.0)
    frame[:] = (
        frame.astype(float) * (1 - mask * alpha) + combined.astype(float) * alpha
    ).clip(0, 255).astype(np.uint8)

TARGET_LINE_ROLES = {
    "attacking_forward",
    "attacking_midfield",
    "defending_defensive",
}


def select_topology_window(topology: dict | None, timestamp_s: float) -> dict | None:
    """Choose the latest overlapping one-second topology window."""
    if not topology:
        return None
    matches = [
        window for window in topology.get("windows", [])
        if float(window["start_s"]) <= timestamp_s <= float(window["end_s"])
    ]
    return max(matches, key=lambda window: float(window["start_s"])) if matches else None


def draw_striped_polygon(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    *,
    stripe_color: Tuple[int, int, int] = (245, 245, 245),
    alpha: float = 0.16,
    spacing: int = 16,
) -> None:
    """Draw a restrained striped polygon without affecting pixels outside it."""
    if len(points) < 3:
        return
    polygon = np.asarray(points, dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    height, width = mask.shape
    stripe_mask = np.zeros_like(mask)
    for x in range(-height, width + height, max(4, int(spacing))):
        cv2.line(stripe_mask, (x, height), (x + height, 0), 255, 2, cv2.LINE_AA)
    overlay[(mask > 0) & (stripe_mask > 0)] = stripe_color
    blended = cv2.addWeighted(frame, 1 - float(alpha), overlay, float(alpha), 0)
    frame[mask > 0] = blended[mask > 0]


def draw_confirmed_topology(
    frame: np.ndarray,
    annotations: list[dict],
    topology_window: dict | None,
    *,
    attacking_team: str,
    homography_valid: bool,
    pitch_to_image_fn: Callable[[Tuple[float, float]], Tuple[int, int] | None] | None,
    style: dict,
    alpha: float | None = None,
) -> list[str]:
    """Draw at most three rule-confirmed lines and their selective zones."""
    if not topology_window or topology_window.get("attacking_team") != attacking_team:
        return []
    by_track = {
        int(annotation["track_id"]): annotation
        for annotation in annotations
        if annotation.get("track_id") is not None
    }
    drawn = []
    line_alpha = float(style["line_alpha"] if alpha is None else alpha)
    lines = [
        line for line in topology_window.get("lines", [])
        if line.get("status") == "renderable"
        and line.get("role") in TARGET_LINE_ROLES
    ][:3]
    for line in lines:
        color = (
            style["attacker_color_bgr"]
            if line["role"].startswith("attacking_")
            else style["defender_color_bgr"]
        )
        points = []
        for member in sorted(line.get("members", []), key=lambda item: item["y_m"]):
            annotation = by_track.get(int(member["track_id"]))
            point = (
                foot_point_from_bbox(annotation.get("bbox_image") or {})
                if annotation else None
            )
            if point is not None:
                points.append(point)
        if len(points) < 2:
            continue
        draw_formation_lines(
            frame,
            points,
            [(index, index + 1) for index in range(len(points) - 1)],
            color=color,
            alpha=line_alpha,
        )
        for point in points:
            draw_player_marker(frame, point, color=color, alpha=line_alpha)
        drawn.append(line["line_id"])

    if not homography_valid or pitch_to_image_fn is None:
        return drawn
    for zone in topology_window.get("zones", []):
        if zone.get("status") != "renderable":
            continue
        supporting = zone.get("supporting_line_ids") or []
        if not all(line_id in drawn for line_id in supporting):
            continue
        points = [
            point for xy in zone.get("polygon_pitch", [])
            if (point := pitch_to_image_fn(tuple(map(float, xy)))) is not None
        ]
        if len(points) < 3:
            continue
        color = (
            style.get("inter_line_color_bgr", (220, 160, 60))
            if zone.get("type") == "inter_line_space"
            else (
                style["attacker_color_bgr"]
                if zone.get("team") == attacking_team
                else style["defender_color_bgr"]
            )
        )
        draw_striped_polygon(
            frame,
            points,
            color,
            stripe_color=style["white_color_bgr"],
            alpha=float(style["zone_alpha"]),
            spacing=int(style["stripe_spacing"]),
        )
    return drawn
