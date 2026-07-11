"""Tactical topology line rendering on original video frames."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay


def delaunay_adjacency(
    positions: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Build edge list from Delaunay triangulation (no crossing edges)."""
    if len(positions) < 3:
        if len(positions) == 2:
            return [(0, 1)]
        return []
    pts = np.array(positions, dtype=float)
    tri = Delaunay(pts)
    edges: set[Tuple[int, int]] = set()
    for simplex in tri.simplices:
        for k in range(3):
            a, b = int(simplex[k]), int(simplex[(k + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


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


def draw_dashed_line(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 15,
) -> None:
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = max(1, int(np.hypot(dx, dy)))
    for i in range(0, dist, dash_length * 2):
        t1 = i / dist
        t2 = min((i + dash_length) / dist, 1.0)
        x1 = int(pt1[0] + dx * t1)
        y1 = int(pt1[1] + dy * t1)
        x2 = int(pt1[0] + dx * t2)
        y2 = int(pt1[1] + dy * t2)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_arrow(
    frame: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.arrowedLine(frame, start, end, color, thickness, cv2.LINE_AA, tipLength=0.15)


def draw_running_path(
    frame: np.ndarray,
    current: Tuple[int, int],
    predicted: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    draw_dashed_line(frame, current, predicted, color, thickness=2)
    draw_arrow(frame, current, predicted, color, thickness=2)


def draw_pressing_line(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 200, 100),
) -> None:
    """Draw horizontal dashed pressing line through projected pitch points."""
    if len(points) < 2:
        return
    sorted_pts = sorted(points, key=lambda p: p[0])
    draw_dashed_line(frame, sorted_pts[0], sorted_pts[-1], color, thickness=2, dash_length=20)
