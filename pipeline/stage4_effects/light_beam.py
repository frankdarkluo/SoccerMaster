"""Perspective-correct cone light beam + foot marker rendering."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_foot_marker(
    frame: np.ndarray,
    center: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 20,
    alpha: float = 0.4,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius * 1.5), color, 1, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(overlay, (0, 0), sigmaX=5)
    cv2.addWeighted(blurred, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_cone_beam(
    frame: np.ndarray,
    origin: Tuple[int, int],
    target: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.3,
    width_base: int = 40,
    spread: float = 0.3,
) -> None:
    """Draw perspective cone beam from origin toward target."""
    ox, oy = origin
    tx, ty = target

    dx, dy = tx - ox, ty - oy
    length = max(1, int(np.hypot(dx, dy)))
    nx, ny = -dy / length, dx / length

    w0 = width_base / 2
    w1 = w0 + length * spread

    pts = np.array([
        [ox + nx * w0, oy + ny * w0],
        [ox - nx * w0, oy - ny * w0],
        [tx - nx * w1, ty - ny * w1],
        [tx + nx * w1, ty + ny * w1],
    ], dtype=np.int32)

    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay, [pts], color)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(15, length // 10))

    mask = overlay.astype(float) / 255.0
    frame[:] = (
        frame.astype(float) * (1 - mask * alpha) + overlay.astype(float) * alpha
    ).clip(0, 255).astype(np.uint8)


def compute_beam_alpha(frame_offset: int, half_duration_frames: int, alpha_max: float) -> float:
    """Fade in/out alpha based on frame distance from event center."""
    if abs(frame_offset) >= half_duration_frames:
        return 0.0
    return alpha_max * (1 - abs(frame_offset) / half_duration_frames)
