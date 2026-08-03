"""FIFA-style vertical spotlight rendering."""
from __future__ import annotations

import cv2
import numpy as np

SPOTLIGHT_COLOR_BGR = (90, 230, 215)


def draw_vertical_beam(
    frame: np.ndarray,
    foot: tuple[int, int],
    color: tuple[int, int, int] = SPOTLIGHT_COLOR_BGR,
    alpha: float = 0.3,
) -> None:
    """Draw a soft yellow-green shaft ending in an elliptical ground glow."""
    height, width = frame.shape[:2]
    x = max(0, min(width - 1, int(foot[0])))
    y = max(0, min(height - 1, int(foot[1])))
    shaft_half = max(10, min(width // 28, 70))
    ground_half = max(32, min(width // 16, 120))

    shaft = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        shaft,
        np.array([
            (x - shaft_half, 0), (x + shaft_half, 0),
            (x + ground_half, y), (x - ground_half, y),
        ], dtype=np.int32),
        210,
        cv2.LINE_AA,
    )
    shaft = cv2.GaussianBlur(shaft, (0, 0), sigmaX=max(8, width // 100))
    shaft = shaft.astype(float) * np.linspace(0.55, 0.9, height)[:, None]

    ground = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        ground,
        (x, y),
        (ground_half, max(8, ground_half // 4)),
        0,
        0,
        360,
        255,
        -1,
    )
    ground = cv2.GaussianBlur(
        ground, (0, 0), sigmaX=max(6, ground_half // 5),
    )

    # ponytail: two masks match the broadcast effect; use particles only for animated dust.
    mask = np.maximum(shaft, ground).clip(0, 255)[:, :, None] / 255.0 * float(alpha)
    tint = np.asarray(color, dtype=float).reshape(1, 1, 3)
    frame[:] = (
        frame.astype(float) * (1 - mask) + tint * mask
    ).clip(0, 255).astype(np.uint8)


def compute_beam_alpha(frame_offset: int, half_duration_frames: int, alpha_max: float) -> float:
    """Fade in/out alpha based on frame distance from event center."""
    if abs(frame_offset) >= half_duration_frames:
        return 0.0
    return alpha_max * (1 - abs(frame_offset) / half_duration_frames)
