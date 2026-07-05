"""Homography projection utilities for pitch↔image coordinate mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_homography(homo_json_path: Path) -> Dict[str, dict]:
    with open(homo_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", {})


def pitch_to_image(point_pitch: Tuple[float, float], h_inv: np.ndarray) -> Optional[Tuple[int, int]]:
    """Project a pitch-coordinate point to image pixel coordinates."""
    px, py = point_pitch
    p = h_inv @ np.array([px, py, 1.0], dtype=float)
    if abs(p[2]) < 1e-9:
        return None
    ix, iy = p[0] / p[2], p[1] / p[2]
    return int(round(ix)), int(round(iy))


def get_h_inv_for_frame(homo_frames: dict, image_id: str) -> Optional[np.ndarray]:
    entry = homo_frames.get(str(image_id))
    if entry is None or not entry.get("valid"):
        return None
    h_inv = entry.get("H_inv")
    if h_inv is None:
        return None
    return np.array(h_inv, dtype=float)
