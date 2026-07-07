"""Tests for pklz_to_json converter."""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json


def _write_minimal_pklz(path: Path, video_id: str, h_valid: bool) -> None:
    h = np.eye(3) if h_valid else None
    image_df = pd.DataFrame([
        {"id": "img1", "frame": 0, "file_path": "img1/000001.jpg", "h": h, "video_id": int(video_id)},
        {"id": "img2", "frame": 1, "file_path": "img1/000002.jpg", "h": None, "video_id": int(video_id)},
    ])
    det_df = pd.DataFrame([
        {
            "image_id": "img1", "track_id": 1, "role": "player", "team": "left",
            "jersey_number": "10", "bbox_ltwh": [100.0, 200.0, 50.0, 80.0],
            "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0,
                           "x_bottom_left": 0.5, "y_bottom_left": 2.0,
                           "x_bottom_right": 1.5, "y_bottom_right": 2.0},
        },
        {
            "image_id": "img2", "track_id": 1, "role": "player", "team": "left",
            "jersey_number": "10", "bbox_ltwh": [110.0, 210.0, 50.0, 80.0],
            "bbox_pitch": {"x_bottom_middle": 99.0, "y_bottom_middle": 99.0,
                           "x_bottom_left": 98.0, "y_bottom_left": 99.0,
                           "x_bottom_right": 100.0, "y_bottom_right": 99.0},
        },
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        with z.open(f"{video_id}.pkl", "w") as f:
            pickle.dump(det_df, f)
        with z.open(f"{video_id}_image.pkl", "w") as f:
            pickle.dump(image_df, f)


def test_invalid_h_strips_bbox_pitch_keeps_bbox_image(tmp_path: Path):
    pklz = tmp_path / "state.pklz"
    out = tmp_path / "out"
    _write_minimal_pklz(pklz, "148", h_valid=True)

    convert_pklz_to_json(pklz, "148", out)

    preds = json.loads((out / "predictions.json").read_text())
    by_image = {a["image_id"]: a for a in preds["annotations"]}

    assert by_image["img1"]["bbox_pitch"] is not None
    assert by_image["img1"]["bbox_image"] is not None
    # frame 1 has invalid h → stale bbox_pitch must not leak
    assert by_image["img2"]["bbox_pitch"] is None
    assert by_image["img2"]["bbox_image"] is not None

    homo = json.loads((out / "homography_per_frame.json").read_text())
    assert homo["frames"]["img1"]["valid"] is True
    assert homo["frames"]["img2"]["valid"] is False
