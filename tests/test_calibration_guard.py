"""Tests for calibration structural guard."""
from __future__ import annotations

import pandas as pd

from pipeline.stage1_inference.calibration_guard import analyze_image_df


def _make_df(frames_kp_h):
    rows = []
    for i, (kp, h_ok) in enumerate(frames_kp_h):
        h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if h_ok else None
        kp_dict = {j: {"x": 0.5, "y": 0.5} for j in range(1, kp + 1)} if kp > 0 else {}
        rows.append({"frame": i, "keypoints": kp_dict, "h": h})
    return pd.DataFrame(rows)


def test_guard_flags_dropout_block():
    # 20 valid frames with 8 kp, then 30-frame block with 0 kp / invalid h
    data = [(8, True)] * 20 + [(0, False)] * 30 + [(8, True)] * 20
    report = analyze_image_df(_make_df(data), video_id="148")
    assert report.flagged
    assert len(report.invalid_blocks) >= 1
    block = report.invalid_blocks[0]
    assert block.start_frame == 20
    assert block.end_frame == 49
    assert block.zero_keypoint_frames == 30


def test_guard_passes_healthy_video():
    data = [(8, True)] * 100
    report = analyze_image_df(_make_df(data), video_id="116")
    assert not report.flagged
    assert report.valid_homography_frames == 100
