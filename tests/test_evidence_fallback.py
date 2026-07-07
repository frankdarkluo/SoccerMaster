"""Tests for Stage-2 evidence image-space fallback (H2)."""
from __future__ import annotations

from pipeline.stage2_events.evidence import _homography_valid, image_space_goal_point


def test_homography_valid_respects_valid_flag():
    homo = {"img1": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "valid": True}}
    assert _homography_valid(homo, "img1") is True
    homo_invalid = {"img2": {"H": None, "valid": False}}
    assert _homography_valid(homo_invalid, "img2") is False


def test_image_space_goal_point_team_direction():
    bbox = {"x_center": 960, "y_center": 540}
    left_gx, _ = image_space_goal_point(bbox, "left", 1920)
    right_gx, _ = image_space_goal_point(bbox, "right", 1920)
    assert left_gx > 960  # attacks right goal → right side of frame
    assert right_gx < 960  # attacks left goal → left side of frame
