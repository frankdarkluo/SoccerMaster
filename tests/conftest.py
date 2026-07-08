import json
from pathlib import Path

import numpy as np
import pytest


def _ann(ann_id, image_id, track_id, role, team, jersey, px, py,
         bx=100, by=200, bw=40, bh=100):
    return {
        "id": str(ann_id), "image_id": str(image_id), "track_id": track_id,
        "supercategory": "object", "category_id": 1,
        "bbox_image": {"x": bx, "y": by, "w": bw, "h": bh,
                       "x_center": bx + bw / 2, "y_center": by + bh / 2},
        "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                       "x_bottom_left": px - 0.3, "y_bottom_left": py,
                       "x_bottom_right": px + 0.3, "y_bottom_right": py},
        "attributes": {"role": role, "team": team, "jersey": jersey},
    }


def _ball_ann(ann_id, image_id, px, py):
    return {
        "id": str(ann_id), "image_id": str(image_id), "track_id": 99,
        "supercategory": "object", "category_id": 4,
        "bbox_image": {"x": 500, "y": 400, "w": 12, "h": 12,
                       "x_center": 506, "y_center": 406},
        "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                       "x_bottom_left": px, "y_bottom_left": py,
                       "x_bottom_right": px, "y_bottom_right": py},
        "attributes": {"role": "ball"},
    }


def _write_predictions(tmp_path, name, n_frames, ball_for, players_for):
    images, annotations = [], []
    aid = 1
    for f in range(1, n_frames + 1):
        image_id = f"3148{f:06d}"
        images.append({"image_id": image_id, "file_name": f"{f:06d}.jpg",
                       "width": 1920, "height": 1080})
        annotations.append(_ball_ann(aid, image_id, *ball_for(f))); aid += 1
        for (tid, role, team, jersey, px, py) in players_for(f):
            annotations.append(_ann(aid, image_id, tid, role, team, jersey, px, py)); aid += 1
    data = {"info": {"name": "FIXT", "fps": 25}, "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "player"}, {"id": 4, "name": "ball"}]}
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def gk_boot_predictions(tmp_path):
    """Keeper #1 (left, own box at x=-44) boots long to #9 at x=+8. 30 frames."""
    def ball_for(f):
        if f <= 10:
            return (-44.0, 0.0)                    # at the keeper's feet
        if f <= 20:
            return (-44.0 + (f - 10) * 5.2, 0.0)   # 5.2 m/frame = 130 m/s spike... scaled below
        return (8.0, 0.0)                          # settled at #9

    # NOTE: 5.2 m/frame is unrealistically fast but safely above KICK_SPEED_MPS;
    # what matters is monotonic travel from own box to midfield.
    def players_for(f):
        return [
            (1, "goalkeeper", "left", "1", -44.0, 0.0),
            (9, "player", "left", "9", 8.0, 0.0),
            (4, "player", "right", "4", 20.0, 5.0),
        ]
    return _write_predictions(tmp_path, "gk_boot.json", 30, ball_for, players_for)


@pytest.fixture
def pass_intercept_predictions(tmp_path):
    """#7 left → #10 left (pass), then #4 right wins it (interception). 25 frames."""
    def holder_for(f):
        if 1 <= f <= 6:
            return 7
        if 9 <= f <= 14:
            return 10
        if 17 <= f <= 22:
            return 4
        return None

    def ball_for(f):
        if f <= 8:
            return (min(f * 1.3, 8.0), 0.0)
        if f <= 16:
            return (8.0 + (f - 8) * 1.0, 0.0)
        return (16.0, 0.0)

    def players_for(f):
        h = holder_for(f)
        out = []
        for (tid, team, jersey, hx) in [(7, "left", "7", 0.0),
                                        (10, "left", "10", 8.0),
                                        (4, "right", "4", 16.0)]:
            px = ball_for(f)[0] if h == tid else hx
            out.append((tid, "player", team, jersey, px, 0.0))
        out.append((50, "referee", None, "", ball_for(f)[0], 1.0))
        return out
    return _write_predictions(tmp_path, "pass_intercept.json", 25, ball_for, players_for)


@pytest.fixture
def homography_file(tmp_path):
    frames = {}
    H = [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]]
    H_inv = np.linalg.inv(np.array(H)).tolist()
    for f in range(1, 31):
        frames[f"3148{f:06d}"] = {"H": H, "H_inv": H_inv, "valid": True}
    p = tmp_path / "homography_per_frame.json"
    p.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return p


@pytest.fixture
def frames_dir(tmp_path):
    import cv2
    d = tmp_path / "img1"
    d.mkdir()
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for f in range(1, 31):
        cv2.imwrite(str(d / f"{f:06d}.jpg"), img)
    return d


class MockVLMAdapter:
    """Injectable stand-in for DoubaoAPIAdapter/QwenLocalAdapter.

    `script` maps a substring of the prompt -> raw JSON string the model returns.
    Records every (prompt, visual_input) call for assertions.
    """
    def __init__(self, script=None, default='{"action": "none", "reason": "quiet"}'):
        self.script = script or {}
        self.default = default
        self.calls = []

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        self.calls.append((prompt, visual_input))
        for key, resp in self.script.items():
            if key in prompt:
                return resp
        return self.default


@pytest.fixture
def mock_adapter():
    return MockVLMAdapter
