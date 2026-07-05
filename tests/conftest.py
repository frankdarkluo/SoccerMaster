import json
from pathlib import Path
import numpy as np
import pytest


def _ann(ann_id, image_id, track_id, role, team, jersey, px, py, bx=100, by=200, bw=40, bh=100):
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


@pytest.fixture
def predictions_file(tmp_path):
    """Scripted clip: same-team pass (#7->#10 left) then interception (#4 right)."""
    images, annotations = [], []
    aid = 1
    # holder schedule: frame -> (holder pitch pos, ball pos following holder)
    # #7 left @ (0,0) frames 1-6, ball in flight 7-8, #10 left @ (8,0) frames 9-14,
    # flight 15-16, #4 right @ (16,0) frames 17-22.
    def holder_for(f):
        if 1 <= f <= 6:
            return (7, "left", "7", 0.0, 0.0)
        if 9 <= f <= 14:
            return (10, "left", "10", 8.0, 0.0)
        if 17 <= f <= 22:
            return (4, "right", "4", 16.0, 0.0)
        return None
    for f in range(1, 26):
        image_id = f"3148{f:06d}"
        images.append({"image_id": image_id, "file_name": f"{f:06d}.jpg",
                       "width": 1920, "height": 1080})
        h = holder_for(f)
        # ball glides between holders so nearest-player = intended holder
        if f <= 8:
            ball = (min(f * 1.3, 8.0), 0.0)
        elif f <= 16:
            ball = (8.0 + (f - 8) * 1.0, 0.0)
        else:
            ball = (16.0, 0.0)
        annotations.append(_ball_ann(aid, image_id, *ball)); aid += 1
        # place the three tracked players every frame; holder sits on the ball
        for (tid, team, jersey, hx, hy) in [(7, "left", "7", 0.0, 0.0),
                                            (10, "left", "10", 8.0, 0.0),
                                            (4, "right", "4", 16.0, 0.0)]:
            # one-frame team flicker on #10 at frame 11 to prove majority vote wins
            eff_team = "right" if (tid == 10 and f == 11) else team
            if h and tid == h[0]:
                px, py = ball  # holder tracks the ball exactly
            else:
                px, py = hx, hy
            annotations.append(_ann(aid, image_id, tid, "player", eff_team, jersey, px, py)); aid += 1
        # a referee near the ball to prove it is excluded from possession
        annotations.append(_ann(aid, image_id, 50, "referee", None, "", ball[0], ball[1] + 1.0)); aid += 1
    data = {"info": {"name": "FIXT", "fps": 25}, "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "player"}, {"id": 4, "name": "ball"}]}
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def homography_file(tmp_path):
    """Identity-ish homography keyed by image_id for every fixture frame."""
    frames = {}
    H = [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]]
    H_inv = np.linalg.inv(np.array(H)).tolist()
    for f in range(1, 26):
        frames[f"3148{f:06d}"] = {"H": H, "H_inv": H_inv, "valid": True}
    p = tmp_path / "homography_per_frame.json"
    p.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return p


@pytest.fixture
def frames_dir(tmp_path):
    """25 tiny black jpgs named 000001.jpg.. so evidence overlays have something to draw on."""
    import cv2
    d = tmp_path / "img1"
    d.mkdir()
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for f in range(1, 26):
        cv2.imwrite(str(d / f"{f:06d}.jpg"), img)
    return d


class MockVLMAdapter:
    """Injectable stand-in for DoubaoAPIAdapter/QwenLocalAdapter.

    `script` maps event_code substring -> raw JSON string the model would emit.
    Records every (prompt, visual_input) call for assertions.
    """
    def __init__(self, script=None, default='{"verdict": "confirm"}'):
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
