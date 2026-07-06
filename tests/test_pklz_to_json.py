import json
import pickle
import zipfile

import numpy as np
import pandas as pd

from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json


def _pitch(x, y):
    return {
        "x_bottom_left": x - 0.1,
        "y_bottom_left": y,
        "x_bottom_right": x + 0.1,
        "y_bottom_right": y,
        "x_bottom_middle": x,
        "y_bottom_middle": y,
    }


def test_invalid_homography_frame_keeps_bbox_image_but_drops_stale_bbox_pitch(tmp_path):
    detections_df = pd.DataFrame([
        {
            "image_id": "img-valid",
            "track_id": 7,
            "role": "player",
            "team": "left",
            "jersey_number": "14",
            "bbox_ltwh": [10, 20, 30, 40],
            "bbox_pitch": _pitch(1.0, 2.0),
        },
        {
            "image_id": "img-invalid",
            "track_id": 8,
            "role": "player",
            "team": "left",
            "jersey_number": "10",
            "bbox_ltwh": [100, 200, 30, 40],
            "bbox_pitch": _pitch(-6.8, 6.1),
        },
    ])
    image_df = pd.DataFrame([
        {
            "id": "img-valid",
            "file_path": "000001.jpg",
            "frame": 0,
            "is_labeled": True,
            "h": np.eye(3),
        },
        {
            "id": "img-invalid",
            "file_path": "000002.jpg",
            "frame": 1,
            "is_labeled": True,
            "h": np.nan,
        },
    ])

    pklz = tmp_path / "state.pklz"
    with zipfile.ZipFile(pklz, "w") as z:
        z.writestr("148.pkl", pickle.dumps(detections_df))
        z.writestr("148_image.pkl", pickle.dumps(image_df))

    predictions_path, homography_path = convert_pklz_to_json(pklz, "148", tmp_path)

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    anns_by_image = {ann["image_id"]: ann for ann in predictions["annotations"]}
    assert anns_by_image["img-valid"]["bbox_pitch"]["x_bottom_middle"] == 1.0
    assert anns_by_image["img-valid"]["bbox_image"]["x_center"] == 25
    assert anns_by_image["img-invalid"]["bbox_pitch"] is None
    assert anns_by_image["img-invalid"]["bbox_image"]["x_center"] == 115

    homography = json.loads(homography_path.read_text(encoding="utf-8"))
    assert homography["frames"]["img-valid"]["valid"] is True
    assert homography["frames"]["img-invalid"]["valid"] is False
