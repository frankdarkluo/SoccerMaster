"""Frame index helpers for Labels-GameState / predictions.json."""
from __future__ import annotations

from typing import Dict, Tuple


def frame_index_from_labels(data: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    image_id_to_frame: Dict[str, int] = {}
    frame_to_image_id: Dict[int, str] = {}
    for image in data.get("images", []):
        image_id = image.get("image_id")
        file_name = image.get("file_name")
        if image_id is None or not file_name:
            continue
        frame_num = int(str(file_name).split(".")[0])
        image_id_to_frame[str(image_id)] = frame_num
        frame_to_image_id[frame_num] = str(image_id)
    return image_id_to_frame, frame_to_image_id
