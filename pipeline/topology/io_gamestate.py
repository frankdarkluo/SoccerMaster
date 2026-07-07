import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    frame: int
    track_id: int
    team: Optional[str]
    role: str
    jersey_number: Optional[int]
    x: float
    y: float


def load_detections(json_path: str) -> List[Detection]:
    """Parse SoccerNet GameState detections with meter-space foot points."""
    with open(json_path) as f:
        data = json.load(f)

    frame_by_image_id = {}
    for image in data.get("images", []):
        image_id = image.get("image_id")
        file_name = image.get("file_name")
        if image_id is None or not file_name:
            continue
        frame_by_image_id[str(image_id)] = int(file_name.split(".")[0])

    dets: List[Detection] = []
    for annotation in data["annotations"]:
        bbox_pitch = annotation.get("bbox_pitch")
        if not isinstance(bbox_pitch, dict):
            continue

        x = bbox_pitch.get("x_bottom_middle")
        y = bbox_pitch.get("y_bottom_middle")
        if x is None or y is None:
            continue

        attrs = annotation.get("attributes", {}) or {}
        jersey = attrs.get("jersey")
        dets.append(
            Detection(
                frame=frame_by_image_id.get(
                    str(annotation["image_id"]), int(annotation["image_id"])
                ),
                track_id=int(annotation.get("track_id", annotation["id"])),
                team=attrs.get("team"),
                role=attrs.get("role", "other"),
                jersey_number=int(jersey) if jersey not in (None, "") else None,
                x=float(x),
                y=float(y),
            )
        )
    return dets
