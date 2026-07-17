"""Load GSR predictions.json into frame-indexed pitch tracks."""
import json
from dataclasses import dataclass, field
from pathlib import Path

BALL_CATEGORY = 4
PERSON_CATEGORIES = {1, 2}


@dataclass
class GsrClip:
    fps: float
    n_frames: int
    ball: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)


def load_gsr(path) -> GsrClip:
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    frame_by_image = {
        str(image["image_id"]): int(Path(image["file_name"]).stem)
        for image in data["images"]
    }
    info = data["info"]
    clip = GsrClip(float(info.get("fps", info.get("frame_rate", 25.0))),
                   int(info["n_frames"]))
    for annotation in data["annotations"]:
        pitch = annotation.get("bbox_pitch") or {}
        x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
        frame = frame_by_image.get(str(annotation.get("image_id")))
        if x is None or y is None or frame is None:
            continue
        attributes = annotation.get("attributes") or {}
        category = annotation.get("category_id")
        if category == BALL_CATEGORY or attributes.get("role") == "ball":
            clip.ball[frame] = (float(x), float(y))
        elif category in PERSON_CATEGORIES:
            clip.players.setdefault(frame, []).append(
                (int(annotation["track_id"]), attributes.get("team"),
                 float(x), float(y)))
    return clip
