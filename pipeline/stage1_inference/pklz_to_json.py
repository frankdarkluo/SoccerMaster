"""Convert TrackLab pklz state to Labels-GameState-compatible JSON + homography export."""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Tuple

import numpy as np

LABELS_CATEGORIES = [
    {"supercategory": "object", "id": 1, "name": "player"},
    {"supercategory": "object", "id": 2, "name": "goalkeeper"},
    {"supercategory": "object", "id": 3, "name": "referee"},
    {"supercategory": "object", "id": 4, "name": "ball"},
    {"supercategory": "pitch", "id": 5, "name": "pitch"},
    {"supercategory": "camera", "id": 6, "name": "camera"},
    {"supercategory": "object", "id": 7, "name": "other"},
]

ROLE_TO_CATEGORY = {
    "player": 1,
    "goalkeeper": 2,
    "referee": 3,
    "ball": 4,
}


def _bbox_pitch(bbox_pitch: Any) -> dict | None:
    if bbox_pitch is None:
        return None
    if isinstance(bbox_pitch, float) and np.isnan(bbox_pitch):
        return None
    if not isinstance(bbox_pitch, dict):
        return None
    x = bbox_pitch.get("x_bottom_middle")
    y = bbox_pitch.get("y_bottom_middle")
    if x is None or y is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(y, float) and np.isnan(y):
        return None
    return {k: float(v) for k, v in bbox_pitch.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


def _bbox_image(bbox_ltwh: Any) -> dict | None:
    if bbox_ltwh is None:
        return None
    if isinstance(bbox_ltwh, float) and np.isnan(bbox_ltwh):
        return None
    values = [_to_serializable(v) for v in bbox_ltwh]
    if len(values) != 4:
        return None
    left, top, width, height = values
    return {
        "x": left,
        "y": top,
        "w": width,
        "h": height,
        "x_center": left + width / 2,
        "y_center": top + height / 2,
    }


def _homography_entry(h: Any) -> dict:
    if h is None:
        return {"H": None, "H_inv": None, "valid": False}
    if isinstance(h, float) and np.isnan(h):
        return {"H": None, "H_inv": None, "valid": False}
    try:
        h_arr = np.array(h, dtype=float)
    except (TypeError, ValueError):
        return {"H": None, "H_inv": None, "valid": False}
    if h_arr.shape != (3, 3) or not np.isfinite(h_arr).all():
        return {"H": None, "H_inv": None, "valid": False}
    try:
        h_inv = np.linalg.inv(h_arr)
    except np.linalg.LinAlgError:
        return {"H": None, "H_inv": None, "valid": False}
    return {
        "H": h_arr.tolist(),
        "H_inv": h_inv.tolist(),
        "valid": True,
    }


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def _ball_annotations_from_labels(labels_path: Path, start_ann_id: int) -> list[dict]:
    """Append GT ball rows when GSR pklz has no ball detections."""
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    balls = []
    ann_id = start_ann_id
    for ann in data.get("annotations", []):
        attrs = ann.get("attributes") or {}
        if attrs.get("role") != "ball":
            continue
        bp = _bbox_pitch(ann.get("bbox_pitch"))
        if bp is None:
            continue
        ann_id += 1
        balls.append({
            "id": str(ann_id),
            "image_id": str(ann["image_id"]),
            "track_id": int(ann.get("track_id", 0)),
            "supercategory": "object",
            "category_id": ROLE_TO_CATEGORY["ball"],
            "bbox_image": ann.get("bbox_image"),
            "bbox_pitch": bp,
            "attributes": {
                "role": "ball",
                "team": attrs.get("team"),
                "jersey": str(attrs.get("jersey") or ""),
            },
        })
    return balls


def convert_pklz_to_json(
    pklz_path: Path,
    video_id: str,
    output_dir: Path,
    fps: int = 25,
    sequence_name: str | None = None,
    ball_labels_path: Path | str | None = None,
) -> Tuple[Path, Path]:
    """
    Convert pklz tracker state to predictions.json + homography_per_frame.json.

    Returns (predictions_json_path, homography_json_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pklz_path = Path(pklz_path)

    with zipfile.ZipFile(pklz_path) as z:
        names = z.namelist()
        if f"{video_id}.pkl" not in names:
            raise FileNotFoundError(f"{video_id}.pkl not in {pklz_path}")
        with z.open(f"{video_id}.pkl") as f:
            detections_df = pickle.load(f)
        with z.open(f"{video_id}_image.pkl") as f:
            image_df = pickle.load(f)

    homography_data: dict[str, dict] = {"frames": {}}
    valid_homography_image_ids = set()
    for _, row in image_df.iterrows():
        image_id = str(row.get("id", row.name))
        entry = _homography_entry(row.get("h"))
        homography_data["frames"][image_id] = entry
        if entry["valid"]:
            valid_homography_image_ids.add(image_id)

    images = []
    for _, row in image_df.iterrows():
        image_id = str(row.get("id", row.name))
        file_path = row.get("file_path", "")
        file_name = Path(file_path).name if file_path else f"{int(row.get('frame', 0)) + 1:06d}.jpg"
        images.append({
            "image_id": image_id,
            "file_name": file_name,
            "width": 1920,
            "height": 1080,
            "is_labeled": bool(row.get("is_labeled", True)),
            "has_labeled_person": True,
            "has_labeled_pitch": True,
            "has_labeled_camera": True,
        })

    annotations = []
    ann_id = 0
    for _, row in detections_df.iterrows():
        image_id = str(row["image_id"])
        bbox_image = _bbox_image(row.get("bbox_ltwh"))
        has_valid_homography = image_id in valid_homography_image_ids
        bp = _bbox_pitch(row.get("bbox_pitch")) if has_valid_homography else None
        if bp is None and (has_valid_homography or bbox_image is None):
            continue
        role = row.get("role")
        ann_id += 1
        annotations.append({
            "id": str(ann_id),
            "image_id": image_id,
            "track_id": int(row["track_id"]),
            "supercategory": "object",
            "category_id": ROLE_TO_CATEGORY.get(role, 7),
            "bbox_image": bbox_image,
            "bbox_pitch": bp,
            "attributes": {
                "role": role,
                "team": row.get("team"),
                "jersey": str(row.get("jersey_number", "")) if row.get("jersey_number") is not None else "",
            },
        })

    has_ball = any(a.get("attributes", {}).get("role") == "ball" for a in annotations)
    if not has_ball and ball_labels_path is not None:
        annotations.extend(_ball_annotations_from_labels(Path(ball_labels_path), ann_id))

    name = sequence_name or f"SNGS-{video_id}"
    predictions = {
        "info": {
            "name": name,
            "n_frames": len(images),
            "fps": fps,
        },
        "images": images,
        "annotations": annotations,
        "categories": LABELS_CATEGORIES,
    }

    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, default=_to_serializable)

    homo_path = output_dir / "homography_per_frame.json"
    with open(homo_path, "w", encoding="utf-8") as f:
        json.dump(homography_data, f, indent=2)

    return pred_path, homo_path
