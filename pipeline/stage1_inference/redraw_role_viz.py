"""Redraw step3 visualization.mp4 from predictions.json (role/team only, no jersey)."""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

from pipeline.utils.video import reencode_to_h264

# BGR — match sn_gamestate pitch radar conventions
ROLE_COLOR = {
    "referee": (2, 210, 238),
    "goalkeeper": (255, 0, 255),
    "other": (0, 255, 0),
    "ball": (0, 0, 0),
}
TEAM_COLOR = {"left": (255, 0, 0), "right": (0, 0, 255)}


def _color(role: str, team: str) -> tuple[int, int, int]:
    if role in ROLE_COLOR:
        return ROLE_COLOR[role]
    if role == "player":
        return TEAM_COLOR.get(team, (200, 200, 200))
    return (180, 180, 180)


def _label(track_id: int, role: str, team: str) -> str:
    if role == "player":
        side = team if team in ("left", "right") else "?"
        return f"ID:{track_id} T:{side}"
    return f"ID:{track_id} {role or '?'}"


def _frames_dir(clip_dir: Path, video_id: str) -> Path:
    pklz = clip_dir / "step1" / "states" / "sn-gamestate.pklz"
    with zipfile.ZipFile(pklz) as z:
        member = f"{video_id}_image.pkl"
        img = pickle.loads(z.read(member))
    return Path(img.iloc[0].file_path).parent


def _anns_by_image(predictions: dict) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for ann in predictions.get("annotations", []):
        by[str(ann["image_id"])].append(ann)
    return by


def redraw_clip(clip_dir: Path, fps: float = 25.0) -> Path:
    clip_dir = Path(clip_dir)
    video_id = f"9{int(clip_dir.name):03d}"
    pred_path = clip_dir / "predictions.json"
    out_path = clip_dir / "step3" / "visualization" / "videos" / f"{video_id}.mp4"
    predictions = json.loads(pred_path.read_text())
    frames_dir = _frames_dir(clip_dir, video_id)
    by_image = _anns_by_image(predictions)
    images = sorted(predictions["images"], key=lambda im: im["file_name"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = None
    try:
        for i, image in enumerate(images):
            frame_path = frames_dir / image["file_name"]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(frame_path)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(tmp),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed: {tmp}")
            for ann in by_image.get(str(image["image_id"]), []):
                bbox = ann.get("bbox_image") or {}
                x, y, bw, bh = bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h")
                if None in (x, y, bw, bh):
                    continue
                attrs = ann.get("attributes") or {}
                role = str(attrs.get("role") or "")
                team = str(attrs.get("team") or "")
                color = _color(role, team)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bw), int(y + bh)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = _label(int(ann["track_id"]), role, team)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"{i + 1}/{len(images)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    reencode_to_h264(tmp, out_path)
    tmp.unlink(missing_ok=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "clips",
        nargs="*",
        type=Path,
        help="Clip dirs under outputs/preprocessing (default: 0001..0015)",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    root = Path("outputs/preprocessing")
    clips = args.clips or [root / f"{i:04d}" for i in range(1, 16)]
    for clip in clips:
        out = redraw_clip(clip, fps=args.fps)
        print(f"OK {clip.name} → {out}")


if __name__ == "__main__":
    main()
