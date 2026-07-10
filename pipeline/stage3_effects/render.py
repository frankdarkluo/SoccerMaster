"""Orchestrate all visual effects onto original frames from img1/."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2

from pipeline.config import PipelineConfig
from pipeline.utils.video import reencode_to_h264
from pipeline.stage3_effects.beam_targets import load_predictions_index
from pipeline.stage3_effects.overlay import apply_frame_overlays
from pipeline.stage3_effects.projection import load_homography

log = logging.getLogger(__name__)


def _sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.jpg"))


def _frame_number_from_path(path: Path) -> int:
    return int(path.stem)


def render_annotated_video(
    frames_dir: Path,
    events_json_path: Path,
    predictions_json_path: Path,
    output_path: Path,
    config: PipelineConfig,
    homography_json_path: Optional[Path] = None,
    reencode_h264: bool = True,
) -> Path:
    """Render full annotated video from raw img1/ frames.

    When reencode_h264 is True (default), the OpenCV mp4v temp file is
    re-encoded to H.264/yuv420p with faststart so Cursor/VSCode can play it.
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(events_json_path, encoding="utf-8") as f:
        events_data = json.load(f)

    frame_to_image_id, anns_by_image = load_predictions_index(predictions_json_path)
    homo_frames = None
    if homography_json_path and Path(homography_json_path).exists():
        homo_frames = load_homography(homography_json_path)

    frame_paths = _sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise FileNotFoundError(f"Could not read frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    tmp_path = output_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.fps),
        (w, h),
    )

    events = events_data.get("events", [])
    for frame_path in frame_paths:
        frame_num = _frame_number_from_path(frame_path)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        apply_frame_overlays(
            frame,
            frame_num,
            events,
            frame_to_image_id,
            anns_by_image,
            homo_frames,
            config,
        )
        writer.write(frame)

    writer.release()

    if reencode_h264 and shutil.which("ffmpeg"):
        reencode_to_h264(tmp_path, output_path)
        tmp_path.unlink(missing_ok=True)
    else:
        if reencode_h264:
            log.warning("ffmpeg not found; keeping OpenCV mp4v output at %s", output_path)
        tmp_path.replace(output_path)

    return output_path
