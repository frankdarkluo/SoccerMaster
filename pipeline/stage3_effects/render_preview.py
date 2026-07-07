"""Render light-beam previews on raw frames from img1/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2

from pipeline.config import PipelineConfig
from pipeline.utils.video import reencode_to_h264
from pipeline.stage3_effects.beam_targets import load_predictions_index
from pipeline.stage3_effects.overlay import apply_frame_overlays
from pipeline.stage3_effects.projection import load_homography


def render_beam_preview_frame(
    frames_dir: Path,
    events_json: Path,
    predictions_json: Path,
    output_image: Path,
    frame_number: int,
    config: Optional[PipelineConfig] = None,
    homography_json: Optional[Path] = None,
) -> Path:
    """Render a single frame with light-beam overlays."""
    config = config or PipelineConfig()
    frames_dir = Path(frames_dir)
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    with open(events_json, encoding="utf-8") as f:
        events_data = json.load(f)

    frame_to_image_id, anns_by_image = load_predictions_index(predictions_json)
    homo_frames = load_homography(homography_json) if homography_json and homography_json.exists() else None

    frame_path = frames_dir / f"{frame_number:06d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    apply_frame_overlays(
        frame,
        frame_number,
        events_data.get("events", []),
        frame_to_image_id,
        anns_by_image,
        homo_frames,
        config,
    )
    cv2.imwrite(str(output_image), frame)
    return output_image


def render_beam_preview_clip(
    frames_dir: Path,
    events_json: Path,
    predictions_json: Path,
    output_video: Path,
    center_frame: int,
    config: Optional[PipelineConfig] = None,
    homography_json: Optional[Path] = None,
) -> Path:
    """Render a short MP4 clip with beams around one event."""
    config = config or PipelineConfig()
    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    with open(events_json, encoding="utf-8") as f:
        events_data = json.load(f)

    frame_to_image_id, anns_by_image = load_predictions_index(predictions_json)
    homo_frames = load_homography(homography_json) if homography_json and homography_json.exists() else None

    beam_half_frames = max(1, int(config.beam_duration_s * config.fps))
    start = max(1, center_frame - beam_half_frames)
    end = center_frame + beam_half_frames

    first_frame = cv2.imread(str(frames_dir / f"{start:06d}.jpg"))
    if first_frame is None:
        raise FileNotFoundError(f"Frame not found: {frames_dir / f'{start:06d}.jpg'}")

    h, w = first_frame.shape[:2]
    tmp_path = output_video.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.fps),
        (w, h),
    )

    events = events_data.get("events", [])
    for frame_num in range(start, end + 1):
        frame_path = frames_dir / f"{frame_num:06d}.jpg"
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

    if not shutil.which("ffmpeg"):
        tmp_path.rename(output_video)
    else:
        reencode_to_h264(tmp_path, output_video)
        tmp_path.unlink(missing_ok=True)

    return output_video
