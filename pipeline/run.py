"""Orchestration for Stage 1 inference and optional Stage 3 effects."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig

log = logging.getLogger(__name__)


def infer_video_id(clip_dir: Path, override: Optional[str] = None) -> str:
    if override:
        return override
    match = re.match(r"SNGS-(\d+)", Path(clip_dir).name, re.IGNORECASE)
    return match.group(1) if match else "001"


def resolve_input_video(config: PipelineConfig) -> Path:
    if config.input_video and Path(config.input_video).exists():
        return Path(config.input_video)
    clip_dir = Path(config.clip_dir)
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        if matches := sorted(clip_dir.glob(pattern)):
            return matches[0]
    if matches := sorted(clip_dir.parent.glob(f"{clip_dir.name}.mp4")):
        return matches[0]
    raise FileNotFoundError(
        f"No input video found for Stage 1. Pass --input-video or place an mp4 in {clip_dir}"
    )


def run_stage1(config: PipelineConfig) -> None:
    from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

    video_id = infer_video_id(config.clip_dir, config.pklz_video_id)
    output_dir = Path(config.output_dir)
    sequence_name = Path(config.clip_dir).name

    if config.existing_pklz_path and Path(config.existing_pklz_path).exists():
        convert_pklz_to_json(
            Path(config.existing_pklz_path),
            video_id,
            output_dir,
            fps=config.fps,
            sequence_name=sequence_name,
            ball_labels_path=(Path(config.clip_dir) / "Labels-GameState.json")
            if (Path(config.clip_dir) / "Labels-GameState.json").is_file()
            else None,
        )
        return

    from pipeline.stage1_inference.run_gsr import run_full_gsr

    frames_dir = Path(config.clip_dir) / "img1"
    n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.is_dir() else 0
    if n_frames:
        log.info("Using existing frames in %s (%d frames)", frames_dir, n_frames)
        fps = config.fps
    else:
        from pipeline.stage1_inference.preprocess import preprocess_video

        video_path = resolve_input_video(config)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Input video is missing or empty: {video_path}")
        sequence = preprocess_video(
            video_path,
            sequence_name=config.sequence_prefix,
            split=config.gsr_split,
        )
        fps = int(sequence["fps"])
        sequence_name = config.sequence_prefix

    labels_path = Path(config.clip_dir) / "Labels-GameState.json"
    convert_pklz_to_json(
        run_full_gsr(config),
        video_id,
        output_dir,
        fps=fps,
        sequence_name=sequence_name,
        ball_labels_path=labels_path if labels_path.is_file() else None,
    )


def run_stage3(config: PipelineConfig) -> None:
    if not config.events_json.is_file():
        raise FileNotFoundError(f"Stage 2B events not found: {config.events_json}")
    if not config.predictions_json.is_file():
        raise FileNotFoundError(f"Stage 1 predictions not found: {config.predictions_json}")

    from pipeline.stage3_effects.render import render_annotated_video
    from pipeline.stage3_effects.topology_analysis import run_topology_analysis

    render_annotated_video(
        frames_dir=config.frames_dir,
        events_json_path=config.events_json,
        predictions_json_path=config.predictions_json,
        output_path=config.annotated_video,
        config=config,
        homography_json_path=config.homography_json if config.homography_json.exists() else None,
    )
    if config.force or not config.topology_json.exists():
        run_topology_analysis(config.predictions_json, config.topology_json, fps=config.fps)
