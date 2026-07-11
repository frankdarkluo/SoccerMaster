"""End-to-end pipeline orchestrator with flexible entry points."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.atomic import atomic_copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def infer_video_id(clip_dir: Path, override: Optional[str] = None) -> str:
    """Infer pklz video_id from clip directory name (e.g. SNGS-061 → 061)."""
    if override:
        return override
    name = Path(clip_dir).name
    match = re.match(r"SNGS-(\d+)", name, re.IGNORECASE)
    if match:
        return match.group(1)
    return "001"


def resolve_input_video(config: PipelineConfig) -> Path:
    """Find the source video for preprocess stage."""
    if config.input_video and Path(config.input_video).exists():
        return Path(config.input_video)

    clip_dir = Path(config.clip_dir)
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        matches = sorted(clip_dir.glob(pattern))
        if matches:
            return matches[0]

    parent_matches = sorted(clip_dir.parent.glob(f"{clip_dir.name}.mp4"))
    if parent_matches:
        return parent_matches[0]

    raise FileNotFoundError(
        f"No input video found for Stage 1. Pass --input-video or place an mp4 in {clip_dir}"
    )


def run_stage1(config: PipelineConfig) -> None:
    from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

    video_id = infer_video_id(config.clip_dir, config.pklz_video_id)
    output_dir = Path(config.output_dir)
    sequence_name = Path(config.clip_dir).name

    if config.existing_pklz_path and Path(config.existing_pklz_path).exists():
        log.info("Using existing pklz: %s (skipping GSR inference)", config.existing_pklz_path)
        convert_pklz_to_json(
            Path(config.existing_pklz_path),
            video_id,
            output_dir,
            fps=config.fps,
            sequence_name=sequence_name,
            ball_labels_path=Path(config.clip_dir) / "Labels-GameState.json"
            if (Path(config.clip_dir) / "Labels-GameState.json").is_file()
            else None,
        )
        return

    from pipeline.stage1_inference.run_gsr import run_full_gsr

    frames_dir = Path(config.clip_dir) / "img1"
    n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.is_dir() else 0

    if n_frames > 0:
        # Dataset clips (e.g. SNGS-148) already ship frames; skip broken/missing mp4.
        log.info(
            "Using existing frames in %s (%d frames), skip video preprocess",
            frames_dir, n_frames,
        )
        fps = config.fps
    else:
        from pipeline.stage1_inference.preprocess import preprocess_video

        video_path = resolve_input_video(config)
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"No frames in {frames_dir} and input video is missing/empty: {video_path}"
            )
        seq_info = preprocess_video(
            video_path,
            sequence_name=config.sequence_prefix,
            split=config.gsr_split,
        )
        fps = int(seq_info["fps"])
        sequence_name = config.sequence_prefix

    pklz_path = run_full_gsr(config)
    labels_path = Path(config.clip_dir) / "Labels-GameState.json"
    convert_pklz_to_json(
        pklz_path,
        video_id,
        output_dir,
        fps=fps,
        sequence_name=sequence_name,
        ball_labels_path=labels_path if labels_path.is_file() else None,
    )

def run_pipeline(config: PipelineConfig, effects: bool = False) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.should_run_stage1():
        log.info("=== Stage 1: SoccerMaster Inference ===")
        run_stage1(config)
    else:
        log.info("Stage 1 skipped: predictions at %s", config.predictions_json)

    if config.existing_predictions_json:
        source = Path(config.existing_predictions_json)
        target = config.output_dir / "predictions.json"
        if source.resolve() != target.resolve():
            atomic_copy(source, target)



    from pipeline.stage2b.run import run_stage2b
    from pipeline.stage3_tts.run import run_stage3_tts
    from pipeline.stage4_effects.run import run_stage4

    result = run_stage2b(
        config.output_dir,
        config.clip_dir,
        mode=config.commentary_mode,
        force=config.force,
    )
    for language in config.languages:
        result = run_stage3_tts(
            config.output_dir,
            language=language,
            force=config.force,
        )
        if effects:
            result = run_stage4(
                config.output_dir,
                language=language,
                config=config,
            )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SoccerMaster inference and commentary pipeline")
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline_run"))
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--existing-predictions-json", type=Path)
    parser.add_argument("--existing-homography-json", type=Path)
    parser.add_argument("--existing-pklz-path", type=Path)
    parser.add_argument("--pklz-video-id")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--lang", nargs="+", default=["en", "zh"])
    parser.add_argument("--mode", choices=["direct", "hybrid"], default="hybrid")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--effects", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        clip_dir=args.clip_dir,
        output_dir=args.output_dir,
        input_video=args.input_video,
        existing_predictions_json=args.existing_predictions_json,
        existing_homography_json=args.existing_homography_json,
        existing_pklz_path=args.existing_pklz_path,
        pklz_video_id=args.pklz_video_id,
        fps=args.fps,
        languages=args.lang,
        commentary_mode=args.mode,
        force=args.force,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(config_from_args(args), effects=args.effects)


if __name__ == "__main__":
    main()
