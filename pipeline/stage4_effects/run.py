"""Render Stage 4 tactical effects and mux the commentary audio into the final video."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stage3_tts.mux import mux_audio_video
from pipeline.stage4_effects.preview import render_preview_video
from pipeline.topology.analysis import analyze_clip

REPO_ROOT = Path(__file__).resolve().parents[2]


def replace_final_video(
    annotated_video: Path,
    audio: Path,
    final_video: Path,
    *,
    mux=mux_audio_video,
) -> Path:
    temporary = final_video.with_name(f".{final_video.stem}.tmp{final_video.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        mux(annotated_video, audio, temporary)
        os.replace(temporary, final_video)
    finally:
        temporary.unlink(missing_ok=True)
    return final_video


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 4 input not found: {path}")
    return path


def run_stage4(
    config: PipelineConfig,
    language: str = "zh",
    *,
    focus_track_id: int | None = None,
    window: str | None = None,
) -> Path:
    output_dir = Path(config.output_dir)
    frames = Path(config.clip_dir) / "img1"
    if not next(frames.glob("*.jpg"), None):
        raise FileNotFoundError(f"Clip frames not found: {frames}")
    predictions = _require_file(output_dir / "predictions.json")
    homography = _require_file(config.homography_json)
    audio = _require_file(config.commentary_audio(language))

    topology = output_dir / "topo.json"
    if config.topology_lines_enabled:
        analyze_clip(
            REPO_ROOT,
            output_dir.name,
            topology,
            fps=config.fps,
            force=config.force,
            preprocessing_root=output_dir.parent,
        )
    annotated = config.annotated_video
    if config.force or not annotated.is_file():
        render_preview_video(
            frames,
            predictions,
            homography,
            annotated,
            topology_json_path=topology if topology.is_file() else None,
            repo_root=REPO_ROOT,
            focus_track_id=focus_track_id,
            window=window,
            fps=config.fps,
        )
    return replace_final_video(annotated, audio, config.final_video(language))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-dir", type=Path, help="defaults to output_dir")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--focus-track-id", type=int)
    parser.add_argument("--window", help='limit graphics to "start:end" seconds')
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = PipelineConfig(
        clip_dir=args.clip_dir or args.output_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(run_stage4(
        config, args.language,
        focus_track_id=args.focus_track_id, window=args.window,
    ))


if __name__ == "__main__":
    main()
