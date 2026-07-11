"""Render Stage 4 effects and atomically upgrade the final video."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stage3_tts.mux import mux_audio_video
from pipeline.stage4_effects.render import render_annotated_video
from pipeline.stage4_effects.topology_analysis import run_topology_analysis


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


def _select_audio(config: PipelineConfig, language: str) -> Path:
    candidates = [
        config.commentary_audio(language),
        config.commentary_audio(language, default=True),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Stage 3 selected audio not found; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def run_stage4(
    output_dir: Path,
    clip_dir: Path | None = None,
    language: str = "zh",
    force: bool | None = None,
    config: PipelineConfig | None = None,
) -> Path:
    output_dir = Path(output_dir)
    if config is None:
        clip_dir = Path(clip_dir) if clip_dir is not None else output_dir
        force = False if force is None else force
        config = PipelineConfig(clip_dir=clip_dir, output_dir=output_dir, force=force)
    else:
        if output_dir != Path(config.output_dir):
            raise ValueError("output_dir must match config.output_dir")
        if clip_dir is not None and Path(clip_dir) != Path(config.clip_dir):
            raise ValueError("clip_dir must match config.clip_dir")
        clip_dir = Path(config.clip_dir)
        force = config.force if force is None else force
    predictions = _require_file(output_dir / "predictions.json")
    events = _require_file(output_dir / "comments" / "events.json")
    frames = clip_dir / "img1"
    if not frames.is_dir() or not next(frames.glob("*.jpg"), None):
        raise FileNotFoundError(f"Clip frames not found: {frames}")
    audio = _select_audio(config, language)

    annotated = config.annotated_video
    final = config.final_video(language)
    homography = config.homography_json

    if force or not annotated.is_file():
        render_annotated_video(
            frames, events, predictions, annotated, config,
            homography_json_path=homography if homography.is_file() else None,
        )
    topology = output_dir / "topo.json"
    if config.topology_lines_enabled and (force or not topology.is_file()):
        run_topology_analysis(predictions, topology, fps=config.fps)
    return replace_final_video(annotated, audio, final)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(run_stage4(
        args.output_dir,
        clip_dir=args.clip_dir,
        language=args.language,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
