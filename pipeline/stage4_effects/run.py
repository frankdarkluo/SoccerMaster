"""Render Stage 4 effects and atomically upgrade the final video."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stage3_tts.mux import mux_audio_video
from pipeline.stage4_effects.preview import render_preview_video
from pipeline.stage4_effects.render import render_annotated_video
from pipeline.topology.analysis import analyze_clip


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

    topology = output_dir / "topo.json"
    if config.topology_lines_enabled:
        analyze_clip(
            Path(__file__).resolve().parents[2],
            output_dir.name,
            topology,
            fps=config.fps,
            force=bool(force),
            preprocessing_root=output_dir.parent,
        )
    if force or not annotated.is_file():
        render_annotated_video(
            frames, events, predictions, annotated, config,
            homography_json_path=homography if homography.is_file() else None,
            topology_json_path=topology if topology.is_file() else None,
        )
    return replace_final_video(annotated, audio, final)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use `python -m pipeline.stage4_effects.run preview --help` for previews.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--force", action="store_true")
    return parser


def build_preview_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a silent tactical preview.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--homography", type=Path, required=True)
    parser.add_argument("--style-profile", type=Path)
    parser.add_argument("--focus-track-id", type=int)
    parser.add_argument("--window", default="21:25")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["preview"]:
        args = build_preview_arg_parser().parse_args(argv[1:])
        print(render_preview_video(
            args.frames_dir,
            args.predictions,
            args.homography,
            args.output,
            style_profile_path=args.style_profile,
            topology_json_path=args.topology,
            repo_root=Path(__file__).resolve().parents[2],
            focus_track_id=args.focus_track_id,
            window=args.window,
            fps=args.fps,
        ))
        return

    args = build_arg_parser().parse_args(argv)
    print(run_stage4(
        args.output_dir,
        clip_dir=args.clip_dir,
        language=args.language,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
