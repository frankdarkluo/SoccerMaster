"""End-to-end pipeline orchestrator with flexible entry points."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig

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
    convert_pklz_to_json(
        pklz_path,
        video_id,
        output_dir,
        fps=fps,
        sequence_name=sequence_name,
    )


def run_stage2(config: PipelineConfig) -> int:
    from pipeline.stage2_events.detector import EventDetector
    from pipeline.stage2_events.schema import EventSchema

    schema = EventSchema()
    detector = EventDetector(
        schema,
        fps=config.fps,
        shot_speed_threshold=config.ball_speed_shot_threshold_mps,
        min_gap_s=config.min_event_gap_s,
    )
    events = detector.detect(str(config.predictions_json))

    if config.verify_events:
        from pipeline.stage2_events.verify import verify_events
        from pipeline.stage4_commentary.adapters.qwen_local import QwenLocalAdapter

        log.info("Verifying possession-derived events with Qwen2.5-VL ...")
        adapter = QwenLocalAdapter(model_path=config.verify_model_path)
        events, audit = verify_events(
            events,
            str(config.predictions_json),
            config.frames_dir,
            config.output_dir,
            adapter,
            fps=config.fps,
            window_s=config.verify_window_s,
            force=config.force,
        )
        dropped = sum(1 for a in audit if not a["kept"])
        log.info("Verification: %d checked, %d dropped", len(audit), dropped)
        if config.cleanup_verify_temp:
            from pipeline.stage2_events.verify import cleanup_verify_artifacts
            cleanup_verify_artifacts(config.output_dir)

    video_info = {
        "source": str(config.frames_dir),
        "fps": config.fps,
        "duration_s": 30.0,
        "total_frames": config.fps * 30,
    }
    detector.write_events_json(events, config.output_dir / "events.json", video_info)
    return len(events)


def run_stage3(config: PipelineConfig) -> None:
    from pipeline.stage3_effects.render import render_annotated_video

    homography_path = config.homography_json if config.homography_json.exists() else None
    render_annotated_video(
        frames_dir=config.frames_dir,
        events_json_path=config.events_json,
        predictions_json_path=config.predictions_json,
        output_path=config.annotated_video,
        config=config,
        homography_json_path=homography_path,
    )


def run_stage4(config: PipelineConfig) -> None:
    from pipeline.stage4_commentary.generate import generate_commentary

    visual = config.annotated_video if config.annotated_video.exists() else None
    if visual is None and config.topdown_video.exists():
        visual = config.topdown_video

    generate_commentary(
        config.events_json,
        config.commentary_json,
        config=config,
        visual_input=visual,
    )


def run_stage5(config: PipelineConfig) -> None:
    from pipeline.stage5_tts.synthesize import synthesize_commentary
    from pipeline.stage5_tts.mux import mux_audio_video

    adapter = _build_tts_adapter(config)

    for lang in config.languages:
        audio = config.output_dir / f"commentary_{lang}.mp3"
        if config.force or not audio.exists():
            lang_adapter = adapter
            if config.tts_backend == "edge_tts":
                from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
                lang_adapter = EdgeTTSAdapter(language=lang)
            synthesize_commentary(
                config.commentary_json,
                config.output_dir,
                language=lang,
                adapter=lang_adapter,
            )
            log.info("TTS audio: %s", audio)

    primary_audio = config.commentary_audio
    if primary_audio.exists() and config.annotated_video.exists():
        mux_audio_video(config.annotated_video, primary_audio, config.final_video)


def _build_tts_adapter(config: PipelineConfig):
    from pipeline.stage4_commentary.generate import load_ark_env
    load_ark_env()

    if config.tts_backend == "doubao_tts":
        from pipeline.stage5_tts.adapters.doubao_tts import DoubaoTTSAdapter
        return DoubaoTTSAdapter()
    if config.tts_backend == "edge_tts":
        from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
        return EdgeTTSAdapter(language=config.tts_language)
    raise ValueError(f"Unknown TTS backend: {config.tts_backend}")


def run_pipeline(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.should_run_stage1():
        log.info("=== Stage 1: SoccerMaster Inference ===")
        run_stage1(config)
        log.info("Stage 1 complete: %s", config.predictions_json)
    else:
        log.info("Stage 1 skipped: predictions at %s", config.predictions_json)

    if config.should_run_stage2():
        log.info("=== Stage 2: Event Detection ===")
        n_events = run_stage2(config)
        log.info("Stage 2 complete: %d events → %s", n_events, config.events_json)
    else:
        log.info("Stage 2 skipped: events at %s", config.events_json)

    if config.should_run_stage3():
        log.info("=== Stage 3: Visual Effects ===")
        run_stage3(config)
        log.info("Stage 3 complete: %s", config.annotated_video)
    else:
        log.info("Stage 3 skipped: %s exists", config.annotated_video)

    if config.should_run_stage4():
        log.info("=== Stage 4: LLM Commentary ===")
        run_stage4(config)
        log.info("Stage 4 complete: %s", config.commentary_json)
    else:
        log.info("Stage 4 skipped: %s exists", config.commentary_json)

    if config.should_run_stage5():
        log.info("=== Stage 5: TTS Voice Synthesis ===")
        run_stage5(config)
        log.info("Stage 5 complete: %s", config.final_video)
    else:
        log.info("Stage 5 skipped: %s exists", config.final_video)

    log.info("Pipeline complete. Outputs in %s", config.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Football Commentary Pipeline")
    parser.add_argument("--clip-dir", type=Path, required=True, help="Clip directory containing img1/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline_run"))
    parser.add_argument("--input-video", type=Path, default=None, help="Raw video for Stage 1 preprocess")
    parser.add_argument("--existing-predictions-json", type=Path, default=None, help="Skip Stage 1")
    parser.add_argument("--existing-homography-json", type=Path, default=None)
    parser.add_argument("--existing-pklz-path", type=Path, default=None, help="Skip GSR inference")
    parser.add_argument("--existing-events-json", type=Path, default=None, help="Skip Stage 2")
    parser.add_argument("--pklz-video-id", type=str, default=None, help="Video id inside pklz (default: from clip name)")
    parser.add_argument(
        "--llm-backend",
        default="doubao",
        choices=["mock", "qwen_local", "doubao", "openai"],
    )
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--lang", nargs="+", default=["en", "zh"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-events",
        action="store_true",
        help="Verify interception/pass events with local Qwen2.5-VL",
    )
    parser.add_argument(
        "--no-cleanup-verify-temp",
        dest="cleanup_verify_temp",
        action="store_false",
        help="Keep verify_clips/ and verify_cache/ after verification (default: delete them)",
    )
    parser.set_defaults(cleanup_verify_temp=True)
    parser.add_argument("--no-topology-lines", action="store_true")
    parser.add_argument(
        "--tts-backend",
        default="doubao_tts",
        choices=["doubao_tts", "edge_tts"],
    )
    parser.add_argument("--tts-language", default="zh", choices=["zh", "en"])
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        clip_dir=args.clip_dir,
        output_dir=args.output_dir,
        input_video=args.input_video,
        existing_predictions_json=args.existing_predictions_json,
        existing_homography_json=args.existing_homography_json,
        existing_pklz_path=args.existing_pklz_path,
        existing_events_json=args.existing_events_json,
        pklz_video_id=args.pklz_video_id,
        llm_backend=args.llm_backend,
        roster_json=args.roster,
        languages=args.lang,
        fps=args.fps,
        force=args.force,
        verify_events=args.verify_events,
        cleanup_verify_temp=args.cleanup_verify_temp,
        topology_lines_enabled=not args.no_topology_lines,
        tts_backend=args.tts_backend,
        tts_language=args.tts_language,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(config_from_args(args))


if __name__ == "__main__":
    main()
