"""Run direct CosyVoice synthesis, timeline assembly, and video muxing."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Optional
from pipeline.config import PipelineConfig


def _video_duration_s(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}")
    return duration


def _load_segments(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    segments = data.get("segments", data.get("commentary", [])) if isinstance(data, dict) else data
    if not segments:
        raise RuntimeError(f"No commentary segments in {path}")
    return segments


def _synthesize_voice(
    output_dir: Path,
    segments: list[dict],
    language: str,
    voice: str,
    audio_path: Path,
    duration_s: float,
    synthesizer,
    prompt_wav: Optional[Path],
    prompt_text: Optional[str],
    force: bool,
) -> Path:
    from pipeline.stage3_tts.synthesize import assemble_timeline, synthesize_fitting_segment

    segment_dir = output_dir / "voice" / "tts_segments" / language / voice
    segment_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, segment in enumerate(segments):
        path = segment_dir / f"segment_{index:03d}.wav"
        if force or not path.is_file():
            start = float(segment.get("timestamp_s", 0.0))
            end = float(segment.get("end_s", duration_s))
            synthesize_fitting_segment(
                segment, language, path, max(end - start, 0.0), synthesizer,
                voice=voice, prompt_wav=prompt_wav, prompt_text=prompt_text,
            )
        paths.append(path)
    return assemble_timeline(segments, paths, audio_path, duration_s)


def _atomic_mux(clip: Path, audio: Path, target: Path) -> Path:
    from pipeline.stage3_tts.mux import mux_audio_video

    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        mux_audio_video(clip, audio, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def run_stage3_tts(
    output_dir: Path,
    language: str = "zh",
    voice: str = "both",
    prompt_wav: Optional[Path] = None,
    prompt_text: Optional[str] = None,
    force: bool = False,
) -> Path:
    from pipeline.stage3_tts.cosyvoice import (
        DEFAULT_PROMPT_TEXT, MODEL_DIR, CosyVoiceSynthesizer,
    )

    output_dir = Path(output_dir)
    config = PipelineConfig(output_dir=output_dir)
    config.voice_dir.mkdir(parents=True, exist_ok=True)
    commentary = output_dir / "comments" / "commentary.json"
    clip = output_dir / "clip.mp4"
    if not commentary.is_file():
        raise FileNotFoundError(f"commentary.json not found: {commentary}")
    if not clip.is_file():
        raise FileNotFoundError(f"clip.mp4 not found: {clip}")

    segments = _load_segments(commentary)
    duration_s = _video_duration_s(clip)
    synthesizer = CosyVoiceSynthesizer(MODEL_DIR)

    default_audio = config.commentary_audio(language, default=True)
    clone_audio = config.commentary_audio(language)
    baseline = config.raw_final_video(language)
    final = config.final_video(language)

    selected_audio = None
    if voice in {"default", "both"}:
        selected_audio = _synthesize_voice(
            output_dir, segments, language, "default", default_audio, duration_s,
            synthesizer, prompt_wav, prompt_text or DEFAULT_PROMPT_TEXT, force,
        )
    if voice in {"clone", "both"}:
        selected_audio = _synthesize_voice(
            output_dir, segments, language, "clone", clone_audio, duration_s,
            synthesizer, prompt_wav, prompt_text, force,
        )
    assert selected_audio is not None
    _atomic_mux(clip, selected_audio, baseline)
    return _atomic_mux(clip, selected_audio, final)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument(
        "--voice", choices=["default", "clone", "both"], default="both"
    )
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(run_stage3_tts(
        args.output_dir,
        language=args.language,
        voice=args.voice,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
