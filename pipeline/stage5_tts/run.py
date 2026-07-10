#!/usr/bin/env python3
"""Stage 5 CLI: synthesize Edge preview, CosyVoice clone, or both."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from pipeline.config import load_env

log = logging.getLogger(__name__)


def select_video(output_dir: Path, explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else [output_dir / "annotated_video.mp4", output_dir / "clip.mp4"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No base video found in {output_dir}; run Stage 2B or pass --video")


def _paths(output_dir: Path, language: str, voice: str) -> tuple[Path, Path, str]:
    suffix = "_en" if language == "en" else ""
    if voice == "preview":
        return (
            output_dir / f"commentary_{language}_default.mp3",
            output_dir / f"raw_final_video{suffix}.mp4",
            f"default_{language}",
        )
    return output_dir / f"commentary_{language}.mp3", output_dir / f"final_video{suffix}.mp4", "wang"


def render_voice(
    output_dir: Path,
    video: Path,
    language: str,
    voice: str,
    force: bool,
) -> Path:
    from pipeline.stage5_tts.mux import mux_audio_video
    from pipeline.stage5_tts.synthesize import synthesize_commentary

    commentary = output_dir / "commentary.json"
    if not commentary.is_file():
        raise FileNotFoundError(f"Stage 2B commentary not found: {commentary}")
    audio, final_video, voice_tag = _paths(output_dir, language, voice)
    if force:
        shutil.rmtree(output_dir / "tts_segments" / language / voice_tag, ignore_errors=True)
        audio.unlink(missing_ok=True)
    if not audio.exists():
        if voice == "preview":
            from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
            adapter = EdgeTTSAdapter(language=language)
        else:
            from pipeline.stage5_tts.adapters.cosyvoice_adapter import CosyVoiceAdapter
            load_env()
            adapter = CosyVoiceAdapter(language=language)
        synthesize_commentary(
            commentary,
            output_dir,
            language=language,
            adapter=adapter,
            events_json=output_dir / "events.json",
            voice_tag=voice_tag,
            audio_path=audio,
        )
    else:
        log.info("Reusing %s", audio)
    mux_audio_video(video, audio, final_video)
    print(final_video)
    return final_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--voice", choices=["preview", "clone", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    video = select_video(args.output_dir, args.video)
    voices = ("preview", "clone") if args.voice == "both" else (args.voice,)
    for voice in voices:
        render_voice(args.output_dir, video, args.language, voice, args.force)


if __name__ == "__main__":
    main()
