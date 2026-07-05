#!/usr/bin/env python3
"""Stage 5 step 2: produce final_video.mp4 with 王楚淇 cloned-voice commentary.

Uses Doubao TTS (seed-audio-1.0 + DOUBAO_TTS_SPEAKER). Replaces the default-voice
track from step 1 with the cloned voice.

Prerequisites:
    - <output-dir>/annotated_video.mp4
    - <output-dir>/commentary.json
    - <output-dir>/events.json  (optional, for energy tiers)
    - .env with DOUBAO_TTS_API_KEY and DOUBAO_TTS_SPEAKER

Outputs:
    - <output-dir>/commentary_{lang}.mp3   (王楚淇)
    - <output-dir>/final_video.mp4

Usage:
    python -m pipeline.stage5_tts.make_final_video --output-dir outputs/SNGS-148
    python -m pipeline.stage5_tts.make_final_video --output-dir outputs/SNGS-148 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def make_final_video(
    output_dir: Path,
    language: str = "zh",
    force: bool = False,
) -> Path:
    """annotated_video + Doubao TTS (王楚淇) → final_video.mp4."""
    from pipeline.stage4_commentary.generate import load_ark_env
    from pipeline.stage5_tts.adapters.doubao_tts import DoubaoTTSAdapter
    from pipeline.stage5_tts.mux import mux_audio_video
    from pipeline.stage5_tts.synthesize import synthesize_commentary

    output_dir = Path(output_dir)
    annotated = output_dir / "annotated_video.mp4"
    commentary_json = output_dir / "commentary.json"
    events_json = output_dir / "events.json"
    audio_path = output_dir / f"commentary_{language}.mp3"
    final_video = output_dir / "final_video.mp4"

    if not annotated.exists():
        raise FileNotFoundError(f"annotated video not found: {annotated}")
    if not commentary_json.exists():
        raise FileNotFoundError(f"commentary.json not found: {commentary_json}")

    if force or not audio_path.exists():
        load_ark_env()
        log.info("Synthesizing 王楚淇 voice (%s) with doubao_tts …", language)
        synthesize_commentary(
            commentary_json,
            output_dir,
            language=language,
            adapter=DoubaoTTSAdapter(),
            events_json=events_json if events_json.exists() else None,
            voice_tag="wang",
            audio_path=audio_path,
        )
    else:
        log.info("Reusing existing clone-voice audio: %s", audio_path)

    log.info("Muxing %s + %s → %s", annotated.name, audio_path.name, final_video.name)
    mux_audio_video(annotated, audio_path, final_video)

    from pipeline.stage5_tts.preview import print_preview_links, write_preview_html

    preview_html = write_preview_html(
        output_dir, final_video, audio_path,
        title="Stage 5 · 王楚淇克隆音",
        html_name="preview_final.html",
    )
    print(f"Wrote {final_video} ({final_video.stat().st_size} bytes, 王楚淇 voice)")
    print_preview_links(final_video, audio_path, preview_html, workspace_root=ROOT)
    return final_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Pipeline output directory (e.g. outputs/SNGS-148)",
    )
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument(
        "--force", action="store_true",
        help="Re-synthesize clone-voice audio even if it already exists",
    )
    args = parser.parse_args()
    make_final_video(args.output_dir, language=args.language, force=args.force)


if __name__ == "__main__":
    main()
