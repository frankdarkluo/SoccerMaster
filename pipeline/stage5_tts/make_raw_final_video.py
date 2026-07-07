#!/usr/bin/env python3
"""Stage 5 step 1: produce raw_final_video with default-voice commentary.

Uses edge-tts (generic male voice) — not the 王楚淇 clone. Output is a full
video with commentary audio, for review before the clone pass.

Chinese → raw_final_video.mp4
English → raw_final_video_en.mp4

Prerequisites:
    - <output-dir>/annotated_video.mp4
    - <output-dir>/commentary.json
    - <output-dir>/events.json  (optional, for energy tiers)

Outputs:
    - <output-dir>/commentary_{lang}_default.mp3
    - <output-dir>/raw_final_video.mp4       (zh)
    - <output-dir>/raw_final_video_en.mp4    (en)

Usage:
    python -m pipeline.stage5_tts.make_raw_final_video --output-dir outputs/SNGS-148
    python -m pipeline.stage5_tts.make_raw_final_video --output-dir outputs/SNGS-148 --language en --force
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


def raw_final_path(output_dir: Path, language: str) -> Path:
    """Chinese keeps the original name; English is raw_final_video_en.mp4."""
    if language == "en":
        return output_dir / "raw_final_video_en.mp4"
    return output_dir / "raw_final_video.mp4"


def make_raw_final_video(
    output_dir: Path,
    language: str = "zh",
    force: bool = False,
) -> Path:
    """annotated_video + edge-tts (default voice) → raw_final_video[_en].mp4."""
    from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
    from pipeline.stage5_tts.mux import mux_audio_video
    from pipeline.stage5_tts.synthesize import synthesize_commentary

    output_dir = Path(output_dir)
    annotated = output_dir / "annotated_video.mp4"
    commentary_json = output_dir / "commentary.json"
    events_json = output_dir / "events.json"
    audio_path = output_dir / f"commentary_{language}_default.mp3"
    raw_final = raw_final_path(output_dir, language)

    if not annotated.exists():
        raise FileNotFoundError(f"annotated video not found: {annotated}")
    if not commentary_json.exists():
        raise FileNotFoundError(f"commentary.json not found: {commentary_json}")

    voice_tag = f"default_{language}"
    if force or not audio_path.exists():
        if force:
            import shutil
            seg_dir = output_dir / "tts_segments" / language / voice_tag
            if seg_dir.exists():
                shutil.rmtree(seg_dir)
            audio_path.unlink(missing_ok=True)
        log.info("Synthesizing default-voice (%s) commentary with edge-tts …", language)
        synthesize_commentary(
            commentary_json,
            output_dir,
            language=language,
            adapter=EdgeTTSAdapter(language=language),
            events_json=events_json if events_json.exists() else None,
            voice_tag=voice_tag,
            audio_path=audio_path,
        )
    else:
        log.info("Reusing existing default-voice audio: %s", audio_path)

    log.info("Muxing %s + %s → %s", annotated.name, audio_path.name, raw_final.name)
    mux_audio_video(annotated, audio_path, raw_final)

    from pipeline.stage5_tts.preview import print_preview_links, write_preview_html

    title = "Stage 5 · Default voice (EN)" if language == "en" else "Stage 5 · 默认音色预览"
    html_name = "preview_default_en.html" if language == "en" else "preview_default.html"
    preview_html = write_preview_html(
        output_dir, raw_final, audio_path,
        title=title,
        html_name=html_name,
    )
    print(f"Wrote {raw_final} ({raw_final.stat().st_size} bytes, default voice, {language})")
    print_preview_links(raw_final, audio_path, preview_html, workspace_root=ROOT)
    return raw_final


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
        help="Re-synthesize default-voice audio even if it already exists",
    )
    args = parser.parse_args()
    make_raw_final_video(args.output_dir, language=args.language, force=args.force)


if __name__ == "__main__":
    main()
