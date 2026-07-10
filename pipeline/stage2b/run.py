#!/usr/bin/env python3
"""Stage 2B CLI: Stage 1 tracking + clip video -> events/commentary JSON."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from pipeline.stage2b.digest import build_tracking_digest
    from pipeline.stage2b.generate_direct import ark_model, generate_direct
    from pipeline.stage2b.video import build_clip_mp4, video_duration_s

    if not args.predictions.is_file():
        raise FileNotFoundError(f"Stage 1 predictions not found: {args.predictions}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_mp4 = args.output_dir / "clip.mp4"
    if args.force or not clip_mp4.exists():
        build_clip_mp4(args.clip_dir / "img1", clip_mp4, fps=args.fps)
    if not args.force and (args.output_dir / "commentary.json").exists():
        log.info("Reusing %s; use --force to regenerate", args.output_dir / "commentary.json")
        return

    duration_s = video_duration_s(clip_mp4)
    log.info("Calling %s with %.1fs video", ark_model(), duration_s)
    output = generate_direct(
        clip_mp4=clip_mp4,
        digest=build_tracking_digest(args.predictions, fps=args.fps),
        duration_s=duration_s,
        fps=args.fps,
        output_dir=args.output_dir,
        languages=["en", "zh"],
    )
    log.info("Stage 2B complete: %s", output)


if __name__ == "__main__":
    main()
