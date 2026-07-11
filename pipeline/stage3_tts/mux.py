"""Merge an audio track into a video file using ffmpeg."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from pipeline.utils.video import is_h264_browser_safe

log = logging.getLogger(__name__)


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Combine *video_path* with *audio_path* into *output_path*.

    Output is always browser/IDE-safe: H.264 yuv420p video + AAC audio,
    with ``faststart`` so Cursor/VS Code can stream-play immediately.
    Re-encodes video only when the input is not already H.264 yuv420p.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio-video muxing")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    copy_video = is_h264_browser_safe(video_path)
    video_codec_args = (
        ["-c:v", "copy"]
        if copy_video
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"]
    )
    if not copy_video:
        log.info("Input video is not H.264/yuv420p — re-encoding for IDE playback")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        *video_codec_args,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    log.info("Muxing %s + %s → %s", video_path.name, audio_path.name, output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr}")

    log.info("Final video: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path
