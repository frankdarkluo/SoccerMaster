"""Small ffmpeg helpers for Stage 2b video observation."""
from __future__ import annotations

import subprocess
from pathlib import Path


def build_clip_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: float = 25.0,
    max_height: int = 720,
) -> Path:
    """Encode numbered PNG/JPEG frames into an H.264 MP4."""
    frames_dir = Path(frames_dir)
    frames = sorted(
        path for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")
    suffix = frames[0].suffix
    if any(path.suffix != suffix for path in frames):
        raise RuntimeError("Frame files must use one image format")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-pattern_type", "glob", "-i", str(frames_dir / f"*{suffix}"),
            "-vf", f"scale=-2:'min({max_height},ih)'",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg frame encoding failed")
    return output_path


def video_duration_s(path: Path) -> float:
    """Return the container duration reported by ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed on {path}") from exc
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed on {path}")
    return duration


def extract_window(source: Path, start_s: float, end_s: float, target: Path) -> Path:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{max(0.0, start_s):.3f}",
            "-i", str(source), "-t", f"{max(0.1, end_s - start_s):.3f}",
            "-c:v", "libx264", "-an", str(target),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg window extraction failed")
    return target
