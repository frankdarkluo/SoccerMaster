"""Build the raw clip mp4 that Doubao watches and final videos mux onto."""
from __future__ import annotations

import subprocess
from pathlib import Path


def build_clip_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: int = 25,
    max_height: int = 720,
) -> Path:
    """Encode img1/ jpg frames into an H.264 mp4, downscaled to max_height."""
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames (*.jpg) found in {frames_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = output_path.with_suffix(".frames.txt")
    list_file.write_text(
        "".join(f"file '{f.resolve()}'\nduration {1 / fps}\n" for f in frames),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"scale=-2:'min({max_height},ih)'",
        "-c:v",
        "libx264",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        list_file.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg not found; install ffmpeg to build Stage 2b clip.mp4") from exc
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip encoding failed:\n{result.stderr[-800:]}")
    return output_path


def video_duration_s(path: Path) -> float:
    """Duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found; install ffmpeg to inspect Stage 2b clip duration") from exc
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}") from exc
