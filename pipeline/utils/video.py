"""Video I/O helpers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, quality: int = 2) -> int:
    """Extract all frames from video as JPEG. Returns frame count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-q:v", str(quality),
        str(output_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(output_dir.glob("*.jpg")))


def get_video_info(video_path: Path) -> dict:
    """Return fps, duration_s, width, height, total_frames."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps_parts = vs["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    duration = float(data["format"]["duration"])
    return {
        "fps": fps,
        "duration_s": duration,
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "total_frames": int(duration * fps),
    }


def get_video_codec(video_path: Path) -> tuple[str, str]:
    """Return (codec_name, pix_fmt) of the first video stream."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    return vs.get("codec_name", ""), vs.get("pix_fmt", "")


def is_h264_browser_safe(video_path: Path) -> bool:
    """True when video is H.264 yuv420p — playable in Cursor/VS Code / browsers."""
    codec, pix_fmt = get_video_codec(video_path)
    return codec == "h264" and pix_fmt == "yuv420p"


def reencode_to_h264(src: Path, dst: Path) -> None:
    """Re-encode MP4 to H.264 with faststart for IDE/browser playback."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-crf", "20", "-preset", "medium",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def encode_video(frame_dir: Path, output_path: Path, fps: float = 25.0) -> None:
    """Encode directory of numbered JPEGs to MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
