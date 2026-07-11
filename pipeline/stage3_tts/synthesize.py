"""Synthesize commentary segments and fit them to event slots."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def audio_duration_s(path: Path) -> float:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required to measure TTS audio")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid audio duration for {path}") from exc
    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration for {path}: {duration}")
    return duration


def synthesize_fitting_segment(
    segment: dict,
    language: str,
    output_path: Path,
    slot_s: float,
    synthesizer,
    probe: Callable[[Path], float] = audio_duration_s,
    **synthesis_kwargs,
) -> Path:
    text = segment.get(f"text_{language}")
    if not text:
        raise ValueError(f"Missing text_{language} in commentary segment")
    synthesizer.synthesize(text, output_path, **synthesis_kwargs)
    duration = probe(output_path)
    if duration <= slot_s + 0.05:
        return output_path

    fallback = segment.get(f"fallback_text_{language}")
    if not fallback:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Missing fallback_text_{language} for overflowing TTS")
    synthesizer.synthesize(fallback, output_path, **synthesis_kwargs)
    duration = probe(output_path)
    if duration <= slot_s + 0.05:
        return output_path
    output_path.unlink(missing_ok=True)
    raise ValueError(
        f"TTS still exceeds slot after fallback: {duration:.2f}s > {slot_s:.2f}s"
    )


def assemble_timeline(
    segments: list[dict], segment_paths: list[Path], output_path: Path,
    duration_s: float,
) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for timeline assembly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"anullsrc=r=24000:cl=mono:d={duration_s}",
    ]
    for path in segment_paths:
        cmd.extend(["-i", str(path)])
    filters = ["[0:a]acopy[base]"]
    inputs = ["[base]"]
    for index, segment in enumerate(segments, 1):
        delay = int(round(float(segment.get("timestamp_s", 0.0)) * 1000))
        label = f"seg{index}"
        filters.append(f"[{index}:a]adelay={delay}|{delay}[{label}]")
        inputs.append(f"[{label}]")
    filters.append(
        f"{''.join(inputs)}amix=inputs={len(inputs)}:duration=first:dropout_transition=0[out]"
    )
    cmd.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]",
        "-t", str(duration_s), "-c:a", "libmp3lame", "-q:a", "2",
        str(output_path),
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg timeline assembly failed:\n{result.stderr}")
    return output_path
