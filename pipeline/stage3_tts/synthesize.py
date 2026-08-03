"""Synthesize commentary segments and fit them to event slots."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Callable


MAX_TEMPO = 1.5


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


def _speed_up_to_fit(path: Path, duration_s: float, slot_s: float) -> bool:
    if slot_s <= 0:
        return False
    tempo = duration_s / slot_s * 1.01
    if tempo > MAX_TEMPO:
        return False
    temporary = path.with_name(f".{path.stem}.tempo{path.suffix}")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-filter:a", f"atempo={tempo:.6f}",
             str(temporary)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg tempo adjustment failed:\n{result.stderr}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def synthesize_fitting_segment(
    segment: dict,
    language: str,
    output_path: Path,
    slot_s: float,
    synthesizer,
    probe: Callable[[Path], float] = audio_duration_s,
    prefer_fallback: bool = False,
    **synthesis_kwargs,
) -> Path:
    primary_field = f"fallback_text_{language}" if prefer_fallback else f"text_{language}"
    primary_text = segment.get(primary_field)
    if not primary_text:
        raise ValueError(f"Missing {primary_field} in commentary segment")

    def synthesize_and_fit(text: str) -> bool:
        synthesizer.synthesize(text, output_path, **synthesis_kwargs)
        duration = probe(output_path)
        if duration <= slot_s + 0.05:
            return True
        return (_speed_up_to_fit(output_path, duration, slot_s)
                and probe(output_path) <= slot_s + 0.05)

    if synthesize_and_fit(primary_text):
        return output_path

    if prefer_fallback:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Preferred fallback TTS exceeds slot: {slot_s:.2f}s slot")

    fallback = segment.get(f"fallback_text_{language}")
    if not fallback:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Missing fallback_text_{language} for overflowing TTS")
    if synthesize_and_fit(fallback):
        return output_path
    output_path.unlink(missing_ok=True)
    raise ValueError(f"TTS still exceeds slot after fallback: {slot_s:.2f}s slot")


def assemble_timeline(
    segments: list[dict], segment_paths: list[Path], output_path: Path,
    duration_s: float,
) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for timeline assembly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "anullsrc=r=24000:cl=mono",
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
