"""Read commentary.json, synthesize TTS per segment, assemble into a timed audio track."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from pipeline.stage5_tts.tts_adapter import (
    ENERGY_ENGAGED,
    ENERGY_EXCITED,
    ENERGY_EXPLOSIVE,
    ENERGY_NORMAL,
    TTSAdapter,
)

log = logging.getLogger(__name__)

# Event codes that map to energy tiers (from events.json event_code).
_GOAL_CODES = ("football.goal", "goal")
_SHOT_CODES = ("football.shoot", "football.shot", "shoot", "shot")

# Text keywords as fallback when event codes are unavailable.
_GOAL_KEYWORDS = ("球进了", "进球", "GOAL!", "GOAL", "goal!")
_SHOT_KEYWORDS = ("射门", "远射", "太精彩", "STRIKE", "shoots", "fires", "SHOT")


# Commentary emits energy ∈ {calm, engaged, excited, explosive}; map onto tiers.
_ENERGY_TIERS = {
    "calm": ENERGY_NORMAL,
    "engaged": ENERGY_ENGAGED,
    "excited": ENERGY_EXCITED,
    "explosive": ENERGY_EXPLOSIVE,
}


def _text_key(language: str) -> str:
    return f"text_{language}"


def _trim_leading_silence(path: Path) -> None:
    """Strip leading silence from an mp3 in place so speech starts at t=0.

    TTS engines pad a short silence before the first phoneme, which makes every
    line land late. Removing it lets us place segments exactly on their event.
    """
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_name(path.stem + "_trim.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af", "silenceremove=start_periods=1:start_duration=0:start_threshold=-45dB",
        "-c:a", "libmp3lame", "-q:a", "2", str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink()


def _audio_duration_s(path: Path) -> float:
    """Return audio duration in seconds via ffprobe (0.0 on failure)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def load_event_code_map(events_json: Optional[Path]) -> dict[str, str]:
    """Map event_id → event_code from events.json."""
    if events_json is None or not Path(events_json).exists():
        return {}
    with open(events_json, encoding="utf-8") as f:
        data = json.load(f)
    return {
        ev["event_id"]: ev.get("event_code", "")
        for ev in data.get("events", [])
        if ev.get("event_id")
    }


def energy_for_segment(
    seg: dict,
    event_codes: Optional[dict[str, str]] = None,
) -> str:
    """Classify segment energy: explicit tag first, then event/text fallback.

    Returns ``explosive`` (goal) > ``excited`` (shot) > ``engaged`` > ``normal``.
    """
    tagged = _ENERGY_TIERS.get(str(seg.get("energy", "")).strip().lower())
    if tagged is not None:
        return tagged

    event_codes = event_codes or {}
    codes = [
        event_codes.get(ref, "").lower()
        for ref in seg.get("events_referenced", [])
    ]
    # also accept bare codes embedded in refs (defensive)
    codes += [ref.lower() for ref in seg.get("events_referenced", [])]

    has_goal = any(any(g in c for g in _GOAL_CODES) for c in codes if c)
    has_shot = any(any(s in c for s in _SHOT_CODES) for c in codes if c)

    text = seg.get("text_zh", "") + " " + seg.get("text_en", "")
    if not has_goal and any(kw in text for kw in _GOAL_KEYWORDS):
        has_goal = True
    if not has_shot and any(kw in text for kw in _SHOT_KEYWORDS):
        has_shot = True

    if has_goal:
        return ENERGY_EXPLOSIVE
    if has_shot:
        return ENERGY_EXCITED
    return ENERGY_NORMAL


def synthesize_segments(
    segments: List[dict],
    output_dir: Path,
    language: str,
    adapter: TTSAdapter,
    event_codes: Optional[dict[str, str]] = None,
    voice_tag: str = "default",
) -> List[Path]:
    """Call TTS for each segment, returning a list of per-segment audio files.

    *voice_tag* isolates caches per voice (e.g. ``default`` vs ``wang``).
    """
    seg_dir = output_dir / "tts_segments" / language / voice_tag
    seg_dir.mkdir(parents=True, exist_ok=True)

    text_key = _text_key(language)
    paths: List[Path] = []

    for i, seg in enumerate(segments):
        text = seg.get(text_key, "")
        if not text:
            log.warning("Segment %d has no %s, skipping TTS", i, text_key)
            paths.append(Path(""))
            continue

        energy = energy_for_segment(seg, event_codes)
        # include energy in filename so cache invalidates when tier changes
        digest = hashlib.sha1(f"{language}\0{energy}\0{text}".encode("utf-8")).hexdigest()[:10]
        out = seg_dir / f"seg_{i:03d}_{energy}_{digest}.mp3"
        if out.exists() and out.stat().st_size > 0:
            log.debug("Reusing cached %s", out)
            paths.append(out)
            continue

        adapter.synthesize(text, out, energy=energy)
        _trim_leading_silence(out)
        paths.append(out)

    return paths


_FADE_S = 0.12


_MAX_STRETCH = 1.35


def _plan_placement(
    starts_wanted: List[float],
    durations: List[float],
    highlights: List[bool],
    total_duration_s: float,
) -> list[tuple[float, float, float]]:
    """Compute (start, tempo, allowed_duration) for each segment.

    Normal lines flow back-to-back: start at ``max(timestamp_s, prev_end)`` and
    spill freely (commentary lag is acceptable — 2026-07-08 decision). Highlight
    lines always start exactly at their ``timestamp_s``. The only hard
    boundaries are a following locked highlight and the end of the clip; a line
    hitting one is first sped up (tempo ≤ _MAX_STRETCH) and only the remainder
    is trimmed via allowed_duration. Words are never deleted from the text.
    """
    n = len(starts_wanted)
    plan: list[tuple[float, float, float]] = []
    prev_end = 0.0
    for i in range(n):
        ts, dur, is_hi = starts_wanted[i], durations[i], highlights[i]
        start = max(ts, 0.0) if is_hi else max(ts, prev_end)

        boundary: Optional[float] = None
        if i + 1 < n and highlights[i + 1]:
            boundary = starts_wanted[i + 1]
        elif i + 1 == n:
            boundary = total_duration_s

        tempo = 1.0
        allowed = dur
        if boundary is not None and start + dur > boundary:
            room = max(boundary - start, 0.0)
            allowed = room
            if room > 0:
                tempo = min(dur / room, _MAX_STRETCH)

        eff_dur = min(dur / tempo, allowed)
        plan.append((start, tempo, allowed))
        prev_end = start + eff_dur
    return plan


def assemble_timeline(
    segments: List[dict],
    seg_paths: List[Path],
    output_path: Path,
    total_duration_s: float = 30.0,
    highlights: Optional[List[bool]] = None,
) -> Path:
    """Place each segment audio on the timeline with back-to-back, highlight-synced timing."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for timeline assembly")

    highlights = highlights or [False] * len(segments)

    valid: list[tuple[Path, float, float, bool]] = []
    for seg, p, is_hi in zip(segments, seg_paths, highlights):
        if p and p.exists() and p.stat().st_size > 0:
            valid.append((p, seg.get("timestamp_s", 0.0), _audio_duration_s(p), is_hi))

    if not valid:
        raise RuntimeError("No valid TTS segments to assemble")

    plan = _plan_placement(
        [ts for _, ts, _, _ in valid],
        [dur for _, _, dur, _ in valid],
        [is_hi for _, _, _, is_hi in valid],
        total_duration_s,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"anullsrc=r=24000:cl=mono:d={total_duration_s}",
    ]
    for p, _, _, _ in valid:
        cmd += ["-i", str(p)]

    filters: list[str] = ["[0]acopy[base]"]
    mix_inputs = ["[base]"]
    for idx, ((start, tempo, allowed), (_, _, dur, _)) in enumerate(zip(plan, valid)):
        if allowed <= 0.05:
            continue
        inp = idx + 1
        label = f"d{idx}"
        chain = f"[{inp}]"
        eff_dur = dur
        if tempo > 1.01:
            chain += f"atempo={tempo:.3f},"
            eff_dur = dur / tempo
        # Trim + fade only for what tempo could not absorb (locked highlight
        # or clip end ahead).
        if eff_dur > allowed + 0.01:
            fade = min(_FADE_S, allowed / 2)
            chain += f"atrim=0:{allowed:.3f},afade=t=out:st={max(allowed - fade, 0):.3f}:d={fade:.3f},"
        delay_ms = int(round(start * 1000))
        chain += f"adelay={delay_ms}|{delay_ms}[{label}]"
        filters.append(chain)
        mix_inputs.append(f"[{label}]")

    mix_str = "".join(mix_inputs)
    filters.append(
        f"{mix_str}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0[out]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-t", str(total_duration_s),
        "-c:a", "libmp3lame", "-q:a", "2",
        str(output_path),
    ]

    log.info("Assembling %d segments → %s", len(valid), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg timeline assembly failed:\n{result.stderr}")

    log.info("Audio track ready: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


def synthesize_commentary(
    commentary_json: Path,
    output_dir: Path,
    language: str = "zh",
    adapter: Optional[TTSAdapter] = None,
    events_json: Optional[Path] = None,
    voice_tag: str = "default",
    audio_path: Optional[Path] = None,
) -> Path:
    """End-to-end: read commentary.json → TTS each segment → assemble timeline.

    *voice_tag* separates segment caches (``default`` for edge-tts,
    ``wang`` for the cloned voice). *audio_path* overrides the default
    ``commentary_{language}.mp3`` output name.
    """
    with open(commentary_json, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("commentary", [])
    if not segments:
        raise RuntimeError(f"No commentary segments in {commentary_json}")

    duration_s = data.get("video_info", {}).get("duration_s", 30.0)

    from pipeline.stage5_tts.pace_filter import filter_segments
    segments = filter_segments(segments)

    events_path = events_json or (Path(output_dir) / "events.json")
    event_codes = load_event_code_map(events_path)

    if adapter is None:
        from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
        adapter = EdgeTTSAdapter(language=language)

    seg_paths = synthesize_segments(
        segments, output_dir, language, adapter,
        event_codes=event_codes, voice_tag=voice_tag,
    )

    highlights = [
        energy_for_segment(seg, event_codes) in (ENERGY_EXCITED, ENERGY_EXPLOSIVE)
        for seg in segments
    ]

    output_audio = Path(audio_path) if audio_path else (
        output_dir / f"commentary_{language}.mp3"
    )
    assemble_timeline(
        segments, seg_paths, output_audio,
        total_duration_s=duration_s, highlights=highlights,
    )
    return output_audio
