"""Direct ARK video-to-events/commentary generation for Stage 2B."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

from pipeline.config import load_env

log = logging.getLogger(__name__)

EVENTS = {
    "football.pass": "pass between teammates",
    "football.shoot": "shot toward goal",
    "football.goal": "goal scored",
    "football.clearance": "defensive clearance",
    "football.interception": "pass intercepted",
    "football.dribble": "controlled carry past pressure",
    "football.tackle": "challenge for the ball",
    "football.pressing": "active pressure on the ball carrier",
    "football.save": "goalkeeper save",
    "football.goal_kick": "goalkeeper restart or long kick",
    "football.buildup": "routine possession buildup",
}
ENERGY_LEVELS = {"calm", "engaged", "excited", "explosive"}
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260428"


def ark_model() -> str:
    load_env()
    return (
        os.environ.get("ARK_VIDEO_MODEL")
        or os.environ.get("ARK_RESPONSES_MODEL")
        or os.environ.get("DOUBAO_MODEL")
        or DEFAULT_ARK_MODEL
    )


def ark_generate(prompt: str, clip_mp4: Path) -> str:
    load_env()
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY or DOUBAO_API_KEY in the environment or .env")
    if not clip_mp4.is_file() or clip_mp4.suffix.lower() != ".mp4":
        raise FileNotFoundError(f"Stage 2B video not found or not mp4: {clip_mp4}")
    from openai import OpenAI

    encoded = base64.b64encode(clip_mp4.read_bytes()).decode("ascii")
    response = OpenAI(
        api_key=api_key,
        base_url=(
            os.environ.get("ARK_BASE_URL")
            or os.environ.get("DOUBAO_BASE_URL")
            or DEFAULT_ARK_BASE_URL
        ),
    ).chat.completions.create(
        model=ark_model(),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
        ]}],
        max_tokens=4096,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def _normalize_segment(segment: dict, index: int) -> dict:
    timestamp = segment.get("timestamp_s", segment.get("start_s", segment.get("t", index * 1.5)))
    refs = segment.get("events_referenced") or segment.get("event_ids") or []
    if isinstance(refs, str):
        refs = [refs]
    normalized = {
        "timestamp_s": float(timestamp),
        "end_s": float(segment.get("end_s", float(timestamp) + 1.5)),
        "text_en": str(segment.get("text_en") or segment.get("en_commentary") or segment.get("en") or ""),
        "text_zh": str(segment.get("text_zh") or segment.get("zh_commentary") or segment.get("zh") or ""),
        "events_referenced": list(refs),
    }
    if str(segment.get("energy", "")).lower() in ENERGY_LEVELS:
        normalized["energy"] = str(segment["energy"]).lower()
    return normalized


def validate_pacing(segments: list[dict], duration_s: float) -> list[str]:
    problems = []
    minimum = max(1, int(duration_s // 5))
    if len(segments) < minimum:
        problems.append(f"only {len(segments)} segments; need at least {minimum}")
    for segment in segments:
        window = segment["end_s"] - segment["timestamp_s"]
        if window > 7:
            problems.append(f"segment {segment['timestamp_s']}-{segment['end_s']}s spans {window:.1f}s; max 7s")
        elif window < 2:
            problems.append(f"segment {segment['timestamp_s']}-{segment['end_s']}s spans {window:.1f}s; min 2s")
    return problems


def build_direct_prompt(digest: str, duration_s: float, languages: list[str]) -> str:
    menu = "\n".join(f"- {code}: {description}" for code, description in EVENTS.items())
    minimum = max(1, int(duration_s // 5))
    return f"""You are a professional football commentator and match analyst.
Watch the attached {duration_s:.0f}-second clip from start to end.

[Reliable Stage 1 tracking digest]
{digest}

Choose event_code only from:
{menu}

Generate {' and '.join(languages).upper()} commentary. Use jersey numbers only
when visible or confirmed by the digest. energy must be calm, engaged, excited,
or explosive. Produce at least {minimum} segments, no segment longer than 7
seconds or shorter than 2 seconds, and cover the clip without dead air.

Return one JSON object only:
{{"events": [{{"event_id": "evt_001", "timestamp_s": 0.0,
"event_code": "football.buildup", "player_jersey": "", "player_team": "left",
"description": "short factual description"}}],
"commentary": [{{"timestamp_s": 0.0, "end_s": 5.0, "text_en": "...",
"text_zh": "...", "energy": "calm", "events_referenced": ["evt_001"]}}]}}
Every referenced event id must exist."""


def parse_direct_output(raw_text: str) -> tuple[list[dict], list[dict]]:
    match = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
    if not match:
        raise ValueError("LLM output contains no JSON object")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output JSON is malformed: {exc}") from exc
    events = [event for event in data.get("events", []) if isinstance(event, dict) and event.get("event_id")]
    invalid_codes = [event.get("event_code") for event in events if event.get("event_code") not in EVENTS]
    if invalid_codes:
        raise ValueError(f"events contain unsupported event codes: {invalid_codes}")
    raw_segments = data.get("commentary")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError('LLM output has no "commentary" array')
    segments = [_normalize_segment(segment, index) for index, segment in enumerate(raw_segments) if isinstance(segment, dict)]
    if not segments:
        raise ValueError("LLM output has no usable commentary entries")
    known_ids = {event["event_id"] for event in events}
    for segment in segments:
        unknown = [ref for ref in segment["events_referenced"] if ref not in known_ids]
        if unknown:
            raise ValueError(f"commentary references unknown event ids: {unknown}")
    return events, segments


def generate_direct(
    clip_mp4: Path,
    digest: str,
    duration_s: float,
    fps: int,
    output_dir: Path,
    languages: list[str],
    generate: Callable[[str, Path], str] = ark_generate,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_direct_prompt(digest, duration_s, languages)

    def attempt(text: str):
        try:
            events, segments = parse_direct_output(generate(text, clip_mp4))
        except ValueError as exc:
            return None, None, [str(exc)]
        return events, segments, validate_pacing(segments, duration_s)

    events, segments, problems = attempt(prompt)
    if problems:
        log.warning("Stage 2B attempt rejected: %s; retrying once", problems)
        events, segments, problems = attempt(
            prompt + "\nFix these previous-response problems and return the full JSON again:\n- " + "\n- ".join(problems)
        )
        if segments is None or problems:
            raise RuntimeError(f"Stage 2B generation failed after retry: {problems}")

    video_info = {"source": str(clip_mp4), "duration_s": duration_s, "fps": fps}
    (output_dir / "events.json").write_text(
        json.dumps({"video_info": video_info, "events": events}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commentary_path = output_dir / "commentary.json"
    commentary_path.write_text(json.dumps({
        "video_info": video_info,
        "model_info": {"name": ark_model(), "backend": "doubao-direct-2b"},
        "language": languages,
        "commentary": segments,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return commentary_path
