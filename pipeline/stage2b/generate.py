"""Direct ARK video observation and local event verification."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

from pipeline.stage2b.events import event_prompt_menu, get_event
from pipeline.stage2b.video import extract_window

DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260428"
ENERGIES = {"calm", "engaged", "excited", "explosive"}
CONFIDENCES = {"low", "medium", "high"}
COMMENTARY_KINDS = {"event", "hybrid", "tactical"}


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def ark_chat(
    prompt: str,
    *,
    video_path: Path | None = None,
    image_paths: Iterable[Path] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call ARK's OpenAI-compatible chat endpoint with inline media."""
    api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY; set it in the environment before Stage 2b.")

    from openai import OpenAI

    content: list[dict] = [{"type": "text", "text": prompt}]
    if video_path is not None:
        content.append({
            "type": "video_url",
            "video_url": {"url": _data_url(Path(video_path), "video/mp4")},
        })
    for path in image_paths or ():
        content.append({
            "type": "image_url",
            "image_url": {"url": _data_url(Path(path), "image/png")},
        })
    client = OpenAI(
        base_url=os.environ.get("ARK_BASE_URL", DEFAULT_ARK_BASE_URL),
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=os.environ.get("ARK_RESPONSES_MODEL", DEFAULT_ARK_MODEL),
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _direct_prompt(digest: str, duration_s: float, languages: list[str]) -> str:
    return f"""Watch the full football clip and return JSON only.
Use event_code only from this closed menu:
{event_prompt_menu()}

Tracking digest (supporting evidence, not a substitute for the video):
{digest}

Clip bounds: 0.0 through {duration_s:.3f} seconds. Requested languages: {languages}.
Return exactly {{"events": [...], "commentary": [...]}}.
Every event requires event_id, start_s, end_s, event_code, player_team,
player_jersey, actors, outcome, confidence, confidence_reasons,
suggested_wording_zh, suggested_wording_en, and energy.
actors must be a JSON array of non-empty strings.
confidence_reasons must be a JSON array of short text reasons,
never a plain string; use [] when there is nothing to note.
suggested_wording_zh must include the event's Chinese name from the menu and
the side as 左侧 or 右侧; suggested_wording_en must include the event name in
English and the side as left or right. Include the jersey number in both
wordings when it is clearly visible.
player_jersey must be the digits of a clearly visible shirt number, or an
empty string when it is not visible; never use placeholders such as "?".
Every commentary item requires kind, timestamp_s, end_s, text_zh, text_en,
fallback_text_zh, fallback_text_en, energy, and events_referenced.
All Chinese and English text and both fallback texts must be non-empty.
Confidence must be exactly one of ["low", "medium", "high"].
Commentary kind must be exactly one of ["event", "hybrid", "tactical"].
Energy must be exactly one of ["calm", "engaged", "excited", "explosive"].
Use unique event ids and reference only ids present in events. Do not fill the
clip with invented generic prose; report only directly supported observations.
"""


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _nonempty(item: dict, fields: Iterable[str], label: str) -> None:
    for field in fields:
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"{label}.{field} must be non-empty text")


def _parse_direct(raw: str, duration_s: float) -> tuple[list[dict], list[dict]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    events, commentary = payload.get("events"), payload.get("commentary")
    if not isinstance(events, list) or not isinstance(commentary, list):
        raise ValueError("events and commentary must be arrays")

    ids: set[str] = set()
    for index, event in enumerate(events):
        label = f"events[{index}]"
        if not isinstance(event, dict):
            raise ValueError(f"{label} must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in ids:
            raise ValueError(f"{label}.event_id must be non-empty and unique")
        ids.add(event_id)
        if get_event(event.get("event_code")) is None:
            raise ValueError(f"{label}.event_code is not in the closed menu")
        start = _number(event.get("start_s"), f"{label}.start_s")
        end = _number(event.get("end_s"), f"{label}.end_s")
        if start < 0 or end < start or end > duration_s:
            raise ValueError(f"{label} times must be ordered within clip bounds")
        _nonempty(
            event,
            ("player_team", "outcome", "suggested_wording_zh", "suggested_wording_en"),
            label,
        )
        if not isinstance(event.get("player_jersey"), str):
            raise ValueError(f"{label}.player_jersey must be text")
        actors = event.get("actors")
        if not isinstance(actors, list) or not actors or any(
            not isinstance(actor, str) or not actor.strip() for actor in actors
        ):
            raise ValueError(f"{label}.actors must be a non-empty text array")
        if "confidence_reasons" not in event:
            raise ValueError(f"{label}.confidence_reasons is missing")
        confidence_reasons = event.get("confidence_reasons")
        if confidence_reasons is None:
            raise ValueError(f"{label}.confidence_reasons must not be null")
        if not isinstance(confidence_reasons, list):
            raise ValueError(
                f"{label}.confidence_reasons must be a list; "
                f"got {type(confidence_reasons).__name__}"
            )
        for reason_index, reason in enumerate(confidence_reasons):
            item_label = f"{label}.confidence_reasons[{reason_index}]"
            if not isinstance(reason, str):
                raise ValueError(f"{item_label} must be text; got {type(reason).__name__}")
            if not reason.strip():
                raise ValueError(f"{item_label} must be non-blank text")
        if event.get("confidence") not in CONFIDENCES:
            raise ValueError(f"{label}.confidence is invalid")
        if event.get("energy") not in ENERGIES:
            raise ValueError(f"{label}.energy is invalid")

    for index, segment in enumerate(commentary):
        label = f"commentary[{index}]"
        if not isinstance(segment, dict):
            raise ValueError(f"{label} must be an object")
        start = _number(segment.get("timestamp_s"), f"{label}.timestamp_s")
        end = _number(segment.get("end_s"), f"{label}.end_s")
        if start < 0 or end < start or end > duration_s:
            raise ValueError(f"{label} times must be ordered within clip bounds")
        _nonempty(
            segment,
            ("text_zh", "text_en", "fallback_text_zh", "fallback_text_en"),
            label,
        )
        if segment.get("kind") not in COMMENTARY_KINDS:
            raise ValueError(f"{label}.kind is invalid")
        if segment.get("energy") not in ENERGIES:
            raise ValueError(f"{label}.energy is invalid")
        refs = segment.get("events_referenced")
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or ref not in ids for ref in refs
        ):
            raise ValueError(f"{label}.events_referenced contains an unknown id")
    return events, commentary


def observe_direct(
    video_path: Path,
    digest: str,
    duration_s: float,
    languages: list[str],
    call: Callable = ark_chat,
) -> tuple[list[dict], list[dict]]:
    """Observe a whole clip, retrying one malformed structured response."""
    prompt = _direct_prompt(digest, duration_s, languages)
    error = ""
    for attempt in range(2):
        reply = call(
            prompt if not error else f"{prompt}\nPrevious response errors: {error}\nReturn corrected JSON only.",
            video_path=Path(video_path),
            temperature=0.7,
        )
        try:
            return _parse_direct(reply, duration_s)
        except ValueError as exc:
            error = str(exc)
            if attempt:
                raise ValueError(f"ARK returned invalid direct observation twice: {error}") from exc
    raise AssertionError("unreachable")



def _parse_verification(raw: str, start_s: float, end_s: float) -> dict:
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid verification JSON: {exc.msg}") from exc
    required = {
        "event_code", "midpoint_s", "player_team", "player_jersey",
        "outcome", "directly_visible", "disagreements",
    }
    if not isinstance(verdict, dict) or not required <= verdict.keys():
        raise ValueError(f"verification requires fields: {sorted(required)}")
    if get_event(verdict["event_code"]) is None:
        raise ValueError("verification event_code is not in the closed menu")
    midpoint = _number(verdict["midpoint_s"], "midpoint_s")
    if midpoint < start_s or midpoint > end_s:
        raise ValueError("midpoint_s must be within the verification window")
    if not isinstance(verdict["directly_visible"], bool):
        raise ValueError("directly_visible must be boolean")
    disagreements = verdict["disagreements"]
    if not isinstance(disagreements, list) or any(
        not isinstance(item, str) or not item.strip() for item in disagreements
    ):
        raise ValueError("disagreements must be an array of non-empty text")
    _nonempty(verdict, ("player_team", "outcome"), "verification")
    if not isinstance(verdict["player_jersey"], str):
        raise ValueError("player_jersey must be text")
    return verdict

def verify_event_window(video_path: Path, event: dict, call: Callable = ark_chat) -> dict:
    """Re-watch an event with one second of context on each side."""
    start = _number(event.get("start_s"), "event.start_s")
    end = _number(event.get("end_s"), "event.end_s")
    with TemporaryDirectory() as directory:
        window = extract_window(
            Path(video_path), start - 1.0, end + 1.0, Path(directory) / "window.mp4"
        )
        prompt = f"""Verify this proposed football event and return JSON only.
Closed event menu:
{event_prompt_menu()}
Proposed event: {json.dumps(event, ensure_ascii=False)}
Return event_code, midpoint_s, player_team, player_jersey, outcome,
directly_visible (boolean), and disagreements (array). Do not infer unseen facts.
"""
        return _parse_verification(
            call(prompt, video_path=window, temperature=0.1),
            max(0.0, start - 1.0),
            end + 1.0,
        )
