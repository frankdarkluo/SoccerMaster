"""Deterministic adapter for tests and offline development."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter

_EVENT_LINE_RE = re.compile(
    r"^t=(?P<ts>[\d.]+)s:\s*\[(?P<code>[^\]]+)\]"
    r"(?:\s+id=(?P<event_id>\S+))?"
    r"(?:\s+#(?P<jersey>\S+))?"
    r"(?:\s+\((?P<team>[^)]+)\))?"
    r"(?:\s+→\s+#(?P<target_jersey>\S+))?"
    r"(?:\s+\((?P<target_team>[^)]+)\))?"
)

_TEMPLATES = {
    "football.goal": (
        "GOAL! A clinical finish finds the net!",
        "进球！一脚精准射门破门得分！",
    ),
    "football.shoot": (
        "A powerful shot toward goal!",
        "一脚有力射门直奔球门！",
    ),
    "football.pass": (
        "A composed pass keeps the move alive.",
        "一记稳健传球延续进攻。",
    ),
    "football.assist": (
        "A decisive assist sets up the chance!",
        "关键助攻创造得分机会！",
    ),
    "football.clearance": (
        "A firm clearance eases the pressure.",
        "一脚解围缓解防守压力。",
    ),
    "football.interception": (
        "An interception turns defense into attack.",
        "一次拦截完成攻防转换。",
    ),
}


def _player_phrase(jersey: Optional[str], team: Optional[str], lang: str) -> str:
    if jersey and team:
        return f"#{jersey} ({team})" if lang == "en" else f"{team}队 #{jersey}"
    if jersey:
        return f"#{jersey}"
    if team:
        return f"the {team} side" if lang == "en" else f"{team}队"
    return ""


def _segment_for_event(match: re.Match, event_id: str, highlight: bool) -> dict:
    ts = float(match.group("ts"))
    code = match.group("code")
    jersey = match.group("jersey")
    team = match.group("team")
    target_jersey = match.group("target_jersey")
    event_id = match.group("event_id") or event_id

    en_base, zh_base = _TEMPLATES.get(
        code,
        (f"{code.replace('football.', '').replace('_', ' ').title()}!", f"{code}。"),
    )
    player_en = _player_phrase(jersey, team, "en")
    player_zh = _player_phrase(jersey, team, "zh")

    if code == "football.pass" and target_jersey:
        en = f"{player_en} finds #{target_jersey}." if player_en else en_base
        zh = f"{player_zh}传给 #{target_jersey}。" if player_zh else zh_base
    elif player_en:
        en = f"{player_en}: {en_base}"
        zh = f"{player_zh}：{zh_base}"
    else:
        en, zh = en_base, zh_base

    if highlight and code != "football.goal":
        en = f"Highlight — {en}"
        zh = f"高光 — {zh}"

    return {
        "timestamp_s": ts,
        "end_s": round(ts + 1.5, 2),
        "text_en": en,
        "text_zh": zh,
        "events_referenced": [event_id],
        "event_code": code,
    }


def commentary_from_prompt(prompt: str) -> str:
    """Build event-aligned bilingual commentary segments from a prompt timeline."""
    segments: List[dict] = []
    event_idx = 0
    for line in prompt.splitlines():
        match = _EVENT_LINE_RE.match(line.strip())
        if not match:
            continue
        event_idx += 1
        event_id = f"evt_{event_idx:03d}"
        highlight = "HIGHLIGHT" in line
        segments.append(_segment_for_event(match, event_id, highlight))

    if not segments:
        return json.dumps([
            {
                "timestamp_s": 0.0,
                "end_s": 25.0,
                "text_en": "Build-up play in midfield.",
                "text_zh": "中场组织进攻。",
                "events_referenced": [],
            },
            {
                "timestamp_s": 25.0,
                "end_s": 30.0,
                "text_en": "GOAL! A brilliant finish!",
                "text_zh": "进球！精彩的射门！",
                "events_referenced": ["evt_001"],
            },
        ], ensure_ascii=False, indent=2)

    # Fill opening gap before first event.
    if segments[0]["timestamp_s"] > 0.5:
        segments.insert(0, {
            "timestamp_s": 0.0,
            "end_s": segments[0]["timestamp_s"],
            "text_en": "The teams settle into shape as the move develops.",
            "text_zh": "双方落位成型，进攻徐徐展开。",
            "events_referenced": [],
        })

    for seg in segments:
        seg["end_s"] = min(float(seg["end_s"]), 30.0)
    segments[-1]["end_s"] = 30.0

    return json.dumps(segments, ensure_ascii=False, indent=2)


class MockLLMAdapter(LLMAdapter):
    def __init__(self, response: str | None = None):
        self.response = response
        self.last_prompt: str | None = None
        self.last_visual_input = None

    def supports_video(self) -> bool:
        return False

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        self.last_prompt = prompt
        self.last_visual_input = self.prepare_visual_input(visual_input)
        if self.response is not None:
            return self.response
        return commentary_from_prompt(prompt)
