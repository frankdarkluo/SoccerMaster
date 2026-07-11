from __future__ import annotations

from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class EventDef:
    code: str
    description: str
    display_name_zh: str
    importance_base: float


_EVENTS = {
    item.code: item
    for item in [
        EventDef("football.corner", "Corner setup or delivery", "角球", 0.35),
        EventDef("football.pass", "Ball delivered to a teammate", "传球", 0.15),
        EventDef("football.clearance", "Defender removes danger", "解围", 0.40),
        EventDef("football.interception", "Opponent wins the ball", "抢断", 0.55),
        EventDef("football.dribble", "Controlled ball carry", "带球", 0.20),
        EventDef("football.tackle", "Challenge wins the ball", "铲抢", 0.60),
        EventDef("football.shoot", "Attempt toward goal", "射门", 0.75),
        EventDef("football.goal", "Ball enters the goal", "进球", 1.00),
        EventDef("football.save", "Goalkeeper prevents a goal", "扑救", 0.80),
        EventDef("football.goal_kick", "Long goalkeeper/defender restart", "开大脚", 0.25),
        EventDef("football.buildup", "Controlled possession advances", "组织推进", 0.10),
        EventDef("football.pressing", "Coordinated pressure", "逼抢", 0.20),
    ]
}


def get_event(code: str) -> Optional[EventDef]:
    return _EVENTS.get(code)


def event_prompt_menu() -> str:
    return "\n".join(
        f"- {event.code}: {event.description} ({event.display_name_zh})"
        for event in _EVENTS.values()
    )
