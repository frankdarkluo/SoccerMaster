#!/usr/bin/env python3
"""Run and report the frozen 37-clip GSR-2025 multi-agent pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.video_models import generate_json, model_name, temporary_compact_clip


GT_PATH = ROOT / "data/grounded_tactics/gsr2025_gt_37.csv"
GLOSSARY_PATH = ROOT / "data/足球战术数据库_词条表_Grid.csv"
VIDEO_ROOT = ROOT / "data/GSR-2025/videos"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs/multiagent_pilot"
PROVIDERS = ("doubao", "gemini")
MODEL_FPS = {"A": 2.0, "B": 4.0, "C": 2.0}
TEMPERATURE = 0.0
MIN_SCORE = 0.5
# Former hard gates are scored terms, not a binary AND. Per-tactic soft weights sum to 1.0, so
# two gates carry 2*GATE_WEIGHT/(2*GATE_WEIGHT+1) of the normalised score. Re-tune if A/B change.
GATE_WEIGHT = 0.25
# Inline base64 upload of the 1080p25 source timed out on 17/37 r0 clips; downscale once per clip.
VIDEO_PREPROCESS = {"height": 720, "crf": 28, "fps": None}
# B reports 3-6 events across 30s (r0 gaps: p25 2.5s, median 3.5s), so the original 1s/2s
# windows made the counter-attack gates unsatisfiable at that granularity.
POSSESSION_JOIN_TOLERANCE_S = 2.0
IMMEDIATE_FORWARD_WINDOW_S = 4.0
# A wall pass returns within about a second; 2.0s is an engineering tolerance for frame-sampled
# timing, not part of any published definition of a one-two.
WALL_PASS_RETURN_S = 2.0
EXPECTED_CLIPS = 37
RULE_VERSION = "multiagent-pilot-rules-v1"
SCHEMA_VERSION = "gsr2025-multiagent-pilot-v1"

TACTICS = (
    "counter-attack",
    "long-ball",
    "one-two",
    "tactical-foul",
    "cutback",
    "run-in-behind",
    "line-breaking-pass",
    "back-line-possession-management",
)
GT_ALIASES = {"back-line": "back-line-possession-management"}

BALL_PROGRESSIONS = (
    "defensive_third -> middle_third",
    "middle_third -> final_third",
    "defensive_third -> final_third",
    "within_defensive_third",
    "within_middle_third",
    "within_final_third",
    "no_clear_progression",
)
SPATIAL_FACTS = (
    "attackers_outnumber_goal_side_defenders",
    "multiple_attackers_run_forward",
    "defenders_not_set",
    "defenders_set",
    "runner_moves_from_last_line_front_to_behind",
    "pass_path_crosses_defensive_line",
    "receiver_controls_between_or_behind_lines",
    "ball_carrier_reaches_left_byline",
    "ball_carrier_reaches_right_byline",
    "backward_pass_target_in_central_penalty_area",
    "first_passer_runs_beyond_nearby_defender",
    "forward_target_or_landing_space_beyond_midfield",
    "teammates_positioned_for_second_ball",
    "ball_circulates_across_back_line",
    "ball_remains_in_own_half",
    "contact_outside_defending_penalty_area",
)
FACT_JUDGEMENTS = ("yes", "no", "not_observable")
ACTIONS = (
    "possession_gain",
    "interception",
    "tackle",
    "clearance",
    "forward_dribble",
    "lateral_dribble",
    "backward_dribble",
    "forward_ground_pass",
    "lateral_ground_pass",
    "backward_ground_pass",
    "forward_aerial_pass",
    "lateral_aerial_pass",
    "backward_aerial_pass",
    "one_touch_return_pass",
    "reception",
    "duel",
    "header",
    "shot",
    "foul",
    "pull_or_block",
    "whistle",
    "throw_in",
    "free_kick",
    "goal_kick",
    "corner",
)
# Relational predicates ("B returns the ball to the player who passed it", "the carrier who reached
# the byline cuts it back") need to say WHO acted. Without an actor field B can only emit a single
# collapsed token such as one_touch_return_pass, which it never did (0/26 in r2_pro) even on clips
# whose GT was one-two. These are within-clip relative ids, not shirt numbers.
ATTACK_ACTORS = ("A1", "A2", "A3", "A4")
DEFEND_ACTORS = ("D1", "D2", "D3", "D4")
ACTORS = ATTACK_ACTORS + DEFEND_ACTORS + ("unknown",)
DEFENSIVE_CHANGES = (
    "retreat_toward_own_goal",
    "recover_behind_ball",
    "collapse_toward_ball",
    "hold_shape",
    "step_up",
    "no_visible_change",
    "not_observable",
)

# A clearance is the defending side hoofing the ball away, not a possession gain by the attacking side.
POSSESSION_ACTIONS = {"possession_gain", "interception", "tackle"}
FORWARD_ACTIONS = {"forward_dribble", "forward_ground_pass", "forward_aerial_pass"}
FORWARD_PASSES = {"forward_ground_pass", "forward_aerial_pass"}
GROUND_PASSES = {"forward_ground_pass", "lateral_ground_pass", "backward_ground_pass"}
PASSES = GROUND_PASSES | {"forward_aerial_pass", "lateral_aerial_pass", "backward_aerial_pass"}
RECEIVE_ACTIONS = {"reception", "header", "shot", "forward_dribble", "lateral_dribble", "backward_dribble"}
BACK_LINE_PASSES = {"lateral_ground_pass", "backward_ground_pass"}
RESTART_ACTIONS = {"throw_in", "free_kick", "goal_kick", "corner"}

A_SCHEMA = {
    "type": "object",
    "properties": {
        "possession_change": {"type": ["number", "null"]},
        "ball_progression": {"type": "string", "enum": list(BALL_PROGRESSIONS)},
        "key_spatial_facts": {
            "type": "object",
            "properties": {fact: {"type": "string", "enum": list(FACT_JUDGEMENTS)} for fact in SPATIAL_FACTS},
            "required": list(SPATIAL_FACTS),
            "additionalProperties": False,
        },
    },
    "required": ["possession_change", "ball_progression", "key_spatial_facts"],
    "additionalProperties": False,
}
B_MAX_EVENTS = 6


B_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array", "maxItems": B_MAX_EVENTS,
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "number"},
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "actor": {"type": "string", "enum": list(ACTORS)},
                },
                "required": ["time", "action", "actor"],
                "additionalProperties": False,
            },
        },
        "defensive_change": {"type": "string", "enum": list(DEFENSIVE_CHANGES)},
    },
    "required": ["events", "defensive_change"],
    "additionalProperties": False,
}

# C arbitrates rather than only flagging errors: it picks among the Rule's top candidates or
# rejects them all, and `final.source` records which of the two produced the answer. As a pure
# error-spotter its flag predicted a wrong Rule label at precision 0.69 against a 0.65 base rate —
# no signal — while the Rule's top-3 held the right answer 73% of the time, so choosing is where
# the headroom is.
C_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": ["string", "null"], "enum": [*TACTICS, None]},
        "rationale": {"type": "string"},
        "objections": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
    },
    "required": ["label", "rationale", "objections"],
    "additionalProperties": False,
}
C_CANDIDATES = 3

A_PROMPT = f"""你是足球视频的空间证据模块。只回答在哪里，以及空间结构有什么关键变化。
不要命名战术，不要推断意图，不要使用队服颜色，不要做人数精确统计。

观看完整30秒视频并输出严格JSON：
- possession_change：与主要连贯片段直接相关、最清晰的一次球权转换秒数，精确到0.5秒；没有则null。
- ball_progression：只能从以下值选择：{', '.join(BALL_PROGRESSIONS)}。
- key_spatial_facts：对下面每一个事实独立判断，必须全部给出，取值只能是 yes / no / not_observable。

判断规则：
- 每条事实单独看，不要因为已经写了某几条 yes 就跳过其余的。
- yes：画面中直接看见该事实成立。
- no：判断所需的区域或瞬间可见，但该事实不成立。
- not_observable：镜头从未拍到判断所需的区域或瞬间。not_observable 是正式答案，不是含糊其辞。
- 没看见就不要写成 yes。

必须逐条回答的事实：{', '.join(SPATIAL_FACTS)}。

只输出JSON。"""

_BASE_B_PROMPT = f"""你是足球视频的事件序列模块。只回答发生了什么、发生时间，以及一次总体防守变化。
不要命名战术，不要推断意图，不要使用队服颜色。

观看完整30秒视频并输出严格JSON：
- events：按时间排序的关键原子动作，最多6条；每条含 time、action、actor 三项。
- time精确到0.5秒；action只能从以下值选择：{', '.join(ACTIONS)}。
- defensive_change只能从以下值选择：{', '.join(DEFENSIVE_CHANGES)}。

actor 的用法（这一项决定能不能看出配合关系，务必认真填）：
- 进攻方球员用 {', '.join(ATTACK_ACTORS)}，防守方球员用 {', '.join(DEFEND_ACTORS)}，实在分不清是谁写 unknown。
- 编号是本片段内部的临时代号，不是球衣号。谁先在片段里出现就先用 A1，下一个新出现的进攻球员用 A2，依此类推。
- 最重要的一条：**同一名球员在整个片段里必须始终用同一个代号**。如果 A1 把球传出去，几秒后球又回到同一个人脚下，那一条的 actor 仍然写 A1。
- 传球类动作的 actor 写传球的人；接球、带球、射门、头球的 actor 写触球的人；抢断、拦截、犯规的 actor 写实施动作的防守球员。
- 只有在你确实分不清球员身份时才写 unknown，不要图省事全写 unknown。

events的取舍规则（条数不是目标）：
- 只记录你在画面中确认看到的动作。没看清就不写，宁可只写2条也不要补满6条。
- 只描述同一段连贯攻防序列内的动作。片段开头或结尾属于上一回合/下一回合的动作，不要混进来。
- time写你实际看到该动作发生的时刻，不要按"应该大概在这里"估算。

动作判别（这几类最容易混）：
- 地面传球（*_ground_pass）指球基本贴地滚动；空中传球（*_aerial_pass）指球明显离地飞行。按球的实际飞行高度判断，不要按传球距离判断。
- header只在球员确实用头触球时使用；用脚停球、用脚解围都不是header。
- goal_kick只指门将在球门区内主罚的定位球；门将手抛球发动进攻不是goal_kick。
- corner只指从角旗区发出的球；球从边线出界后重新开球是throw_in。
- free_kick不包括中圈开球。
- one_touch_return_pass只表示一脚把球回给原传球者，不等于战术标签。

只输出JSON。"""

B_PROMPT = _BASE_B_PROMPT


def _schema_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} fields are not strict")


def _half_second(value: Any, duration_s: float = 30.0) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= float(value) <= duration_s
        and abs(float(value) * 2 - round(float(value) * 2)) < 1e-9
    )


def validate_a(payload: dict[str, Any]) -> dict[str, Any]:
    _schema_keys(payload, {"possession_change", "ball_progression", "key_spatial_facts"}, "A")
    if payload["possession_change"] is not None and not _half_second(payload["possession_change"]):
        raise ValueError("A.possession_change must be null or a 0.5-second timestamp")
    if payload["ball_progression"] not in BALL_PROGRESSIONS:
        raise ValueError("invalid A.ball_progression")
    facts = payload["key_spatial_facts"]
    if not isinstance(facts, dict) or set(facts) != set(SPATIAL_FACTS):
        raise ValueError("A.key_spatial_facts must contain exactly the 16 spatial facts")
    if any(facts[fact] not in FACT_JUDGEMENTS for fact in SPATIAL_FACTS):
        raise ValueError("invalid A.key_spatial_facts")
    return payload


def validate_b(payload: dict[str, Any]) -> dict[str, Any]:
    _schema_keys(payload, {"events", "defensive_change"}, "B")
    events = payload["events"]
    if not isinstance(events, list) or len(events) > B_MAX_EVENTS:
        raise ValueError(f"B.events must contain 0-{B_MAX_EVENTS} events")
    previous = -1.0
    for event in events:
        _schema_keys(event, {"time", "action", "actor"}, "B event")
        if not _half_second(event["time"]):
            raise ValueError("B event time must use 0.5-second precision")
        if float(event["time"]) < previous:
            raise ValueError("B.events must be time ordered")
        if event["action"] not in ACTIONS:
            raise ValueError("invalid B event action")
        if event["actor"] not in ACTORS:
            raise ValueError("invalid B event actor")
        previous = float(event["time"])
    if payload["defensive_change"] not in DEFENSIVE_CHANGES:
        raise ValueError("invalid B.defensive_change")
    return payload


def validate_c(payload: dict[str, Any], candidates: Iterable[str] = ()) -> dict[str, Any]:
    _schema_keys(payload, {"label", "rationale", "objections"}, "C")
    objections = payload["objections"]
    if not isinstance(objections, list) or len(objections) > 2:
        raise ValueError("invalid C objections")
    if payload["label"] is not None and payload["label"] not in TACTICS:
        raise ValueError("invalid C label")
    allowed = set(candidates)
    if allowed and payload["label"] is not None and payload["label"] not in allowed:
        raise ValueError("C label must be one of the offered candidates or null")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("C rationale must be a non-empty string")
    if any(not isinstance(item, str) or not re.match(r"^[AB]\.[A-Za-z0-9_\[\].]+\s*[:：]\s*\S", item) for item in objections):
        raise ValueError("C objections must target a concrete A.* or B.* field")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_interval(value: str) -> dict[str, float]:
    match = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", value.strip())
    if not match:
        raise ValueError(f"invalid GT interval: {value!r}")
    sm, ss, em, es = map(int, match.groups())
    start, end = float(sm * 60 + ss), float(em * 60 + es)
    if ss >= 60 or es >= 60 or not 0 <= start < end <= 30:
        raise ValueError(f"GT interval outside 30-second clip: {value!r}")
    return {"start_s": start, "end_s": end}


def load_gt(path: Path = GT_PATH, video_root: Path = VIDEO_ROOT) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if len(raw_rows) != EXPECTED_CLIPS or len({row["clip"] for row in raw_rows}) != EXPECTED_CLIPS:
        raise ValueError(f"expected {EXPECTED_CLIPS} unique GT clips, got {len(raw_rows)} rows")
    rows = []
    for raw in raw_rows:
        tactic = GT_ALIASES.get(raw["tactic"], raw["tactic"])
        if tactic not in TACTICS:
            raise ValueError(f"unknown GT tactic: {raw['tactic']!r}")
        video = video_root / f"{raw['clip']}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"missing video for {raw['clip']}: {video}")
        rows.append({
            "clip": raw["clip"],
            "tactic": tactic,
            "interval": parse_interval(raw["interval"]),
            "video_path": video,
        })
    return rows


CRITERIA_FIELDS = (
    "id（英文标识）",
    "视频侧-参与者",
    "视频侧-空间约束",
    "视频侧-时间窗",
    "视频侧-事件序列",
    "反例① 描述",
    "反例① 为何不算",
    "反例② 描述",
    "反例② 为何不算",
    "易混淆① 对象",
    "易混淆① 区分点",
    "易混淆② 对象",
    "易混淆② 区分点",
    "易混淆③ 对象",
    "易混淆③ 区分点",
)


def load_criteria(path: Path = GLOSSARY_PATH) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(CRITERIA_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("glossary is missing required video-side fields")
        rows = {
            row["id（英文标识）"].strip(): {field: (row.get(field) or "").strip() for field in CRITERIA_FIELDS}
            for row in reader
            if row["id（英文标识）"].strip() in TACTICS
        }
    if set(rows) != set(TACTICS):
        raise ValueError(f"glossary missing tactics: {sorted(set(TACTICS) - set(rows))}")
    return {tactic: rows[tactic] for tactic in TACTICS}


def _event_index(events: list[dict[str, Any]], actions: set[str], *, after: float | None = None) -> int | None:
    return next((i for i, event in enumerate(events) if event["action"] in actions and (after is None or event["time"] >= after)), None)


def _event_after(events: list[dict[str, Any]], index: int | None, actions: set[str]) -> int | None:
    if index is None:
        return None
    return next((i for i in range(index + 1, len(events)) if events[i]["action"] in actions), None)


def _fact(a: dict[str, Any], value: str) -> bool:
    # not_observable is a real answer, but unseen is not satisfied.
    return a["key_spatial_facts"].get(value) == "yes"


def _crosses_third(a: dict[str, Any]) -> bool:
    return " -> " in a["ball_progression"]



def _interval(start: float | None, end: float | None) -> dict[str, float] | None:
    if start is None or end is None or start >= end:
        return None
    return {"start_s": round(max(0.0, start), 3), "end_s": round(min(30.0, end), 3)}


def _rule_result(
    tactic: str,
    hard: dict[str, bool],
    soft: dict[str, tuple[float, bool]],
    indices: Iterable[int] = (),
    interval: dict[str, float] | None = None,
) -> dict[str, Any]:
    # The former hard gates are now heavily weighted score terms rather than a binary AND.
    # r2_pro measurement: gating zeroed 16/26 clips whose GT was otherwise top-ranked, and
    # dropping the gate lifted top-1 from 35% to 58%. `hard_pass` is kept for diagnostics only.
    earned = sum(GATE_WEIGHT for passed in hard.values() if passed)
    earned += sum(weight for weight, matched in soft.values() if matched)
    total = GATE_WEIGHT * len(hard) + sum(weight for weight, _ in soft.values())
    return {
        "tactic": tactic,
        "hard_pass": all(hard.values()),
        "score": round(earned / total, 4) if total else 0.0,
        "detail": {
            "hard": hard,
            "gate_weight": GATE_WEIGHT,
            "soft": {name: {"weight": weight, "matched": matched} for name, (weight, matched) in soft.items()},
        },
        "supporting_event_indices": sorted(set(indices)),
        "interval": interval,
    }


def _counter_attack(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events, change = b["events"], a["possession_change"]
    possession = None if change is None else next((i for i, e in enumerate(events) if e["action"] in POSSESSION_ACTIONS and abs(e["time"] - change) <= POSSESSION_JOIN_TOLERANCE_S), None)
    first_ball = None if change is None else next((i for i, e in enumerate(events) if e["time"] >= change and e["action"] not in POSSESSION_ACTIONS), None)
    no_restart = change is not None and not any(e["action"] in RESTART_ACTIONS and change - 2 <= e["time"] <= change for e in events)
    immediate = first_ball is not None and events[first_ball]["action"] in FORWARD_ACTIONS and events[first_ball]["time"] - change <= IMMEDIATE_FORWARD_WINDOW_S
    terminal = max((i for i, e in enumerate(events) if change is not None and change <= e["time"] <= change + 20), default=None)
    end = events[terminal]["time"] if terminal is not None else None
    return _rule_result(
        "counter-attack",
        {"possession_change": possession is not None, "immediate_forward_no_restart": immediate and no_restart},
        {
            "rapid_vertical_progression": (0.30, _crosses_third(a) and end is not None and end - change <= 20),
            "multiple_attackers_run_forward": (0.25, _fact(a, "multiple_attackers_run_forward")),
            "defence_recovering": (0.20, b["defensive_change"] in {"retreat_toward_own_goal", "recover_behind_ball"}),
            "final_third_before_set": (0.25, a["ball_progression"] in {"middle_third -> final_third", "defensive_third -> final_third"} and _fact(a, "defenders_not_set")),
        },
        [i for i in (possession, first_ball, terminal) if i is not None],
        _interval(change, end),
    )


def _long_ball(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    pass_i = _event_index(events, {"forward_aerial_pass"})
    first_i = _event_after(events, pass_i, {"reception", "duel", "header"})
    terminal = None if pass_i is None else max((i for i in range(pass_i + 1, len(events)) if events[i]["action"] in {"reception", "duel", "header", "shot"}), default=None)
    duration = None if pass_i is None or terminal is None else events[terminal]["time"] - events[pass_i]["time"]
    return _rule_result(
        "long-ball",
        {"forward_aerial_pass": pass_i is not None, "clear_forward_target": _fact(a, "forward_target_or_landing_space_beyond_midfield")},
        {
            "crosses_third": (0.30, _crosses_third(a)),
            "first_ball_contested": (0.25, first_i is not None),
            "second_ball_structure": (0.25, _fact(a, "teammates_positioned_for_second_ball")),
            "duration_6_12": (0.20, duration is not None and 6 <= duration <= 12),
        },
        [i for i in (pass_i, first_i, terminal) if i is not None],
        _interval(events[pass_i]["time"] if pass_i is not None else None, events[terminal]["time"] if terminal is not None else None),
    )


def _pass_then_return(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Find (first pass by X, ball coming back to X) using actor identity, not a collapsed token.

    B never emitted `one_touch_return_pass` (0/26 in r2_pro) even on GT one-two clips, because
    "returned it to whoever passed it" is a relation across two events that a single action token
    cannot carry. With actors the relation is derivable, so the token is only a fallback here.
    """
    for i, first in enumerate(events):
        if first["action"] not in PASSES or first["actor"] == "unknown":
            continue
        for j in range(i + 1, len(events)):
            back = events[j]
            if back["actor"] == first["actor"] or back["actor"] == "unknown":
                continue
            if back["action"] not in PASSES | {"one_touch_return_pass"}:
                continue
            returned = next(
                (k for k in range(j, len(events))
                 if events[k]["actor"] == first["actor"] and events[k]["action"] in RECEIVE_ACTIONS),
                None,
            )
            if returned is not None or back["action"] == "one_touch_return_pass":
                return i, j
    return None, None


def _weak_return(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Fallback tier: X passes, a different player plays it straight back, receiver unconfirmed.

    On the Codex probe B consistently described the wall pass as X->Y and then attributed the
    continuation to Y, never recording the ball arriving back at X (SNGS-137/184/198 all show it).
    Re-identifying X across sparse frames is the failing step, not the passing itself, so this
    tier accepts the evidence without the closing leg — and `confirmed_return_to_passer` below
    keeps it scoring strictly under a fully observed X->Y->X.
    """
    for i, first in enumerate(events):
        if first["action"] not in PASSES:
            continue
        for j in range(i + 1, len(events)):
            back = events[j]
            if back["action"] not in PASSES | {"one_touch_return_pass"}:
                continue
            if back["time"] - first["time"] > WALL_PASS_RETURN_S:
                break
            # An explicit one_touch_return_pass already asserts "played straight back", so it
            # counts whatever the actor bookkeeping says.
            if (back["action"] == "one_touch_return_pass"
                    or back["actor"] != first["actor"]
                    or "unknown" in (back["actor"], first["actor"])):
                return i, j
    return None, None


def _one_two(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    pass_i, return_i = _pass_then_return(events)
    confirmed = return_i is not None
    if not confirmed:
        pass_i, return_i = _weak_return(events)
    receive_i = _event_after(events, return_i, RECEIVE_ACTIONS)
    end_i = receive_i if receive_i is not None else return_i
    duration = None if pass_i is None or end_i is None else events[end_i]["time"] - events[pass_i]["time"]
    quick = pass_i is not None and return_i is not None and events[return_i]["time"] - events[pass_i]["time"] <= 2.5
    return _rule_result(
        "one-two",
        {"pass_then_return": pass_i is not None and return_i is not None,
         "first_passer_runs_beyond": _fact(a, "first_passer_runs_beyond_nearby_defender")},
        {
            # The defining leg. Weak evidence alone cannot reach the score a confirmed X->Y->X does,
            # so an ordinary two-pass move never scores like a real wall pass.
            "confirmed_return_to_passer": (0.30, confirmed),
            "quick_return": (0.25, quick),
            "reception_after_return": (0.20, receive_i is not None),
            "forward_progression": (0.15, _crosses_third(a)),
            "duration_2_5": (0.10, duration is not None and 2 <= duration <= 5),
        },
        [i for i in (pass_i, return_i, receive_i) if i is not None],
        _interval(events[pass_i]["time"] if pass_i is not None else None,
                  events[end_i]["time"] if end_i is not None else None),
    )


def _tactical_foul(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events, change = b["events"], a["possession_change"]
    foul_i = None if change is None else next((i for i, e in enumerate(events) if e["time"] >= change and e["time"] - change <= 6 and e["action"] in {"foul", "pull_or_block"}), None)
    move_i = None if change is None or foul_i is None else next((i for i, e in enumerate(events[:foul_i]) if e["time"] >= change and e["action"] in FORWARD_ACTIONS), None)
    whistle_i = _event_after(events, foul_i, {"whistle"})
    end_i = whistle_i if whistle_i is not None else foul_i
    duration = None if change is None or end_i is None else events[end_i]["time"] - change
    return _rule_result(
        "tactical-foul",
        {"transition_then_foul": change is not None and foul_i is not None, "outside_penalty_area": _fact(a, "contact_outside_defending_penalty_area")},
        {
            "opponent_accelerates": (0.25, move_i is not None),
            "pull_or_block": (0.30, foul_i is not None and events[foul_i]["action"] == "pull_or_block"),
            "defence_not_set": (0.20, _fact(a, "defenders_not_set") or b["defensive_change"] in {"retreat_toward_own_goal", "recover_behind_ball"}),
            "whistle": (0.15, whistle_i is not None),
            "duration_2_6": (0.10, duration is not None and 2 <= duration <= 6),
        },
        [i for i in (move_i, foul_i, whistle_i) if i is not None],
        _interval(change, events[end_i]["time"] if end_i is not None else None),
    )


def _cutback(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    pass_i = _event_index(events, {"backward_ground_pass"})
    dribble_i = None if pass_i is None else next((i for i in range(pass_i - 1, -1, -1) if events[i]["action"] == "forward_dribble" and events[pass_i]["time"] - events[i]["time"] <= 3), None)
    terminal = _event_after(events, pass_i, {"reception", "shot"})
    start = events[dribble_i]["time"] if dribble_i is not None else max(0.0, events[pass_i]["time"] - 1) if pass_i is not None else None
    end = events[terminal]["time"] if terminal is not None else None
    byline = _fact(a, "ball_carrier_reaches_left_byline") or _fact(a, "ball_carrier_reaches_right_byline")
    return _rule_result(
        "cutback",
        {"reaches_byline": byline, "backward_ground_pass": pass_i is not None},
        {
            "central_penalty_target": (0.40, _fact(a, "backward_pass_target_in_central_penalty_area")),
            "reception_or_shot": (0.25, terminal is not None),
            "defence_collapses": (0.15, b["defensive_change"] == "collapse_toward_ball"),
            "duration_3_8": (0.20, dribble_i is not None and end is not None and 3 <= end - start <= 8),
        },
        [i for i in (dribble_i, pass_i, terminal) if i is not None],
        _interval(start, end),
    )


def _run_in_behind(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    pass_i = _event_index(events, FORWARD_PASSES)
    receive_i = _event_after(events, pass_i, {"reception"})
    start = max(0.0, events[pass_i]["time"] - 2) if pass_i is not None else None
    end = events[receive_i]["time"] if receive_i is not None else None
    duration = None if start is None or end is None else end - start
    return _rule_result(
        "run-in-behind",
        {"runner_moves_behind": _fact(a, "runner_moves_from_last_line_front_to_behind"), "forward_pass_and_reception": pass_i is not None and receive_i is not None},
        {
            "crosses_last_line": (0.30, _fact(a, "pass_path_crosses_defensive_line")),
            "receiver_behind_line": (0.25, _fact(a, "receiver_controls_between_or_behind_lines")),
            "forward_progression": (0.20, _crosses_third(a)),
            "duration_3_6": (0.15, duration is not None and 3 <= duration <= 6),
            "defence_recovering": (0.10, b["defensive_change"] in {"retreat_toward_own_goal", "recover_behind_ball"}),
        },
        [i for i in (pass_i, receive_i) if i is not None],
        _interval(start, end),
    )


def _line_breaking_pass(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    pass_i = _event_index(events, FORWARD_PASSES)
    receive_i = _event_after(events, pass_i, {"reception"})
    start = max(0.0, events[pass_i]["time"] - 1) if pass_i is not None else None
    end = events[receive_i]["time"] if receive_i is not None else None
    duration = None if start is None or end is None else end - start
    return _rule_result(
        "line-breaking-pass",
        {"pass_crosses_line": _fact(a, "pass_path_crosses_defensive_line"), "receiver_behind_or_between": _fact(a, "receiver_controls_between_or_behind_lines")},
        {
            "forward_pass": (0.35, pass_i is not None),
            "reception": (0.25, receive_i is not None),
            "forward_progression": (0.25, _crosses_third(a)),
            "duration_2_6": (0.15, duration is not None and 2 <= duration <= 6),
        },
        [i for i in (pass_i, receive_i) if i is not None],
        _interval(start, end),
    )


def _back_line(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    events = b["events"]
    passes = [i for i, event in enumerate(events) if event["action"] in BACK_LINE_PASSES]
    duration = events[passes[-1]]["time"] - events[passes[0]]["time"] if len(passes) >= 2 else None
    no_forward = not any(event["action"] in FORWARD_ACTIONS for event in events)
    return _rule_result(
        "back-line-possession-management",
        {"three_lateral_or_backward_passes": len(passes) >= 3, "ball_remains_own_half": _fact(a, "ball_remains_in_own_half")},
        {
            "back_line_circulation": (0.40, _fact(a, "ball_circulates_across_back_line")),
            "within_defensive_third": (0.25, a["ball_progression"] == "within_defensive_third"),
            "no_forward_action": (0.20, no_forward),
            "duration_10_25": (0.15, duration is not None and 10 <= duration <= 25),
        },
        passes,
        None,
    )


RULES = (
    _counter_attack,
    _long_ball,
    _one_two,
    _tactical_foul,
    _cutback,
    _run_in_behind,
    _line_breaking_pass,
    _back_line,
)


def classify(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    validate_a(a)
    validate_b(b)
    ranked = sorted((rule(a, b) for rule in RULES), key=lambda item: (-item["score"], item["tactic"]))
    qualified = [item for item in ranked if item["score"] >= MIN_SCORE]
    result: dict[str, Any] = {"label": None, "score": None, "interval": None, "reason": "no_match"}
    if qualified:
        top_score = qualified[0]["score"]
        tied = [item for item in qualified if item["score"] == top_score]
        if len(tied) > 1:
            result.update(reason="tie", score=top_score, tied=[item["tactic"] for item in tied])
        else:
            winner = tied[0]
            result.update(
                label=winner["tactic"], score=winner["score"], interval=winner["interval"],
                reason="matched", supporting_event_indices=winner["supporting_event_indices"], detail=winner["detail"],
            )
    result["ranked"] = ranked
    return result


def c_candidates(rule: dict[str, Any]) -> list[str]:
    return [item["tactic"] for item in rule["ranked"][:C_CANDIDATES]]


def _candidate_text(rule: dict[str, Any], criteria: dict[str, dict[str, str]]) -> str:
    """Only the operational video-side criteria.

    Feeding C the glossary's prose (triggers, counter-examples, confusables) made it read the
    entry as a conjunctive checklist and reject candidates for missing any one trait — on r3 it
    changed 7/19 clips, 5 correct->wrong and 0 wrong->correct, reinstating by hand the hard gates
    P0 had removed. With only these four fields the Codex probe changed 0 correct answers to wrong
    across 32 arbitrations and recovered 3.
    """
    sections = []
    for candidate in rule["ranked"][:C_CANDIDATES]:
        row = criteria[candidate["tactic"]]
        lines = [f"候选：{candidate['tactic']}（rule 打分 {candidate['score']}）"]
        for field in ("视频侧-参与者", "视频侧-空间约束", "视频侧-时间窗", "视频侧-事件序列"):
            if row.get(field):
                lines.append(f"  {field}: {row[field]}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_c_prompt(a: dict[str, Any], b: dict[str, Any], rule: dict[str, Any], criteria: dict[str, dict[str, str]]) -> str:
    names = "、".join(c_candidates(rule))
    return f"""你是最终裁决模块。A和B是只描述画面、不给战术名的模块，Rule是一套写死的打分规则。
对照完整30秒视频，从候选里选出最符合画面的那个战术。

重要的判断尺度：
- 下面每个候选给出的是「典型特征」，不是「必要条件清单」。不要因为某一条典型特征没出现就否决整个候选。
- 判断依据是哪个候选与画面的整体吻合度最高，而不是哪个候选挑不出毛病。
- Rule 的打分是有信息量的先验。只有当你有明确的画面证据表明它选错了，才改选别的。
- 只有当所有候选都与画面明显矛盾时才填 null。宁可选一个不完美但最接近的，也不要轻易全盘否定。

label只能是以下之一，或者null：{names}
rationale：一到两句话，引用画面里的具体依据。
objections：A或B中与画面明确不符的字段，最多两条，格式“字段路径: 具体错误”，路径以A.或B.开头。
格式示例：B.events[2].action: 该时刻球员用脚停球，不是头球
没有明确错误就留空，不要凑数。

A={json.dumps(a, ensure_ascii=False, sort_keys=True)}
B={json.dumps(b, ensure_ascii=False, sort_keys=True)}

候选及其典型特征：
{_candidate_text(rule, criteria)}

只输出严格JSON：label、rationale、objections。"""


Generator = Callable[..., tuple[dict[str, Any], list[dict[str, Any]], int]]


def run_clip(
    video_path: Path,
    criteria: dict[str, dict[str, str]],
    generator: Generator = generate_json,
    *,
    provider: str = "doubao",
) -> dict[str, Any]:
    def call(upload: Path, role: str, prompt: str, schema: dict[str, Any], validator: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        # Gemini's Interactions API takes neither fps nor temperature; Ark takes both.
        tuning = {"fps": MODEL_FPS[role], "temperature": TEMPERATURE} if provider == "doubao" else {}
        payload, usage, attempts = generator(provider, prompt, schema, video_path=upload, **tuning)
        validator(payload)
        return {
            "output": payload,
            "usage": usage,
            "attempts": attempts,
            "elapsed_s": round(time.monotonic() - started, 3),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "fps": tuning.get("fps"),
        }

    with temporary_compact_clip(video_path, **VIDEO_PREPROCESS) as upload:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(call, upload, "A", A_PROMPT, A_SCHEMA, validate_a)
            future_b = executor.submit(call, upload, "B", B_PROMPT, B_SCHEMA, validate_b)
            agent_a, agent_b = future_a.result(), future_b.result()
        rule = classify(agent_a["output"], agent_b["output"])
        c_prompt = build_c_prompt(agent_a["output"], agent_b["output"], rule, criteria)
        candidates = c_candidates(rule)
        # A C failure must never discard a valid A/B/Rule prediction; `final` falls back to the Rule.
        try:
            agent_c = call(upload, "C", c_prompt, C_SCHEMA, lambda p: validate_c(p, candidates))
        except Exception as exc:
            agent_c = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
    verdict = agent_c.get("output") if isinstance(agent_c, dict) else None
    final = {
        "label": verdict["label"] if verdict else rule.get("label"),
        "source": "arbiter" if verdict else "rule",
        "candidates": candidates,
        "interval": next((item["interval"] for item in rule["ranked"]
                          if verdict and item["tactic"] == verdict["label"]), rule.get("interval")),
    }
    return {"agents": {"A": agent_a, "B": agent_b, "C": agent_c}, "rule": rule, "final": final}


def manifest_signature(
    criteria: dict[str, dict[str, str]],
    gt_path: Path = GT_PATH,
    glossary_path: Path = GLOSSARY_PATH,
    *,
    provider: str = "doubao",
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name(provider),
        "provider": provider,
        "fps": MODEL_FPS,
        "temperature": TEMPERATURE,
        "min_score": MIN_SCORE,
        "gate_weight": GATE_WEIGHT,
        "video_preprocess": VIDEO_PREPROCESS,
        "possession_join_tolerance_s": POSSESSION_JOIN_TOLERANCE_S,
        "immediate_forward_window_s": IMMEDIATE_FORWARD_WINDOW_S,
        "rule_version": RULE_VERSION,
        "rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": {
            "A": hashlib.sha256(A_PROMPT.encode()).hexdigest(),
            "B": hashlib.sha256(B_PROMPT.encode()).hexdigest(),
        },
        "gt_sha256": sha256_file(gt_path),
        "glossary_sha256": sha256_file(glossary_path),
        "criteria_snapshot_sha256": _json_hash(criteria),
    }
    return {**value, "signature": _json_hash(value)}


def ensure_manifest(
    run_dir: Path,
    run_id: str,
    criteria: dict[str, dict[str, str]],
    *,
    provider: str = "doubao",
) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    current = manifest_signature(criteria, provider=provider)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != current["signature"]:
            raise RuntimeError(f"{run_id} manifest does not match current data/prompt/rules; use a new run-id")
        return existing
    manifest = {
        **current,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_clips": EXPECTED_CLIPS,
        "criteria_snapshot": criteria,
    }
    write_json(path, manifest)
    return manifest


def run_pilot(
    run_id: str,
    *,
    provider: str = "doubao",
    clip_ids: Iterable[str] | None = None,
    retry_failed: bool = False,
    generator: Generator = generate_json,
    output_root: Path = OUTPUT_ROOT,
    gt_path: Path = GT_PATH,
    glossary_path: Path = GLOSSARY_PATH,
    video_root: Path = VIDEO_ROOT,
) -> dict[str, int]:
    rows = load_gt(gt_path, video_root)
    criteria = load_criteria(glossary_path)
    run_dir = output_root / run_id
    ensure_manifest(run_dir, run_id, criteria, provider=provider)
    selected = set(clip_ids or (row["clip"] for row in rows))
    known = {row["clip"] for row in rows}
    if unknown := selected - known:
        raise ValueError(f"unknown clip ids: {sorted(unknown)}")
    counts = Counter(selected=len(selected), success=0, failed=0, skipped=0)
    for gt in rows:
        if gt["clip"] not in selected:
            continue
        target = run_dir / "clips" / f"{gt['clip']}.json"
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("status") == "success" or not retry_failed:
                counts["skipped"] += 1
                continue
        started = time.monotonic()
        try:
            value = run_clip(gt["video_path"], criteria, generator, provider=provider)
            result = {"status": "success", **value}
        except Exception as exc:
            result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        result.update({
            "schema_version": SCHEMA_VERSION,
            "clip": gt["clip"],
            "video_path": gt["video_path"].relative_to(ROOT).as_posix() if gt["video_path"].is_relative_to(ROOT) else gt["video_path"].as_posix(),
            "video_sha256": sha256_file(gt["video_path"]),
            "gt": {"tactic": gt["tactic"], "interval": gt["interval"]},
            "elapsed_s": round(time.monotonic() - started, 3),
        })
        write_json(target, result)
        counts[result["status"]] += 1
    return dict(counts)


def temporal_iou(predicted: dict[str, Any] | None, expected: dict[str, Any]) -> float:
    if not predicted:
        return 0.0
    intersection = max(0.0, min(predicted["end_s"], expected["end_s"]) - max(predicted["start_s"], expected["start_s"]))
    union = max(predicted["end_s"], expected["end_s"]) - min(predicted["start_s"], expected["start_s"])
    return intersection / union if union else 0.0


def classification_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_clip = {result["clip"]: result for result in results}
    gt_rows = load_gt()
    pairs = []
    confusion_columns = [*TACTICS, "tie", "no_match", "arbiter_reject", "failed", "missing"]
    confusion = {tactic: {column: 0 for column in confusion_columns} for tactic in TACTICS}
    temporal = []
    success = 0
    for gt in gt_rows:
        result = by_clip.get(gt["clip"])
        if result is None:
            predicted, bucket, interval = None, "missing", None
        elif result["status"] != "success":
            predicted, bucket, interval = None, "failed", None
        else:
            success += 1
            # Score the system's actual output: the arbiter's verdict when there is one, else the
            # Rule. `rule` is absent from C-only results, which share this function.
            rule = result.get("rule", {})
            chosen = result.get("final") or rule
            predicted, interval = chosen.get("label"), chosen.get("interval")
            if predicted:
                bucket = predicted
            elif rule.get("reason") in {"tie", "no_match"}:
                bucket = rule["reason"]
            else:
                # The Rule matched something but the arbiter rejected every candidate.
                bucket = "arbiter_reject"
        pairs.append((gt["tactic"], predicted))
        confusion[gt["tactic"]][bucket] += 1
        if gt["tactic"] != "back-line-possession-management":
            temporal.append(temporal_iou(interval if predicted == gt["tactic"] else None, gt["interval"]))
    per_class = {}
    for tactic in TACTICS:
        tp = sum(truth == tactic and prediction == tactic for truth, prediction in pairs)
        fp = sum(truth != tactic and prediction == tactic for truth, prediction in pairs)
        fn = sum(truth == tactic and prediction != tactic for truth, prediction in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[tactic] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    correct = sum(truth == prediction for truth, prediction in pairs)
    accuracy = correct / len(pairs) if pairs else 0.0
    majority = max(Counter(truth for truth, _ in pairs).values()) / len(pairs) if pairs else 0.0
    return {
        "n_expected": len(pairs),
        "n_success": success,
        "correct_n": correct,
        "accuracy": accuracy,
        "macro_f1": statistics.fmean(item["f1"] for item in per_class.values()),
        "majority_baseline": majority,
        "accuracy_lift_over_majority": accuracy - majority,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "temporal": {
            "eligible_n": len(temporal),
            "mean_iou": statistics.fmean(temporal) if temporal else 0.0,
            "tiou_at_0_5": sum(value >= 0.5 for value in temporal) / len(temporal) if temporal else 0.0,
            "missing_interval_counts_as_zero": True,
        },
    }



def _render_review(results: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    lines = [
        "# GSR-2025 Multi-Agent Pilot Review",
        "",
        f"macro-F1={metrics['macro_f1']:.3f} · accuracy={metrics['correct_n']}/{metrics['n_expected']} ({metrics['accuracy']:.1%}) · mean tIoU={metrics['temporal']['mean_iou']:.3f}",
        "",
    ]
    for result in sorted(results, key=lambda item: item["clip"]):
        gt = result["gt"]
        lines.extend([f"## {result['clip']}", "", f"GT: {gt['tactic']} [{gt['interval']['start_s']:.1f}–{gt['interval']['end_s']:.1f}]", f"Video: `{result['video_path']}`", ""])
        if result["status"] != "success":
            lines.extend([f"FAILED: {result['error_type']}: {result['error']}", ""])
            continue
        a = result["agents"]["A"]["output"]
        b = result["agents"]["B"]["output"]
        c = result["agents"]["C"].get("output")
        rule = result["rule"]
        lines.extend(["A:", f"- possession change: {a['possession_change']}", f"- {a['ball_progression']}"])
        lines.extend(
            f"- {name}: {judgement}"
            for name, judgement in a["key_spatial_facts"].items()
            if judgement != "no"
        )
        lines.extend(["", "B:"])
        lines.extend(f"- {event['time']:.1f} {event['actor']} {event['action']}" for event in b["events"])
        lines.extend([f"- defence: {b['defensive_change']}", ""])
        interval = rule.get("interval")
        interval_text = f" [{interval['start_s']:.1f}–{interval['end_s']:.1f}]" if interval else ""
        label = rule.get("label") or rule.get("reason")
        lines.extend([f"Rule: {label}{interval_text}, score={rule.get('score')}", "", "C:"])
        if c is None:
            lines.append(f"- Unavailable: {result['agents']['C'].get('error_type')}")
        else:
            lines.append(f"- verdict: {c['label'] or 'none of the candidates'} — {c['rationale']}")
            lines.extend([f"- {item}" for item in c["objections"]] or ["- No objection"])
        final = result.get("final")
        if final:
            lines.extend(["", f"FINAL: {final['label'] or 'no_match'} (via {final['source']}, candidates={'/'.join(final['candidates'])})"])
        lines.append("")
    return "\n".join(lines)


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"missing manifest in {run_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "clips").glob("*.json"))]


def report(run_id: str, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    run_dir = output_root / run_id
    results = load_results(run_dir)
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8",
    )
    metrics = classification_metrics(results)
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "review.md").write_text(_render_review(results, metrics), encoding="utf-8")
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--provider", required=True, choices=PROVIDERS)
    run.add_argument("--clip-id", action="append", dest="clip_ids")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--allow-external-upload", action="store_true")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run-id", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run_pilot(
            args.run_id, provider=args.provider,
            clip_ids=args.clip_ids, retry_failed=args.retry_failed,
        )
    else:
        value = report(args.run_id)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
