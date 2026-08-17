"""FIFA Game Library 人工 grounding 标注的共享读写逻辑。

模板由 scripts/build_fifa_annotation_tasks.py 生成，回收后由
scripts/check_annotations.py 校验、scripts/xlsx_to_jsonl.py 转换。
三者共用这里的词表解析与行解析，避免各写一份走样。
"""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import openpyxl

MAIN_SHEET = "主表"
DETAIL_SHEET = "事件明细"

# 与战术标签冗余 / 孤立片段判断不了，不发给标注者，见标注规范。
DROPPED_GROUPS = ("pattern_of_play", "context_tag")

# reference_football.csv 没有无球跑动类事件，本项目先行扩展。
EXTRA_EVENTS = (
    ("run_in_behind", "跑向身后（本项目扩展）"),
    ("off_ball_run", "无球跑动（本项目扩展）"),
    ("receive", "接球（本项目扩展）"),
)

# 阶段/层级推导不出、但本质是集体形态的战术，同样走 who_group。
COLLECTIVE_SHAPE_TACTICS = frozenset({"buildup-structure"})

# 「视频侧-*」四栏在 JSONL 里的稳定 id。
EVIDENCE_AXES = (
    ("video_participants", "ev_participants"),
    ("video_space", "ev_space"),
    ("video_time", "ev_time"),
    ("video_sequence", "ev_sequence"),
)

MIN_STEPS, MAX_STEPS = 3, 6

_CODE_IN_PARENS = re.compile(r"\(([a-z0-9_]+)\)\s*$")
_STEP_SEPARATOR = re.compile(r"\s*(?:->|→|—>)\s*")
_NULLS = {"", "null", "none", "n/a", "na", "无", "-"}


# --------------------------------------------------------------------- 小工具

def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def is_null(value: Any) -> bool:
    return clean(value).lower() in _NULLS


def extract_code(value: Any) -> str:
    """下拉显示成「传球(pass)」，自由文本里写成「pass」，两种都要认。"""
    text = clean(value)
    m = _CODE_IN_PARENS.search(text)
    return m.group(1) if m else text


def split_steps(sequence: Any) -> list[str]:
    text = clean(sequence)
    if not text:
        return []
    return [s.strip() for s in _STEP_SEPARATOR.split(text) if s.strip()]


def step_event_code(step: str) -> str:
    """一步写作「角色@队伍 事件code」，事件 code 是最后一个词。"""
    parts = clean(step).split()
    return extract_code(parts[-1]) if parts else ""


def step_actor(step: str) -> str:
    parts = clean(step).split()
    return " ".join(parts[:-1]) if len(parts) > 1 else ""


def role_of(actor: Any) -> str:
    """`10号左边后卫@白队` -> `10号左边后卫`，用于和事件序列里的写法比对。"""
    return clean(actor).split("@", 1)[0].strip()


# --------------------------------------------------------------------- 词表

def load_reference(reference_csv: Path) -> tuple[list[tuple[str, str]], "OrderedDict[str, dict]"]:
    """返回 (事件 [(code, 中文名)], 限定词组 OrderedDict)。"""
    with Path(reference_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if clean(r.get("event_id"))]

    events: "OrderedDict[str, str]" = OrderedDict()
    for r in rows:
        if r["event_id"].startswith("_TAG"):
            continue
        if r["level_hint"] == "core" or r["family"] == "possession":
            events[r["code"]] = r["display_name_cn"]
    for code, label in EXTRA_EVENTS:
        events[code] = label

    groups: "OrderedDict[str, dict]" = OrderedDict()
    for g in rows:
        if g["event_id"] != "_TAG_GROUP" or g["code"] in DROPPED_GROUPS:
            continue
        applies = [a.strip() for a in
                   re.sub(r"^applies_to:", "", g["description"]).split(",") if a.strip()]
        values = [f'{v["display_name_cn"]}({v["code"]})'
                  for v in rows if v["event_id"] == "_TAG_VALUE" and v["family"] == g["code"]]
        groups[g["code"]] = {
            "label": g["display_name_cn"],
            "exclusive": g["level_hint"] == "exclusive",
            "applies_to": applies,
            "values": values,
        }
    return list(events.items()), groups


def load_tactics(grid_csv: Path) -> dict[str, dict]:
    """词条 id -> 优先级/层级/阶段/提示文本。"""
    out: dict[str, dict] = {}
    with Path(grid_csv).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            tid = clean(r.get("id（英文标识）"))
            if not tid or clean(r.get("优先级")) not in ("P0", "P1", "P2"):
                continue
            features = [ln.strip() for ln in clean(r.get("可观察特征")).splitlines() if ln.strip()]
            out[tid] = {
                "priority": clean(r.get("优先级")),
                "name_cn": clean(r.get("中文名")),
                "layer": clean(r.get("层级 layer")),
                "phase": clean(r.get("阶段 phase")),
                "hint": "；".join([clean(r.get("一句话定义"))] + features[:3]).strip("；"),
                "actors": clean(r.get("视频侧-参与者")),
                "sequence": clean(r.get("视频侧-事件序列")),
            }
    return out


def needs_group(tactic: dict, tactic_id: str) -> bool:
    """防守组织 / 集体形态类战术，who_group 必填，四个角色槽多为 null。"""
    return (tactic["phase"] in ("defensive_org", "defensive_transition")
            or tactic["layer"] == "structure"
            or tactic_id in COLLECTIVE_SHAPE_TACTICS)


def derive_pattern_of_play(tactic_id: str, tactic: dict | None) -> str:
    """pattern_of_play 与战术标签冗余，不让人填，这里从阶段推。"""
    if tactic_id == "counter-attack":
        return "fast_break"
    phase = tactic["phase"] if tactic else ""
    if phase == "set_piece":
        return "set_piece"
    if phase == "attacking_transition":
        return "fast_break"
    return "open_play"


# --------------------------------------------------------------------- 读回收的表

def _header_index(ws) -> dict[str, int]:
    return {clean(c.value): c.column for c in ws[1] if clean(c.value)}


def read_workbook(path: Path) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    """读回收的模板，返回 (卡片列表, 按主键分组的事件步)。

    只保留标注者动过的行：主表看 annotator/时间/证据是否有内容，明细表看 actor/event_code。
    """
    wb = openpyxl.load_workbook(path, data_only=True)

    ws = wb[MAIN_SHEET]
    h = _header_index(ws)
    cards = []
    for row in range(2, ws.max_row + 1):
        rec = {name: ws.cell(row, col).value for name, col in h.items()}
        rec["_row"] = row
        touched = any(not is_null(rec.get(k)) for k in
                      ("annotator", "start_sec", "end_sec", "ev_participants",
                       "ev_space", "ev_time", "ev_sequence", "dispute"))
        if touched:
            cards.append(rec)

    steps: dict[tuple[str, str], list[dict]] = {}
    if DETAIL_SHEET in wb.sheetnames:
        wd = wb[DETAIL_SHEET]
        hd = _header_index(wd)
        for row in range(2, wd.max_row + 1):
            rec = {name: wd.cell(row, col).value for name, col in hd.items()}
            rec["_row"] = row
            if is_null(rec.get("actor")) and is_null(rec.get("event_code")):
                continue
            key = (clean(rec.get("video_path")), clean(rec.get("tactic_label_given")))
            steps.setdefault(key, []).append(rec)

    for group in steps.values():
        group.sort(key=lambda r: (r.get("step_no") or 0, r["_row"]))
    return cards, steps


def card_key(card: dict) -> tuple[str, str]:
    return clean(card.get("video_path")), clean(card.get("tactic_label_given"))


def qualifier_columns(groups: "OrderedDict[str, dict]") -> dict[str, str]:
    """事件明细表的列名（中文）-> 组 code。"""
    return {g["label"]: code for code, g in groups.items()}


def iter_filled(values: Iterable[Any]) -> list[str]:
    return [clean(v) for v in values if not is_null(v)]
