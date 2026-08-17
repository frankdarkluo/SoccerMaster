#!/usr/bin/env python3
"""把回收的标注表转成 JSONL，字段结构与喂给模型的 prompt 同构。

同构的部分（可以直接和模型输出逐字段比对）：
    tactic / temporal_grounding / spatial_grounding / evidence / tactical_report
人工独有的扩展字段，一律带 non_scored 标记，不参与打分：
    who / result / event_sequence / commentary_zh

pattern_of_play 与战术标签冗余，没让标注者填，这里从阶段自动推导并标 derived。

用法：
    python3 scripts/xlsx_to_jsonl.py 回收的表.xlsx --out 标注.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fifa_grounding import (  # noqa: E402
    EVIDENCE_AXES,
    card_key,
    clean,
    derive_pattern_of_play,
    event_zh_to_code,
    is_null,
    join_zone,
    load_reference,
    load_tactics,
    read_workbook,
    split_steps,
    step_actor,
    step_event_code,
)

DEFAULT_DATA = ROOT / "data"
GRID_REL = Path("足球战术数据库_词条表_Grid.csv")
REFERENCE_REL = Path("reference_football.csv")


def text_or_none(value) -> str | None:
    return None if is_null(value) else clean(value)


def number_or_none(value) -> float | None:
    if is_null(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_steps(rows: list[dict], sequence_steps: list[str],
                groups: dict, zh_to_code: dict) -> list[dict]:
    """明细表有行就用明细表，没有就退回主表事件序列解析出的步。

    表里填的全是中文，这里统一映射回 reference_football.csv 的英文 code。
    """
    if rows:
        out = []
        for r in rows:
            qualifiers = {}
            for group_code, group in groups.items():
                value = r.get(group["label"])
                if is_null(value):
                    continue
                qualifiers[group_code] = group["zh_to_code"].get(clean(value), clean(value))
            out.append({
                "step_no": r.get("step_no"),
                "is_key_step": clean(r.get("is_key_step")) == "是",
                "actor": text_or_none(r.get("actor")),
                "event_code": zh_to_code.get(clean(r.get("event_code"))) or None,
                "qualifiers": qualifiers,
            })
        return out
    return [{
        "step_no": i,
        "is_key_step": False,
        "actor": step_actor(s) or None,
        "event_code": zh_to_code.get(step_event_code(s)) or None,
        "qualifiers": {},
    } for i, s in enumerate(sequence_steps, start=1)]


def to_record(card: dict, rows: list[dict], tactics: dict,
              groups: dict, zh_to_code: dict) -> dict:
    tid = clean(card.get("tactic_id"))
    tactic = tactics.get(tid)
    steps = split_steps(card.get("ev_sequence"))

    evidence = [{"evidence_id": axis, "text": clean(card.get(column))}
                for axis, column in EVIDENCE_AXES if not is_null(card.get(column))]

    return {
        # --- 与模型 prompt 同构 ---
        "tactic": text_or_none(card.get("tactic_label_given")),
        "temporal_grounding": {
            "start_sec": number_or_none(card.get("start_sec")),
            "end_sec": number_or_none(card.get("end_sec")),
        },
        # 表里是「横区 / 纵道」两列中文，这里拼回受控词表原样的 `final_third, left_halfspace`
        "spatial_grounding": {
            "origin_zone": join_zone(card.get("origin_third"), card.get("origin_channel")),
            "target_zone": join_zone(card.get("target_third"), card.get("target_channel")),
        },
        "evidence": evidence,
        "tactical_report": text_or_none(card.get("ev_sequence")),

        # --- 溯源 ---
        "video_path": clean(card.get("video_path")),
        "tactic_id": tid or None,
        "tactic_priority": tactic["priority"] if tactic else None,
        "annotator": text_or_none(card.get("annotator")),
        "date": clean(card.get("date")) or None,
        "visibility": text_or_none(card.get("visibility")),
        "dispute": text_or_none(card.get("dispute")),

        # --- 人工独有，不参与打分 ---
        "who": {
            "non_scored": True,
            "team": text_or_none(card.get("who_team")),
            "initiator": text_or_none(card.get("who_initiator")),
            "carriers": text_or_none(card.get("who_carriers")),
            "finisher": text_or_none(card.get("who_finisher")),
            "defenders": text_or_none(card.get("who_defenders")),
            "group": text_or_none(card.get("who_group")),
        },
        "result": {
            "non_scored": True,
            "value": text_or_none(card.get("result")),
            "note": text_or_none(card.get("result_note")),
        },
        "event_sequence": {
            "non_scored": True,
            "steps": build_steps(rows, steps, groups, zh_to_code),
        },
        "commentary_zh": {
            "non_scored": True,
            "text": text_or_none(card.get("commentary_zh")),
        },
        "pattern_of_play": {
            "derived": True,
            "value": derive_pattern_of_play(tid, tactic),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()

    data = Path(args.data_dir)
    tactics = load_tactics(data / GRID_REL)
    event_list, groups = load_reference(data / REFERENCE_REL)
    zh_to_code = event_zh_to_code(event_list)

    cards, steps_by_key = read_workbook(Path(args.workbook))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for card in cards:
            record = to_record(card, steps_by_key.get(card_key(card), []),
                               tactics, groups, zh_to_code)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已写入 {out}（{len(cards)} 条）")


if __name__ == "__main__":
    main()
