#!/usr/bin/env python3
"""逐行硬校验回收的标注表，输出打回清单。

只有一名标注者，算不了标注者间一致率，质量保证靠这里的硬校验 + 人工抽检。
每条规则都对应标注规范里的一条，报错信息直接指到规范条目，方便标注者自己改。

用法：
    python3 scripts/check_annotations.py 回收的表.xlsx
    python3 scripts/check_annotations.py 回收的表.xlsx --json 打回清单.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fifa_grounding import (  # noqa: E402
    MAX_STEPS,
    MIN_STEPS,
    card_key,
    clean,
    extract_code,
    is_null,
    load_reference,
    load_tactics,
    needs_group,
    qualifier_columns,
    read_workbook,
    role_of,
    split_steps,
    step_event_code,
)

DEFAULT_DATA = ROOT / "data"
GRID_REL = Path("足球战术数据库_词条表_Grid.csv")
REFERENCE_REL = Path("reference_football.csv")

HAS_DIGIT = re.compile(r"\d")
MULTI_VALUE = re.compile(r"[,，;；\n]")


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, sheet: str, row: int, rule: str, detail: str, key: tuple) -> None:
        self.items.append({
            "sheet": sheet, "row": row, "rule": rule, "detail": detail,
            "video_path": key[0], "tactic_label_given": key[1],
        })


def check_card(card: dict, tactics: dict, report: Report) -> None:
    key = card_key(card)
    row = card["_row"]
    tid = clean(card.get("tactic_id"))
    tactic = tactics.get(tid)

    # 规则 4：时间区间必须落在片段里
    start, end = card.get("start_sec"), card.get("end_sec")
    duration = card.get("duration_s")
    if is_null(start) or is_null(end):
        report.add("主表", row, "时间区间缺失", "start_sec / end_sec 必填", key)
    else:
        try:
            s, e = float(start), float(end)
            if s >= e:
                report.add("主表", row, "时间区间非法", f"start_sec({s}) 必须小于 end_sec({e})", key)
            elif duration and e > float(duration) + 0.01:
                report.add("主表", row, "时间区间越界",
                           f"end_sec({e}) 超过片段时长({duration})", key)
        except (TypeError, ValueError):
            report.add("主表", row, "时间区间非数字", f"start={start!r} end={end!r}", key)

    # 规则 1：证据必须含几何/时序/人数，最容易违反的是空间栏写成纯形容词
    if is_null(card.get("ev_space")):
        report.add("主表", row, "证据缺失", "ev_space 必填（规则 1）", key)
    elif not HAS_DIGIT.search(clean(card.get("ev_space"))):
        report.add("主表", row, "纯形容词",
                   f"ev_space 里没有任何数字，疑似纯形容词：{clean(card.get('ev_space'))[:40]}", key)

    # 规则 8：防守 / 集体形态类战术必须写群体栏
    if tactic and needs_group(tactic, tid) and is_null(card.get("who_group")):
        report.add("主表", row, "who_group 缺失",
                   f"{tid} 是防守/集体形态类战术，who_group 必填（规则 8）", key)

    # 规则 9 / 10：事件序列的步数与事件 code
    steps = split_steps(card.get("ev_sequence"))
    if not steps:
        report.add("主表", row, "事件序列缺失", "ev_sequence 必填（规则 9）", key)
        return steps
    if not MIN_STEPS <= len(steps) <= MAX_STEPS:
        report.add("主表", row, "步数越界",
                   f"事件序列 {len(steps)} 步，应在 {MIN_STEPS}-{MAX_STEPS} 之间", key)
    return steps


def check_sequence_events(card, steps, events, report) -> None:
    key, row = card_key(card), card["_row"]
    for i, step in enumerate(steps, start=1):
        code = step_event_code(step)
        if code == "cross":
            report.add("主表", row, "传中不是事件",
                       f"第 {i} 步写了 cross，应写 pass 并在明细表选「传中(cross)」（规则 10）", key)
        elif code not in events:
            report.add("主表", row, "事件 code 非法",
                       f"第 {i} 步的「{code}」不在事件词表里（规则 10）", key)


def check_who_matches_sequence(card, steps, report) -> None:
    """who_* 与事件序列是故意冗余的，交叉校验就是零成本质检。"""
    key, row = card_key(card), card["_row"]
    if not steps:
        return
    for field, step, label in (("who_initiator", steps[0], "第一步"),
                               ("who_finisher", steps[-1], "最后一步")):
        value = card.get(field)
        if is_null(value):
            continue                      # 防守类战术四槽为 null 是正常的
        role = role_of(value)
        if role and role not in clean(step):
            report.add("主表", row, f"{field} 与事件序列对不上",
                       f"{field}=「{role}」没有出现在事件序列的{label}：{clean(step)[:40]}", key)


def check_steps(card, steps, rows, events, groups, qual_cols, report) -> None:
    key, row = card_key(card), card["_row"]
    if not rows:
        report.add("事件明细", row, "明细表缺行", "主表填了事件序列，明细表却没有对应行", key)
        return

    if len(rows) != len(steps):
        report.add("事件明细", rows[0]["_row"], "步数不一致",
                   f"主表 {len(steps)} 步，明细表 {len(rows)} 行", key)

    keys = [r for r in rows if clean(r.get("is_key_step")) == "是"]
    if len(keys) != 1:
        report.add("事件明细", rows[0]["_row"], "关键步数量不对",
                   f"每张卡应恰好 1 步 is_key_step=是，实际 {len(keys)} 步", key)

    for r in rows:
        code = extract_code(r.get("event_code"))
        if code == "cross":
            report.add("事件明细", r["_row"], "传中不是事件",
                       "event_code 写了 cross，应写 pass 并在「传球方向」选传中", key)
        elif code and code not in events:
            report.add("事件明细", r["_row"], "事件 code 非法", f"「{code}」不在事件词表里", key)

        for column, group_code in qual_cols.items():
            value = r.get(column)
            if is_null(value):
                continue
            group = groups[group_code]
            if code and code not in group["applies_to"]:
                report.add("事件明细", r["_row"], "限定词越界",
                           f"「{column}」只适用于 {'/'.join(group['applies_to'])}，"
                           f"但这一步是 {code}", key)
            if group["exclusive"] and MULTI_VALUE.search(clean(value)):
                report.add("事件明细", r["_row"], "互斥组填了多个值",
                           f"「{column}」是互斥组，只能选一个：{clean(value)[:40]}", key)
            if clean(r.get("is_key_step")) != "是":
                report.add("事件明细", r["_row"], "非关键步填了限定词",
                           f"只有 is_key_step=是 的那一步需要填「{column}」", key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--json", dest="json_out", help="把打回清单另存为 JSON")
    args = ap.parse_args()

    data = Path(args.data_dir)
    tactics = load_tactics(data / GRID_REL)
    event_list, groups = load_reference(data / REFERENCE_REL)
    events = {code for code, _ in event_list}
    qual_cols = qualifier_columns(groups)

    cards, steps_by_key = read_workbook(Path(args.workbook))
    report = Report()

    for card in cards:
        steps = check_card(card, tactics, report) or []
        check_sequence_events(card, steps, events, report)
        check_who_matches_sequence(card, steps, report)
        check_steps(card, steps, steps_by_key.get(card_key(card), []),
                    events, groups, qual_cols, report)

    if not cards:
        print("这份表里还没有任何已标注的行。")
        return

    print(f"已标注 {len(cards)} 张卡，打回 {len(report.items)} 条")
    by_rule: dict[str, int] = {}
    for item in report.items:
        by_rule[item["rule"]] = by_rule.get(item["rule"], 0) + 1
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {rule}")
    for item in report.items[:40]:
        print(f"\n[{item['sheet']} 第 {item['row']} 行] {item['rule']}\n"
              f"  {Path(item['video_path']).name} / {item['tactic_label_given']}\n"
              f"  {item['detail']}")
    if len(report.items) > 40:
        print(f"\n……还有 {len(report.items) - 40} 条，用 --json 导出完整清单")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整清单已写入 {args.json_out}")

    sys.exit(1 if report.items else 0)


if __name__ == "__main__":
    main()
