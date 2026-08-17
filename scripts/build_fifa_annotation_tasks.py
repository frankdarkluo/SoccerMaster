#!/usr/bin/env python3
"""生成 FIFA Game Library 的人工 grounding 标注模板（xlsx）。

输入（都在 data/ 下，只读）：
  FIFA Game Library/FIFA_Game_Library_examples.csv  片段与文件名标签
  足球战术数据库_词条表_Grid.csv                      词条定义、可观察特征、视频侧-*
  足球战术数据库.xlsx → 受控词表 sheet                zones 受控词表
  reference_football.csv                            事件 code 与细粒度限定词组

输出：四个 sheet 的 xlsx
  主表      一张卡一行，联合主键 = (video_path, tactic_label_given)，按 P0 优先排序
  事件明细  ev_sequence 的每一步一行，只有关键步要填限定词
  反馈表    标注者边标边记，不阻塞
  选项      各下拉的取值来源（数据验证引用这里的区域）

规则见 data/annotations/足球战术人工标注注意事项_v2.pdf。

用法：
    python3 scripts/build_fifa_annotation_tasks.py \
        --out data/annotations/templates/fifa_grounding_template.xlsx
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fifa_grounding import (  # noqa: E402
    COLLECTIVE_SHAPE_TACTICS,
    DROPPED_GROUPS,
    EXTRA_EVENTS,
    load_reference,
    load_tactics,
    needs_group,
)

DEFAULT_DATA = ROOT / "data"

EXAMPLES_REL = Path("FIFA Game Library") / "FIFA_Game_Library_examples.csv"
GRID_REL = Path("足球战术数据库_词条表_Grid.csv")
DB_XLSX_REL = Path("足球战术数据库.xlsx")
REFERENCE_REL = Path("reference_football.csv")

# 绝不能出现在发给标注者的文件里：模型产物会锚定标注结果。
FORBIDDEN_COLUMNS = ("description_zh", "evidence_source", "evidence_spans_json")

RESULT_OPTIONS = ["进球", "射正", "射偏", "传中", "获得定位球", "丢失球权", "被解围", "无明确结果"]
VISIBILITY_OPTIONS = ["清晰", "部分遮挡", "无法判断"]
TEAM_OPTIONS = ["进攻方", "防守方"]
YESNO_OPTIONS = ["是", "否"]
FEEDBACK_KINDS = ["字段说明看不懂", "下拉里没有合适取值", "标签与画面不符", "耗时过长", "其他"]

STEPS_PREFILLED = 3          # 每张卡预先铺 3 行（最少步数），不够自己插行
DV_MAX_ROW = 8000            # 数据验证覆盖到的行，留出插行余量

GREY = PatternFill("solid", fgColor="EFF2F5")
HEAD = PatternFill("solid", fgColor="1F3B54")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10)
WRAP = Alignment(vertical="top", wrap_text=True)
TOP = Alignment(vertical="top")


# --------------------------------------------------------------------------- 读取

def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_zones(db_xlsx: Path) -> list:
    ws = openpyxl.load_workbook(db_xlsx, read_only=True)["受控词表"]
    text = "\n".join(str(c) for row in ws.iter_rows(values_only=True) for c in row if c)
    channels = ["left_wing", "left_halfspace", "center", "right_halfspace", "right_wing"]
    thirds = ["defensive_third", "middle_third", "final_third"]
    specials = ["zone_14", "penalty_area", "six_yard_box", "all"]
    missing = [v for v in channels + thirds + specials if v not in text]
    if missing:
        raise SystemExit(f"受控词表 sheet 里找不到这些 zone 取值，请核对：{missing}")
    return [f"{t}, {c}" for t in thirds for c in channels] + specials


def load_cards(examples: Path, tactics: dict) -> list:
    """一张卡 = (片段, 文件名给的一个标签)。去重后按 P0 优先排序。"""
    order = {"P0": 0, "P1": 1, "P2": 2}
    cards = []
    for r in read_csv(examples):
        if r["duplicate_of"].strip() or r["label_role"] not in ("primary", "secondary"):
            continue
        tid = r["tactic_id"].strip()
        t = tactics.get(tid)
        cards.append({
            "video_path": r["video_path"],
            "duration_s": r["duration_s"],
            "label": r["source_label"] or r["raw_label"],
            "tactic_id": tid,
            "priority": t["priority"] if t else "",
            "hint": t["hint"] if t else "",
            "actors": t["actors"] if t else "",
            "sequence": t["sequence"] if t else "",
            "group_required": "是" if t and needs_group(t, tid) else "否",
        })
    cards.sort(key=lambda c: (order.get(c["priority"], 9), c["video_path"], c["label"]))
    return cards


# --------------------------------------------------------------------------- 写表

def write_header(ws, columns: list, prefilled: int) -> None:
    for i, (name, width) in enumerate(columns, start=1):
        cell = ws.cell(1, i, name)
        cell.fill = HEAD
        cell.font = HEAD_FONT
        cell.alignment = WRAP
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = ws.cell(2, prefilled + 1)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"


def add_options_sheet(wb, lists: "OrderedDict[str, list]"):
    """把所有下拉取值写进「选项」sheet，返回 名称 -> 区域引用。

    必须走区域引用而不是内联列表：Excel 的内联列表上限 255 字符，zones 有 19 项放不下。
    """
    ws = wb.create_sheet("选项")
    refs = {}
    for col, (name, values) in enumerate(lists.items(), start=1):
        letter = get_column_letter(col)
        ws.cell(1, col, name).font = Font(bold=True)
        for row, v in enumerate(values, start=2):
            ws.cell(row, col, v)
        ws.column_dimensions[letter].width = max(12, min(40, max(len(str(v)) for v in values) + 2))
        refs[name] = f"='选项'!${letter}$2:${letter}${len(values) + 1}"
    ws.sheet_state = "hidden"
    return refs


def attach_dropdown(ws, ref: str, col: int, last_row: int) -> None:
    dv = DataValidation(type="list", formula1=ref, allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv)
    letter = get_column_letter(col)
    dv.add(f"{letter}2:{letter}{last_row}")


def build_main(wb, cards, refs):
    columns = [
        ("video_path", 58), ("duration_s", 10), ("tactic_label_given", 24), ("tactic_id", 20),
        ("tactic_hint", 52), ("参与者提示", 44), ("序列提示", 44), ("who_group_必填", 12),
        ("annotator", 10), ("date", 11),
        ("start_sec", 9), ("end_sec", 9), ("origin_zone", 24), ("target_zone", 24),
        ("who_team", 16), ("who_initiator", 18), ("who_carriers", 18), ("who_finisher", 18),
        ("who_defenders", 22), ("who_group", 30),
        ("ev_participants", 30), ("ev_space", 30), ("ev_time", 26), ("ev_sequence", 44),
        ("result", 14), ("result_note", 18), ("commentary_zh", 34), ("visibility", 12),
        ("dispute", 26),
    ]
    prefilled = 8
    ws = wb.create_sheet("主表", 0)
    write_header(ws, columns, prefilled)

    for row, c in enumerate(cards, start=2):
        values = [c["video_path"], float(c["duration_s"]) if c["duration_s"] else None,
                  c["label"], c["tactic_id"], c["hint"], c["actors"], c["sequence"],
                  c["group_required"]]
        for i, v in enumerate(values, start=1):
            cell = ws.cell(row, i, v)
            cell.fill = GREY
            cell.alignment = WRAP
        for i in range(prefilled + 1, len(columns) + 1):
            ws.cell(row, i).alignment = TOP

    last = len(cards) + 1
    for name, col in [("zones", 13), ("zones", 14), ("who_team", 15),
                      ("result", 25), ("visibility", 28)]:
        attach_dropdown(ws, refs[name], col, max(last, DV_MAX_ROW))
    return ws


def build_detail(wb, cards, refs, groups):
    columns = [("video_path", 58), ("tactic_label_given", 24), ("step_no", 8),
               ("is_key_step", 12), ("actor", 22), ("event_code", 20)]
    columns += [(g["label"], 18) for g in groups.values()]

    ws = wb.create_sheet("事件明细")
    write_header(ws, columns, 3)

    row = 2
    for c in cards:
        for step in range(1, STEPS_PREFILLED + 1):
            for i, v in enumerate([c["video_path"], c["label"], step], start=1):
                cell = ws.cell(row, i, v)
                cell.fill = GREY
                cell.alignment = WRAP
            row += 1

    attach_dropdown(ws, refs["is_key_step"], 4, DV_MAX_ROW)
    attach_dropdown(ws, refs["event_code"], 6, DV_MAX_ROW)
    for i, code in enumerate(groups, start=7):
        attach_dropdown(ws, refs[f"tag_{code}"], i, DV_MAX_ROW)
    return ws


def build_feedback(wb, refs):
    columns = [("日期", 12), ("问题类型", 22), ("涉及片段（可留空）", 50),
               ("具体描述", 60), ("你的建议", 40)]
    ws = wb.create_sheet("反馈表")
    write_header(ws, columns, 0)
    attach_dropdown(ws, refs["feedback_kind"], 2, 500)
    ws.cell(2, 4, "例：who_carriers 说「超过 3 人写群体」，但没说群体要不要带队伍名").alignment = WRAP
    return ws


# --------------------------------------------------------------------------- 主流程

def assert_no_model_columns(wb) -> None:
    """模型产物绝不能出现在发给标注者的文件里。"""
    for ws in wb.worksheets:
        headers = [c.value for c in ws[1] if c.value]
        leaked = [h for h in headers if h in FORBIDDEN_COLUMNS]
        if leaked:
            raise SystemExit(f"sheet「{ws.title}」泄漏了模型产物列：{leaked}")


def build(data_dir: Path, out: Path) -> dict:
    tactics = load_tactics(data_dir / GRID_REL)
    cards = load_cards(data_dir / EXAMPLES_REL, tactics)
    zones = load_zones(data_dir / DB_XLSX_REL)
    events, groups = load_reference(data_dir / REFERENCE_REL)

    lists = OrderedDict([
        ("zones", zones),
        ("who_team", TEAM_OPTIONS),
        ("result", RESULT_OPTIONS),
        ("visibility", VISIBILITY_OPTIONS),
        ("is_key_step", YESNO_OPTIONS),
        ("feedback_kind", FEEDBACK_KINDS),
        ("event_code", [f"{label}({code})" for code, label in events]),
    ])
    for code, g in groups.items():
        lists[f"tag_{code}"] = g["values"]

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    refs = add_options_sheet(wb, lists)
    build_main(wb, cards, refs)
    build_detail(wb, cards, refs, groups)
    build_feedback(wb, refs)
    wb.move_sheet("主表", offset=-wb.sheetnames.index("主表"))
    assert_no_model_columns(wb)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    counts = {p: sum(1 for c in cards if c["priority"] == p) for p in ("P0", "P1")}
    return {
        "cards": len(cards),
        "p0": counts["P0"],
        "p1": counts["P1"],
        "unmapped": len(cards) - counts["P0"] - counts["P1"],
        "events": len(events),
        "groups": len(groups),
        "detail_rows": len(cards) * STEPS_PREFILLED,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA),
                    help="data/ 目录，默认取仓库里的 data/")
    args = ap.parse_args()

    s = build(Path(args.data_dir), Path(args.out))
    print(f"已生成 {args.out}")
    print(f"  卡片 {s['cards']}（P0 {s['p0']} / P1 {s['p1']} / 无映射 {s['unmapped']}）")
    print(f"  事件明细预铺 {s['detail_rows']} 行（每卡 {STEPS_PREFILLED} 步）")
    print(f"  事件 code {s['events']} 个，限定词组 {s['groups']} 组"
          f"（已砍 {'、'.join(DROPPED_GROUPS)}）")


if __name__ == "__main__":
    main()
