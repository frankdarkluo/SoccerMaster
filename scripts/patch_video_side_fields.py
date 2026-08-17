#!/usr/bin/env python3
"""把 halfspace-penetration / overlap-underlap 两条词条的「视频侧-*」四栏与「参考来源」
写进 足球战术数据库.xlsx（词条表 sheet）和同步导出的 _词条表_Grid.csv。

内容依据见 data/annotations/词条表补充_视频侧参与者_draft.md，每个角色都能追到公开来源。
状态一律写 draft，由负责人审核后改 reviewed。

用法：
    python3 scripts/patch_video_side_fields.py [--dry-run]

运行前请先备份（data/.backup/ 下已有带时间戳的副本）。
"""

import csv
import sys
from pathlib import Path

import openpyxl

DATA = Path("/home/zhangzheng/guoqing/SoccerMaster/data")
XLSX = DATA / "足球战术数据库.xlsx"
GRID = DATA / "足球战术数据库_词条表_Grid.csv"

AUTHOR = "Claude（草稿，待审）"
DATE = "2026-08-17"

PATCH = {
    "halfspace-penetration": {
        "视频侧-参与者":
            "肋部接球者（中场/前腰，或内切的边锋）、送出渗透球的传球者（中卫、后腰，或从边路回敲的球员）、"
            "在同侧拉开宽度把对方边后卫引出的边路球员、被中锋钉住而无法出防的中卫，"
            "以及在肋部接球者与持球人之间被迫二选一的边后卫。",
        "视频侧-空间约束":
            "接球点位于左/右半空间纵道内，距底线约 20-35 米；渗透传球线路斜切穿过对方中场线与防线之间的间隙；"
            "同侧边路有己方球员站位提供宽度，使对方边后卫与相邻中卫之间的横向间距被拉开。",
        "视频侧-时间窗":
            "在持球人于禁区前沿停顿、抬头观察的时序窗口内完成出球；接球者的斜向切入与传球出脚基本同步"
            "（跑动早于或与出球同时启动，而不是出球之后才启动）。",
        "视频侧-事件序列":
            "边路球员拉开宽度 → 对方边后卫被引出、与中卫拉开间距 → 中锋钉住同侧中卫 → "
            "接球者斜向切入肋部 → 传球者送出穿透中场线与防线之间的球 → 接球者在两线之间接球。",
        "参考来源":
            "https://learning.coachesvoice.com/cv/half-spaces-football-tactics-explained/\n"
            "https://totalfootballanalysis.com/article/tactical-theory-using-half-spaces-progress-ball-tactical-analysis-tactics",
    },
    "overlap-underlap": {
        "视频侧-参与者":
            "持球人（通常是边锋或边中场）、做套边或内切套上跑动的队友（多为边后卫，也可能是中场或前插的中卫）、"
            "在持球人与跑动者之间被迫二选一的对方边后卫（由此形成 2v1）；"
            "内切套上时另需标出被拉动、需要决定由谁盯防的中卫。",
        "视频侧-空间约束":
            "发生在同侧边路与半空间的结合地带，多在前场 30 米区域至底线附近；套边跑动路线在持球人外侧、贴近边线，"
            "内切套上路线穿过对方边后卫与中卫之间的间隙；同一边路同时出现 2 名本方进攻球员，形成 2 打 1。",
        "视频侧-时间窗":
            "跑动者在持球人形成稳定控球（护球或内切吸引防守）之后启动，从持球人身后加速绕前并越过球；"
            "从启动到越过持球人是秒级的连续动作，其间控球权不发生变更。",
        "视频侧-事件序列":
            "边锋接球并护球/内切吸引对方边后卫贴身对位 → 队友从持球人身后启动，走外侧（套边）或走内侧（内切套上）→ "
            "对方边后卫被迫在持球人与跑动者之间二选一 → 持球人分球给跑动者 → 跑动者传中或射门。",
        "参考来源":
            "https://tacticalfootballanalysis.com/tactical-theory-overlapping-and-underlapping-runs-tactical-analysis-tactics/\n"
            "https://the-footballanalyst.com/overlaps-vs-underlaps-football-tactics-explained/",
    },
}

COMMON = {"状态": "draft", "填写人": AUTHOR, "更新日期": DATE}


def patch_xlsx(dry: bool) -> None:
    wb = openpyxl.load_workbook(XLSX)
    ws = wb["词条表"]
    header = {c.value: c.column for c in ws[1] if c.value}
    idcol = header["id（英文标识）"]

    for row in range(1, ws.max_row + 1):
        tid = ws.cell(row, idcol).value
        if tid not in PATCH:
            continue
        for field, value in {**PATCH[tid], **COMMON}.items():
            cell = ws.cell(row, header[field])
            old = (cell.value or "").strip() if isinstance(cell.value, str) else cell.value
            if old:
                print(f"  跳过（已有内容）{tid} / {field}: {str(old)[:40]}...")
                continue
            print(f"  写入 {tid} / {field} ({len(value)} 字)")
            if not dry:
                cell.value = value
    if not dry:
        wb.save(XLSX)
        print(f"已保存 {XLSX.name}")


def patch_csv(dry: bool) -> None:
    with GRID.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    idcol = idx["id（英文标识）"]

    for row in rows[1:]:
        if len(row) <= idcol or row[idcol] not in PATCH:
            continue
        tid = row[idcol]
        for field, value in {**PATCH[tid], **COMMON}.items():
            i = idx[field]
            while len(row) <= i:
                row.append("")
            if row[i].strip():
                continue
            row[i] = value
        print(f"  CSV 已更新 {tid}")

    if not dry:
        with GRID.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        print(f"已保存 {GRID.name}")


def main() -> None:
    dry = "--dry-run" in sys.argv
    print("== xlsx ==")
    patch_xlsx(dry)
    print("== csv ==")
    patch_csv(dry)
    if dry:
        print("\n(dry-run，未写盘)")


if __name__ == "__main__":
    main()
