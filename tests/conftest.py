from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import pytest

from scripts import build_fifa_annotation_tasks as build

REAL_DATA = Path("/home/zhangzheng/guoqing/SoccerMaster/data")

EXAMPLE_FIELDS = [
    "video_path", "source_folder", "filename", "duration_s", "sha256", "duplicate_of",
    "raw_label", "tactic_id", "label_status", "description_zh", "description_source",
    "canonical_video_path", "analysis_result_path", "source_label", "label_kind",
    "label_role", "mapping_status", "evidence_spans_json",
]

GRID_FIELDS = [
    "优先级", "id（英文标识）", "中文名", "层级 layer", "阶段 phase", "一句话定义",
    "可观察特征", "视频侧-参与者", "视频侧-事件序列",
]


def _example(video: str, label: str, *, role: str, tactic_id: str = "",
             duplicate_of: str = "") -> dict:
    row = dict.fromkeys(EXAMPLE_FIELDS, "")
    row.update({
        "video_path": f"FIFA Game Library/{video}",
        "duration_s": "24.5",
        "duplicate_of": duplicate_of,
        "raw_label": label,
        "source_label": label,
        "tactic_id": tactic_id,
        "label_kind": "tactic",
        "label_role": role,
        # 模型产物：必须被模板丢弃
        "description_zh": "模型写的描述，不该出现在模板里",
        "evidence_spans_json": '[{"start_s":0.0,"end_s":24.5}]',
    })
    return row


def _grid(tactic_id: str, priority: str, *, phase: str, layer: str = "pattern") -> dict:
    row = dict.fromkeys(GRID_FIELDS, "")
    row.update({
        "优先级": priority,
        "id（英文标识）": tactic_id,
        "中文名": tactic_id,
        "层级 layer": layer,
        "阶段 phase": phase,
        "一句话定义": f"{tactic_id} 的一句话定义",
        "可观察特征": "特征一\n特征二\n特征三\n特征四",
        "视频侧-参与者": "传球者、接球者、被穿越的防守者。",
        "视频侧-事件序列": "A 传球 → B 接球",
    })
    return row


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """一个最小的 data/ 目录：三张 P0 卡、一张 P1 卡、一张无映射卡，外加一行重复片段与一行 context 标签。"""
    root = tmp_path / "data"
    (root / "FIFA Game Library").mkdir(parents=True)

    examples = [
        _example("out of possession/a.mp4", "Low-block", role="primary", tactic_id="low-block"),
        _example("In possession/b.mp4", "Breaking lines", role="primary",
                 tactic_id="line-breaking-pass"),
        _example("In possession/b.mp4", "Switching play", role="secondary",
                 tactic_id="width-and-depth"),
        _example("In possession/c.mp4", "Offering to receive", role="primary"),
        _example("In possession/f.mp4", "Build-up shape", role="primary",
                 tactic_id="buildup-structure"),
        _example("In possession/d.mp4", "Dup", role="primary", tactic_id="low-block",
                 duplicate_of="deadbeef"),
        _example("In possession/e.mp4", "Algeria (2021)", role="context"),
    ]
    with (root / build.EXAMPLES_REL).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXAMPLE_FIELDS)
        w.writeheader()
        w.writerows(examples)

    grid = [
        _grid("low-block", "P0", phase="defensive_org"),
        _grid("line-breaking-pass", "P0", phase="progression"),
        _grid("width-and-depth", "P1", phase="all"),
        _grid("buildup-structure", "P0", phase="buildup"),
    ]
    with (root / build.GRID_REL).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRID_FIELDS)
        w.writeheader()
        w.writerows(grid)

    # 受控词表与事件表直接复用真实文件：它们是词表的唯一来源，抄一份反而会失真。
    wb = openpyxl.load_workbook(REAL_DATA / build.DB_XLSX_REL)
    for name in list(wb.sheetnames):
        if name != "受控词表":
            wb.remove(wb[name])
    wb.save(root / build.DB_XLSX_REL)

    (root / build.REFERENCE_REL).write_bytes((REAL_DATA / build.REFERENCE_REL).read_bytes())
    return root
