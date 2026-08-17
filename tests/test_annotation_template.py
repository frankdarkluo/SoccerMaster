from __future__ import annotations

import csv
import re
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


@pytest.fixture()
def workbook(data_dir: Path, tmp_path: Path):
    out = tmp_path / "template.xlsx"
    stats = build.build(data_dir, out)
    return openpyxl.load_workbook(out), stats


def _headers(ws) -> list[str]:
    return [c.value for c in ws[1] if c.value]


def test_card_counts_and_p0_first(workbook):
    wb, stats = workbook
    assert stats["cards"] == 5          # 重复片段与 context 标签都被剔除
    assert stats["p0"] == 3
    assert stats["p1"] == 1
    assert stats["unmapped"] == 1

    ws = wb["主表"]
    tactic_ids = [ws.cell(r, 4).value for r in range(2, 2 + stats["cards"])]
    # P0 三张排最前（同优先级内按 video_path 排，不保证战术顺序），然后 P1，最后无映射
    assert set(tactic_ids[:3]) == {"low-block", "line-breaking-pass", "buildup-structure"}
    assert tactic_ids[3] == "width-and-depth"
    assert not tactic_ids[4]


def test_video_path_is_relative_to_library(workbook):
    wb, stats = workbook
    ws = wb["主表"]
    paths = [ws.cell(r, 1).value for r in range(2, 2 + stats["cards"])]
    assert paths and all(p.startswith("FIFA Game Library/") for p in paths)


def test_model_outputs_never_leak(workbook):
    wb, _ = workbook
    for ws in wb.worksheets:
        headers = _headers(ws)
        assert not set(headers) & set(build.FORBIDDEN_COLUMNS)
        flat = [v for row in ws.iter_rows(values_only=True) for v in row if v]
        assert not any("模型写的描述" in str(v) for v in flat)
        assert not any('[{"start_s"' in str(v) for v in flat)   # evidence_spans_json 的载荷


def test_dropped_groups_absent_from_detail(workbook):
    wb, _ = workbook
    headers = _headers(wb["事件明细"])
    for label in ("进攻模式", "情境标签"):
        assert label not in headers


def _validated_columns(ws) -> set[str]:
    cols = set()
    for dv in ws.data_validations.dataValidation:
        for rng in str(dv.sqref).split():
            cols.add(re.match(r"([A-Z]+)", rng).group(1))
    return cols


def test_dropdowns_present(workbook):
    wb, _ = workbook
    # origin_zone(M) / target_zone(N) / who_team(O) / result(Y) / visibility(AB)
    assert {"M", "N", "O", "Y", "AB"} <= _validated_columns(wb["主表"])

    detail = wb["事件明细"]
    assert len(detail.data_validations.dataValidation) == 2 + 15   # is_key_step + event_code + 15 组
    assert all(str(dv.formula1).startswith("='选项'!") for dv in detail.data_validations.dataValidation)


def test_detail_rows_match_cards(workbook):
    wb, stats = workbook
    ws = wb["事件明细"]
    assert stats["detail_rows"] == stats["cards"] * build.STEPS_PREFILLED
    steps = [ws.cell(r, 3).value for r in range(2, 2 + stats["detail_rows"])]
    assert steps[:build.STEPS_PREFILLED] == list(range(1, build.STEPS_PREFILLED + 1))


def test_group_required_flag_follows_phase(workbook):
    """防守组织 / 集体形态类战术才要求填 who_group。"""
    wb, stats = workbook
    ws = wb["主表"]
    by_tactic = {ws.cell(r, 4).value: ws.cell(r, 8).value
                 for r in range(2, 2 + stats["cards"])}
    assert by_tactic["low-block"] == "是"             # defensive_org
    assert by_tactic["line-breaking-pass"] == "否"    # progression
    # phase=buildup、layer=pattern，靠阶段推导不出来，必须走显式集体形态清单
    assert by_tactic["buildup-structure"] == "是"


def test_extension_events_are_marked(workbook):
    wb, _ = workbook
    ws = wb["选项"]
    header = [c.value for c in ws[1]]
    col = header.index("event_code") + 1
    codes = [ws.cell(r, col).value for r in range(2, ws.max_row + 1) if ws.cell(r, col).value]
    assert any("run_in_behind" in c and "本项目扩展" in c for c in codes)
    assert any("pass" in c for c in codes)
    assert not any(c.endswith("(cross)") for c in codes)   # 传中是限定词，不是事件
