from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pytest

from scripts import build_fifa_annotation_tasks as build

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
