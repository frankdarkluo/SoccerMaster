from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pytest

from openpyxl.utils import get_column_letter

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
    main_cols = _validated_columns(wb["主表"])
    keys = list(build.MAIN_COLUMNS)
    for key in ("origin_third", "origin_channel", "target_third", "target_channel",
                "who_team", "result", "visibility"):
        assert get_column_letter(keys.index(key) + 1) in main_cols, key

    detail = wb["事件明细"]
    assert len(detail.data_validations.dataValidation) == 2 + 15   # 是否关键步 + 事件 + 15 组


def test_dropdown_formulas_are_valid_inline_lists(workbook):
    """v1 的下拉全废了，就是因为 formula1 带了前导 =，Excel 静默丢弃整条验证。"""
    wb, _ = workbook
    assert "选项" not in wb.sheetnames          # 不再跨表引用
    seen = 0
    for ws in wb.worksheets:
        for dv in ws.data_validations.dataValidation:
            formula = str(dv.formula1)
            assert not formula.startswith("="), f"{ws.title}: formula1 不能带前导 = → {formula[:40]}"
            assert formula.startswith('"') and formula.endswith('"'), formula[:40]
            assert len(formula) <= 255, f"{ws.title}: 内联下拉超 255 字符（{len(formula)}）"
            seen += 1
    assert seen >= 20


def test_all_headers_are_chinese(workbook):
    wb, _ = workbook
    for name in ("主表", "事件明细", "反馈表"):
        for header in _headers(wb[name]):
            assert re.search(r"[\u4e00-\u9fff]", header), f"{name} 的表头「{header}」不是中文"


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


def test_event_dropdown_is_chinese_with_extension_mark(workbook):
    wb, _ = workbook
    detail = wb["事件明细"]
    col = list(build.DETAIL_COLUMNS).index("event_code") + 1
    dv = next(d for d in detail.data_validations.dataValidation
              if get_column_letter(col) in str(d.sqref))
    labels = str(dv.formula1).strip('"').split(",")
    assert "传球" in labels
    assert "跑向身后*" in labels          # * = 本项目扩展
    assert "传中" not in labels           # 传中是限定词取值，不是事件
    assert not any("(" in x for x in labels)
