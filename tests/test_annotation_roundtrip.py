from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import openpyxl
import pytest

from pipeline import fifa_grounding as fg
from scripts import build_fifa_annotation_tasks as build
from scripts import check_annotations as check
from scripts import xlsx_to_jsonl as convert

ROOT = Path(__file__).resolve().parents[1]


def _col(ws, name: str) -> int:
    for cell in ws[1]:
        if cell.value == name:
            return cell.column
    raise AssertionError(f"找不到列 {name}")


def _fill(ws, row: int, values: dict) -> None:
    for name, value in values.items():
        ws.cell(row, _col(ws, name), value)


def _row_of(ws, tactic_id: str) -> int:
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, _col(ws, "词条id")).value == tactic_id:
            return r
    raise AssertionError(f"主表里没有 {tactic_id}")


def _detail_rows(ws, video_path: str) -> list[int]:
    col = _col(ws, "视频路径")
    return [r for r in range(2, ws.max_row + 1) if ws.cell(r, col).value == video_path]


@pytest.fixture()
def filled(data_dir: Path, tmp_path: Path) -> Path:
    """生成模板并填两张卡：一张完全合规，一张埋了 5 个错。"""
    path = tmp_path / "filled.xlsx"
    build.build(data_dir, path)
    wb = openpyxl.load_workbook(path)
    main, detail = wb["主表"], wb["事件明细"]

    # --- 合规的一张：line-breaking-pass（进攻链，不要求 who_group）---
    good = _row_of(main, "line-breaking-pass")
    good_path = main.cell(good, _col(main, "视频路径")).value
    _fill(main, good, {
        "标注人": "小王", "标注日期": "2026-08-18",
        "开始秒": 8.0, "结束秒": 14.0,
        "起始区_横区": "中场", "起始区_纵道": "中路",
        "结束区_横区": "前场", "结束区_纵道": "左肋",
        "队伍": "进攻方", "发起者": "后腰@蓝队", "终结者": "前腰@蓝队",
        "关键防守者": "两名中场@白队",
        "证据_参与者": "传球者 1 人、被穿越的防守者 2 人、接球者 1 人",
        "证据_空间": "穿越白队两条线之间约 12 米的空当",
        "证据_时间": "从出脚到接球约 1.5 秒",
        "事件序列": "后腰@蓝队 传球 -> 前腰@蓝队 接球* -> 前腰@蓝队 过人",
        "结果": "无明确结果", "画面清晰度": "清晰",
    })
    rows = _detail_rows(detail, good_path)
    for i, r in enumerate(rows[:3]):
        _fill(detail, r, {
            "角色": ["后腰@蓝队", "前腰@蓝队", "前腰@蓝队"][i],
            "事件": ["传球", "接球*", "过人"][i],
            "是否关键步": "是" if i == 0 else "否",
        })
    _fill(detail, rows[0], {"传球方向/意图": "直塞", "触球技术": "脚弓/脚内侧"})

    # --- 有问题的一张：low-block（防守类，要求 who_group）---
    bad = _row_of(main, "low-block")
    _fill(main, bad, {
        "标注人": "小王", "标注日期": "2026-08-18",
        "开始秒": 5.0, "结束秒": 99.0,               # 越界：片段只有 24.5 秒
        "队伍": "防守方",
        "发起者": "中卫@红队",                       # 与事件序列第一步对不上
        "证据_参与者": "全队回撤",
        "证据_空间": "站得很紧凑",                    # 纯形容词，没有数字
        "证据_时间": "持续一段时间",
        "事件序列": "边锋@白队 传中 -> 后腰@红队 拦截",   # 只有 2 步 + 传中不是事件
        # 群体描述 故意留空
        "画面清晰度": "清晰",
    })
    wb.save(path)
    return path


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_checker_flags_every_planted_error(filled: Path, data_dir: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    proc = _run("check_annotations.py", str(filled),
                "--data-dir", str(data_dir), "--json", str(out))
    assert proc.returncode == 1, proc.stdout + proc.stderr

    items = json.loads(out.read_text(encoding="utf-8"))
    rules = {i["rule"] for i in items}
    assert "时间区间越界" in rules
    assert "纯形容词" in rules
    assert "who_group 缺失" in rules
    assert "步数越界" in rules
    assert "传中不是事件" in rules
    assert "发起者 与事件序列对不上" in rules

    # 合规的那张卡一条都不该被打回
    assert not [i for i in items if "Breaking lines" in i["tactic_label_given"]]


def test_checker_passes_a_clean_sheet(data_dir: Path, tmp_path: Path):
    path = tmp_path / "empty.xlsx"
    build.build(data_dir, path)
    proc = _run("check_annotations.py", str(path), "--data-dir", str(data_dir))
    assert proc.returncode == 0
    assert "还没有任何已标注的行" in proc.stdout


def test_jsonl_is_isomorphic_to_the_model_prompt(filled: Path, data_dir: Path, tmp_path: Path):
    out = tmp_path / "annotations.jsonl"
    proc = _run("xlsx_to_jsonl.py", str(filled), "--out", str(out),
                "--data-dir", str(data_dir))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2                     # 只有填过的两张卡

    good = next(r for r in records if r["tactic_id"] == "line-breaking-pass")
    # 与模型 prompt 同构的五个字段
    assert good["tactic"] == "Breaking lines"
    assert good["temporal_grounding"] == {"start_sec": 8.0, "end_sec": 14.0}
    # 中文两列必须拼回受控词表的英文原值
    assert good["spatial_grounding"]["origin_zone"] == "middle_third, center"
    assert good["spatial_grounding"]["target_zone"] == "final_third, left_halfspace"
    assert {e["evidence_id"] for e in good["evidence"]} == {
        "video_participants", "video_space", "video_time", "video_sequence"}
    assert good["tactical_report"]

    # 人工独有字段一律标 non_scored
    for field in ("who", "result", "event_sequence", "commentary_zh"):
        assert good[field]["non_scored"] is True

    steps = good["event_sequence"]["steps"]
    assert [s["event_code"] for s in steps] == ["pass", "receive", "dribble"]
    assert sum(s["is_key_step"] for s in steps) == 1
    # 表里填中文，导出必须是英文 code
    assert steps[0]["qualifiers"] == {
        "foot_technique": "inside_foot", "pass_direction": "through_ball"}


def test_pattern_of_play_is_derived_not_asked(filled: Path, data_dir: Path, tmp_path: Path):
    out = tmp_path / "annotations.jsonl"
    _run("xlsx_to_jsonl.py", str(filled), "--out", str(out), "--data-dir", str(data_dir))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for r in records:
        assert r["pattern_of_play"]["derived"] is True
        assert r["pattern_of_play"]["value"] in {"open_play", "set_piece", "fast_break"}


def test_step_parsing_accepts_both_arrow_forms():
    assert fg.split_steps("A pass -> B receive") == ["A pass", "B receive"]
    assert fg.split_steps("A pass → B receive") == ["A pass", "B receive"]
    assert fg.step_event_code("后腰@蓝队 pass") == "pass"
    assert fg.extract_code("传球(pass)") == "pass"
    assert fg.role_of("10号左边后卫@白队") == "10号左边后卫"


def test_module_exposes_what_the_scripts_import():
    for name in ("MIN_STEPS", "MAX_STEPS", "EVIDENCE_AXES", "read_workbook"):
        assert hasattr(fg, name)
    assert check.HAS_DIGIT.search("约 12 米")
    assert convert.number_or_none("null") is None
