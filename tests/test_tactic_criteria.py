from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.tactics_qa import criteria

REPO = Path(__file__).resolve().parent.parent
PROMPT_DOC = REPO / "docs/specs/2026-08-29-ai-tactic-scoring-prompt-v3.md"


def _payload(clip_id: str = "AA-1") -> dict:
    """一条合规输出：全部 12 类，每类全部证据键。"""
    return {
        "clip_id": clip_id,
        "teams": {"A": "白", "B": "蓝"},
        "scores": {
            tid: {
                "team": "A",
                "evidence": {k: 0.5 for k in criteria.evidence_keys(tid)},
                "window_start_s": 1.0,
                "window_end_s": 9.0,
                "note": "",
            }
            for tid in criteria.tactic_ids()
        },
    }


def test_schema_requires_every_tactic_and_every_evidence_key():
    schema = criteria.response_schema()
    scores = schema["properties"]["scores"]
    assert sorted(scores["required"]) == sorted(criteria.tactic_ids())
    assert scores["additionalProperties"] is False
    for tid in criteria.tactic_ids():
        evidence = scores["properties"][tid]["properties"]["evidence"]
        assert evidence["required"] == criteria.evidence_keys(tid)
        assert evidence["additionalProperties"] is False


def test_prompt_doc_has_not_drifted_from_criteria_json():
    # 判据块是生成物。漂移了就重新生成：python3 -m pipeline.tactics_qa.criteria
    assert criteria.prompt_block() in PROMPT_DOC.read_text(encoding="utf-8")


def test_validate_accepts_wellformed_payload():
    criteria.validate(_payload())


def _drop_key(payload, tactic):
    payload["scores"][tactic]["evidence"].pop(criteria.evidence_keys(tactic)[0])


def _set_first(payload, tactic, value):
    payload["scores"][tactic]["evidence"][criteria.evidence_keys(tactic)[0]] = value


# 判据键名会随判据修订而变，测试按位置取键，不写死名字。
@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["scores"].pop("one-two"), "缺少"),
    (lambda p: p["scores"].update({"tiki-taka": p["scores"]["one-two"]}), "多出"),
    (lambda p: _drop_key(p, "one-two"), "键不匹配"),
    (lambda p: p["scores"]["one-two"]["evidence"].update({"made_up": 0.5}), "键不匹配"),
    (lambda p: _set_first(p, "cutback", 1.5), "越界"),
    (lambda p: _set_first(p, "cutback", "high"), "不是数值"),
    (lambda p: p["scores"]["high-press"].update({"window_start_s": 9.0,
                                                 "window_end_s": 1.0}), "窗口反向"),
    (lambda p: p["scores"].clear(), "不匹配"),
])
def test_validate_rejects_malformed_payload(mutate, message):
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        criteria.validate(payload)


def test_validate_rejects_window_past_clip_duration():
    with pytest.raises(ValueError, match="超出片段时长"):
        criteria.validate(_payload(), duration_s=5.0)


def _run(src: Path, dst: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/score_to_candidates.py"), str(src), str(dst)],
        capture_output=True, text=True,
    )


def test_script_writes_candidates_for_valid_input(tmp_path: Path):
    src, dst = tmp_path / "scores.jsonl", tmp_path / "cand.csv"
    src.write_text(json.dumps(_payload(), ensure_ascii=False) + "\n", encoding="utf-8")

    result = _run(src, dst)

    assert result.returncode == 0, result.stderr
    assert dst.exists()
    assert len(dst.read_text(encoding="utf-8-sig").strip().splitlines()) > 1


def test_script_quarantines_invalid_rows_instead_of_passing_them_through(tmp_path: Path):
    """静默放行会让某一类在该 clip 上凭空消失，而 CSV 看起来完全正常。"""
    src, dst = tmp_path / "scores.jsonl", tmp_path / "cand.csv"
    bad = _payload("AA-bad")
    bad["scores"].pop("long-ball")
    src.write_text(
        json.dumps(_payload("AA-good"), ensure_ascii=False) + "\n"
        + json.dumps(bad, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = _run(src, dst)

    assert result.returncode == 1
    rejects = dst.with_suffix(dst.suffix + ".rejects.txt").read_text(encoding="utf-8")
    assert "AA-bad" in rejects and "long-ball" in rejects
    assert "AA-bad" not in dst.read_text(encoding="utf-8-sig")
    assert "AA-good" in dst.read_text(encoding="utf-8-sig")


def test_criterion_keys_are_stable_names_not_positional():
    """E1..En 这种位置编号是脆的：插一条判据，已落盘数据里的键含义就悄悄变了。"""
    keys = [k for tid in criteria.tactic_ids() for k in criteria.evidence_keys(tid)]
    assert keys, "没有读到任何判据"
    assert not any(k.startswith("E") and k[1:].isdigit() for k in keys)
    assert all(criteria.KEY_PATTERN.fullmatch(k) for k in keys)


def test_load_rejects_non_snake_case_keys(tmp_path: Path):
    doc = json.loads(criteria.CRITERIA_PATH.read_text(encoding="utf-8"))
    doc["tactics"][0]["evidence"][0]["key"] = "E1"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="snake_case"):
        criteria.load(bad)


def test_registry_cross_references_are_named():
    """能与 criterion_registry.json 一对一对上的判据，保留原名以便日后对账。"""
    doc = criteria.load()
    refs = [e["registry_criterion_id"]
            for t in doc["tactics"] for e in t["evidence"]
            if "registry_criterion_id" in e]
    assert refs, "至少应保留几条与 registry 同名的判据"
    assert all(isinstance(r, str) and r for r in refs)
