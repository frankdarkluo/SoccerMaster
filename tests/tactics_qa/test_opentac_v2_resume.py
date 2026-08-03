import json
from pathlib import Path

import pytest

from pipeline.tactics_qa import opentac


def _observation():
    return {
        "sequence_id": "s1", "start_s": 1.0, "end_s": 4.0,
        "state_before_zh": "双方保持阵型", "position_or_action_zh": "白队向前传球",
        "observed_change_zh": "黑队转身", "space_origin": "existing",
        "space_evidence_zh": "两线之间已有空档", "opponent_constraint_zh": "黑队回追",
        "space_change_zh": "纵向空档被利用", "functional_effect_zh": "白队向前推进",
        "terminal_result_zh": "继续控球", "mechanism_chain_zh": "传球后黑队回追，白队利用空档推进",
        "confidence": 80, "confidence_reasons": ["过程连续可见"], "evidence_gaps": [],
    }


def _nomination():
    return {
        "candidates": [{
            "rank": 1, "tactic_id": "run-in-behind", "confidence": 80,
            "matched_cues": ["1. 跑动者起步时位于最后一线身前或与其平齐"],
            "supporting_sequence_ids": ["s1"],
            "evidence_spans": [{"start_s": 1.5, "end_s": 3.0}],
        }],
    }


def _kwargs(root, output_root):
    return {
        "output_root": output_root,
        "glossary_path": root / "data/足球战术数据库_词条表_Grid.csv",
        "source_rows": root / "benchmark/tactical_prototypes/source_rows.jsonl",
        "ground_truth": root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl",
        "repo_root": root,
        "provider": "doubao",
        "tactic_ids": ["run-in-behind"],
        "clip_uids": ["event_clips:0013"],
    }


def test_retry_failed_reuses_locked_observation_and_retries_only_nomination(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    calls = []

    def fake_generate(_provider, _prompt, _schema, **kwargs):
        calls.append(kwargs)
        if "video_path" in kwargs:
            return {"observations": [_observation()]}, [], 1
        if len(calls) == 2:
            raise RuntimeError("nomination failed")
        return _nomination(), [], 1

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64)
    monkeypatch.setattr(opentac, "generate_json", fake_generate)

    first = opentac.run_one_shot_v2(**_kwargs(root, tmp_path))
    second = opentac.run_one_shot_v2(**_kwargs(root, tmp_path), retry_failed=True)

    assert first["failed"] == first["observation_calls"] == first["nomination_calls"] == 1
    assert second["success"] == second["nomination_calls"] == 1
    assert second["observation_calls"] == 0
    assert len(calls) == 3
    assert calls[0]["retries"] == calls[1]["retries"] == 0
    assert calls[2]["retries"] == 2
    record = json.loads((tmp_path / "video_one_shot_v2/doubao/run-in-behind/0013.json").read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert record["calls"]["observation"]["status"] == "success"
    assert record["calls"]["nomination"]["status"] == "success"


def test_observation_failure_does_not_call_nomination(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    calls = []

    def fake_generate(_provider, _prompt, _schema, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("observation failed")

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64)
    monkeypatch.setattr(opentac, "generate_json", fake_generate)

    result = opentac.run_one_shot_v2(**_kwargs(root, tmp_path))

    assert result["failed"] == result["observation_calls"] == 1
    assert result["nomination_calls"] == 0
    assert len(calls) == 1
    record = json.loads((tmp_path / "video_one_shot_v2/doubao/run-in-behind/0013.json").read_text(encoding="utf-8"))
    assert record["failed_stage"] == "observation"


def test_observation_validation_rejects_string_lists_and_non_text_fields():
    cards = {"run-in-behind": {
        "tactic_id": "run-in-behind", "name_zh": "打身后 / 反越位跑位",
        "name_en": "run in behind", "observable_cues": ["cue"], "triggers": [],
    }}
    payload = {"observations": [_observation()]}
    payload["observations"][0]["confidence_reasons"] = "不是数组"
    with pytest.raises(ValueError, match="reasons or gaps"):
        opentac.validate_one_shot_v2_observations(payload, cards, 12.0)

    payload = {"observations": [{**_observation(), "state_before_zh": 1}]}
    with pytest.raises(ValueError, match="text fields"):
        opentac.validate_one_shot_v2_observations(payload, cards, 12.0)


def test_validation_failure_preserves_provider_usage_and_attempts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    invalid = {**_observation(), "position_or_action_zh": "run-in-behind"}
    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64)
    monkeypatch.setattr(
        opentac, "generate_json",
        lambda *_args, **_kwargs: ({"observations": [invalid]}, [{"total_tokens": 7}], 2),
    )

    result = opentac.run_one_shot_v2(**_kwargs(root, tmp_path))

    assert result["failed"] == 1
    record = json.loads((tmp_path / "video_one_shot_v2/doubao/run-in-behind/0013.json").read_text(encoding="utf-8"))
    call = record["calls"]["observation"]
    assert call["attempts"] == 2
    assert call["api_usage"] == [{"total_tokens": 7}]


def test_nomination_validation_failure_preserves_provider_usage_and_attempts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    calls = 0

    def fake_generate(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if "video_path" in kwargs:
            return {"observations": [_observation()]}, [{"total_tokens": 5}], 1
        invalid = _nomination()
        invalid["candidates"][0]["supporting_sequence_ids"] = ["unknown"]
        return invalid, [{"total_tokens": 11}], 2

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64)
    monkeypatch.setattr(opentac, "generate_json", fake_generate)

    result = opentac.run_one_shot_v2(**_kwargs(root, tmp_path))

    assert calls == 2
    assert result["failed"] == 1
    record = json.loads((tmp_path / "video_one_shot_v2/doubao/run-in-behind/0013.json").read_text(encoding="utf-8"))
    call = record["calls"]["nomination"]
    assert call["attempts"] == 2
    assert call["api_usage"] == [{"total_tokens": 11}]


def test_v2_report_metadata_rejects_stale_result(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]

    def fake_generate(*_args, **kwargs):
        if "video_path" in kwargs:
            return {"observations": [_observation()]}, [], 1
        return _nomination(), [], 1

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("b" if "FIFA Game Library" in str(path) else "a") * 64)
    monkeypatch.setattr(opentac, "generate_json", fake_generate)
    opentac.run_one_shot_v2(**_kwargs(root, tmp_path))
    record = json.loads((tmp_path / "video_one_shot_v2/doubao/run-in-behind/0013.json").read_text(encoding="utf-8"))
    cards = {
        key: value
        for key, value in opentac.load_glossary(root / "data/足球战术数据库_词条表_Grid.csv").items()
        if value["priority"] == "P0"
    }

    opentac._validate_one_shot_v2_metadata(record, cards, "doubao")
    record["calls"]["nomination"]["call_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nomination metadata"):
        opentac._validate_one_shot_v2_metadata(record, cards, "doubao")
