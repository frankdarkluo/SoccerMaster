import json

import pytest

from pipeline.stage2b import tactical_commentary


ANALYSIS = {
    "fact_index": 0,
    "evidence_spans": [{"start_s": 0.0, "end_s": 2.0, "observation": "白队向前推进"}],
    "state_before": "双方尚未落位",
    "position_or_action": "白队夺球后立即向前",
    "observed_change": "黑队转身回撤",
    "space_origin": "created",
    "space_evidence": "防守间距扩大",
    "opponent_constraint": "黑队只能回追",
    "space_change": "纵向通道打开",
    "functional_effect": "白队快速推进",
    "optional_dilemma": None,
    "terminal_sequence": [],
    "mechanism_chain": "向前跑动迫使对手回撤并打开通道",
    "mechanism_result": "effective",
    "play_result": "unclear",
    "confidence": 90,
    "confidence_reasons": ["过程连续可见"],
    "evidence_gaps": ["未呈现终结"],
    "topology_fact_ids": ["topology:w000000"],
}
TEXT = "白队夺球后立即向前推进，多名球员同步前插迫使黑队整线转身回撤。白队由此利用扩大的纵向通道推进过中线，但片段没有呈现后续终结。"


@pytest.mark.parametrize("provider", ["doubao", "gemini"])
def test_two_stage_runner_has_no_single_stage_output(tmp_path, monkeypatch, provider):
    review = tmp_path / "review.jsonl"
    review.write_text(json.dumps({
        "clip_uid": "event_clips:0006", "verdict": "correct",
        "tactic_id": "fast_break_pattern", "tactic_zh": "快速反击",
        "window": {"start_s": 0.0, "end_s": 2.0}, "evidence_text": "白队向前推进",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    exemplar = tmp_path / "exemplar.json"
    exemplar.write_text(json.dumps({
        "schema_version": "deep-commentary-exemplar-v2",
        "locked_fact": {}, "response": {},
    }), encoding="utf-8")
    clip = tmp_path / "preprocessing/0006/clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    calls = []
    prompts = []

    def fake_analyze(_repo, _clip_uid, output_path, **_):
        topology = {
            "position_source": "gt",
            "labels_path": "Labels-GameState.json",
            "windows": [{
                "window_id": "w000000",
                "start_s": 0.0,
                "end_s": 1.0,
                "attacking_team": "left",
                "lines": [{
                    "line_id": "line-1",
                    "team": "left",
                    "role": "attacking_forward",
                    "status": "renderable",
                    "support_ratio": 1.0,
                    "members": [{
                        "track_id": 4,
                        "jersey_number": 4,
                    }],
                    "evidence_gaps": [],
                }],
                "zones": [],
            }],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(topology), encoding="utf-8")
        return topology

    def fake_generate(selected, prompt, schema, *, video_path=None, **_):
        calls.append((selected, video_path))
        prompts.append(prompt)
        return (ANALYSIS if video_path else {"text_zh": TEXT}), [], 1

    monkeypatch.setattr(tactical_commentary, "generate_json", fake_generate)
    monkeypatch.setattr(tactical_commentary, "analyze_clip", fake_analyze)
    monkeypatch.setattr(
        tactical_commentary, "get_video_info", lambda _: {"duration_s": 12.0},
    )
    output = tmp_path / "out"
    tactical_commentary.run(
        review, tmp_path / "preprocessing", output, exemplar,
        repo_root=tmp_path, topology_root=tmp_path / "topology",
        provider=provider, clip_uids=["event_clips:0006"],
    )
    record = json.loads((output / provider / "0006.json").read_text())
    assert calls == [(provider, clip), (provider, None)]
    assert record["commentary_zh"] == TEXT
    assert "single_call" not in record
    assert "text_zh" not in record["analysis"]
    assert "locked_visible_topology" in prompts[0]
    assert record["topology"]["position_source"] == "gt"
