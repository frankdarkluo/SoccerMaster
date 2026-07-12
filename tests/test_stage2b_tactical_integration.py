import json
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.stage2b.hybrid import (
    audit_commentary,
    commentary_windows,
    compose_hybrid,
)
from pipeline.stage2b.kb import load_catalog
from pipeline.stage2b.run import generate_tactical_artifacts, run_stage2b


def _write_tracking(path: Path) -> Path:
    data = {
        "info": {
            "version": "1.3",
            "game_id": "dev",
            "id": "9001",
            "name": "SNGS-9001",
            "frame_rate": 25,
            "seq_length": 2,
            "im_dir": "img1",
            "im_ext": ".jpg",
        },
        "images": [
            {"image_id": "1", "file_name": "000001.jpg"},
            {"image_id": "2", "file_name": "000002.jpg"},
        ],
        "annotations": [],
        "categories": [],
    }
    for image_id, offset in (("1", 0.0), ("2", 0.1)):
        data["annotations"].extend(
            [
                {
                    "image_id": image_id,
                    "track_id": 10,
                    "attributes": {
                        "role": "player",
                        "team": "left",
                        "jersey": "10",
                    },
                    "bbox_pitch": {
                        "x_bottom_middle": 1.0 + offset,
                        "y_bottom_middle": 2.0,
                    },
                },
                {
                    "image_id": image_id,
                    "track_id": 20,
                    "attributes": {
                        "role": "player",
                        "team": "right",
                        "jersey": "8",
                    },
                    "bbox_pitch": {
                        "x_bottom_middle": -1.0 - offset,
                        "y_bottom_middle": -2.0,
                    },
                },
                {
                    "image_id": image_id,
                    "track_id": 99,
                    "attributes": {"role": "ball"},
                    "bbox_pitch": {
                        "x_bottom_middle": offset,
                        "y_bottom_middle": 0.0,
                    },
                },
            ]
        )
    path.write_text(json.dumps(data))
    return path


def _direct() -> list[dict]:
    return [
        {
            "kind": "event",
            "timestamp_s": 0.0,
            "end_s": 1.0,
            "text_zh": "直接解说。",
            "text_en": "Direct commentary.",
            "fallback_text_zh": "直接解说。",
            "fallback_text_en": "Direct commentary.",
            "energy": "calm",
            "events_referenced": [],
            "tactical_facts_referenced": [],
            "event_claims": [],
            "tactical_claims": [],
        }
    ]


def _fact(
    *,
    fact_id: str = "fact_001",
    concept_id: str = "overlap_run",
    decision_time_s: float = 2.2,
    verified_window: dict | None = None,
) -> dict:
    return {
        "fact_id": fact_id,
        "clip_id": "SNGS-9001",
        "concept_id": concept_id,
        "recipe": f"{concept_id}@1",
        "state_source": "gsr_prediction",
        "proposed_window": {"start_s": 1.0, "end_s": 2.0},
        "verified_window": verified_window
        or {"start_s": 1.1, "end_s": 2.1},
        "window_resolution": {
            "method": "deterministic_anchor_snap",
            "tolerance_s": 0.5,
        },
        "decision_time_s": decision_time_s,
        "team_id": "team_0",
        "attack_direction": "left_to_right",
        "actors": {"carrier_track_id": 10, "runner_track_id": 11},
        "verified_claim_levels": ["observation"],
        "resolved_evidence": [],
        "quality_flags": [],
        "provenance": {},
    }


def _tactical_segment(
    *,
    fact_id: str = "fact_001",
    concept_id: str = "overlap_run",
    claim_level: str = "observation",
    timestamp_s: float = 2.2,
    end_s: float = 4.0,
) -> dict:
    return {
        "kind": "tactical",
        "timestamp_s": timestamp_s,
        "end_s": end_s,
        "text_zh": "可见套边跑动。",
        "text_en": "A visible overlap run.",
        "fallback_text_zh": "可见套边。",
        "fallback_text_en": "Visible overlap.",
        "energy": "engaged",
        "events_referenced": [],
        "tactical_facts_referenced": [fact_id],
        "event_claims": [],
        "tactical_claims": [
            {
                "fact_id": fact_id,
                "concept_id": concept_id,
                "claim_level": claim_level,
            }
        ],
    }


def test_config_exposes_state_proposal_and_fact_paths(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)

    assert config.tactical_state_json == tmp_path / "comments/tactical_state.json"
    assert (
        config.tactical_proposals_json
        == tmp_path / "comments/tactical_proposals.json"
    )
    assert (
        config.verified_tactical_facts_json
        == tmp_path / "comments/verified_tactical_facts.json"
    )
    assert not hasattr(config, "tactical_candidates_json")


def test_disabled_catalog_skips_tactical_llm_call(tmp_path):
    tracking = _write_tracking(tmp_path / "predictions.json")
    calls = 0

    def forbidden_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("tactical LLM must not be called")

    state, audits, facts = generate_tactical_artifacts(
        tracking,
        clip_id="SNGS-9001",
        state_source="gsr_prediction",
        duration_s=10.0,
        events=[],
        catalog=load_catalog(),
        allowed_concept_ids=set(),
        call=forbidden_call,
    )

    assert state["schema_version"] == "tactical-state-v1"
    assert audits == []
    assert facts == []
    assert calls == 0


def test_run_writes_empty_proposals_and_facts_and_preserves_direct(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    clip_dir = tmp_path / "SNGS-9001"
    output.mkdir()
    clip_dir.mkdir()
    _write_tracking(output / "predictions.json")
    (clip_dir / "clip.mp4").write_bytes(b"fixture")
    direct = _direct()

    monkeypatch.setattr(
        "pipeline.stage2b.run.observe_direct",
        lambda *args, **kwargs: ([], direct),
    )
    monkeypatch.setattr(
        "pipeline.stage2b.run.video_duration_s", lambda path: 10.0
    )
    monkeypatch.setattr(
        "pipeline.stage2b.run._reconcile_direct",
        lambda events, commentary, duration_s=None: commentary,
    )

    result = run_stage2b(
        output,
        clip_dir,
        mode="hybrid",
        force=True,
        call=lambda *args, **kwargs: pytest.fail("unexpected tactical call"),
    )
    config = PipelineConfig(output_dir=output, clip_dir=clip_dir)

    assert json.loads(result.read_text()) == direct
    assert json.loads(config.tactical_proposals_json.read_text()) == []
    assert json.loads(config.verified_tactical_facts_json.read_text()) == []
    assert json.loads(config.tactical_state_json.read_text())["clip_id"] == "SNGS-9001"


def test_tactical_prompt_projects_event_teams_to_canonical_ids(tmp_path):
    tracking = _write_tracking(tmp_path / "predictions.json")
    prompts = []

    def capture(prompt, **kwargs):
        prompts.append(prompt)
        return "[]"

    generate_tactical_artifacts(
        tracking,
        clip_id="SNGS-9001",
        state_source="ground_truth",
        duration_s=10.0,
        events=[{"event_id": "evt_1", "player_team": "left"}],
        catalog=load_catalog(),
        allowed_concept_ids={"overlap_run"},
        call=capture,
    )

    assert len(prompts) == 1
    assert '"player_team": "team_0"' in prompts[0]
    assert '"player_team": "left"' not in prompts[0]


def test_composer_prompt_contains_facts_not_raw_proposals_or_private_recipe():
    prompts = []

    def capture(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(_direct())

    facts = [_fact()]
    result = compose_hybrid(
        [],
        _direct(),
        facts,
        {"overlap_run"},
        commentary_windows(facts, 10.0),
        10.0,
        call=capture,
    )

    assert result == _direct()
    assert '"verified_tactical_facts"' in prompts[0]
    assert '"fact_id": "fact_001"' in prompts[0]
    assert "normalized_proposal" not in prompts[0]
    assert '"parameters"' not in prompts[0]


def test_commentary_rejects_raw_proposal_reference():
    segment = _tactical_segment()
    segment["tactical_proposals_referenced"] = ["proposal_001"]

    errors = audit_commentary(
        [segment], [], [_fact()], {"overlap_run"}, 10.0
    )

    assert any("raw proposal" in error for error in errors)


@pytest.mark.parametrize(
    "facts,enabled,claim_level,expected",
    [
        ([], {"overlap_run"}, "observation", "missing fact"),
        ([_fact()], set(), "observation", "disabled concept"),
        ([_fact()], {"overlap_run"}, "effect", "observation"),
    ],
)
def test_commentary_rejects_missing_fact_disabled_concept_and_non_observation(
    facts, enabled, claim_level, expected
):
    errors = audit_commentary(
        [_tactical_segment(claim_level=claim_level)],
        [],
        facts,
        enabled,
        10.0,
    )

    assert any(expected in error for error in errors)


def test_commentary_cannot_start_before_fact_decision_time():
    errors = audit_commentary(
        [_tactical_segment(timestamp_s=2.0)],
        [],
        [_fact(decision_time_s=2.2)],
        {"overlap_run"},
        10.0,
    )

    assert any("decision_time_s" in error for error in errors)


@pytest.mark.parametrize("status", ["failed", "unsupported"])
def test_checker_failure_or_unsupported_falls_back_to_direct(status):
    direct = _direct()

    def invalid_tactical_response(prompt, **kwargs):
        return json.dumps([_tactical_segment()])

    result = compose_hybrid(
        [],
        direct,
        [],
        {"overlap_run"},
        [],
        10.0,
        call=invalid_tactical_response,
    )

    assert result is direct


def test_commentary_slots_do_not_bound_verified_evidence_windows():
    fact = _fact(
        decision_time_s=8.0,
        verified_window={"start_s": 0.0, "end_s": 8.0},
    )
    windows = commentary_windows([fact], 10.0)

    assert windows == [
        {
            "window_id": "window_001",
            "fact_id": "fact_001",
            "start_s": 8.0,
            "end_s": 10.0,
        }
    ]
    assert (
        audit_commentary(
            [
                _tactical_segment(
                    timestamp_s=8.0,
                    end_s=10.0,
                )
            ],
            [],
            [fact],
            {"overlap_run"},
            10.0,
        )
        == []
    )
