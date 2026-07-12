from copy import deepcopy

import pytest

from pipeline.relations.state import (
    build_tactical_state,
    canonical_team_map,
    project_proposal_events,
    project_proposal_state,
)
from pipeline.stage2b.digest import FrameData


def _frames(count: int = 25) -> list[FrameData]:
    return [
        FrameData(
            frame_id=frame_id,
            ball_xy=(float(frame_id), 0.5),
            players=[
                {
                    "track_id": 10,
                    "x": float(frame_id),
                    "y": 1.0,
                    "role": "player",
                    "team": "left",
                    "jersey": "10",
                },
                {
                    "track_id": 20,
                    "x": -float(frame_id),
                    "y": -1.0,
                    "role": "player",
                    "team": "right",
                    "jersey": "8",
                },
            ],
        )
        for frame_id in range(1, count + 1)
    ]


def _state(**kwargs) -> dict:
    return build_tactical_state(
        _frames(),
        fps=25.0,
        clip_id="SNGS-9001",
        state_source="ground_truth",
        **kwargs,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def test_canonical_team_map_is_recorded_and_direction_independent():
    assert canonical_team_map({"right", "left"}) == {
        "left": "team_0",
        "right": "team_1",
    }
    recorded = {"left": "team_1", "right": "team_0"}
    state = _state(
        source_team_map=recorded,
        attack_directions={
            "team_0": {
                "value": "left_to_right",
                "source": "reviewed_metadata",
                "confidence": 1.0,
            },
            "team_1": {
                "value": "right_to_left",
                "source": "reviewed_metadata",
                "confidence": 1.0,
            },
        },
    )

    assert state["provenance"]["team_mapping"]["source_to_canonical"] == recorded
    assert state["teams"]["team_0"]["source_team_label"] == "right"
    assert state["teams"]["team_0"]["attack_direction"] == "left_to_right"


@pytest.mark.parametrize(
    "recorded",
    [
        {"left": "team_0"},
        {"left": "team_0", "right": "team_0"},
        {"left": "team_0", "right": "team_2"},
        {"left": "team_0", "other": "team_1"},
    ],
)
def test_invalid_or_partial_recorded_map_fails(recorded):
    with pytest.raises(ValueError):
        canonical_team_map({"left", "right"}, recorded=recorded)


def test_state_retains_25_hz_player_and_ball_measurements():
    state = _state()

    assert state["fps"] == 25.0
    assert state["n_frames"] == 25
    assert len(state["measurements"]["tracks"]["10"]["samples"]) == 25
    assert len(state["measurements"]["tracks"]["20"]["samples"]) == 25
    assert len(state["measurements"]["ball"]) == 25
    assert state["measurements"]["tracks"]["10"]["samples"][0] == {
        "frame_id": 1,
        "t": 0.0,
        "x": 1.0,
        "y": 1.0,
    }
    assert state["measurements"]["tracks"]["10"]["samples"][-1]["t"] == 0.96


def test_attack_direction_defaults_unknown_and_requires_explicit_provenance():
    state = _state()

    assert state["teams"]["team_0"]["attack_direction"] == "unknown"
    assert state["teams"]["team_1"]["attack_direction"] == "unknown"

    with pytest.raises(ValueError, match="source"):
        _state(
            attack_directions={
                "team_0": {"value": "left_to_right", "confidence": 1.0}
            }
        )


def test_normalized_context_defaults_to_explicit_unknowns():
    assert _state()["normalized_context"] == {
        "possession_phase": {"team_id": None, "value": "unknown"},
        "restart_state": {"value": "unknown"},
        "progression_state": {"team_id": None, "value": "unknown"},
        "danger_state": {"team_id": None, "value": "unknown"},
    }


def test_valid_source_labels_and_normalized_context_can_be_supplied():
    state = _state(
        source_labels={
            "gentac": {
                "macro_type": "transition",
                "subtype": "progression",
                "confidence": 0.82,
                "provenance": "rule_v1",
            }
        },
        normalized_context={
            "possession_phase": {
                "team_id": "team_0",
                "value": "transition",
                "source": "reviewed_rule_v1",
                "confidence": 0.9,
            },
            "restart_state": {
                "value": "open_play",
                "source": "reviewed_rule_v1",
                "confidence": 0.9,
            },
            "progression_state": {
                "team_id": "team_0",
                "value": "progressing",
                "source": "reviewed_rule_v1",
                "confidence": 0.8,
            },
            "danger_state": {
                "team_id": "team_0",
                "value": "neutral",
                "source": "reviewed_rule_v1",
                "confidence": 0.8,
            },
        },
    )

    assert state["source_labels"]["gentac"]["macro_type"] == "transition"
    assert state["normalized_context"]["possession_phase"] == {
        "team_id": "team_0",
        "value": "transition",
    }
    assert (
        state["provenance"]["normalized_context"]["possession_phase"]["source"]
        == "reviewed_rule_v1"
    )


@pytest.mark.parametrize(
    "context",
    [
        {
            "possession_phase": {
                "team_id": "team_0",
                "value": "invalid",
                "source": "rule",
                "confidence": 1.0,
            }
        },
        {
            "progression_state": {
                "team_id": "team_0",
                "value": "progressing",
            }
        },
        {
            "danger_state": {
                "team_id": "left",
                "value": "threat",
                "source": "rule",
                "confidence": 1.0,
            }
        },
    ],
)
def test_invalid_context_enum_or_missing_provenance_fails(context):
    with pytest.raises(ValueError):
        _state(normalized_context=context)


def test_state_contains_quality_and_provenance_but_no_concepts():
    state = _state(provenance={"tracking_source": "fixture"})

    assert state["quality"]["ball_coverage"] == 1.0
    assert state["provenance"]["tracking_source"] == "fixture"
    forbidden = ("concept", "candidate", "overlap", "counter_attack", "verified")
    assert not [
        key
        for key in _walk_keys(state)
        if any(token in key.casefold() for token in forbidden)
    ]


def test_proposal_projection_is_at_most_12_5_hz():
    projected = project_proposal_state(_state(), max_hz=12.5)
    samples = projected["measurements"]["tracks"]["10"]["samples"]

    assert len(samples) == 13
    assert [sample["frame_id"] for sample in samples[:3]] == [1, 3, 5]
    assert "source_team_label" not in str(projected)
    assert "team_mapping" not in str(projected)


def test_proposal_events_use_canonical_team_ids_without_mutating_event_spine():
    state = _state()
    events = [
        {
            "event_id": "evt_1",
            "event_code": "football.pass",
            "player_team": "left",
            "opponent_team": "right",
            "start_s": 1.0,
            "end_s": 2.0,
        }
    ]
    original = deepcopy(events)

    projected = project_proposal_events(events, state=state)

    assert projected[0]["player_team"] == "team_0"
    assert projected[0]["opponent_team"] == "team_1"
    assert events == original
    assert {
        key: value
        for key, value in projected[0].items()
        if key not in {"player_team", "opponent_team"}
    } == {
        key: value
        for key, value in original[0].items()
        if key not in {"player_team", "opponent_team"}
    }
