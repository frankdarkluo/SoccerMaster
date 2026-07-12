from copy import deepcopy

import pytest

from pipeline.stage2b.tactics import process_tactical_proposals


def _catalog(
    *,
    concept_id: str = "overlap_run",
    roles: list[str] | None = None,
    constraints: dict[str, str] | None = None,
) -> dict:
    roles = roles or ["carrier_track_id", "runner_track_id"]
    constraints = constraints or {role: "same" for role in roles}
    return {
        "catalog_version": "tactical-kb-v2",
        "concepts": [
            {
                "id": concept_id,
                "required_actor_roles": roles,
                "actor_team_constraints": constraints,
                "allowed_claim_levels": ["observation"],
                "maturity": "audited",
                "production_enabled": False,
                "recipe": {
                    "id": concept_id,
                    "version": 1,
                    "checker": "fake_checker",
                    "parameters": {"window_tolerance_s": 0.5},
                },
            }
        ],
    }


def _state() -> dict:
    return {
        "schema_version": "tactical-state-v1",
        "clip_id": "SNGS-9001",
        "state_source": "gsr_prediction",
        "teams": {
            "team_0": {
                "source_team_label": "left",
                "attack_direction": "left_to_right",
            },
            "team_1": {
                "source_team_label": "right",
                "attack_direction": "right_to_left",
            },
        },
        "measurements": {
            "tracks": {
                "10": {"track_id": 10, "team_id": "team_0", "samples": []},
                "11": {"track_id": 11, "team_id": "team_0", "samples": []},
                "20": {"track_id": 20, "team_id": "team_1", "samples": []},
            },
            "ball": [],
        },
        "quality": {"ball_coverage": 1.0},
        "provenance": {"tracking_source": "fixture"},
    }


def _proposal(**updates) -> dict:
    proposal = {
        "concept_id": "overlap_run",
        "proposed_window": {"start_s": 1.0, "end_s": 2.0},
        "team_id": "team_0",
        "actors": {"carrier_track_id": 10, "runner_track_id": 11},
    }
    proposal.update(updates)
    return proposal


def _passed_checker(proposal, concept, state):
    return {
        "status": "passed",
        "verified_window": {"start_s": 1.1, "end_s": 2.1},
        "window_resolution": {
            "method": "deterministic_anchor_snap",
            "tolerance_s": 0.5,
        },
        "decision_time_s": 2.2,
        "resolved_evidence": [{"metric": "fixture", "value": 1.0}],
        "quality_flags": ["synthetic"],
        "reason": "fixture_pass",
    }


def _process(raw, **kwargs):
    return process_tactical_proposals(
        raw,
        catalog=kwargs.pop("catalog", _catalog()),
        tactical_state=kwargs.pop("tactical_state", _state()),
        duration_s=kwargs.pop("duration_s", 10.0),
        allowed_concept_ids=kwargs.pop(
            "allowed_concept_ids", {"overlap_run"}
        ),
        checkers=kwargs.pop("checkers", {"fake_checker": _passed_checker}),
        **kwargs,
    )


def test_unknown_fields_are_rejected_and_logged():
    proposal = _proposal(extra_field="malicious")

    audits, facts = _process([proposal])

    assert facts == []
    assert audits[0]["proposal_status"] == "rejected"
    assert "unknown_fields" in audits[0]["reasons"]
    assert audits[0]["raw"] == proposal
    assert audits[0]["normalized_proposal"] is None


@pytest.mark.parametrize(
    "field",
    [
        "evidence_queries",
        "predicate",
        "threshold",
        "recipe",
        "verified",
        "fact_id",
    ],
)
def test_model_predicate_recipe_threshold_and_verified_fields_are_rejected(field):
    proposal = _proposal()
    proposal[field] = True

    audits, facts = _process([proposal])

    assert facts == []
    assert audits[0]["proposal_status"] == "rejected"
    assert "unknown_fields" in audits[0]["reasons"]


@pytest.mark.parametrize(
    "proposal,reason",
    [
        (_proposal(concept_id="unknown"), "unknown_concept"),
        (_proposal(team_id="left"), "invalid_team_id"),
        (
            _proposal(
                actors={"carrier_track_id": 10, "runner_track_id": 999}
            ),
            "unknown_actor_track",
        ),
        (
            _proposal(actors={"carrier_track_id": 10}),
            "invalid_actor_roles",
        ),
    ],
)
def test_unknown_concept_team_track_and_missing_actor_are_rejected(
    proposal, reason
):
    audits, facts = _process([proposal])

    assert facts == []
    assert reason in audits[0]["reasons"]


def test_concept_outside_explicit_allowed_set_is_rejected():
    audits, facts = _process([_proposal()], allowed_concept_ids=set())

    assert facts == []
    assert "concept_not_allowed" in audits[0]["reasons"]


@pytest.mark.parametrize(
    "catalog,proposal,reason",
    [
        (
            _catalog(),
            _proposal(
                actors={"carrier_track_id": 10, "runner_track_id": 10}
            ),
            "actor_tracks_not_distinct",
        ),
        (
            _catalog(),
            _proposal(
                actors={"carrier_track_id": 10, "runner_track_id": 20}
            ),
            "actor_team_constraint_failed",
        ),
        (
            _catalog(
                concept_id="high_press",
                roles=["presser_track_id", "opponent_carrier_track_id"],
                constraints={
                    "presser_track_id": "same",
                    "opponent_carrier_track_id": "opponent",
                },
            ),
            {
                "concept_id": "high_press",
                "proposed_window": {"start_s": 1.0, "end_s": 2.0},
                "team_id": "team_0",
                "actors": {
                    "presser_track_id": 10,
                    "opponent_carrier_track_id": 11,
                },
            },
            "actor_team_constraint_failed",
        ),
    ],
)
def test_actor_roles_are_distinct_and_follow_same_or_opponent_constraints(
    catalog, proposal, reason
):
    audits, facts = _process(
        [proposal],
        catalog=catalog,
        allowed_concept_ids={proposal["concept_id"]},
    )

    assert facts == []
    assert reason in audits[0]["reasons"]


def test_checker_pass_rebuilds_a_new_fact_from_whitelisted_fields():
    raw = _proposal()

    audits, facts = _process([raw])

    assert audits[0]["verification_status"] == "passed"
    assert audits[0]["fact_id"] == "fact_001"
    assert len(facts) == 1
    fact = facts[0]
    assert fact["fact_id"] == "fact_001"
    assert fact["concept_id"] == "overlap_run"
    assert fact["recipe"] == "overlap_run@1"
    assert fact["actors"] == raw["actors"]
    assert "verified" not in fact
    assert "raw" not in fact
    assert fact is not raw


def test_verified_window_cannot_escape_recipe_tolerance():
    def escaping_checker(proposal, concept, state):
        result = _passed_checker(proposal, concept, state)
        result["verified_window"] = {"start_s": 0.0, "end_s": 3.0}
        return result

    audits, facts = _process(
        [_proposal()], checkers={"fake_checker": escaping_checker}
    )

    assert facts == []
    assert audits[0]["verification_status"] == "unsupported"
    assert "verified_window_outside_tolerance" in audits[0]["reasons"]


@pytest.mark.parametrize("outcome", ["failed", "unsupported", "exception"])
def test_checker_failed_unsupported_and_exception_produce_no_fact(outcome):
    def checker(proposal, concept, state):
        if outcome == "exception":
            raise RuntimeError("fixture")
        result = _passed_checker(proposal, concept, state)
        result.update(
            status=outcome,
            verified_window=None,
            decision_time_s=None,
            reason=f"fixture_{outcome}",
        )
        return result

    audits, facts = _process(
        [_proposal()], checkers={"fake_checker": checker}
    )

    assert facts == []
    assert audits[0]["verification_status"] in {
        "failed",
        "unsupported",
    }
    assert audits[0]["reasons"]


def test_fact_is_observation_only_and_traceable_to_recipe_and_state():
    _, facts = _process([_proposal()])
    fact = facts[0]

    assert fact["clip_id"] == "SNGS-9001"
    assert fact["state_source"] == "gsr_prediction"
    assert fact["proposed_window"] == {"start_s": 1.0, "end_s": 2.0}
    assert fact["verified_window"] == {"start_s": 1.1, "end_s": 2.1}
    assert fact["window_resolution"] == {
        "method": "deterministic_anchor_snap",
        "tolerance_s": 0.5,
    }
    assert fact["decision_time_s"] == 2.2
    assert fact["team_id"] == "team_0"
    assert fact["attack_direction"] == "left_to_right"
    assert fact["verified_claim_levels"] == ["observation"]
    assert fact["resolved_evidence"] == [
        {"metric": "fixture", "value": 1.0}
    ]
    assert fact["quality_flags"] == ["synthetic"]
    assert fact["provenance"]["proposal_id"] == "proposal_001"
    assert fact["provenance"]["recipe_version"] == 1


def test_no_registered_production_checker_means_no_fact():
    audits, facts = _process([_proposal()], checkers={})

    assert facts == []
    assert audits[0]["proposal_status"] == "accepted"
    assert audits[0]["verification_status"] == "not_run"
    assert "checker_not_registered" in audits[0]["reasons"]
