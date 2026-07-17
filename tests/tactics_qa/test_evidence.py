import pytest

from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent, from_dict


def test_round_trip():
    evidence = ClipEvidence(
        clip_uid="soccernetgs:SNGS-001",
        tactic_id="fast_break_pattern",
        events=[PossessionEvent(t=1.0, team="attacking", kind="kickoff")],
        delivery_kind="long_clearance",
        corner_landing_zone=None,
        source="human_review_text",
    )
    data = evidence.to_dict()
    assert from_dict(data) == evidence
    assert data["events"][0] == {"t": 1.0, "team": "attacking", "kind": "kickoff"}


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        PossessionEvent(t=0.0, team="attacking", kind="bicycle_kick")
