from pipeline.tactics_qa.agreement import claim_agreement
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent

WINDOW = {"start_s": 10.0, "end_s": 20.0}


def test_field_agreement_and_missing():
    hand = ClipEvidence("t:a", "fast_break_pattern",
                        [PossessionEvent(10, "attacking", "kickoff")],
                        delivery_kind="long_clearance")
    auto = ClipEvidence("t:a", "fast_break_pattern",
                        [PossessionEvent(10.2, "attacking", "kickoff")],
                        source="gsr_trajectory")
    result = claim_agreement(hand, auto, WINDOW)
    assert result["restart_at_origin"] == "match"
    assert result["delivery_kind"] == "auto_missing"
    assert result["has_controlled_regain"] == result["chaotic"] == "match"


def test_verdict_level_agreement():
    hand = ClipEvidence("t:b", "corner-near-far-post", corner_landing_zone="center")
    auto = ClipEvidence("t:b", "corner-near-far-post",
                        corner_landing_zone="near_post", source="gsr_trajectory")
    result = claim_agreement(hand, auto, WINDOW)
    assert result["corner_landing_zone"] == "mismatch"
    assert result["checker_verdicts_equal"] is False


def test_ambiguous_side_is_missing_not_mismatch():
    hand = ClipEvidence(
        "t:c", "fast_break_pattern",
        [PossessionEvent(10, "attacking", "controlled_recovery")])
    auto = ClipEvidence(
        "t:c", "fast_break_pattern", source="gsr_trajectory",
        notes="attacking side ambiguous: no possession events emitted")
    result = claim_agreement(hand, auto, WINDOW)
    assert result["has_controlled_regain"] == "auto_missing"
    assert result["chaotic"] == "auto_missing"


def test_restart_at_wrong_time_is_mismatch_not_missing():
    hand = ClipEvidence(
        "t:d", "fast_break_pattern",
        [PossessionEvent(10, "attacking", "kickoff")])
    auto = ClipEvidence(
        "t:d", "fast_break_pattern",
        [PossessionEvent(15, "attacking", "kickoff")], source="gsr_trajectory")
    assert claim_agreement(hand, auto, WINDOW)["restart_at_origin"] == "mismatch"
