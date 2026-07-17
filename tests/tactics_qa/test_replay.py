from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent as E
from pipeline.tactics_qa.replay import replay


def test_replay_counts_flips_and_false_vetoes():
    review = [
        {"clip_uid": "t:a", "tactic_id": "fast_break_pattern", "verdict": "wrong",
         "window": {"start_s": 10.0, "end_s": 20.0}, "root_cause": "restart_delivery"},
        {"clip_uid": "t:b", "tactic_id": "fast_break_pattern", "verdict": "correct",
         "window": {"start_s": 10.0, "end_s": 20.0}, "root_cause": None},
    ]
    evidence = {
        ("t:a", "fast_break_pattern"): ClipEvidence(
            "t:a", "fast_break_pattern", [E(10.0, "attacking", "kickoff")]),
        ("t:b", "fast_break_pattern"): ClipEvidence(
            "t:b", "fast_break_pattern", [E(10.0, "attacking", "interception")]),
    }
    report = replay(review, evidence)
    stats = report["per_tactic"]["fast_break_pattern"]
    assert stats["n_claims"] == 2
    assert stats["precision_before"] == 0.5
    assert stats["precision_after"] == 1.0
    assert stats["flipped_errors"] == ["t:a"]
    assert stats["false_vetoes"] == []
