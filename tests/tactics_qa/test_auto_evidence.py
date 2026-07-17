from pipeline.tactics_qa.auto_evidence import attacking_side, build_claim_evidence
from pipeline.tactics_qa.gsr_io import GsrClip


def test_attacking_side_or_abstain():
    runs = [{"team": "left", "start_s": 0, "end_s": 3},
            {"team": "right", "start_s": 3.5, "end_s": 12}]
    assert attacking_side(runs, {"start_s": 4, "end_s": 10}) == "right"
    tied = [{"team": "left", "start_s": 4, "end_s": 7},
            {"team": "right", "start_s": 7.2, "end_s": 10}]
    assert attacking_side(tied, {"start_s": 4, "end_s": 10}) is None


def test_sparse_clip_marks_unknowns():
    ball = {f: (0.0, 0.0) for f in range(10)}
    clip = GsrClip(25, 10, ball, {})
    evidence = build_claim_evidence(clip, "t:x", "fast_break_pattern",
                                    {"start_s": 5, "end_s": 15})
    assert evidence.delivery_kind is None
    assert evidence.corner_landing_zone is None
    assert evidence.source == "gsr_trajectory"
