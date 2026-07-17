"""Real-data anchor for the SNGS-116 corner around 6.2 seconds."""
import os
import pytest

PRED = "outputs/SNGS-116/predictions.json"
pytestmark = pytest.mark.skipif(not os.path.exists(PRED),
                                reason="SNGS-116 GSR outputs not present")


def test_corner_restart_detected():
    from pipeline.tactics_qa.gsr_io import load_gsr
    from pipeline.tactics_qa.kinematics import dead_ball_periods, detect_kicks
    from pipeline.tactics_qa.restart_infer import infer_restarts
    clip = load_gsr(PRED)
    restarts = infer_restarts(detect_kicks(clip.ball, clip.fps),
                              dead_ball_periods(clip.ball, clip.fps))
    assert any(r.kind == "corner" and 4 <= r.t <= 8.5 for r in restarts), restarts


def test_corner_landing_zone_is_near_post():
    from pipeline.tactics_qa.auto_evidence import build_claim_evidence
    from pipeline.tactics_qa.gsr_io import load_gsr
    evidence = build_claim_evidence(
        load_gsr(PRED), "soccernetgs:SNGS-116", "corner-near-far-post",
        {"start_s": 3.0, "end_s": 9.0})
    assert evidence.delivery_kind == "corner_cross"
    assert evidence.corner_landing_zone == "near_post", evidence.notes
