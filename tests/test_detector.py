from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.detector import EventDetector


def _detect(predictions_file):
    return EventDetector(EventSchema(), fps=25).detect(str(predictions_file))


def test_same_team_switch_is_a_pass_at_kick_frame(predictions_file):
    events = _detect(predictions_file)
    passes = [e for e in events if e.event_code == "football.pass"]
    assert len(passes) == 1
    p = passes[0]
    assert p.player_jersey == "7" and p.player_team == "left"
    assert p.target_jersey == "10" and p.target_team == "left"
    # action-moment timestamp: passer's last-possession frame (#7 held through 6)
    assert p.frame_id == 6
    assert abs(p.timestamp_s - 6 / 25) < 1e-6
