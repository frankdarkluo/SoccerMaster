from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.detector import EventDetector
from pipeline.stage2_events.possession import resolve_team_by_track, possession_segments
from pipeline.stage2_events.types import FrameData


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


def test_cross_team_switch_is_interception_at_win_frame(predictions_file):
    events = _detect(predictions_file)
    ints = [e for e in events if e.event_code == "football.interception"]
    assert len(ints) == 1
    e = ints[0]
    assert e.player_jersey == "4" and e.player_team == "right"
    assert e.frame_id == 17  # winner's first-possession frame (action moment)


def test_no_phantom_interception_storm(predictions_file):
    events = _detect(predictions_file)
    # exactly one real cross-team switch; the old code produced ~9
    assert sum(e.event_code == "football.interception" for e in events) == 1
    assert sum(e.event_code == "football.pass" for e in events) == 1


def _carry_frames():
    frames = []
    for f in range(1, 11):
        bx = (f - 1) * 1.0  # carrier advances 9m over the segment
        players = [
            {"track_id": 7, "x": bx, "y": 0.0, "role": "player", "team": "left", "jersey": "7"},
            {"track_id": 3, "x": 3.0, "y": 0.5, "role": "player", "team": "right", "jersey": "3"},
        ]
        fd = FrameData(frame_id=f, ball_xy=(bx, 0.0))
        fd.players = players
        frames.append(fd)
    return frames


def test_dribble_on_long_carry_past_opponent():
    frames = _carry_frames()
    det = EventDetector(EventSchema(), fps=25)
    team = resolve_team_by_track(frames)
    segs = possession_segments(frames, team)
    dribbles = det._dribbles(segs, frames, team)
    assert len(dribbles) == 1
    assert dribbles[0].player_jersey == "7" and dribbles[0].player_team == "left"
