import json as _json

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.detector import EventDetector, dedup_events, write_events_json
from pipeline.stage2_events.possession import resolve_team_by_track, possession_segments
from pipeline.stage2_events.types import FrameData
from pipeline.stage2_events.detector import compose_assists


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


def _shot_frames():
    # #9 left holds ball near right goal (x≈48) frames 1-4, then ball rockets to
    # x=52.5 (goal line) over frames 5-6: high speed + crosses line.
    frames = []
    seq = {1: 47.0, 2: 47.5, 3: 48.0, 4: 48.0, 5: 50.5, 6: 52.5}
    for f in range(1, 7):
        bx = seq[f]
        fd = FrameData(frame_id=f, ball_xy=(bx, 0.0))
        fd.players = [
            {"track_id": 9, "x": min(bx, 48.0), "y": 0.0, "role": "player",
             "team": "left", "jersey": "9"},
        ]
        frames.append(fd)
    return frames


def test_shot_attributes_shooter_and_goal_attributes_scorer():
    from pipeline.stage2_events.schema import EventSchema
    frames = _shot_frames()
    det = EventDetector(EventSchema(), fps=25, shot_speed_threshold=10.0)
    team = resolve_team_by_track(frames)
    segs = possession_segments(frames, team)
    ball_pos = {f.frame_id: f.ball_xy for f in frames}
    vel = det._velocities(ball_pos)
    shots = det._shots(vel, ball_pos, segs)
    goals = det._goals(ball_pos, segs)
    assert shots and shots[0].player_jersey is None     # emit shot even without last holder
    assert any(s.player_jersey == "9" for s in shots)   # shooter attributed once holder exists
    assert goals and goals[0].player_jersey == "9"      # scorer attributed (not null)


def _clearance_frames():
    # ball deep in left's own half (x≈-40), a defender clears it fast toward halfway.
    frames = []
    seq = {1: -40.0, 2: -40.0, 3: -40.0, 4: -34.0, 5: -28.0}  # fast move away from own goal
    for f in range(1, 6):
        bx = seq[f]
        fd = FrameData(frame_id=f, ball_xy=(bx, 0.0))
        fd.players = [{"track_id": 5, "x": max(bx, -40.0), "y": 0.0, "role": "player",
                       "team": "left", "jersey": "5"}]
        frames.append(fd)
    return frames


def test_clearance_is_attributed():
    from pipeline.stage2_events.schema import EventSchema
    frames = _clearance_frames()
    det = EventDetector(EventSchema(), fps=25)
    team = resolve_team_by_track(frames)
    segs = possession_segments(frames, team, min_frames=3)
    ball_pos = {f.frame_id: f.ball_xy for f in frames}
    vel = det._velocities(ball_pos)
    clears = det._clearances(vel, ball_pos, segs)
    assert clears
    assert clears[0].player_jersey == "5"   # was null in the old code


def test_dedup_keeps_higher_importance_within_gap():
    from pipeline.stage2_events.schema import EventSchema
    det = EventDetector(EventSchema(), fps=25)
    a = det._make("football.shoot", 100, confidence=0.95)   # t=4.0
    a.importance = 0.2
    b = det._make("football.shoot", 105, confidence=0.1)    # t=4.2 (< 1.0s gap)
    b.importance = 0.9
    c = det._make("football.pass", 50, confidence=0.8)      # t=2.0
    kept = dedup_events([a, b, c])
    assert [e.timestamp_s for e in kept] == sorted(e.timestamp_s for e in kept)
    assert len([e for e in kept if e.event_code == "football.shoot"]) == 1
    assert next(e for e in kept if e.event_code == "football.shoot") is b


def test_write_raw_dump(tmp_path, predictions_file):
    from pipeline.stage2_events.schema import EventSchema
    events = EventDetector(EventSchema(), fps=25).detect(str(predictions_file))
    out = tmp_path / "events_detected.json"
    write_events_json(events, out, {"source": "FIXT", "fps": 25})
    data = _json.loads(out.read_text())
    assert data["schema_version"] == "v3-20260319"
    assert {e["event_code"] for e in data["events"]} >= {"football.pass", "football.interception"}


def test_assist_composed_from_pass_then_goal():
    from pipeline.stage2_events.schema import EventSchema
    det = EventDetector(EventSchema(), fps=25)
    p = det._make("football.pass", 100, player_jersey="7", player_team="left",
                  target_jersey="9", target_team="left")   # t=4.0
    g = det._make("football.goal", 150, player_jersey="9", player_team="left")  # t=6.0
    out = compose_assists([p, g])
    assists = [e for e in out if e.event_code == "football.assist"]
    assert len(assists) == 1
    assert assists[0].player_jersey == "7"       # passer credited


def test_no_assist_when_pass_receiver_did_not_score():
    from pipeline.stage2_events.schema import EventSchema
    det = EventDetector(EventSchema(), fps=25)
    p = det._make("football.pass", 100, player_jersey="7", player_team="left",
                  target_jersey="8", target_team="left")
    g = det._make("football.goal", 150, player_jersey="9", player_team="left")
    out = compose_assists([p, g])
    assert not [e for e in out if e.event_code == "football.assist"]
