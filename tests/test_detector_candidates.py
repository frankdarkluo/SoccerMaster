from pipeline.stage2_events.detector import detect_candidates, load_frames
from pipeline.stage2_events.possession import (
    possession_segments,
    resolve_role_by_track,
    resolve_team_by_track,
)
from pipeline.stage2_events.types import FrameData


def _candidates(path):
    frames = load_frames(str(path))
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    return detect_candidates(frames, segs, team, role, fps=25)


def test_role_by_track_majority_vote(gk_boot_predictions):
    frames = load_frames(str(gk_boot_predictions))
    role = resolve_role_by_track(frames)
    assert role[1] == "goalkeeper"
    assert role[9] == "player"


def test_gk_boot_yields_kick_candidate_with_gk_facts(gk_boot_predictions):
    cands = _candidates(gk_boot_predictions)
    kicks = [c for c in cands if "kick" in c.signals]
    assert kicks, "keeper's boot must produce a kick candidate"
    k = kicks[0]
    assert k.role == "goalkeeper" and k.jersey == "1" and k.team == "left"
    assert k.ball_speed_mps and k.ball_speed_mps >= 8.0
    assert k.next_holder and k.next_holder["jersey"] == "9"
    assert not hasattr(k, "event_code")


def test_cross_team_switch_yields_possession_win(pass_intercept_predictions):
    cands = _candidates(pass_intercept_predictions)
    wins = [c for c in cands if "possession_win" in c.signals]
    assert len(wins) == 1
    w = wins[0]
    assert w.jersey == "4" and w.team == "right"
    assert w.prev_holder and w.prev_holder["team"] == "left"


def test_same_team_switch_yields_kick_not_win(pass_intercept_predictions):
    cands = _candidates(pass_intercept_predictions)
    kicks = [c for c in cands if "kick" in c.signals and c.jersey == "7"]
    assert kicks
    assert kicks[0].next_holder["jersey"] == "10"


def _pressure_frames():
    """#3 right sits 2m from holder #7 left for 15 frames."""
    frames = []
    for f in range(1, 21):
        fd = FrameData(frame_id=f, ball_xy=(0.0, 0.0))
        fd.players = [
            {"track_id": 7, "x": 0.0, "y": 0.0, "role": "player", "team": "left", "jersey": "7"},
            {"track_id": 3, "x": 2.0, "y": 0.0, "role": "player", "team": "right", "jersey": "3"},
        ]
        frames.append(fd)
    return frames


def test_sustained_pressure_yields_pressure_candidate():
    frames = _pressure_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    pressures = [c for c in cands if "pressure" in c.signals]
    assert pressures
    p = pressures[0]
    assert p.jersey == "3" and p.team == "right"
    assert p.prev_holder and p.prev_holder["jersey"] == "7"


def _carry_frames():
    """#7 left carries 9m with opponent #3 nearby."""
    frames = []
    for f in range(1, 11):
        bx = (f - 1) * 1.0
        fd = FrameData(frame_id=f, ball_xy=(bx, 0.0))
        fd.players = [
            {"track_id": 7, "x": bx, "y": 0.0, "role": "player", "team": "left", "jersey": "7"},
            {"track_id": 3, "x": 3.0, "y": 0.5, "role": "player", "team": "right", "jersey": "3"},
        ]
        frames.append(fd)
    return frames


def test_long_carry_past_opponent_yields_carry_candidate():
    frames = _carry_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    assert [c for c in cands if "carry" in c.signals]


def _goal_line_frames():
    """Ball crosses the +x goal line inside the posts."""
    frames = []
    seq = {1: 47.0, 2: 47.0, 3: 47.0, 4: 50.0, 5: 52.4}
    for f in range(1, 6):
        fd = FrameData(frame_id=f, ball_xy=(seq[f], 0.0))
        fd.players = [
            {"track_id": 9, "x": 47.0, "y": 0.0, "role": "player", "team": "left", "jersey": "9"}
        ]
        frames.append(fd)
    return frames


def test_goal_line_crossing_yields_geometry_candidate():
    frames = _goal_line_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    assert [c for c in cands if "goal_line" in c.signals]
