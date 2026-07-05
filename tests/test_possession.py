from pipeline.stage2_events.detector import load_frames
from pipeline.stage2_events.possession import resolve_team_by_track, possession_segments


def test_team_by_track_majority_vote_beats_flicker(predictions_file):
    frames = load_frames(str(predictions_file))
    team = resolve_team_by_track(frames)
    # #10 flickered to "right" on exactly one frame; majority is "left".
    assert team[10] == "left"
    assert team[7] == "left"
    assert team[4] == "right"
    # referee has no team vote
    assert team.get(50) is None


def test_possession_segments_are_stable_and_exclude_referee(predictions_file):
    frames = load_frames(str(predictions_file))
    team = resolve_team_by_track(frames)
    segs = possession_segments(frames, team)
    # exactly three stable holders in order: #7 left, #10 left, #4 right
    holders = [(s.track_id, s.team) for s in segs]
    assert holders == [(7, "left"), (10, "left"), (4, "right")]
    # #7 kick frame = last frame it held the ball (action moment for the pass)
    assert segs[0].end_fid == 6
    # #4 wins possession at first stable frame (interception action moment)
    assert segs[2].start_fid == 17
    # referee (track 50) is never a holder
    assert all(s.track_id != 50 for s in segs)
