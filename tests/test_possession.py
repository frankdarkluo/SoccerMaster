from pipeline.stage2_events.detector import load_frames
from pipeline.stage2_events.possession import resolve_team_by_track


def test_team_by_track_majority_vote_beats_flicker(predictions_file):
    frames = load_frames(str(predictions_file))
    team = resolve_team_by_track(frames)
    # #10 flickered to "right" on exactly one frame; majority is "left".
    assert team[10] == "left"
    assert team[7] == "left"
    assert team[4] == "right"
    # referee has no team vote
    assert team.get(50) is None
