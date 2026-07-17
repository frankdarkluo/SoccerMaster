from pipeline.tactics_qa.kinematics import Kick
from pipeline.tactics_qa.restart_infer import classify_restart_location, infer_restarts


def test_location_classification():
    assert classify_restart_location((0.5, -0.3)) == "kickoff"
    assert classify_restart_location((49.0, 3.0)) == "goal_kick"
    assert classify_restart_location((-49.5, -5.0)) == "goal_kick"
    assert classify_restart_location((53.8, -36.6)) == "corner"
    assert classify_restart_location((10.0, -34.5)) == "throw_in"
    assert classify_restart_location((20.0, 10.0)) == "free_kick"
    assert classify_restart_location((60.0, 0.0)) is None


def test_restart_requires_recent_dead_ball():
    kicks = [Kick(6.2, (53.8, -36.6), 20.0)]
    restarts = infer_restarts(kicks, [(3.0, 6.1)])
    assert len(restarts) == 1 and restarts[0].kind == "corner"
    assert infer_restarts(kicks, []) == []
    assert infer_restarts(kicks, [(0.0, 2.0)]) == []
