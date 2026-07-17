from pipeline.tactics_qa.kinematics import ball_speeds, dead_ball_periods, detect_kicks


def stationary_then_kick():
    ball = {f: (50.0, -33.0) for f in range(50)}
    ball.update({f: (50.0 - i, -33.0 + i * 0.4)
                 for i, f in enumerate(range(50, 75))})
    return ball


def test_speeds_kick_and_dead_period():
    ball = stationary_then_kick()
    speeds = ball_speeds(ball, 25)
    assert speeds[10] < 0.5 and speeds[60] > 15.0
    kicks = detect_kicks(ball, 25)
    assert len(kicks) == 1 and 1.8 <= kicks[0].t <= 2.4
    assert abs(kicks[0].xy[0] - 50.0) < 3.5 and kicks[0].speed_mps > 8.0
    periods = dead_ball_periods(ball, 25)
    assert len(periods) == 1 and periods[0][0] <= 0.2 and 1.8 <= periods[0][1] <= 2.2


def test_gap_does_not_fabricate_kick():
    ball = {f: (0.0, 0.0) for f in range(25)}
    ball.update({f: (30.0, 0.0) for f in range(50, 75)})
    assert detect_kicks(ball, 25) == []


def test_empty_track():
    assert ball_speeds({}, 25) == {}
    assert dead_ball_periods({}, 25) == []
