from pipeline.tactics_qa.delivery_infer import classify_delivery, corner_landing_zone
from pipeline.tactics_qa.kinematics import Kick
from pipeline.tactics_qa.restart_infer import Restart


def test_delivery_classes():
    corner = Kick(6.2, (53.8, -36.6), 20.0)
    assert classify_delivery(corner, (48, -5), 1,
                             [Restart(6.2, corner.xy, "corner")]) == "corner_cross"
    assert classify_delivery(Kick(10, (40, -28), 18), (47, -2), 1, []) == "cross"
    assert classify_delivery(Kick(3, (-40, 5), 28), (10, 0), 1, []) == "long_clearance"
    assert classify_delivery(Kick(8, (0, 5), 12), (20, 3), 1, []) == "open_play_pass"
    assert classify_delivery(Kick(8, (0, 5), 12), None, 1, []) is None


def test_corner_landing_zones():
    assert corner_landing_zone(-36.6, (50, -5.2)) == "near_post"
    assert corner_landing_zone(-36.6, (50, 6)) == "far_post"
    assert corner_landing_zone(-36.6, (50, 0.5)) == "center"
    assert corner_landing_zone(-36.6, None) is None
