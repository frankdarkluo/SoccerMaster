from pipeline.stage2_events.types import Candidate, Classification


def test_candidate_to_dict_roundtrip():
    c = Candidate(
        candidate_id="cand_001", frame_id=250, timestamp_s=10.0,
        signals=["kick", "possession_change"], strength=0.75,
        track_id=1, jersey="1", team="left", role="goalkeeper",
        ball_speed_mps=13.2, ball_xy=(-44.0, 0.0), ball_direction="toward_opponent_goal",
        prev_holder=None,
        next_holder={"track_id": 9, "jersey": "9", "team": "left",
                     "role": "player", "start_fid": 271},
    )
    d = c.to_dict()
    assert d["candidate_id"] == "cand_001"
    assert d["signals"] == ["kick", "possession_change"]
    assert d["role"] == "goalkeeper"
    assert d["next_holder"]["jersey"] == "9"
    assert d["ball_speed_mps"] == 13.2


def test_classification_defaults():
    cls = Classification()
    assert cls.action == "none"
    assert cls.outcome is None
    assert cls.tags == {}
