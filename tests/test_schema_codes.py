from pipeline.stage2_events.schema import EventSchema


def test_goal_kick_and_buildup_exist():
    schema = EventSchema()
    gk = schema.get_event("football.goal_kick")
    assert gk is not None
    assert gk.display_name_cn == "开大脚"
    assert 0 < gk.importance_base < 0.5
    bu = schema.get_event("football.buildup")
    assert bu is not None
    assert bu.display_name_cn == "控球推进"
    assert 0 < bu.importance_base < 0.3


def test_pattern_of_play_is_visual_not_computable():
    schema = EventSchema()
    assert "pattern_of_play" in schema.visual_tag_groups()
    assert "pattern_of_play" not in schema.computable_tag_groups()
    # geometric groups stay computable
    assert "pitch_zone" in schema.computable_tag_groups()
