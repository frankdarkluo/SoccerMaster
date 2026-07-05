from pipeline.stage2_events.types import Event, Verdict
from pipeline.stage2_events.verify import apply_verdict, build_verify_prompt, parse_verdict, normalize_outcome
from pipeline.stage2_events.schema import EventSchema


def test_parse_verdict_extracts_json():
    raw = 'blah {"verdict": "confirm", "outcome": "success"} trailing'
    v = parse_verdict(raw)
    assert v.verdict == "confirm" and v.outcome == "success"


def test_parse_verdict_malformed_is_uncertain():
    v = parse_verdict("the model rambled with no json")
    assert v.verdict == "uncertain"


def test_parse_verdict_bad_json_is_uncertain():
    v = parse_verdict('{"verdict": "confirm", oops}')  # invalid json
    assert v.verdict == "uncertain"


def test_normalize_outcome_synonyms():
    assert normalize_outcome("succeeded") == "success"
    assert normalize_outcome("intercepted") is None  # unknown stays None
    assert normalize_outcome("failed") == "failure"


def _ev(code="football.pass", **kw):
    d = dict(event_id="evt_001", timestamp_s=4.0, frame_id=100, event_code=code,
             display_name_en="Pass", display_name_cn="传球", importance=0.15,
             player_jersey="7", player_team="left", confidence=0.7)
    d.update(kw)
    return Event(**d)


def test_reject_drops_event():
    assert apply_verdict(_ev(), Verdict(verdict="reject"), EventSchema()) is None


def test_uncertain_keeps_at_half_confidence():
    e = apply_verdict(_ev(confidence=0.8), Verdict(verdict="uncertain"), EventSchema())
    assert e is not None and e.confidence == 0.4 and e.tags["verified"] == "uncertain"


def test_confirm_failure_halves_importance_and_tags_outcome():
    e = apply_verdict(_ev(importance=0.4), Verdict(verdict="confirm", outcome="failure"),
                      EventSchema())
    assert e.tags["outcome"] == "failure" and e.importance == 0.2
    assert e.tags["verified"] == "true"


def test_family_constrained_reclassify_allowed():
    e = apply_verdict(_ev(code="football.interception"),
                      Verdict(verdict="confirm", corrected_event_code="football.pass"),
                      EventSchema())
    assert e.event_code == "football.pass"
    assert e.display_name_cn == "传球"        # re-pulled from schema


def test_out_of_family_reclassify_ignored():
    e = apply_verdict(_ev(code="football.pass"),
                      Verdict(verdict="confirm", corrected_event_code="football.goal"),
                      EventSchema())
    assert e.event_code == "football.pass"   # goal not in duel family -> ignored


def test_confirm_updates_actor_attribution():
    e = apply_verdict(_ev(), Verdict(verdict="confirm", actor_jersey="11", actor_team="right"),
                      EventSchema())
    assert e.player_jersey == "11" and e.player_team == "right"


def test_pass_receiver_updates_target_to_actor_team():
    e = apply_verdict(_ev(), Verdict(verdict="confirm", receiver_jersey="9"), EventSchema())
    assert e.target_jersey == "9" and e.target_team == "left"


def test_prompt_mentions_outcome_and_reclassify_options():
    prompt = build_verify_prompt(_ev(code="football.interception"))
    assert "outcome" in prompt and "corrected_event_code" in prompt
    # only in-family retypes offered
    assert "football.clearance" in prompt and "football.goal" not in prompt


from pipeline.stage2_events.detector import EventDetector
from pipeline.stage2_events.verify import verify_events, cleanup_verify_artifacts


def test_verify_events_end_to_end_with_mock(tmp_path, predictions_file, homography_file,
                                             frames_dir, mock_adapter):
    events = EventDetector(EventSchema(), fps=25).detect(str(predictions_file))
    adapter = mock_adapter(script={
        "拦截": '{"verdict": "reject", "reason": "no"}',    # drop interception
        "传球": '{"verdict": "confirm", "outcome": "success"}',  # keep pass
    })
    out_dir = tmp_path / "out"
    verified, audit = verify_events(
        events, str(predictions_file), frames_dir, out_dir, adapter,
        str(homography_file), fps=25, window_s=0.2, force=False)
    codes = [e.event_code for e in verified]
    assert "football.pass" in codes
    assert "football.interception" not in codes         # rejected -> dropped
    assert (out_dir / "events_verification.json").exists()
    # audit records verdict per checked event
    assert any(a["verdict"] == "reject" and a["kept"] is False for a in audit)
    # audit is self-describing: actor jersey/team present on every checked row
    assert all("player_jersey" in a and "player_team" in a for a in audit)
    int_row = next(a for a in audit if a["event_code"] == "football.interception")
    assert int_row["player_team"] in ("left", "right")


def test_verify_cache_skips_second_call(tmp_path, predictions_file, homography_file,
                                        frames_dir, mock_adapter):
    events = [e for e in EventDetector(EventSchema(), 25).detect(str(predictions_file))
              if e.event_code == "football.pass"]
    adapter = mock_adapter(default='{"verdict": "confirm"}')
    out_dir = tmp_path / "out"
    verify_events(events, str(predictions_file), frames_dir, out_dir, adapter,
                  str(homography_file), fps=25, window_s=0.2)
    n_first = len(adapter.calls)
    verify_events(events, str(predictions_file), frames_dir, out_dir, adapter,
                  str(homography_file), fps=25, window_s=0.2, force=False)
    assert len(adapter.calls) == n_first   # cached -> no new adapter calls
