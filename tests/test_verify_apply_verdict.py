from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Event, Verdict
from pipeline.stage2_events.verify import apply_verdict


def _make_event(event_code: str) -> Event:
    schema = EventSchema()
    ev_def = schema.get_event(event_code)
    return Event(
        event_id="evt_test",
        timestamp_s=7.36,
        frame_id=184,
        event_code=event_code,
        display_name_en=ev_def.display_name_en,
        display_name_cn=ev_def.display_name_cn,
        importance=ev_def.importance_base,
        player_jersey="7",
        player_team="left",
        confidence=0.8,
    )


def test_reject_with_no_correction_is_dropped():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(verdict="reject", corrected_event_code=None, reason="no action")

    result = apply_verdict(event, verdict, schema)

    assert result is None


def test_reject_with_valid_family_correction_is_reclassified_and_kept():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(
        verdict="reject",
        corrected_event_code="football.pass",
        reason="attacking free kick delivered into the box, not a defensive clearance",
    )

    result = apply_verdict(event, verdict, schema)

    assert result is not None
    assert result.event_code == "football.pass"
    assert result.tags.get("verified") == "true"
    assert result.tags.get("reclassified") == "true"


def test_reject_with_correction_outside_family_is_dropped():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(
        verdict="reject",
        corrected_event_code="football.goal",
        reason="not actually a duel action at all",
    )

    result = apply_verdict(event, verdict, schema)

    assert result is None


def test_reject_with_correction_same_as_original_is_dropped():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(
        verdict="reject",
        corrected_event_code="football.clearance",
        reason="model echoed the same code without real reclassification",
    )

    result = apply_verdict(event, verdict, schema)

    assert result is None


def test_uncertain_is_always_dropped_even_with_correction():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(
        verdict="uncertain",
        corrected_event_code="football.pass",
        reason="ambiguous frames",
    )

    result = apply_verdict(event, verdict, schema)

    assert result is None


def test_confirm_still_applies_correction_as_before():
    schema = EventSchema()
    event = _make_event("football.clearance")
    verdict = Verdict(
        verdict="confirm",
        corrected_event_code="football.pass",
        reason="confirmed pass",
    )

    result = apply_verdict(event, verdict, schema)

    assert result is not None
    assert result.event_code == "football.pass"
