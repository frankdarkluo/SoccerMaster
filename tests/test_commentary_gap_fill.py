import json

import pytest

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage4_commentary.prompt_builder import build_commentary_prompt


@pytest.fixture
def schema():
    return EventSchema()


def _write_events(tmp_path, events, duration_s=30.0):
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps({
            "video_info": {"duration_s": duration_s},
            "schema_version": "1.0",
            "events": events,
        }),
        encoding="utf-8",
    )
    return path


def _write_audit(tmp_path, audit):
    path = tmp_path / "events_verification.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def test_no_audit_path_behaves_as_before(tmp_path, schema):
    events_path = _write_events(tmp_path, [
        {"event_id": "evt_001", "timestamp_s": 6.2, "event_code": "football.pass"},
    ])
    prompt = build_commentary_prompt(events_path, schema, ["en"])
    assert "[Unconfirmed Activity]" not in prompt


def test_short_gap_does_not_surface_rejected_events(tmp_path, schema):
    events_path = _write_events(tmp_path, [
        {"event_id": "evt_001", "timestamp_s": 6.2, "event_code": "football.pass"},
        {"event_id": "evt_057", "timestamp_s": 10.0, "event_code": "football.shoot"},
    ])
    audit_path = _write_audit(tmp_path, [
        {
            "event_id": "evt_089", "event_code": "football.clearance", "timestamp_s": 8.0,
            "player_jersey": "", "player_team": "left", "verdict": "reject",
            "corrected_event_code": "football.pass", "reason": "actually a pass", "kept": False,
        },
    ])
    prompt = build_commentary_prompt(
        events_path, schema, ["en"], verification_audit_path=audit_path, gap_threshold_s=8.0,
    )
    assert "[Unconfirmed Activity]" not in prompt


def test_long_gap_surfaces_rejected_event_with_correction(tmp_path, schema):
    events_path = _write_events(tmp_path, [
        {"event_id": "evt_037", "timestamp_s": 7.28, "event_code": "football.shoot"},
        {"event_id": "evt_057", "timestamp_s": 23.48, "event_code": "football.shoot"},
    ])
    audit_path = _write_audit(tmp_path, [
        {
            "event_id": "evt_006", "event_code": "football.interception", "timestamp_s": 9.64,
            "player_jersey": "", "player_team": "right", "verdict": "reject",
            "corrected_event_code": None, "reason": "no interception attempt, ball fully controlled", "kept": False,
        },
        {
            "event_id": "evt_wedge", "event_code": "football.dribble", "timestamp_s": 15.0,
            "player_jersey": "9", "player_team": "left", "verdict": "reject",
            "corrected_event_code": "football.pass", "reason": "looked more like a pass attempt", "kept": False,
        },
    ])
    prompt = build_commentary_prompt(
        events_path, schema, ["en"], verification_audit_path=audit_path, gap_threshold_s=8.0,
    )
    assert "[Unconfirmed Activity]" in prompt
    assert "evt_wedge" in prompt
    assert "evt_006" not in prompt


def test_kept_events_are_never_surfaced_as_unconfirmed(tmp_path, schema):
    events_path = _write_events(tmp_path, [
        {"event_id": "evt_037", "timestamp_s": 7.28, "event_code": "football.shoot"},
        {"event_id": "evt_057", "timestamp_s": 23.48, "event_code": "football.shoot"},
    ])
    audit_path = _write_audit(tmp_path, [
        {
            "event_id": "evt_004", "event_code": "football.pass", "timestamp_s": 15.0,
            "player_jersey": "1", "player_team": "right", "verdict": "reject",
            "corrected_event_code": "football.clearance", "reason": "goalkeeper clearance", "kept": True,
        },
    ])
    prompt = build_commentary_prompt(
        events_path, schema, ["en"], verification_audit_path=audit_path, gap_threshold_s=8.0,
    )
    assert "[Unconfirmed Activity]" not in prompt
