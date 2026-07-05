import json

from pipeline.config import PipelineConfig
from pipeline.run import run_stage2
from pipeline.stage2_events.detector import (
    EventDetector,
    compose_assists,
    dedup_events,
    load_frames,
    write_events_json,
)
from pipeline.stage2_events.enricher import enrich_events
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.verify import verify_events


def test_full_stage2_flow_detect_verify_compose_enrich(
    tmp_path,
    predictions_file,
    homography_file,
    frames_dir,
    mock_adapter,
):
    frames = load_frames(str(predictions_file))
    raw = EventDetector(EventSchema(), 25).detect(str(predictions_file))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_events_json(raw, out_dir / "events_detected.json", {"source": "FIXT", "fps": 25})

    adapter = mock_adapter(default='{"verdict": "confirm", "outcome": "success"}')
    verified, _ = verify_events(
        raw,
        str(predictions_file),
        frames_dir,
        out_dir,
        adapter,
        str(homography_file),
        fps=25,
        window_s=0.2,
    )

    final = enrich_events(dedup_events(compose_assists(verified)), frames)
    write_events_json(final, out_dir / "events.json", {"source": "FIXT", "fps": 25})

    data = json.loads((out_dir / "events.json").read_text())
    codes = {e["event_code"] for e in data["events"]}
    assert "football.pass" in codes
    pass_ev = next(e for e in data["events"] if e["event_code"] == "football.pass")
    assert pass_ev["tags"].get("verified") == "true"


def test_run_stage2_wires_config_verify_and_min_gap(
    tmp_path,
    predictions_file,
    homography_file,
    frames_dir,
    mock_adapter,
    monkeypatch,
):
    import pipeline.run as run_module
    import pipeline.stage2_events.detector as detector_module

    original_detector = detector_module.EventDetector
    seen = {}

    class SpyEventDetector(original_detector):
        def __init__(self, schema, fps, **kwargs):
            seen["fps"] = fps
            seen["kwargs"] = dict(kwargs)
            super().__init__(schema, fps, **kwargs)

    monkeypatch.setattr(detector_module, "EventDetector", SpyEventDetector)
    monkeypatch.setattr(
        run_module,
        "_build_verify_adapter",
        lambda config: mock_adapter(default='{"verdict": "confirm", "outcome": "success"}'),
    )

    output_dir = tmp_path / "out"
    config = PipelineConfig(
        clip_dir=tmp_path,
        output_dir=output_dir,
        existing_predictions_json=predictions_file,
        existing_homography_json=homography_file,
        fps=25,
        min_event_gap_s=2.5,
        verify_events=True,
        cleanup_verify_temp=False,
    )

    count = run_stage2(config)

    assert seen["fps"] == 25
    assert seen["kwargs"]["min_gap_s"] == 2.5
    assert seen["kwargs"]["shot_speed_threshold"] == config.ball_speed_shot_threshold_mps
    assert count > 0
    assert (output_dir / "events_detected.json").exists()
    assert (output_dir / "events.json").exists()
    assert (output_dir / "events_verification.json").exists()

    events_data = json.loads((output_dir / "events.json").read_text())
    assert "football.pass" in {event["event_code"] for event in events_data["events"]}

    audit_data = json.loads((output_dir / "events_verification.json").read_text())
    assert audit_data
