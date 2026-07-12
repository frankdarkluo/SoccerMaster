import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline.stage4_effects.render import load_effect_events
from pipeline.stage4_effects.run import replace_final_video

from pipeline.stage2b.digest import build_tracking_digest, load_frames
from pipeline.stage2b.events import event_prompt_menu, get_event
from pipeline.stage2b import generate, run as stage2b_run
from pipeline.stage2b.generate import ark_chat, observe_direct, verify_event_window
from pipeline.stage2b.run import run_stage2b
from pipeline.config import PipelineConfig
from pipeline.run import run_pipeline
from pipeline.relations.query import predicate_passes, resolve_query
from pipeline.stage2b.hybrid import audit_commentary
from pipeline.stage3_tts import run as stage3_tts_run
from pipeline.stage3_tts.cosyvoice import CosyVoiceSynthesizer, MODEL_ARTIFACTS


def write_predictions(path: Path) -> Path:
    path.write_text(json.dumps({
        "images": [{"image_id": "000001", "file_name": "000001.jpg"}],
        "annotations": [
            {
                "image_id": "000001", "track_id": 1,
                "bbox_pitch": {"x_bottom_middle": -44.0, "y_bottom_middle": 0.0},
                "attributes": {"role": "goalkeeper", "team": "left", "jersey": "1"},
            },
            {
                "image_id": "000001", "track_id": 99,
                "bbox_pitch": {"x_bottom_middle": -43.8, "y_bottom_middle": 0.0},
                "attributes": {"role": "ball"},
            },
        ],
    }), encoding="utf-8")
    return path





@pytest.mark.parametrize("preloaded", [False, True])
def test_cosyvoice_blocks_wetext_only_while_constructing_model(
        tmp_path, monkeypatch, preloaded):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for artifact in MODEL_ARTIFACTS:
        (model_dir / artifact).touch()

    cosyvoice = ModuleType("cosyvoice")
    cli = ModuleType("cosyvoice.cli")
    cosyvoice_module = ModuleType("cosyvoice.cli.cosyvoice")
    original_wetext = ModuleType("wetext") if preloaded else None
    if preloaded:
        monkeypatch.setitem(sys.modules, "wetext", original_wetext)
        monkeypatch.setitem(sys.modules, "wetext.preloaded", ModuleType("wetext.preloaded"))
    else:
        monkeypatch.delitem(sys.modules, "wetext", raising=False)
    before = {name: module for name, module in sys.modules.items()
              if name == "wetext" or name.startswith("wetext.")}

    class FakeAutoModel:
        def __init__(self, model_dir):
            assert "wetext" in sys.modules
            assert sys.modules["wetext"] is None
            sys.modules["wetext.constructed"] = ModuleType("wetext.constructed")
            self.model_dir = model_dir

    cosyvoice_module.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "cosyvoice", cosyvoice)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", cosyvoice_module)

    synthesizer = CosyVoiceSynthesizer(model_dir)

    after = {name: module for name, module in sys.modules.items()
             if name == "wetext" or name.startswith("wetext.")}
    assert synthesizer.model.model_dir == str(model_dir)
    assert after == before
    if preloaded:
        assert sys.modules["wetext"] is original_wetext


def test_pipeline_stages_external_predictions_for_stage2b(tmp_path, monkeypatch):
    external = tmp_path / "external.json"
    external.write_bytes(b"selected predictions")
    output = tmp_path / "output"
    output.mkdir()
    (output / "predictions.json").write_bytes(b"stale predictions")

    def fake_stage2b(output_dir, clip_dir, **kwargs):
        assert (output_dir / "predictions.json").read_bytes() == external.read_bytes()
        return output_dir / "comments" / "commentary.json"

    monkeypatch.setattr("pipeline.stage2b.run.run_stage2b", fake_stage2b)
    result = run_pipeline(PipelineConfig(
        output_dir=output, clip_dir=tmp_path,
        existing_predictions_json=external,
            languages=[],
    ))
    assert result == output / "comments" / "commentary.json"


def _stub_event_verification(monkeypatch):
    monkeypatch.setattr(stage2b_run, "verify_event_window", lambda *args, **kwargs: {
        "event_code": "football.corner", "midpoint_s": 2.5,
        "player_team": "left", "player_jersey": "", "outcome": "corner_taken",
        "directly_visible": True, "disagreements": [],
    })


def test_stage2b_offline_writes_comments_contract(tmp_path, monkeypatch):
    output = tmp_path / "SNGS-116"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    reply = json.dumps({
        "events": [{
            "event_id": "evt_001", "start_s": 0.0, "end_s": 5.0,
            "event_code": "football.corner", "player_team": "left",
            "player_jersey": "", "actors": ["left_team"],
            "outcome": "corner_taken", "confidence": "medium",
            "confidence_reasons": ["directly_visible"],
            "suggested_wording_zh": "左侧球队准备主罚角球。",
            "suggested_wording_en": "The left team prepares the corner.",
            "energy": "engaged",
        }],
        "commentary": [{
            "kind": "event", "timestamp_s": 0.0, "end_s": 5.0,
            "text_zh": "左侧球队准备主罚角球。",
            "text_en": "The left team prepares the corner.",
            "fallback_text_zh": "左侧球队主罚角球。",
            "fallback_text_en": "The left team takes the corner.",
            "energy": "engaged", "events_referenced": ["evt_001"],
        }],
    })
    _stub_event_verification(monkeypatch)
    result = run_stage2b(
        output, tmp_path, mode="direct", force=True,
        call=lambda prompt, **kwargs: reply, duration_s=30.0,
    )
    assert result == output / "comments" / "commentary.json"
    assert (output / "comments" / "events.json").is_file()
    assert (output / "comments" / "event_spine.json").is_file()
def test_stage2b_catalog_and_digest_smoke(tmp_path):
    predictions = write_predictions(tmp_path / "predictions.json")
    frames = load_frames(predictions)
    assert frames[0].players[0]["jersey"] == "1"
    assert get_event("football.corner").importance_base >= 0.35
    assert "football.corner" in event_prompt_menu()
    digest = build_tracking_digest(predictions, fps=25.0)
    assert "left" in digest
    assert "#1" in digest


def test_direct_observer_preserves_corner():
    reply = json.dumps({
        "events": [{
            "event_id": "evt_001", "start_s": 0.0, "end_s": 5.0,
            "event_code": "football.corner", "player_team": "left",
            "player_jersey": "", "actors": ["left_team"],
            "outcome": "corner_taken", "confidence": "medium",
            "confidence_reasons": ["directly_visible"],
            "suggested_wording_zh": "左侧球队准备主罚角球。",
            "suggested_wording_en": "The left team prepares the corner.",
            "energy": "engaged",
        }],
        "commentary": [{
            "kind": "event", "timestamp_s": 0.0, "end_s": 5.0,
            "text_zh": "左侧球队准备主罚角球。",
            "text_en": "The left team prepares the corner.",
            "fallback_text_zh": "左侧球队主罚角球。",
            "fallback_text_en": "The left team takes a corner.",
            "energy": "engaged", "events_referenced": ["evt_001"],
        }],
    })

    def fake_call(prompt, **kwargs):
        assert "football.corner" in prompt
        assert kwargs["temperature"] == 0.7
        return reply

    events, commentary = observe_direct(
        Path("clip.mp4"), "digest", 30.0, ["en", "zh"], call=fake_call
    )
    assert events[0]["event_code"] == "football.corner"
    assert commentary[0]["events_referenced"] == ["evt_001"]


def test_direct_observer_accepts_empty_confidence_reasons():
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append(prompt)
        return direct_reply(confidence_reasons=[])

    events, _ = observe_direct(
        Path("clip.mp4"), "digest", 10.0, ["en"], call=fake_call
    )
    assert events[0]["confidence_reasons"] == []
    assert len(calls) == 1


@pytest.mark.parametrize(("value", "expected"), [
    (None, "events[0].confidence_reasons must not be null"),
    ("model-private-text", "events[0].confidence_reasons must be a list; got str"),
    ([7], "events[0].confidence_reasons[0] must be text; got int"),
    (["model-private-text", "  "],
     "events[0].confidence_reasons[1] must be non-blank text"),
])
def test_direct_observer_reports_confidence_reasons_shape_without_values(value, expected):
    with pytest.raises(ValueError) as exc_info:
        generate._parse_direct(direct_reply(confidence_reasons=value), 10.0)

    assert str(exc_info.value) == expected
    assert "model-private-text" not in str(exc_info.value)


def test_direct_prompt_declares_confidence_reasons_array_shape():
    prompt = generate._direct_prompt("digest", 10.0, ["en"])

    assert "confidence_reasons must be a JSON array of short text reasons" in prompt
    assert "never a plain string" in prompt


def test_direct_observer_reports_missing_confidence_reasons_without_values():
    payload = json.loads(direct_reply())
    payload["events"][0].pop("confidence_reasons")

    with pytest.raises(ValueError) as exc_info:
        generate._parse_direct(json.dumps(payload), 10.0)

    assert str(exc_info.value) == "events[0].confidence_reasons is missing"


def direct_reply(**event_updates):
    event = {
        "event_id": "evt_001", "start_s": 1.0, "end_s": 4.0,
        "event_code": "football.corner", "player_team": "left",
        "player_jersey": "", "actors": ["left_team"],
        "outcome": "corner_taken", "confidence": "medium",
        "confidence_reasons": ["directly_visible"],
        "suggested_wording_zh": "左侧球队主罚角球。",
        "suggested_wording_en": "The left team takes a corner.",
        "energy": "engaged",
    }
    event.update(event_updates)
    return json.dumps({
        "events": [event],
        "commentary": [{
            "kind": "event", "timestamp_s": 1.0, "end_s": 4.0,
            "text_zh": "左侧球队主罚角球。",
            "text_en": "The left team takes a corner.",
            "fallback_text_zh": "左侧主罚角球。",
            "fallback_text_en": "A corner for the left team.",
            "energy": "engaged", "events_referenced": ["evt_001"],
        }],
    })




def test_atomic_json_failure_preserves_previous_commentary(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    comments = output / "comments"
    comments.mkdir()
    final = comments / "commentary.json"
    final.write_bytes(b"previous final")
    original_write_text = Path.write_text

    def fail_commentary_write(path, data, *args, **kwargs):
        if '"kind": "event"' in data:
            path.write_bytes(b"partial")
            raise OSError("injected write failure")
        return original_write_text(path, data, *args, **kwargs)

    _stub_event_verification(monkeypatch)
    monkeypatch.setattr(Path, "write_text", fail_commentary_write)
    with pytest.raises(OSError, match="injected write failure"):
        run_stage2b(
            output, tmp_path, mode="direct", force=True,
            call=lambda prompt, **kwargs: direct_reply(), duration_s=30.0,
        )
    assert final.read_bytes() == b"previous final"
    assert not list(comments.glob(".commentary.json.*"))


def test_atomic_fallback_copy_failure_preserves_previous_commentary(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    comments = output / "comments"
    comments.mkdir()
    final = comments / "commentary.json"
    final.write_bytes(b"previous final")

    def fail_relations(*args, **kwargs):
        raise RuntimeError("injected relations failure")

    def fail_copy(source, target, *args, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("injected copy failure")

    _stub_event_verification(monkeypatch)
    monkeypatch.setattr(stage2b_run, "generate_tactical_artifacts", fail_relations)
    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        run_stage2b(
            output, tmp_path, mode="hybrid", force=True,
            call=lambda prompt, **kwargs: direct_reply(), duration_s=30.0,
        )
    assert final.read_bytes() == b"previous final"
    assert not list(comments.glob(".commentary.json.*"))
def test_direct_observer_retries_invalid_schema_once():
    replies = iter([direct_reply(actors=None), direct_reply()])
    prompts = []

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return next(replies)

    events, _ = observe_direct(Path("clip.mp4"), "digest", 10.0, ["en"], call=fake_call)
    assert events[0]["actors"] == ["left_team"]
    assert len(prompts) == 2
    assert "Previous response errors" in prompts[1]


def test_direct_observer_rejects_invalid_schema_twice():
    def fake_call(prompt, **kwargs):
        return direct_reply(confidence="certain")

    with pytest.raises(ValueError, match="invalid direct observation twice"):
        observe_direct(Path("clip.mp4"), "digest", 10.0, ["en"], call=fake_call)


@pytest.mark.parametrize("updates", [
    {"player_team": 1},
    {"player_jersey": None},
    {"actors": [1]},
    {"outcome": ""},
    {"confidence": "certain"},
    {"confidence_reasons": [1]},
    {"energy": "intense"},
])
def test_direct_observer_rejects_required_event_types_and_enums(updates):
    with pytest.raises(ValueError):
        generate._parse_direct(direct_reply(**updates), 10.0)


def test_direct_observer_rejects_invalid_commentary_kind():
    payload = json.loads(direct_reply())
    payload["commentary"][0]["kind"] = "generic"
    with pytest.raises(ValueError, match="kind"):
        generate._parse_direct(json.dumps(payload), 10.0)


@pytest.mark.parametrize("updates", [
    {"midpoint_s": 99.0},
    {"player_team": ""},
    {"outcome": ""},
    {"disagreements": [123]},
])
def test_verification_rejects_semantically_invalid_values(updates):
    verdict = {
        "event_code": "football.corner", "midpoint_s": 2.5,
        "player_team": "left", "player_jersey": "", "outcome": "corner_taken",
        "directly_visible": True, "disagreements": [],
    }
    verdict.update(updates)
    with pytest.raises(ValueError):
        generate._parse_verification(json.dumps(verdict), 0.0, 5.0)


def test_verify_event_window_uses_low_temperature_and_cleans_up(monkeypatch):
    seen = {}

    def fake_extract(source, start, end, target):
        target.write_bytes(b"window")
        seen.update(start=start, end=end, target=target)
        return target

    def fake_call(prompt, **kwargs):
        assert kwargs["temperature"] == 0.1
        assert kwargs["video_path"].exists()
        return json.dumps({
            "event_code": "football.corner", "midpoint_s": 2.5,
            "player_team": "left", "player_jersey": "", "outcome": "corner_taken",
            "directly_visible": True, "disagreements": [],
        })

    monkeypatch.setattr(generate, "extract_window", fake_extract)
    verdict = verify_event_window(
        Path("clip.mp4"), {"start_s": 1.0, "end_s": 4.0}, call=fake_call
    )
    assert verdict["directly_visible"] is True
    assert (seen["start"], seen["end"]) == (0.0, 5.0)
    assert not seen["target"].exists()


def test_ark_chat_builds_configured_media_request(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    image = tmp_path / "frame.png"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("ARK_API_KEY", "secret")
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.test/v3")
    monkeypatch.setenv("ARK_RESPONSES_MODEL", "video-model")
    assert ark_chat("watch", video_path=video, image_paths=[image], temperature=0.2) == "ok"
    assert captured["client"] == {"base_url": "https://ark.test/v3", "api_key": "secret"}
    assert captured["request"]["model"] == "video-model"
    content = captured["request"]["messages"][0]["content"]
    assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_direct_observer_rejects_non_string_event_reference():
    payload = json.loads(direct_reply())
    payload["commentary"][0]["events_referenced"] = [{}]
    with pytest.raises(ValueError, match="events_referenced"):
        generate._parse_direct(json.dumps(payload), 10.0)


def test_relation_predicates_are_computed_by_code():
    relations = {"snapshots": [{
        "t": 20.0, "ball": {"speed": 4.0}, "players": [],
        "teams": {"right": {"opp_line_x": -24.0, "n_within_15m_of_ball": 4}},
    }]}
    query = {
        "t0": 18.0, "t1": 23.0, "team": "right",
        "quantity": "opp_line_x", "agg": "mean",
    }
    result = resolve_query(relations, query)
    assert result == {"value": -24.0, "n_samples": 1}
    assert predicate_passes(result, {"op": "<=", "threshold": -20.0})
    assert not predicate_passes(result, {"op": ">", "threshold": -20.0})
    assert resolve_query(relations, {**query, "agg": "median"})["value"] is None


def _required_corner(start=5.0, end=10.0):
    return {
        "event_id": "evt_001", "start_s": start, "end_s": end,
        "event_code": "football.corner", "player_team": "left",
        "outcome": "corner_taken", "confidence": "high",
        "suggested_wording_zh": "左侧主罚角球。",
        "suggested_wording_en": "The left team takes the corner.",
    }


def _segment(kind="event", tactic_refs=None, fallback_zh="左侧主罚角球。",
             fallback_en="The left team takes the corner."):
    return {
        "kind": kind, "timestamp_s": 5.0, "end_s": 10.0,
        "text_zh": "左侧主罚角球。", "text_en": "The left team takes the corner.",
        "fallback_text_zh": fallback_zh, "fallback_text_en": fallback_en,
        "energy": "engaged", "events_referenced": ["evt_001"],
        "tactical_facts_referenced": [], "tactical_claims": [],
        "event_claims": [{
            "event_id": "evt_001", "event_code": "football.corner",
            "player_team": "left", "outcome": "corner_taken",
            "assertion_strength": "certain",
        }],
    }


from pipeline.stage3_tts.synthesize import synthesize_fitting_segment


class FakeSynthesizer:
    def __init__(self):
        self.texts = []

    def synthesize(self, text, output_path, **kwargs):
        self.texts.append(text)
        output_path.write_bytes(b"audio")
        return output_path

def test_tts_retries_once_with_event_fallback(tmp_path):
    segment = {
        "text_zh": "右路迅速转移，蓝队的横向移动被彻底拉开。",
        "fallback_text_zh": "右路把球转向远端。",
    }
    durations = iter([6.2, 2.1])
    synth = FakeSynthesizer()
    out = synthesize_fitting_segment(
        segment, "zh", tmp_path / "segment.wav", 3.0, synth,
        probe=lambda path: next(durations),
        voice="default",
        prompt_wav=Path("prompt.wav"),
        prompt_text=None,
    )
    assert out.is_file()
    assert synth.texts == [segment["text_zh"], segment["fallback_text_zh"]]


def test_tts_segment_cache_is_owned_by_voice_tree(tmp_path, monkeypatch):
    output = tmp_path / "output"
    audio = output / "voice" / "commentary_zh.mp3"
    segments = [{"text_zh": "左侧完成传球。", "timestamp_s": 0.0, "end_s": 2.0}]
    synthesized = []

    def fake_synthesize(segment, language, path, slot_s, synthesizer, **kwargs):
        synthesized.append(path)
        path.write_bytes(b"audio")
        return path

    def fake_assemble(items, paths, target, duration_s):
        assert items == segments
        assert paths == synthesized
        assert duration_s == 2.0
        return target

    monkeypatch.setattr(
        "pipeline.stage3_tts.synthesize.synthesize_fitting_segment", fake_synthesize,
    )
    monkeypatch.setattr("pipeline.stage3_tts.synthesize.assemble_timeline", fake_assemble)

    result = stage3_tts_run._synthesize_voice(
        output, segments, "zh", "default", audio, 2.0,
        FakeSynthesizer(), None, None, force=True,
    )

    assert result == audio
    assert synthesized == [output / "voice" / "tts_segments" / "zh" / "default" /
                           "segment_000.wav"]


def test_stage4_keeps_old_final_when_mux_fails(tmp_path):
    annotated = tmp_path / "annotated.mp4"
    audio = tmp_path / "commentary.mp3"
    final = tmp_path / "final_video.mp4"
    annotated.write_bytes(b"video")
    audio.write_bytes(b"audio")
    final.write_bytes(b"old")

    temporary = None

    def fail_mux(video, sound, target):
        nonlocal temporary
        temporary = target
        assert target.suffix == ".mp4"
        target.write_bytes(b"partial")
        raise RuntimeError("mux failed")

    with pytest.raises(RuntimeError, match="mux failed"):
        replace_final_video(annotated, audio, final, mux=fail_mux)
    assert final.read_bytes() == b"old"
    assert temporary is not None and not temporary.exists()


def test_effects_load_canonical_stage2b_events(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([{
        "event_id": "evt_001", "start_s": 2.0, "end_s": 4.0,
        "event_code": "football.corner", "confidence": "high",
    }]), encoding="utf-8")
    events = load_effect_events(path)
    assert events == [{
        "event_id": "evt_001", "start_s": 2.0, "end_s": 4.0,
        "event_code": "football.corner", "confidence": "high",
        "timestamp_s": 3.0, "importance": 0.35,
    }]


def test_stage2b_runner_verifies_key_event_and_applies_state_conflict(tmp_path, monkeypatch):
    output = tmp_path / "SNGS-116"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    calls = []

    def fake_extract(source, start, end, target):
        target.write_bytes(b"window")
        return target

    def fake_call(prompt, **kwargs):
        calls.append(kwargs["temperature"])
        if kwargs["temperature"] == 0.1:
            return json.dumps({
                "event_code": "football.corner", "midpoint_s": 2.5,
                "player_team": "right", "player_jersey": "",
                "outcome": "corner_taken", "directly_visible": True,
                "disagreements": [],
            })
        return direct_reply(player_team="right", confidence="high")

    monkeypatch.setattr(generate, "extract_window", fake_extract)
    run_stage2b(
        output, tmp_path, mode="direct", force=True,
        call=fake_call, duration_s=30.0,
    )
    event = json.loads((output / "comments" / "events.json").read_text())[0]
    assert calls == [0.7, 0.1]
    assert event["verification"]["directly_visible"] is True
    assert event["state_ok"] is False
    assert event["confidence"] == "low"
    commentary = json.loads((output / "comments" / "commentary.json").read_text())
    assert commentary == []


def test_stage2b_hybrid_fallback_cannot_restore_post_verification_low_event(
        tmp_path, monkeypatch):
    output = tmp_path / "SNGS-116"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")

    def fake_extract(source, start, end, target):
        target.write_bytes(b"window")
        return target

    def fake_call(prompt, **kwargs):
        if kwargs["temperature"] == 0.1:
            return json.dumps({
                "event_code": "football.corner", "midpoint_s": 2.5,
                "player_team": "right", "player_jersey": "",
                "outcome": "corner_taken", "directly_visible": True,
                "disagreements": [],
            })
        return direct_reply(player_team="right", confidence="high")

    monkeypatch.setattr(generate, "extract_window", fake_extract)
    monkeypatch.setattr(
        stage2b_run, "compose_hybrid",
        lambda events, direct, *args, **kwargs: direct,
    )
    run_stage2b(
        output, tmp_path, mode="hybrid", force=True,
        call=fake_call, duration_s=30.0,
    )
    assert json.loads((output / "comments" / "commentary_direct.json").read_text()) == []
    assert json.loads((output / "comments" / "commentary.json").read_text()) == []


def test_direct_reconciliation_keeps_required_events_without_low_or_disputed_prose():
    events = [
        {
            **_required_corner(0.0, 4.0),
            "confidence": "high", "confidence_reasons": ["directly_visible"],
        },
        {
            "event_id": "evt_002", "start_s": 4.0, "end_s": 7.0,
            "event_code": "football.pass", "player_team": "left",
            "player_jersey": "9", "outcome": "pass_completed",
            "confidence": "medium", "confidence_reasons": ["directly_visible"],
            "suggested_wording_zh": "左侧9号完成传球。",
            "suggested_wording_en": "Left number 9 completes the pass.",
            "energy": "engaged", "verification": {
                "directly_visible": True, "disagreements": ["jersey unclear"],
            },
        },
        {
            "event_id": "evt_003", "start_s": 7.0, "end_s": 9.0,
            "event_code": "football.pressing", "player_team": "right",
            "player_jersey": "", "outcome": "press_attempted",
            "confidence": "low", "confidence_reasons": [],
            "suggested_wording_zh": "右侧高压逼抢。",
            "suggested_wording_en": "Right-side high press.",
            "energy": "engaged",
        },
    ]
    direct = [{
        "kind": "event", "timestamp_s": 0.0, "end_s": 9.0,
        "text_zh": "左侧主罚角球，随后9号传球，右侧高压逼抢。",
        "text_en": "Left corner, then number 9 passes under a right-side high press.",
        "fallback_text_zh": "左侧主罚角球，随后完成传球。",
        "fallback_text_en": "Left corner, followed by a completed pass.",
        "energy": "engaged",
        "events_referenced": ["evt_001", "evt_002", "evt_003"],
    }]

    reconciled = stage2b_run._reconcile_direct(events, direct)

    assert reconciled[0]["events_referenced"] == ["evt_001", "evt_002"]
    rendered = reconciled[0]["text_zh"] + reconciled[0]["text_en"]
    assert "高压" not in rendered and "high press" not in rendered
    assert "9号" not in rendered and "number 9" not in rendered
    assert {claim["event_id"] for claim in reconciled[0]["event_claims"]} == {
        "evt_001", "evt_002",
    }


def test_direct_reconciliation_rebuilds_noncompliant_high_confidence_wording():
    event = {
        **_required_corner(0.0, 4.0),
        "suggested_wording_zh": "皮球送入禁区。",
        "suggested_wording_en": "The ball is whipped into the box.",
    }
    direct = [_segment()]

    reconciled = stage2b_run._reconcile_direct([event], direct, 10.0)

    assert "角球" in reconciled[0]["fallback_text_zh"]
    assert "corner" in reconciled[0]["fallback_text_en"].lower()
    assert "left" in reconciled[0]["fallback_text_en"].lower()


def test_direct_reconciliation_uses_short_structured_fallback_for_required_event():
    event = {
        **_required_corner(0.0, 5.0),
        "player_jersey": "9",
        "suggested_wording_zh": "左侧9号迅速走向角旗区，准备把这个角球送入禁区。",
        "suggested_wording_en": (
            "Left No. 9 strides to the flag and prepares to deliver this corner."
        ),
    }

    reconciled = stage2b_run._reconcile_direct([event], [_segment()], 10.0)
    segment = reconciled[0]

    assert segment["fallback_text_zh"] != segment["text_zh"]
    assert segment["fallback_text_en"] != segment["text_en"]
    assert len(segment["fallback_text_zh"]) < len(segment["text_zh"])
    assert len(segment["fallback_text_en"]) < len(segment["text_en"])
    assert not audit_commentary(reconciled, [event], [], set(), 10.0)
    for token in ("左", "9", "角球"):
        assert token in segment["fallback_text_zh"]
    for token in ("left", "9", "corner"):
        assert token in segment["fallback_text_en"].lower()


def test_direct_reconciliation_includes_undisputed_jersey_in_wording():
    event = {
        "event_id": "evt_001", "start_s": 0.0, "end_s": 4.0,
        "event_code": "football.pass", "player_team": "left",
        "player_jersey": "9", "outcome": "pass_completed",
        "confidence": "medium", "confidence_reasons": ["directly_visible"],
        "suggested_wording_zh": "左侧完成传球。",
        "suggested_wording_en": "The left team completes a pass.",
        "energy": "engaged",
    }
    direct = [_segment()]

    reconciled = stage2b_run._reconcile_direct([event], direct, 10.0)

    assert "9" in reconciled[0]["fallback_text_zh"]
    assert "9" in reconciled[0]["fallback_text_en"]


def test_direct_reconciliation_omits_placeholder_jersey():
    event = {
        **_required_corner(0.0, 4.0),
        "player_jersey": "?",
        "suggested_wording_zh": "皮球送入禁区。",
        "suggested_wording_en": "The ball is whipped into the box.",
    }
    direct = [_segment()]

    reconciled = stage2b_run._reconcile_direct([event], direct, 10.0)

    assert "?" not in reconciled[0]["fallback_text_zh"]
    assert "?" not in reconciled[0]["fallback_text_en"]


def test_direct_prompt_requires_structured_suggested_wording():
    prompt = generate._direct_prompt("digest", 10.0, ["en"])

    assert "suggested_wording_zh must" in prompt
    assert "left or right" in prompt


def test_stage2b_runner_schedules_missing_required_event_in_free_subinterval(
        tmp_path, monkeypatch):
    output = tmp_path / "SNGS-116"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    first = {**json.loads(direct_reply())["events"][0],
             "start_s": 0.0, "end_s": 6.0, "confidence": "high"}
    second = {**first, "event_id": "evt_002", "start_s": 4.0, "end_s": 8.0,
              "event_code": "football.clearance", "outcome": "danger_cleared",
              "suggested_wording_zh": "左侧完成解围。",
              "suggested_wording_en": "The left team makes a clearance."}
    direct = [{
        **json.loads(direct_reply())["commentary"][0],
        "timestamp_s": 0.0, "end_s": 6.0,
    }]
    monkeypatch.setattr(stage2b_run, "observe_direct", lambda *args, **kwargs: (
        [first, second], direct,
    ))
    monkeypatch.setattr(stage2b_run, "verify_event_window", lambda path, event, **kwargs: {
        "event_code": event["event_code"],
        "midpoint_s": (event["start_s"] + event["end_s"]) / 2,
        "player_team": event["player_team"], "player_jersey": "",
        "outcome": event["outcome"], "directly_visible": True,
        "disagreements": [],
    })
    run_stage2b(output, tmp_path, mode="direct", force=True, duration_s=10.0)
    commentary = json.loads((output / "comments" / "commentary.json").read_text())
    assert [(item["timestamp_s"], item["end_s"]) for item in commentary] == [
        (0.0, 6.0), (6.0, 8.0),
    ]
    assert {ref for item in commentary for ref in item["events_referenced"]} == {
        "evt_001", "evt_002",
    }
    assert all(item["end_s"] > item["timestamp_s"] for item in commentary)


def test_direct_reconciliation_merges_fully_occupied_overlapping_required_events():
    first = {
        **_required_corner(0.0, 10.0), "confidence": "high",
        "suggested_wording_zh": "左侧球员走到角旗区，准备把角球送入禁区。",
        "suggested_wording_en": "The left player approaches the flag to deliver the corner.",
    }
    second = {**first, "event_id": "evt_002", "start_s": 2.0, "end_s": 5.0,
              "event_code": "football.pass", "outcome": "pass_completed",
              "suggested_wording_zh": "随后左侧球员稳稳地完成了这次传球。",
              "suggested_wording_en": "The left player then completes the pass with control."}
    third = {**first, "event_id": "evt_003", "start_s": 4.0, "end_s": 8.0,
             "event_code": "football.clearance", "outcome": "danger_cleared",
             "suggested_wording_zh": "紧接着左侧球员及时完成解围，化解危险。",
             "suggested_wording_en": "The left player follows with a timely clearance from danger."}
    direct = [{
        **_segment(), "timestamp_s": 0.0, "end_s": 10.0,
        "events_referenced": ["evt_001"],
    }]

    reconciled = stage2b_run._reconcile_direct([first, second, third], direct, 10.0)

    assert len(reconciled) == 1
    assert reconciled[0]["events_referenced"] == ["evt_001", "evt_002", "evt_003"]
    assert reconciled[0]["end_s"] > reconciled[0]["timestamp_s"]
    assert {claim["event_id"] for claim in reconciled[0]["event_claims"]} == {
        "evt_001", "evt_002", "evt_003",
    }

    segment = reconciled[0]
    assert segment["fallback_text_zh"] != segment["text_zh"]
    assert segment["fallback_text_en"] != segment["text_en"]
    assert len(segment["fallback_text_zh"]) < len(segment["text_zh"])
    assert len(segment["fallback_text_en"]) < len(segment["text_en"])
    assert not audit_commentary(reconciled, [first, second, third], [], set(), 10.0)
    for token in ("角球", "传球", "解围"):
        assert token in segment["fallback_text_zh"]
    for token in ("corner", "pass", "clearance"):
        assert token in segment["fallback_text_en"].lower()


def test_stage2b_cache_reuse_is_mode_aware(tmp_path, monkeypatch):
    output = tmp_path / "SNGS-116"
    comments = output / "comments"
    comments.mkdir(parents=True)
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    (comments / "commentary.json").write_text("[]", encoding="utf-8")
    (comments / "commentary_direct.json").write_text("[]", encoding="utf-8")
    (comments / "tactical_state.json").write_text("{}", encoding="utf-8")
    (comments / "tactical_proposals.json").write_text("[]", encoding="utf-8")
    (comments / "verified_tactical_facts.json").write_text("[]", encoding="utf-8")
    (comments / "relations.json").write_text("{}", encoding="utf-8")
    observed = []

    def fake_observe(*args, **kwargs):
        observed.append(True)
        payload = json.loads(direct_reply(
            event_code="football.pass", confidence="low",
        ))
        return payload["events"], payload["commentary"]

    monkeypatch.setattr(stage2b_run, "observe_direct", fake_observe)
    run_stage2b(output, tmp_path, mode="direct", force=False, duration_s=30.0)
    assert observed == [True]
