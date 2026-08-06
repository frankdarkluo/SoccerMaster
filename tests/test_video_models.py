from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pipeline import video_models


def test_gemini_rotates_three_keys_and_suspends_rate_limited_key(monkeypatch):
    fake_keys = ("fake-a", "fake-b", "fake-c")
    for name, value in zip(video_models._GEMINI_KEYS, fake_keys):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GEMINI_KEY_ROTATION_OFFSET", raising=False)
    monkeypatch.setattr(video_models, "_GEMINI_INDEX", 0)
    monkeypatch.setattr(video_models, "_GEMINI_SUSPENDED", set())

    responses = iter([
        SimpleNamespace(status_code=429, headers={"retry-after": "15"}, text="quota exceeded"),
        SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            json=lambda: {
                "steps": [{
                    "type": "model_output",
                    "content": [{"type": "text", "text": '{"ok": true}'}],
                }],
            },
        ),
    ])
    used_keys = []

    def post(*_, **kwargs):
        used_keys.append(kwargs["headers"]["x-goog-api-key"])
        return next(responses)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(
        post=post, Timeout=lambda *_args, **_kwargs: None,
    ))
    sleeps = []
    payload, _, attempts = video_models.generate_json(
        "gemini", "prompt", {}, retries=1, sleep=sleeps.append,
    )

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleeps == [15.0]
    assert used_keys == ["fake-a", "fake-b"]
    assert video_models._GEMINI_SUSPENDED == {"GEMINI_API_KEY"}
    assert [video_models._gemini_key()[1] for _ in range(2)] == ["fake-c", "fake-b"]


@pytest.mark.parametrize(("retry_after", "body", "expected_delay"), [
    ("Wed, 21 Oct 2015 07:28:00 GMT", "retry in 4.5s", 4.5),
    ("not-a-delay", "quota exceeded", 1.0),
])
def test_gemini_invalid_retry_after_falls_back_without_parse_error(
    monkeypatch, retry_after, body, expected_delay,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-a")
    monkeypatch.setenv("GEMINI_API_KEY1", "fake-b")
    monkeypatch.setattr(video_models, "_GEMINI_INDEX", 0)
    monkeypatch.setattr(video_models, "_GEMINI_SUSPENDED", set())
    responses = iter([
        SimpleNamespace(
            status_code=429,
            headers={"retry-after": retry_after},
            text=body,
        ),
        SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            json=lambda: {
                "steps": [{
                    "type": "model_output",
                    "content": [{"type": "text", "text": '{"ok": true}'}],
                }],
            },
        ),
    ])
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(
        post=lambda *_args, **_kwargs: next(responses),
        Timeout=lambda *_args, **_kwargs: None,
    ))
    sleeps = []

    payload, _, attempts = video_models.generate_json(
        "gemini", "prompt", {}, retries=1, sleep=sleeps.append,
    )

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleeps == [expected_delay]


def test_gemini_omits_array_bounds_without_mutating_schema(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(video_models, "_GEMINI_INDEX", 0)
    monkeypatch.setattr(video_models, "_GEMINI_SUSPENDED", set())
    sent = {}

    def post(*_, **kwargs):
        sent.update(kwargs["json"])
        return SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            json=lambda: {
                "steps": [{
                    "type": "model_output",
                    "content": [{"type": "text", "text": '{"items": []}'}],
                }],
            },
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(
        post=post, Timeout=lambda *_args, **_kwargs: None,
    ))
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 30,
                "items": {"type": "string"},
            },
        },
        "required": ["items"],
    }

    assert video_models._gemini("prompt", schema, None)[0] == {"items": []}
    sent_items = sent["response_format"]["schema"]["properties"]["items"]
    assert "minItems" not in sent_items
    assert "maxItems" not in sent_items
    assert schema["properties"]["items"]["maxItems"] == 30


@pytest.mark.parametrize("error_name", ["ConnectError", "ReadTimeout"])
def test_gemini_transport_errors_are_retried(monkeypatch, error_name):
    calls = []
    error_type = type(error_name, (RuntimeError,), {})

    def flaky(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise error_type("transient")
        return {"ok": True}, []

    monkeypatch.setattr(video_models, "_gemini", flaky)
    sleeps = []
    payload, _, attempts = video_models.generate_json(
        "gemini", "prompt", {}, retries=1, sleep=sleeps.append,
    )

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleeps == [1]


def test_doubao_sdk_rate_limit_is_retried(monkeypatch):
    calls = []

    class RateLimitError(RuntimeError):
        pass

    def flaky(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise RateLimitError("retry")
        return {"ok": True}, []

    monkeypatch.setattr(video_models, "_doubao", flaky)
    payload, _, attempts = video_models.generate_json(
        "doubao", "prompt", {}, retries=1, sleep=lambda _: None,
    )
    assert payload == {"ok": True}
    assert attempts == 2


def test_retry_exhaustion_records_attempt_count(monkeypatch):
    class ReadTimeout(RuntimeError):
        pass

    def timeout(*_args, **_kwargs):
        raise ReadTimeout("timeout")

    monkeypatch.setattr(video_models, "_doubao", timeout)
    with pytest.raises(ReadTimeout) as caught:
        video_models.generate_json(
            "doubao", "prompt", {}, retries=2, sleep=lambda _: None,
        )

    assert caught.value.attempts == 3


def test_doubao_text_json_uses_shared_transport(monkeypatch):
    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
    )
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **_: response),
    ))
    monkeypatch.delenv("ARK_RESPONSES_MODEL", raising=False)
    monkeypatch.setattr(video_models, "_doubao_client", lambda: client)
    payload, usage = video_models._doubao("prompt", {}, None)
    assert payload == {"ok": True}
    assert usage == []


def test_doubao_video_uses_direct_base64_response_input(tmp_path, monkeypatch):
    sent = {}
    response = SimpleNamespace(usage=None, output_text='{"ok": true}')

    def create(**kwargs):
        sent.update(kwargs)
        return response

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.delenv("ARK_RESPONSES_MODEL", raising=False)
    monkeypatch.setattr(video_models, "_doubao_client", lambda: client)
    video = tmp_path / "neutral.mp4"
    video.write_bytes(b"video-bytes")

    payload, usage = video_models._doubao("prompt", {}, video)

    assert payload == {"ok": True}
    assert usage == []
    content = sent["input"][0]["content"]
    assert content[0]["type"] == "input_video"
    assert content[0]["fps"] == 2
    assert content[0]["video_url"] == "data:video/mp4;base64,dmlkZW8tYnl0ZXM="
    assert content[1] == {"type": "input_text", "text": "prompt"}
    assert sent["temperature"] == 0.0


def test_doubao_honors_explicit_fps_and_temperature(tmp_path, monkeypatch):
    sent = {}
    response = SimpleNamespace(usage=None, output_text='{"ok": true}')
    client = SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: sent.update(kwargs) or response,
    ))
    monkeypatch.delenv("ARK_RESPONSES_MODEL", raising=False)
    monkeypatch.setattr(video_models, "_doubao_client", lambda: client)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")

    video_models.generate_json("doubao", "prompt", {}, video_path=video, fps=8.0, temperature=0.5)

    assert sent["input"][0]["content"][0]["fps"] == 8.0
    assert sent["temperature"] == 0.5


def test_gemini_rejects_interactions_api_fps_field(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(video_models, "_GEMINI_INDEX", 0)
    monkeypatch.setattr(video_models, "_GEMINI_SUSPENDED", set())
    sent = {}

    def post(*_, **kwargs):
        sent.update(kwargs["json"])
        return SimpleNamespace(
            status_code=200, headers={}, text="",
            json=lambda: {"steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"ok": true}'}]}]},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, Timeout=lambda *_a, **_k: None))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")

    with pytest.raises(ValueError, match="has no fps field"):
        video_models.generate_json("gemini", "prompt", {}, video_path=video, fps=6.0, temperature=0.7)

    assert sent == {}


def test_gemini_omits_video_metadata_when_fps_not_given(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(video_models, "_GEMINI_INDEX", 0)
    monkeypatch.setattr(video_models, "_GEMINI_SUSPENDED", set())
    sent = {}

    def post(*_, **kwargs):
        sent.update(kwargs["json"])
        return SimpleNamespace(
            status_code=200, headers={}, text="",
            json=lambda: {"steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"ok": true}'}]}]},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, Timeout=lambda *_a, **_k: None))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")

    video_models.generate_json("gemini", "prompt", {}, video_path=video)

    assert "video_metadata" not in sent["input"][0]
    assert "generation_config" not in sent


def test_temporary_window_clip_trims_with_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"window")

    monkeypatch.setattr(video_models.subprocess, "run", fake_run)
    with video_models.temporary_window_clip(source, 4.0, 9.0, playback_slowdown=6) as window:
        assert window.read_bytes() == b"window"

    assert not window.exists()
    command = commands[0]
    assert command[command.index("-ss") + 1] == "4.000"
    assert command[command.index("-to") + 1] == "9.000"
    assert command[command.index("-vf") + 1] == "setpts=6*PTS"


def test_temporary_window_clip_rejects_empty_or_inverted_range(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="start_s < end_s"):
        with video_models.temporary_window_clip(source, 5.0, 5.0):
            pass


def test_generate_json_keeps_single_video_api(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    seen = []
    monkeypatch.setattr(
        video_models,
        "_doubao",
        lambda _prompt, _schema, path, **_kwargs: seen.append(path) or ({"ok": True}, []),
    )

    video_models.generate_json("doubao", "prompt", {}, video_path=video)

    assert seen == [video]



def test_doubao_client_honors_base_and_timeout_env(monkeypatch):
    calls = {}
    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
    )

    def create(**kwargs):
        calls["model"] = kwargs["model"]
        return response

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))

    def openai(**kwargs):
        calls.update(kwargs)
        return client

    monkeypatch.setenv("ARK_API_KEY", "fake-key")
    monkeypatch.setenv("ARK_BASE_URL", "  ")
    monkeypatch.setenv("ARK_RESPONSES_MODEL", "  ")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai))
    monkeypatch.setenv("ARK_TIMEOUT_S", "90")

    payload, _ = video_models._doubao("prompt", {}, None)

    assert payload == {"ok": True}
    assert calls["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert calls["timeout"] == 90.0
    assert calls["model"] == video_models.DOUBAO_MODEL
