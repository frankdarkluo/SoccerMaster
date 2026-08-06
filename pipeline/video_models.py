"""Minimal Doubao/Gemini JSON generation shared by tactical experiments."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
import re
import subprocess
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

DOUBAO_MODEL = "doubao-seed-2-0-lite-260428"
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TIMEOUT_S = 300.0
TRANSPORT_RETRIES = 2
_GEMINI_KEYS = ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2")
_GEMINI_LOCK = threading.Lock()
_GEMINI_INDEX = 0
_GEMINI_SUSPENDED: set[str] = set()


class RetryableProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


@contextmanager
def temporary_window_clip(video_path: Path, start_s: float, end_s: float, *, playback_slowdown: float = 1.0) -> Iterator[Path]:
    """Yield a temporary trim of `video_path` covering [start_s, end_s], re-encoded for accurate cuts."""
    if not start_s < end_s:
        raise ValueError(f"window bounds must satisfy start_s < end_s, got {start_s}..{end_s}")
    if playback_slowdown < 1:
        raise ValueError("playback_slowdown must be at least 1")
    with tempfile.TemporaryDirectory(prefix="window-clip-") as directory:
        target = Path(directory) / "window.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
                "-i", str(video_path), "-an",
                *(["-vf", f"setpts={playback_slowdown:g}*PTS"] if playback_slowdown != 1 else []),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-movflags", "+faststart", str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        yield target

def model_name(provider: str) -> str:
    if provider == "doubao":
        return DOUBAO_MODEL
    if provider == "gemini":
        return GEMINI_MODEL
    raise ValueError(f"unknown provider: {provider}")


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.DOTALL)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def _doubao_client():
    api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise RuntimeError("ARK_API_KEY is not set")
    timeout_s = float(os.environ.get("ARK_TIMEOUT_S", TIMEOUT_S))
    if timeout_s <= 0:
        raise RuntimeError("ARK_TIMEOUT_S must be positive")
    from openai import OpenAI

    return OpenAI(
        base_url=(
            os.environ.get("ARK_BASE_URL", "").strip()
            or "https://ark.cn-beijing.volces.com/api/v3"
        ),
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )


def _doubao(
    prompt: str,
    schema: dict[str, Any],
    video_path: Path | None,
    *,
    fps: float = 2.0,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configured = os.environ.get("ARK_RESPONSES_MODEL", "").strip() or DOUBAO_MODEL
    if configured != DOUBAO_MODEL:
        raise ValueError(f"ARK_RESPONSES_MODEL must be {DOUBAO_MODEL}, got {configured}")
    client = _doubao_client()
    if video_path is None:
        response = client.chat.completions.create(
            model=configured,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=12000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tactical_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        usage = [response.usage.model_dump(exclude_none=True)] if response.usage else []
        return _json_object(response.choices[0].message.content or ""), usage

    content = [{
        "type": "input_video",
        "video_url": "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode("ascii"),
        "fps": fps,
    }]
    response = client.responses.create(
        model=configured,
        input=[{
            "role": "user",
            "content": content + [{"type": "input_text", "text": prompt}],
        }],
        temperature=temperature,
        max_output_tokens=12000,
        text={"format": {
            "type": "json_schema",
            "name": "tactical_response",
            "strict": True,
            "schema": schema,
        }},
    )
    usage = [response.usage.model_dump(exclude_none=True)] if response.usage else []
    return _json_object(response.output_text), usage


def _gemini_key() -> tuple[str, str]:
    keys, seen = [], set()
    for name in _GEMINI_KEYS:
        value = os.environ.get(name, "").strip()
        if value and value not in seen:
            keys.append((name, value))
            seen.add(value)
    if not keys:
        raise RuntimeError("no Gemini API key is set")
    try:
        offset = int(os.environ.get("GEMINI_KEY_ROTATION_OFFSET", "0"))
    except ValueError as exc:
        raise RuntimeError("GEMINI_KEY_ROTATION_OFFSET must be non-negative") from exc
    if offset < 0:
        raise RuntimeError("GEMINI_KEY_ROTATION_OFFSET must be non-negative")
    global _GEMINI_INDEX
    with _GEMINI_LOCK:
        for _ in keys:
            name, key = keys[(offset + _GEMINI_INDEX) % len(keys)]
            _GEMINI_INDEX += 1
            if name not in _GEMINI_SUSPENDED:
                return name, key
    raise RuntimeError("all configured Gemini API keys are suspended after rate limits")


def _gemini_text(response: dict[str, Any]) -> str:
    parts = [
        part["text"]
        for step in response.get("steps", [])
        if step.get("type") == "model_output"
        for part in step.get("content", [])
        if part.get("type") == "text" and isinstance(part.get("text"), str)
    ]
    if not parts:
        raise ValueError("Gemini response has no text output")
    return "".join(parts)


def _gemini_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _gemini_schema(item)
            for key, item in value.items()
            if key not in {"minItems", "maxItems"}
        }
    if isinstance(value, list):
        return [_gemini_schema(item) for item in value]
    return value


def _gemini(
    prompt: str,
    schema: dict[str, Any],
    video_path: Path | None,
    *,
    fps: float | None = None,
    temperature: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import httpx

    if fps is not None:
        raise ValueError("Gemini Interactions API has no fps field; preprocess video playback instead")
    if temperature is not None:
        raise ValueError("Gemini 3.6 Flash does not support temperature")
    inputs: list[dict[str, Any]] = []
    if video_path is not None:
        inputs.append({
            "type": "video",
            "data": base64.b64encode(video_path.read_bytes()).decode("ascii"),
            "mime_type": "video/mp4",
            "resolution": "high",
        })
    inputs.append({"type": "text", "text": prompt})
    key_name, api_key = _gemini_key()
    response = httpx.post(
        os.environ.get("GEMINI_INTERACTIONS_URL", GEMINI_URL),
        headers={"x-goog-api-key": api_key},
        json={
            "model": GEMINI_MODEL,
            "input": inputs,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": _gemini_schema(schema),
            },
        },
        timeout=httpx.Timeout(TIMEOUT_S, connect=60.0),
    )
    if response.status_code == 429:
        with _GEMINI_LOCK:
            _GEMINI_SUSPENDED.add(key_name)
    if response.status_code == 429 or response.status_code >= 500:
        retry_after = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", response.headers.get("retry-after", ""))
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", response.text, re.IGNORECASE)
        raise RetryableProviderError(
            f"Gemini HTTP {response.status_code}: {response.text[:500]}",
            response.status_code,
            float(retry_after.group(1)) if retry_after else float(match.group(1)) if match else None,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:1000]}")
    body = response.json()
    usage = [body["usage"]] if isinstance(body.get("usage"), dict) else []
    return _json_object(_gemini_text(body)), usage


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return (
        status == 429
        or isinstance(status, int) and status >= 500
        or type(exc).__name__ in {
            "APITimeoutError", "APIConnectionError", "RateLimitError",
            "InternalServerError", "ConnectError", "ConnectTimeout",
            "ReadTimeout", "WriteTimeout", "PoolTimeout",
        }
    )


def generate_json(
    provider: str,
    prompt: str,
    schema: dict[str, Any],
    *,
    video_path: Path | None = None,
    fps: float | None = None,
    temperature: float | None = None,
    retries: int = TRANSPORT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    call = _doubao if provider == "doubao" else _gemini if provider == "gemini" else None
    if call is None:
        raise ValueError(f"unknown provider: {provider}")
    kwargs: dict[str, Any] = {}
    if temperature is not None or provider == "doubao":
        kwargs["temperature"] = temperature if temperature is not None else 0.0
    if fps is not None:
        kwargs["fps"] = fps
    for attempt in range(retries + 1):
        try:
            payload, usage = call(prompt, schema, video_path, **kwargs)
            return payload, usage, attempt + 1
        except Exception as exc:
            if not isinstance(exc, RetryableProviderError) and not _retryable(exc):
                raise
            if attempt == retries:
                exc.attempts = attempt + 1
                raise
            delay = max(2**attempt, getattr(exc, "retry_after_s", None) or 0.0)
            # ponytail: cap retry sleeps at 60s; use a scheduler if longer deferrals become necessary.
            sleep(min(delay, 60.0))
    raise AssertionError("unreachable")
