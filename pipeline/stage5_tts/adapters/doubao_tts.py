"""Volcengine / Doubao TTS adapter (seed-audio-1.0 HTTP create API).

Auth and request shape follow docs/豆包语音_音频生成HTTP_1783059684.pdf:
  POST https://openspeech.bytedance.com/api/v3/tts/create
  Header: X-Api-Key  (new console API Key)
  Body: model=seed-audio-1.0, text_prompt, speaker, audio_config
"""
from __future__ import annotations

import base64
import logging
import os
import uuid
from pathlib import Path

import requests

from pipeline.stage5_tts.tts_adapter import (
    ENERGY_EXCITED,
    ENERGY_EXPLOSIVE,
    ENERGY_NORMAL,
    TTSAdapter,
)

log = logging.getLogger(__name__)

_ENDPOINT = "https://openspeech.bytedance.com/api/v3/tts/create"
_MODEL = "seed-audio-1.0"

# Map energy tiers to audio_config (speech_rate/loudness_rate: -50..100,
# pitch_rate: -12..12). Values approximate grill-me option B.
_PROSODY = {
    ENERGY_NORMAL: {"speech_rate": 15, "pitch_rate": 0, "loudness_rate": 0},
    ENERGY_EXCITED: {"speech_rate": 25, "pitch_rate": 3, "loudness_rate": 10},
    ENERGY_EXPLOSIVE: {"speech_rate": 35, "pitch_rate": 5, "loudness_rate": 20},
}

_STYLE_PREFIX = {
    ENERGY_NORMAL: "",
    ENERGY_EXCITED: "用激动的足球解说语气说：",
    ENERGY_EXPLOSIVE: "用非常激动爆发的足球解说语气，像进球瞬间那样高亢地说：",
}


class DoubaoTTSAdapter(TTSAdapter):
    """Calls the Volcengine seed-audio-1.0 HTTP create API.

    Required env vars (or constructor args):
        DOUBAO_TTS_API_KEY  – API Key from 控制台 > API Key管理
        DOUBAO_TTS_SPEAKER  – speaker ID (``S_xxx`` for cloned voices)
    """

    def __init__(
        self,
        api_key: str | None = None,
        speaker: str | None = None,
    ) -> None:
        # Prefer new-console API Key; fall back to legacy ACCESS_KEY name.
        self.api_key = (
            api_key
            or os.environ.get("DOUBAO_TTS_API_KEY", "")
            or os.environ.get("DOUBAO_TTS_ACCESS_KEY", "")
        )
        self.speaker = speaker or os.environ.get("DOUBAO_TTS_SPEAKER", "")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        energy: str = ENERGY_NORMAL,
    ) -> Path:
        if not (self.api_key and self.speaker):
            raise RuntimeError(
                "Doubao TTS requires DOUBAO_TTS_API_KEY and DOUBAO_TTS_SPEAKER. "
                "Set them in .env or pass to constructor."
            )

        prosody = _PROSODY.get(energy, _PROSODY[ENERGY_NORMAL])
        style = _STYLE_PREFIX.get(energy, "")
        text_prompt = f"{style}{text}" if style else text

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        # speaker must be top-level (not under references) for cloned voices.
        body = {
            "model": _MODEL,
            "text_prompt": text_prompt,
            "speaker": self.speaker,
            "audio_config": {
                "format": "mp3",
                "sample_rate": 24000,
                "speech_rate": prosody["speech_rate"],
                "pitch_rate": prosody["pitch_rate"],
                "loudness_rate": prosody["loudness_rate"],
            },
        }

        log.debug(
            "Doubao TTS request: speaker=%s energy=%s text=%s…",
            self.speaker, energy, text[:40],
        )
        resp = requests.post(_ENDPOINT, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        # Success responses may omit code and only return audio/url.
        code = payload.get("code")
        if code is not None and code != 0:
            raise RuntimeError(
                f"Doubao TTS error code {code}: {payload.get('message', payload)}"
            )

        audio_b64 = payload.get("audio")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
        elif payload.get("url"):
            audio_resp = requests.get(payload["url"], timeout=60)
            audio_resp.raise_for_status()
            audio_bytes = audio_resp.content
        else:
            raise RuntimeError(
                f"Doubao TTS returned no audio. Response: {str(payload)[:500]}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        log.info(
            "Doubao TTS [%s] → %s (%d bytes, %.2fs)",
            energy,
            output_path,
            output_path.stat().st_size,
            float(payload.get("duration") or payload.get("original_duration") or 0),
        )
        return output_path
