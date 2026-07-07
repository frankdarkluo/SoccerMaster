#!/usr/bin/env python3
"""Smoke-test Doubao TTS credentials from .env."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.stage4_commentary.generate import load_ark_env  # noqa: E402


def probe_unidirectional() -> None:
    load_ark_env()
    api_key = os.environ.get("DOUBAO_TTS_API_KEY", "")
    speaker = os.environ.get("DOUBAO_TTS_SPEAKER", "")
    resources = [
        "seed-icl-2.0",
        "volc.seedicl.default",
        "seed-tts-2.0",
        "seed-tts-1.0",
        "volc.service_type.10029",
        "volc.service_type.10050",
    ]
    for resource in resources:
        for model_type in (None, 4):
            additions = {}
            if model_type is not None:
                additions["model_type"] = model_type
            body = {
                "user": {"uid": "soccermaster"},
                "req_params": {
                    "text": "球进了！",
                    "speaker": speaker,
                    "audio_params": {"format": "mp3", "sample_rate": 24000},
                },
            }
            if additions:
                body["req_params"]["additions"] = json.dumps(additions)
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
                "X-Api-Resource-Id": resource,
            }
            resp = requests.post(
                "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
                headers=headers,
                json=body,
                timeout=30,
            )
            label = f"{resource}/mt={model_type}"
            print(f"uni/{label}: {resp.status_code} {resp.text[:180]!r}")


def probe_create() -> None:
    load_ark_env()
    api_key = os.environ.get("DOUBAO_TTS_API_KEY", "")
    speaker = os.environ.get("DOUBAO_TTS_SPEAKER", "")
    bodies = {
        "refs-speaker": {
            "model": "seed-audio-1.0",
            "text_prompt": "球进了！",
            "references": [{"speaker": speaker}],
            "audio_config": {"format": "mp3", "sample_rate": 24000},
        },
        "top-level-speaker": {
            "model": "seed-audio-1.0",
            "text_prompt": "球进了！",
            "speaker": speaker,
            "audio_config": {"format": "mp3", "sample_rate": 24000},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
    }
    for name, body in bodies.items():
        resp = requests.post(
            "https://openspeech.bytedance.com/api/v3/tts/create",
            headers=headers,
            json=body,
            timeout=60,
        )
        print(f"create/{name}: {resp.status_code} {resp.text[:220]!r}")


if __name__ == "__main__":
    print("=== seed-audio create API ===")
    probe_create()
    print("=== legacy unidirectional API ===")
    probe_unidirectional()
