"""Doubao (Volcengine ARK) API adapter."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter

DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260428"


class DoubaoAPIAdapter(LLMAdapter):
    """OpenAI-compatible client against Volcengine ARK.

    Credentials (in priority order):
      ARK_API_KEY / ARK_BASE_URL / ARK_RESPONSES_MODEL
      DOUBAO_API_KEY / DOUBAO_BASE_URL / DOUBAO_MODEL
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get(
            "ARK_RESPONSES_MODEL",
            os.environ.get("DOUBAO_MODEL", DEFAULT_ARK_MODEL),
        )
        self.api_key = api_key or os.environ.get(
            "ARK_API_KEY",
            os.environ.get("DOUBAO_API_KEY", ""),
        )
        self.base_url = (
            base_url
            or os.environ.get("ARK_BASE_URL")
            or os.environ.get("DOUBAO_BASE_URL")
            or DEFAULT_ARK_BASE_URL
        )

    def supports_video(self) -> bool:
        # Seed lite is text-first; avoid uploading full videos by default.
        return False

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Missing ARK_API_KEY (or DOUBAO_API_KEY). "
                "Set it in the environment or a local .env file."
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        visual_input = self.prepare_visual_input(visual_input)

        content: list[dict] = [{"type": "text", "text": prompt}]
        if isinstance(visual_input, list):
            for img_path in visual_input[:12]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
