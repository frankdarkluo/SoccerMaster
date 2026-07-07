"""GPT API adapter (frame-based visual input)."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter


class OpenAIAPIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def supports_video(self) -> bool:
        return False

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        visual_input = self.prepare_visual_input(visual_input)

        content: list[dict] = []
        if isinstance(visual_input, list):
            for img_path in visual_input[:30]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
