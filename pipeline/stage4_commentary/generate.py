"""High-level commentary generation orchestration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from pipeline.config import REPO_ROOT, PipelineConfig
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage4_commentary.llm_adapter import LLMAdapter
from pipeline.stage4_commentary.postprocess import (
    parse_commentary_output,
    write_commentary_json,
)
from pipeline.stage4_commentary.prompt_builder import build_commentary_prompt


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def load_ark_env() -> None:
    """Load ARK / Doubao credentials from repo-root .env if present."""
    _load_dotenv(REPO_ROOT / ".env")


def build_adapter(backend: str = "doubao") -> LLMAdapter:
    """Construct an LLM adapter. Default backend is Doubao (Volcengine ARK)."""
    load_ark_env()
    if backend == "mock":
        from pipeline.stage4_commentary.adapters.mock import MockLLMAdapter
        return MockLLMAdapter()
    if backend == "qwen_local":
        from pipeline.stage4_commentary.adapters.qwen_local import QwenLocalAdapter
        return QwenLocalAdapter()
    if backend == "doubao":
        from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter
        return DoubaoAPIAdapter()
    if backend == "openai":
        from pipeline.stage4_commentary.adapters.openai_api import OpenAIAPIAdapter
        return OpenAIAPIAdapter()
    raise ValueError(f"Unknown LLM backend: {backend}")


def generate_commentary(
    events_json_path: Path,
    output_path: Path,
    adapter: Optional[LLMAdapter] = None,
    config: Optional[PipelineConfig] = None,
    topo_json_path: Optional[Path] = None,
    visual_input: Optional[Union[Path, List[Path]]] = None,
    schema: Optional[EventSchema] = None,
) -> Path:
    """Build prompt, call LLM adapter (Doubao by default), write commentary.json."""
    config = config or PipelineConfig()
    adapter = adapter or build_adapter(config.llm_backend)
    schema = schema or EventSchema()

    roster = None
    if config.roster_json and Path(config.roster_json).exists():
        with open(config.roster_json, encoding="utf-8") as f:
            roster = json.load(f)

    prompt = build_commentary_prompt(
        events_json_path,
        schema,
        config.languages,
        topo_json_path=topo_json_path,
        roster=roster,
    )
    raw_output = adapter.generate(prompt, visual_input)
    segments = parse_commentary_output(raw_output)

    model_name = getattr(adapter, "model", config.llm_backend)
    video_info = {
        "source": str(config.clip_dir),
        "duration_s": 30.0,
        "fps": config.fps,
    }
    model_info = {
        "name": model_name,
        "backend": config.llm_backend,
        "temperature": config.llm_temperature,
    }
    return write_commentary_json(
        segments,
        output_path,
        video_info,
        model_info,
        config.languages,
    )
