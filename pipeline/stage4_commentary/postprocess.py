"""Parse and validate LLM output into commentary.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List


def _normalize_segment(seg: dict, index: int) -> dict:
    """Map alternate LLM field names onto the pipeline schema."""
    text_en = (
        seg.get("text_en")
        or seg.get("en_commentary")
        or seg.get("en")
        or seg.get("commentary_en")
        or ""
    )
    text_zh = (
        seg.get("text_zh")
        or seg.get("zh_commentary")
        or seg.get("zh")
        or seg.get("commentary_zh")
        or ""
    )
    timestamp_s = seg.get("timestamp_s", seg.get("start_s", seg.get("t", index * 1.5)))
    end_s = seg.get("end_s", float(timestamp_s) + 1.5)
    events_referenced = seg.get("events_referenced") or seg.get("event_ids") or []
    if isinstance(events_referenced, str):
        events_referenced = [events_referenced]
    out = {
        "timestamp_s": float(timestamp_s),
        "end_s": float(end_s),
        "text_en": str(text_en),
        "text_zh": str(text_zh),
        "events_referenced": list(events_referenced),
    }
    if seg.get("event_code"):
        out["event_code"] = seg["event_code"]
    return out


def parse_commentary_output(raw_text: str) -> List[dict]:
    """Extract JSON array of commentary segments from LLM output."""
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                return [
                    _normalize_segment(seg, i)
                    for i, seg in enumerate(data)
                    if isinstance(seg, dict)
                ]
        except json.JSONDecodeError:
            pass
    return [{
        "timestamp_s": 0.0,
        "end_s": 30.0,
        "text_en": raw_text,
        "text_zh": raw_text,
        "events_referenced": [],
    }]


def parse_visual_tags(raw_text: str) -> dict:
    """Extract JSON dict of visual tags from LLM output."""
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def parse_tactical_analysis(raw_text: str) -> dict:
    """Extract tactical analysis JSON from LLM output."""
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"text_en": raw_text, "text_zh": raw_text, "key_factors": []}


def write_commentary_json(
    commentary_segments: List[dict],
    output_path: Path,
    video_info: dict,
    model_info: dict,
    languages: List[str],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info,
        "model_info": model_info,
        "language": languages,
        "commentary": commentary_segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path
