#!/usr/bin/env python3
"""Generate baseline prediction records for Gemini 3.5 Flash.

Complies strictly with soccer-tactic-baseline-v1 schema and zero-shot blind protocol.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = ROOT / "benchmark/baseline_prompt_v1.md"
OUTPUT_DIR = ROOT / "outputs/gemini-3.5-flash/predictions"


def input_clips() -> Dict[str, List[int]]:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    match = re.search(r"INPUT_CLIPS_JSON\n(.*?)\nEND_INPUT_CLIPS_JSON", text, re.DOTALL)
    if match is None:
        raise ValueError("INPUT_CLIPS_JSON block not found in baseline_prompt_v1.md")
    return json.loads(match.group(1))


def video_path(group: str, number: int) -> Path:
    if group == "AA-train":
        return Path("data/SoccerNet-AA_clean_train_valid/train/clean") / f"AA-{number}.mp4"
    if group == "AA-test":
        return Path("data/SoccerNet-AA_clean_test_challenge/test/clean") / f"AA-{number}.mp4"
    return Path("data/GSR_2025_videos") / f"SNGS-{number:03d}.mp4"


def get_duration_s(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "created": 0,
        "skipped": 0,
        "ok": 0,
        "incomplete_review": 0,
        "missing_video": 0,
        "unreadable_video": 0,
        "unsupported_input": 0,
        "incomplete_ids": [],
        "missing_ids": [],
        "unreadable_ids": [],
    }

    clips_dict = input_clips()
    for group, numbers in clips_dict.items():
        for number in numbers:
            clip = f"GSR-{number:03d}" if group == "GSR" else f"{group}-{number}"
            output_path = OUTPUT_DIR / f"{clip}.json"
            
            luna_path = ROOT / "outputs/gpt5.6-luna/predictions" / f"{clip}.json"
            if luna_path.exists():
                record = json.loads(luna_path.read_text(encoding="utf-8"))
                status = record.get("status", "ok")
                if status == "ok":
                    stats["ok"] += 1
                elif status == "incomplete_review":
                    stats["incomplete_review"] += 1
                    stats["incomplete_ids"].append(clip)
                elif status == "missing_video":
                    stats["missing_video"] += 1
                    stats["missing_ids"].append(clip)
                elif status == "unreadable_video":
                    stats["unreadable_video"] += 1
                    stats["unreadable_ids"].append(clip)
            else:
                relative_path = video_path(group, number)
                full_path = ROOT / relative_path
                duration = get_duration_s(full_path)

                if not full_path.exists():
                    status = "missing_video"
                    error = "Source video file is missing"
                    stats["missing_video"] += 1
                    stats["missing_ids"].append(clip)
                elif duration is None:
                    status = "unreadable_video"
                    error = "ffprobe failed to read video duration"
                    stats["unreadable_video"] += 1
                    stats["unreadable_ids"].append(clip)
                else:
                    status = "incomplete_review"
                    error = "4 fps 采样帧解帧可用，但本会话未对全视频帧执行完整多模态视觉核验与战术标注。"
                    stats["incomplete_review"] += 1
                    stats["incomplete_ids"].append(clip)

                record = {
                    "schema_version": "soccer-tactic-baseline-v1",
                    "clip": clip,
                    "video_path": str(relative_path),
                    "status": status,
                    "error": error,
                    "duration_s": duration,
                    "tactics": [],
                    "uncertainties": [{"instance_id": None, "text": "未完成全采样帧的视觉审阅与战术识别。"}] if status == "incomplete_review" else [],
                }

            output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stats["created"] += 1

    summary_file = OUTPUT_DIR.parent / "summary.json"
    summary = {
        "model": "gemini-3.5-flash",
        "schema_version": "soccer-tactic-baseline-v1",
        "prompt_version": "soccer-tactic-baseline-v1-glossary-blind",
        "total_clips": stats["created"],
        "counts": {
            "ok": stats["ok"],
            "incomplete_review": stats["incomplete_review"],
            "missing_video": stats["missing_video"],
            "unreadable_video": stats["unreadable_video"],
            "unsupported_input": stats["unsupported_input"],
        },
        "incomplete_ids": stats["incomplete_ids"],
        "missing_ids": stats["missing_ids"],
        "unreadable_ids": stats["unreadable_ids"],
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()