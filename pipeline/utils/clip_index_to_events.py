"""Convert clip_index.csv rows to events.json fixture format."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

ACTION_TO_EVENT_CODE = {
    "shoot": "football.shoot",
    "goal": "football.goal",
    "clearance": "football.clearance",
    "foul": "football.foul",
    "corner": "football.set_piece",
    "free_kick": "football.set_piece",
    "kick_off": "football.set_piece",
    "penalty": "football.set_piece",
    "offside": None,
    "yellow_card": None,
    "substitution": None,
}

ACTION_IMPORTANCE = {
    "goal": 1.0,
    "shoot": 0.55,
    "penalty": 0.9,
    "clearance": 0.4,
    "foul": 0.35,
    "corner": 0.2,
    "free_kick": 0.2,
    "kick_off": 0.1,
}

ACTION_CN = {
    "shoot": "射门",
    "goal": "进球",
    "clearance": "解围",
    "foul": "犯规",
    "corner": "角球",
    "free_kick": "任意球",
    "kick_off": "开球",
    "penalty": "点球",
}


def clip_index_row_to_event(
    sample_id: str,
    event_time_sec: float,
    normalized_action: str,
    fps: int = 25,
) -> Optional[dict]:
    """Convert one clip_index.csv row to an events.json event dict."""
    event_code = ACTION_TO_EVENT_CODE.get(normalized_action)
    if event_code is None:
        return None

    timestamp_s = round(event_time_sec)
    frame_id = int(timestamp_s * fps)

    return {
        "event_id": "evt_001",
        "timestamp_s": float(timestamp_s),
        "frame_id": frame_id,
        "event_code": event_code,
        "display_name_en": normalized_action.replace("_", " ").title(),
        "display_name_cn": ACTION_CN.get(normalized_action, normalized_action),
        "importance": ACTION_IMPORTANCE.get(normalized_action, 0.3),
        "player_jersey": None,
        "player_team": None,
        "tags": {},
        "confidence": 1.0,
        "description_hint": f"from clip_index.csv: {sample_id}",
    }


def generate_events_fixture(
    clip_index_csv: Path,
    sample_id: str,
    output_path: Path,
    fps: int = 25,
) -> Path:
    """Generate events.json fixture for a specific test clip from clip_index.csv."""
    events = []
    with open(clip_index_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample_id"].strip() != sample_id:
                continue
            ev = clip_index_row_to_event(
                sample_id=row["sample_id"].strip(),
                event_time_sec=float(row["event_time_sec"]),
                normalized_action=row["normalized_action"].strip(),
                fps=fps,
            )
            if ev:
                events.append(ev)

    data = {
        "video_info": {
            "source": f"{sample_id}/img1/",
            "fps": fps,
            "duration_s": 30.0,
            "total_frames": fps * 30,
        },
        "schema_version": "v3-20260319",
        "events": events,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path
