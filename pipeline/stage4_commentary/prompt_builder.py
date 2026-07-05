"""Build LLM prompts from events.json + topo.json + schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.stage2_events.schema import EventSchema


def build_commentary_prompt(
    events_json_path: Path,
    schema: EventSchema,
    languages: List[str],
    topo_json_path: Optional[Path] = None,
    roster: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    with open(events_json_path, encoding="utf-8") as f:
        events_data = json.load(f)

    parts = []

    lang_str = " and ".join(languages).upper()
    parts.append(f"""You are a professional football commentator. Generate second-by-second commentary for this 30-second match clip.

RULES:
1. Use ONLY the timestamps from the event list. Never invent timestamps.
2. Use the EXACT terminology from the tag display names shown in parentheses.
3. For gaps between events, describe formations, positioning, build-up play.
4. Refer to players by jersey number (or name if roster provided).
5. Generate BOTH {lang_str} commentary for each segment.
6. For events marked HIGHLIGHT, use more excited/vivid language.
7. Output a valid JSON array only. Each object MUST use these keys:
   timestamp_s (number), end_s (number), text_en (string), text_zh (string),
   events_referenced (array of event id strings, e.g. ["evt_070"]).
   Do not use alternate key names.

PACING (critical for TTS):
8. Each segment will be spoken aloud. Keep text short enough to fit the time window.
   Budget: Chinese ≤ 5 characters per second, English ≤ 3.3 words per second.
   Example: a 3-second window → max ~15 Chinese chars or ~10 English words.
9. Do NOT create segments shorter than 2 seconds. Merge rapid events into one segment.
10. Lines are spoken back-to-back with no pauses, so fill each window fully — do not leave the window half empty.""")

    parts.append("\n[Event Definitions]")
    parts.append(schema.event_definitions_for_prompt())

    parts.append("\n[Event Timeline]")
    for ev in events_data.get("events", []):
        event_id = ev.get("event_id", "")
        line = f"t={ev['timestamp_s']}s: [{ev['event_code']}]"
        if event_id:
            line += f" id={event_id}"
        if ev.get("player_jersey"):
            line += f" #{ev['player_jersey']}"
        if ev.get("player_team"):
            line += f" ({ev['player_team']})"
        if ev.get("target_jersey"):
            line += f" → #{ev['target_jersey']} ({ev.get('target_team', '')})"
        tags = ev.get("tags", {})
        if tags:
            tag_parts = []
            for key, value in tags.items():
                tg = schema.get_tag_group(key)
                if tg:
                    tv = next((item for item in tg.values if item.code == value), None)
                    if tv:
                        tag_parts.append(f"{key}={value}({tv.display_name_cn})")
                    else:
                        tag_parts.append(f"{key}={value}")
                else:
                    tag_parts.append(f"{key}={value}")
            line += f"\n  Tags: {', '.join(tag_parts)}"
        if ev.get("importance", 0) >= 0.5:
            line += "  ⚡ HIGHLIGHT"
        parts.append(line)

    if topo_json_path and topo_json_path.exists():
        parts.append("\n[Formation Context]")
        with open(topo_json_path, encoding="utf-8") as f:
            topo = json.load(f)
        records = topo if isinstance(topo, list) else topo.get("records", [])
        for record in records[:6]:
            t = record.get("window_start_s", "?")
            team = record.get("team", "?")
            height = record.get("block_height_m")
            depth = record.get("block_depth_m")
            parts.append(f"t~{t}s {team}: height={height}m, depth={depth}m")

    if roster:
        parts.append("\n[Player Roster]")
        for team, players in roster.items():
            parts.append(f"{team}: {json.dumps(players, ensure_ascii=False)}")

    return "\n".join(parts)


def build_visual_tag_prompt(event: dict, schema: EventSchema) -> str:
    """Build prompt for Step 1: LLM fills visual tags from video."""
    event_code = event["event_code"]
    vocab = schema.tag_vocabulary_for_prompt(event_code)
    if not vocab:
        return ""

    return f"""You are a football video analyst. For the event below, watch the video clip and fill in the visual tags. Use ONLY values from the provided vocabulary. Output JSON only, no explanation.

[Event]
t={event['timestamp_s']}s: {event_code} by #{event.get('player_jersey', '?')} ({event.get('player_team', '?')})

[Tag Vocabulary]
{vocab}"""


def build_tactical_reasoning_prompt(event: dict, topo_before: dict, topo_at: dict) -> str:
    """Build prompt for Step 3: tactical reasoning for goals/shots."""
    return f"""You are a football tactical analyst. Explain WHY this event succeeded or was dangerous.

Analyze:
1. What defensive weakness was exploited?
2. What attacking movement created the opportunity?
3. Which players' positioning was critical?
4. Could the defense have prevented it?

Use the topology metrics with specific numbers to support your analysis.

[Formation Data: Before event]
{json.dumps(topo_before, indent=2, ensure_ascii=False)}

[Formation Data: At event]
{json.dumps(topo_at, indent=2, ensure_ascii=False)}

[Event Details]
{json.dumps(event, indent=2, ensure_ascii=False)}

Output JSON with keys: text_en, text_zh, key_factors (list of strings)."""
