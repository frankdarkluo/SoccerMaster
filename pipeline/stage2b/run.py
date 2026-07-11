"""Stage 2B direct and hybrid commentary command."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from pipeline.atomic import atomic_copy, atomic_write_json
from pipeline.config import PipelineConfig
from pipeline.relations.build import build_relations
from pipeline.relations.radar import render_radar_frames
from pipeline.stage2b.digest import build_tracking_digest, load_frames
from pipeline.stage2b.events import get_event
from pipeline.stage2b.generate import ark_chat, observe_direct, verify_event_window
from pipeline.stage2b.hybrid import (
    _fallback_preserves_event, assign_confidence, audit_commentary,
    candidate_windows, compose_hybrid, concise_event_wording, structured_wording,
    verify_candidates,
)
from pipeline.stage2b.video import video_duration_s

log = logging.getLogger(__name__)


def _clip_path(output_dir: Path, clip_dir: Path) -> Path:
    for path in (output_dir / "clip.mp4", clip_dir / "clip.mp4"):
        if path.is_file():
            return path
    raise FileNotFoundError("clip.mp4 not found in output or clip directory")


def _propose_candidates(relations: dict, events: list[dict], windows: list[dict], duration_s: float,
                        call: Callable) -> list[dict]:
    prompt = "Propose sparse tactical candidates from these computed relations. Return a JSON array only.\n"
    prompt += json.dumps({"events": events, "relations": relations,
                          "approved_windows": windows,
                          "duration_s": duration_s}, ensure_ascii=False)
    try:
        proposed = json.loads(call(prompt, temperature=0.2))
    except (TypeError, json.JSONDecodeError):
        proposed = []
    return verify_candidates(relations, proposed, windows)


def _cache_complete(config: PipelineConfig, mode: str) -> bool:
    if not config.commentary_json.is_file():
        return False
    hybrid = (
        config.commentary_direct_json.is_file()
        and config.tactical_candidates_json.is_file()
        and config.relations_json.is_file()
    )
    return hybrid if mode == "hybrid" else not hybrid


def _state_ok(event: dict, frames: list, fps: float) -> bool:
    """Reject a restart only when available tracking shows the other team on the ball."""
    if event.get("event_code") not in {"football.corner", "football.goal_kick"}:
        return True
    team = event.get("player_team")
    if team not in {"left", "right"}:
        return True
    holders = []
    start, end = float(event["start_s"]) - 1.0, float(event["end_s"]) + 1.0
    for frame in frames:
        time_s = (frame.frame_id - 1) / fps
        if not start <= time_s <= end or frame.ball_xy is None:
            continue
        bx, by = frame.ball_xy
        nearby = [
            player for player in frame.players
            if player.get("team") in {"left", "right"}
            and (player["x"] - bx) ** 2 + (player["y"] - by) ** 2 <= 9.0
        ]
        if nearby:
            holder = min(nearby, key=lambda player:
                         (player["x"] - bx) ** 2 + (player["y"] - by) ** 2)
            holders.append(holder["team"])
    return not holders or team in holders


def _verify_events(clip: Path, events: list[dict], direct: list[dict], frames: list,
                   fps: float, call: Callable) -> list[dict]:
    narrated = {
        event_id for segment in direct for event_id in segment.get("events_referenced", [])
    }
    for event in events:
        definition = get_event(event.get("event_code"))
        reasons = event.get("confidence_reasons", [])
        eligible = (
            event.get("confidence") == "high"
            or (definition is not None and definition.importance_base >= 0.35)
            or any("ambigu" in reason or "disagree" in reason for reason in reasons)
            or (event.get("event_id") in narrated and event.get("confidence") != "low")
        )
        if not eligible:
            continue
        verification = verify_event_window(clip, event, call=call)
        state_ok = _state_ok(event, frames, fps)
        event["verification"] = verification
        event["state_ok"] = state_ok
        event["confidence"] = assign_confidence(event, verification, state_ok)
        event["confidence_reasons"] = list(dict.fromkeys([
            *reasons,
            *( ["two_pass_agreement"] if event["confidence"] == "high" else [] ),
        ]))
    return events


def _required_event(event: dict) -> bool:
    if event.get("confidence") == "high":
        return True
    visible = (
        "directly_visible" in event.get("confidence_reasons", [])
        or event.get("verification", {}).get("directly_visible") is True
    )
    return event.get("confidence") == "medium" and visible


def _event_wording(event: dict, segment: dict) -> tuple[str, str, str, str]:
    zh, en = structured_wording(event)
    if event.get("confidence") == "high":
        suggested_zh = event.get("suggested_wording_zh") or segment.get("text_zh", "")
        suggested_en = event.get("suggested_wording_en") or segment.get("text_en", "")
        probe = {"fallback_text_zh": suggested_zh, "fallback_text_en": suggested_en}
        if _fallback_preserves_event(probe, event):
            zh, en = suggested_zh, suggested_en
    fallback_zh, fallback_en = concise_event_wording(event)
    return zh, en, fallback_zh, fallback_en


def _reconcile_direct(events: list[dict], direct: list[dict], duration_s=None) -> list[dict]:
    """Rebuild direct narration from final required events only."""
    required = {
        event.get("event_id"): event for event in events
        if isinstance(event, dict) and _required_event(event)
    }
    reconciled = []
    covered = set()
    for source in direct:
        refs = [ref for ref in source.get("events_referenced", []) if ref in required]
        if not refs:
            continue
        wordings = [
            _event_wording(required[ref], source)
            for ref in refs
        ]
        segment = dict(source)
        segment.update(
            kind="event",
            text_zh="".join(wording[0] for wording in wordings),
            text_en=" ".join(wording[1] for wording in wordings),
            fallback_text_zh="".join(wording[2] for wording in wordings),
            fallback_text_en=" ".join(wording[3] for wording in wordings),
            events_referenced=refs,
            tactical_candidates_referenced=[],
            event_claims=[{
                "event_id": ref,
                "event_code": required[ref].get("event_code"),
                "player_team": required[ref].get("player_team"),
                "outcome": required[ref].get("outcome"),
                "assertion_strength": (
                    "certain" if required[ref].get("confidence") == "high" else "qualified"
                ),
            } for ref in refs],
        )
        if segment.get("end_s", 0) > segment.get("timestamp_s", 0):
            reconciled.append(segment)
        covered.update(refs)
    reconciled.sort(key=lambda segment: segment["timestamp_s"])
    for index in range(1, len(reconciled)):
        reconciled[index]["timestamp_s"] = max(
            reconciled[index]["timestamp_s"], reconciled[index - 1]["end_s"]
        )
    reconciled = [
        segment for segment in reconciled
        if segment["end_s"] > segment["timestamp_s"]
    ]
    covered = {
        ref for segment in reconciled for ref in segment["events_referenced"]
    }
    for event_id, event in required.items():
        if event_id in covered:
            continue
        zh, en, fallback_zh, fallback_en = _event_wording(event, {})
        start, end = float(event["start_s"]), float(event["end_s"])
        limit = float(duration_s) if duration_s is not None else max(
            [end, *(segment["end_s"] for segment in reconciled), 0.1]
        )
        if end <= start:
            end = min(limit, start + 0.1)
            if end <= start:
                start = max(0.0, end - 0.1)
        cursor = start
        free = None
        overlapping = []
        for segment in reconciled:
            segment_start, segment_end = segment["timestamp_s"], segment["end_s"]
            overlap = min(end, segment_end) - max(start, segment_start)
            if overlap > 0:
                overlapping.append((overlap, segment))
            if segment_end <= cursor or segment_start >= end:
                continue
            if segment_start > cursor:
                free = (cursor, min(segment_start, end))
                break
            cursor = max(cursor, segment_end)
        if free is None and cursor < end:
            free = (cursor, end)
        claim = {
            "event_id": event_id, "event_code": event.get("event_code"),
            "player_team": event.get("player_team"), "outcome": event.get("outcome"),
            "assertion_strength": (
                "certain" if event.get("confidence") == "high" else "qualified"
            ),
        }
        if free is None:
            if not overlapping:
                raise ValueError(f"required event {event_id} has no positive commentary slot")
            segment = max(overlapping, key=lambda item: item[0])[1]
            segment["text_zh"] += zh
            segment["text_en"] += " " + en
            segment["fallback_text_zh"] += fallback_zh
            segment["fallback_text_en"] += " " + fallback_en
            segment["events_referenced"].append(event_id)
            segment["event_claims"].append(claim)
            covered.add(event_id)
            continue
        reconciled.append({
            "kind": "event", "timestamp_s": free[0], "end_s": free[1],
            "text_zh": zh, "text_en": en,
            "fallback_text_zh": fallback_zh, "fallback_text_en": fallback_en,
            "energy": event.get("energy", "engaged"),
            "events_referenced": [event_id], "tactical_candidates_referenced": [],
            "event_claims": [claim],
        })
        reconciled.sort(key=lambda segment: segment["timestamp_s"])
        covered.add(event_id)
    duration = float(duration_s) if duration_s is not None else max(
        [segment["end_s"] for segment in reconciled] or [0.1]
    )
    errors = audit_commentary(reconciled, events, [], duration)
    if errors:
        raise ValueError("invalid reconciled direct commentary: " + "; ".join(errors))
    return reconciled


def run_stage2b(output_dir, clip_dir, mode="hybrid", force=False,
                call=ark_chat, duration_s=None) -> Path:
    """Write the Stage 2B comments contract and return final commentary path."""
    output_dir, clip_dir = Path(output_dir), Path(clip_dir)
    if mode not in {"direct", "hybrid"}:
        raise ValueError("mode must be 'direct' or 'hybrid'")
    config = PipelineConfig(output_dir=output_dir, clip_dir=clip_dir)
    if not force and _cache_complete(config, mode):
        return config.commentary_json

    predictions = output_dir / "predictions.json"
    if not predictions.is_file():
        raise FileNotFoundError(f"missing predictions: {predictions}")
    clip = _clip_path(output_dir, clip_dir)
    duration = float(duration_s) if duration_s is not None else video_duration_s(clip)
    digest = build_tracking_digest(predictions, fps=float(config.fps))
    events, direct = observe_direct(
        clip, digest, duration, config.languages, call=call,
    )
    frames = load_frames(predictions)
    events = _verify_events(clip, events, direct, frames, float(config.fps), call)
    direct = _reconcile_direct(events, direct, duration)
    atomic_write_json(config.events_json, events)
    atomic_write_json(config.event_spine_json, events)

    if mode == "direct":
        for path in (config.commentary_direct_json, config.tactical_candidates_json,
                     config.relations_json):
            path.unlink(missing_ok=True)
        return atomic_write_json(config.commentary_json, direct)

    atomic_write_json(config.commentary_direct_json, direct)
    try:
        relations = build_relations(
            frames, fps=float(config.fps), snapshot_hz=config.snapshot_hz,
        )
        atomic_write_json(config.relations_json, relations)
        render_radar_frames(relations, config.radar_dir, hz=config.radar_hz)
    except Exception:
        log.exception("Stage 2B relations/radar failed; using direct commentary")
        atomic_copy(config.commentary_direct_json, config.commentary_json)
        return config.commentary_json

    windows = candidate_windows(events, duration, "incomplete_attack")
    candidates = _propose_candidates(relations, events, windows, duration, call)
    atomic_write_json(config.tactical_candidates_json, candidates)
    commentary = compose_hybrid(
        events, direct, candidates, windows, duration, call=call,
    )
    return atomic_write_json(config.commentary_json, commentary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage 2B football commentary")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["direct", "hybrid"], default="hybrid")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    clip_dir = args.clip_dir or args.output_dir
    result = run_stage2b(args.output_dir, clip_dir, mode=args.mode, force=args.force)
    print(result)


if __name__ == "__main__":
    main()
