"""VLM verification of rule-engine candidates.

Rules propose candidates (timestamp + actor); a VLM (Doubao API default, Qwen
local switchable) watches a structurally-annotated frame-burst and returns a
Verdict: did the action happen (confirm/reject/uncertain), did it succeed
(success/failure), corrected actor, and an optional family-constrained retype.

Fail-hard: no try/except around infra or logic. The ONLY guard is model-text
-> JSON parsing, which maps malformed output to a deterministic 'uncertain'.
Per-candidate verdict cache makes crash+rerun cheap.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.stage2_events.evidence import build_bbox_index, build_evidence_frames, load_homography
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Event
from pipeline.stage2_events.types import Verdict

DUEL_CODES = {"football.pass", "football.clearance", "football.dribble", "football.interception"}
SCORING_CODES = {"football.shoot", "football.goal"}
VERIFY_CODES = DUEL_CODES | SCORING_CODES
FAMILY = {
    "football.pass": DUEL_CODES,
    "football.clearance": DUEL_CODES,
    "football.dribble": DUEL_CODES,
    "football.interception": DUEL_CODES,
    "football.shoot": SCORING_CODES,
    "football.goal": SCORING_CODES,
}

_SUCCESS = {
    "success",
    "successful",
    "succeeded",
    "true",
    "yes",
    "1",
    "complete",
    "completed",
    "won",
}
_FAILURE = {
    "failure",
    "failed",
    "fail",
    "unsuccessful",
    "false",
    "no",
    "0",
    "incomplete",
    "lost",
}


def normalize_outcome(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _SUCCESS:
        return "success"
    if s in _FAILURE:
        return "failure"
    return None


def parse_verdict(raw: str) -> Verdict:
    """Extract the first JSON object from model text. Narrow guard: malformed -> uncertain."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return Verdict(verdict="uncertain", reason="no-json")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(verdict="uncertain", reason="bad-json")
    verdict = str(data.get("verdict") or "uncertain").strip().lower()
    if verdict not in ("confirm", "reject", "uncertain"):
        verdict = "uncertain"
    actor_jersey = data.get("actor_jersey")
    receiver_jersey = data.get("receiver_jersey")
    corrected_event_code = data.get("corrected_event_code")
    return Verdict(
        verdict=verdict,
        outcome=normalize_outcome(data.get("outcome")),
        actor_jersey=(str(actor_jersey).strip() or None) if actor_jersey else None,
        actor_team=data.get("actor_team") if data.get("actor_team") in ("left", "right") else None,
        receiver_jersey=(str(receiver_jersey).strip() or None) if receiver_jersey else None,
        corrected_event_code=(
            (str(corrected_event_code).strip() or None) if corrected_event_code else None
        ),
        reason=str(data.get("reason") or "")[:400],
    )


def build_verify_prompt(event: Event) -> str:
    jersey = event.player_jersey or "unknown"
    team = event.player_team or "unknown"
    name, name_cn = event.display_name_en, event.display_name_cn
    family = sorted(FAMILY.get(event.event_code, {event.event_code}))
    retypes = ", ".join(family)
    outcome_line = (
        f"2) outcome — if confirmed, did the {name} SUCCEED? "
        "success = won/completed (interception wins the ball; pass reaches a teammate; "
        "dribble beats the man; clearance removes danger). "
        "failure = attempted but lost/incomplete. Omit unless verdict is confirm.\n"
        if event.event_code in DUEL_CODES
        else "2) outcome — leave null for this event type.\n"
    )
    return (
        "You are a football video analyst. A tracking system flagged a candidate "
        f'"{name}" ({name_cn}) at t={event.timestamp_s:.2f}s by the highlighted player '
        f"(red box): jersey #{jersey}, {team} team. Frames show the ball (green circle) "
        "and, for shots, a yellow arrow toward the goal.\n"
        "Answer these questions:\n"
        f"1) verdict — did this player ATTEMPT a {name} here? confirm / reject / uncertain.\n"
        f"{outcome_line}"
        f"3) corrected_event_code — if it is actually a different action, pick ONE of: "
        f"{retypes}. Else null.\n"
        "4) actor_jersey / actor_team(left|right) — correct the actor if the red box is wrong.\n"
        "Output ONLY JSON:\n"
        '{"verdict": "confirm|reject|uncertain", "outcome": "success|failure|null", '
        '"corrected_event_code": "<one of the codes or null>", '
        '"actor_jersey": "<number or empty>", "actor_team": "left|right|", '
        '"receiver_jersey": "<number or empty>", "reason": "<short>"}'
    )


def apply_verdict(event: Event, verdict: Verdict, schema: EventSchema) -> Optional[Event]:
    """Dispose a candidate per verdict. Returns None to drop. Mutates+returns otherwise."""
    if verdict.verdict == "reject":
        return None

    if verdict.verdict == "uncertain":
        event.confidence = round(event.confidence * 0.5, 2)
        event.tags["verified"] = "uncertain"
        return event

    event.tags["verified"] = "true"

    new_code = verdict.corrected_event_code
    if new_code and new_code != event.event_code and new_code in FAMILY.get(event.event_code, set()):
        ev_def = schema.get_event(new_code)
        event.event_code = new_code
        event.display_name_en = ev_def.display_name_en
        event.display_name_cn = ev_def.display_name_cn
        event.importance = ev_def.importance_base

    if event.event_code in DUEL_CODES and verdict.outcome is not None:
        event.tags["outcome"] = verdict.outcome
        if verdict.outcome == "failure":
            event.confidence = round(event.confidence * 0.5, 2)
            event.importance = round(event.importance * 0.5, 2)

    if verdict.actor_jersey:
        event.player_jersey = verdict.actor_jersey
    if verdict.actor_team in ("left", "right"):
        event.player_team = verdict.actor_team
    if verdict.receiver_jersey and event.event_code == "football.pass":
        event.target_jersey = verdict.receiver_jersey
        event.target_team = event.player_team
    return event


VERIFY_TEMP_DIRS = ("verify_cache", "verify_clips")


def cleanup_verify_artifacts(output_dir: Path) -> None:
    for name in VERIFY_TEMP_DIRS:
        p = Path(output_dir) / name
        if p.is_dir():
            shutil.rmtree(p)


def verify_events(
    events: List[Event],
    predictions_json_path: str,
    frames_dir: Path,
    output_dir: Path,
    adapter,
    homography_path: str,
    fps: int = 25,
    window_s: float = 0.5,
    force: bool = False,
) -> Tuple[List[Event], List[dict]]:
    output_dir = Path(output_dir)
    cache_dir = output_dir / "verify_cache"
    clip_dir = output_dir / "verify_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    schema = EventSchema()
    bbox_index = build_bbox_index(predictions_json_path)
    homo = load_homography(homography_path)

    verified: List[Event] = []
    audit: List[dict] = []

    for event in events:
        if event.event_code not in VERIFY_CODES or event.track_id is None:
            verified.append(event)
            continue

        cache_path = cache_dir / f"{event.event_id}.json"
        if cache_path.exists() and not force:
            verdict = Verdict(**json.loads(cache_path.read_text(encoding="utf-8")))
        else:
            frames = build_evidence_frames(
                event,
                frames_dir,
                bbox_index,
                homo,
                predictions_json_path,
                clip_dir / event.event_id,
                fps=fps,
                window_s=window_s,
            )
            if not frames:
                verdict = Verdict(verdict="uncertain", reason="no-frames")
            else:
                raw = adapter.generate(build_verify_prompt(event), frames)
                verdict = parse_verdict(raw)
            cache_path.write_text(
                json.dumps(verdict.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        actor_jersey = event.player_jersey
        actor_team = event.player_team
        kept = apply_verdict(event, verdict, schema)
        audit.append({
            "event_id": event.event_id,
            "event_code": event.event_code,
            "timestamp_s": event.timestamp_s,
            "player_jersey": actor_jersey,
            "player_team": actor_team,
            "verdict": verdict.verdict,
            "outcome": verdict.outcome,
            "corrected_event_code": verdict.corrected_event_code,
            "reason": verdict.reason,
            "kept": kept is not None,
        })
        if kept is not None:
            verified.append(kept)

    (output_dir / "events_verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return verified, audit
