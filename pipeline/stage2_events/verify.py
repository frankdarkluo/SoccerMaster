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
from typing import Optional

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
