"""Verified tactical facts and event-first hybrid commentary."""
from __future__ import annotations

import json
import math
from typing import Callable

from pipeline.stage2b.generate import ark_chat
from pipeline.stage2b.events import get_event

ENERGIES = {"calm", "engaged", "excited", "explosive"}
KINDS = {"event", "hybrid", "tactical"}
ASSERTION_STRENGTHS = {"certain", "qualified"}
EVENT_ALIASES_EN = {
    "football.corner": ("corner",),
    "football.pass": ("pass",),
    "football.clearance": ("clearance", "clear"),
    "football.interception": ("interception", "intercept"),
    "football.dribble": ("dribble",),
    "football.tackle": ("tackle",),
    "football.shoot": ("shoot", "shot"),
    "football.goal": ("goal",),
    "football.save": ("save",),
    "football.goal_kick": ("goal kick", "goal-kick"),
    "football.buildup": ("buildup", "build-up", "build up"),
    "football.pressing": ("pressing", "press"),
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _midpoint(event):
    return (float(event.get("start_s", 0.0)) + float(event.get("end_s", 0.0))) / 2.0


def _compatible(left, right):
    return not left or not right or left == right


def assign_confidence(event, verification, state_ok):
    """Assign an evidence-gated confidence tier to a proposed event."""
    if not isinstance(event, dict) or not isinstance(verification, dict):
        return "low"
    conflicts = (
        not _compatible(event.get("event_code"), verification.get("event_code"))
        or not _compatible(event.get("player_team"), verification.get("player_team"))
        or not _compatible(event.get("outcome"), verification.get("outcome"))
        or (bool(event.get("player_jersey"))
            and event.get("player_jersey") != verification.get("player_jersey"))
        or bool(verification.get("disagreements"))
        or state_ok is False
    )
    midpoint = verification.get("midpoint_s")
    if _finite(midpoint) and abs(_midpoint(event) - midpoint) > 1.0:
        conflicts = True
    if conflicts:
        return "low"
    high = (
        event.get("event_code") == verification.get("event_code")
        and _finite(midpoint)
        and abs(_midpoint(event) - midpoint) <= 1.0
        and _compatible(event.get("player_team"), verification.get("player_team"))
        and _compatible(event.get("outcome"), verification.get("outcome"))
        and (not event.get("player_jersey")
             or event.get("player_jersey") == verification.get("player_jersey"))
        and verification.get("directly_visible") is True
        and verification.get("disagreements") == []
        and state_ok is True
    )
    return "high" if high else "medium"


def _required_event(event):
    if event.get("confidence") == "high":
        return True
    reasons = event.get("confidence_reasons", [])
    verification = event.get("verification", {})
    return event.get("confidence") == "medium" and (
        "directly_visible" in reasons or verification.get("directly_visible") is True
    )


def commentary_windows(facts, duration_s):
    """Schedule speech after fact decision times without changing evidence windows."""
    if not _finite(duration_s) or duration_s <= 0:
        return []
    eligible = [
        fact for fact in facts if isinstance(fact, dict)
        and isinstance(fact.get("fact_id"), str)
        and _finite(fact.get("decision_time_s"))
    ] if isinstance(facts, list) else []
    eligible.sort(key=lambda fact: (fact["decision_time_s"], fact["fact_id"]))
    windows = []
    cursor = 0.0
    for fact in eligible:
        start = max(cursor, float(fact["decision_time_s"]))
        end = min(float(duration_s), start + 5.0)
        if end <= start:
            continue
        windows.append({
            "window_id": f"window_{len(windows) + 1:03d}",
            "fact_id": fact["fact_id"],
            "start_s": start,
            "end_s": end,
        })
        cursor = end
    return windows

def _event_map(events):
    return {
        event.get("event_id"): event for event in events
        if isinstance(event, dict) and isinstance(event.get("event_id"), str)
    }



def _known_jersey(event):
    """A jersey only counts when it holds digits; '?' style placeholders don't."""
    jersey = str(event.get("player_jersey") or "")
    return jersey if any(char.isdigit() for char in jersey) else ""


def structured_wording(event):
    """Deterministic wording carrying every token the fallback audit checks."""
    definition = get_event(event.get("event_code"))
    if definition is None:
        return "可见事件。", "A visible event."
    team = event.get("player_team")
    zh_team = {"left": "左侧", "right": "右侧"}.get(team, "")
    en_team = {"left": "Left", "right": "Right"}.get(team, "The visible team")
    jersey = _known_jersey(event)
    if event.get("verification", {}).get("disagreements"):
        jersey = ""
    zh_jersey = f"{jersey}号" if jersey else ""
    en_jersey = f"No. {jersey} " if jersey else ""
    event_name = definition.code.split(".")[-1].replace("_", " ")
    zh = f"{zh_team}{zh_jersey}{definition.display_name_zh}。"
    en = f"{en_team} {en_jersey}{event_name}: {definition.description}."
    return zh, en


def concise_event_wording(event):
    """Short deterministic fallback preserving audited structured event facts."""
    definition = get_event(event.get("event_code"))
    if definition is None:
        return "可见事件。", "Visible event."
    team = event.get("player_team")
    zh_team = {"left": "左侧", "right": "右侧"}.get(team, "")
    en_team = {"left": "Left", "right": "Right"}.get(team, "Visible")
    jersey = _known_jersey(event)
    if event.get("verification", {}).get("disagreements"):
        jersey = ""
    zh_jersey = f"{jersey}号" if jersey else ""
    en_jersey = f" No. {jersey}" if jersey else ""
    event_name = definition.code.split(".")[-1].replace("_", " ")
    return f"{zh_team}{zh_jersey}{definition.display_name_zh}。", f"{en_team}{en_jersey} {event_name}."


def _fallback_preserves_event(segment, event):
    definition = get_event(event.get("event_code"))
    if definition is None:
        return False
    fallback_en = segment.get("fallback_text_en", "").lower()
    fallback_zh = segment.get("fallback_text_zh", "")
    event_aliases_en = EVENT_ALIASES_EN.get(definition.code, ())
    if not any(alias in fallback_en for alias in event_aliases_en) or definition.display_name_zh not in fallback_zh:
        return False
    team = event.get("player_team")
    if team in {"left", "right"}:
        if team not in fallback_en or {"left": "左", "right": "右"}[team] not in fallback_zh:
            return False
    jersey = _known_jersey(event)
    disputed = bool(event.get("verification", {}).get("disagreements"))
    return (not jersey or (event.get("confidence") == "medium" and disputed)
            or (jersey in fallback_en and jersey in fallback_zh))

def _fallback_event_only(segment, event, referenced_events=None):
    allowed_events = referenced_events or [event]
    # Gating stays vacuous when no suggested wording (or medium structured
    # phrase) exists; structured-fact words are always acceptable content.
    gating_en = [item.get("suggested_wording_en", "") for item in allowed_events]
    gating_zh = [item.get("suggested_wording_zh", "") for item in allowed_events]
    allowed_en, allowed_zh = list(gating_en), list(gating_zh)
    for item in allowed_events:
        zh, en = structured_wording(item)
        concise_zh, concise_en = concise_event_wording(item)
        allowed_zh.extend((zh, concise_zh))
        allowed_en.extend((en, concise_en))
        if item.get("confidence") == "medium":
            gating_zh.append(zh)
            gating_en.append(en)
    def en_words(texts):
        return {word.strip(".,!?;:").lower()
                for text in texts for word in text.split()}
    def zh_chars(texts):
        return {char for text in texts for char in text if "一" <= char <= "鿿"}
    fallback_en = en_words([segment.get("fallback_text_en", "")])
    fallback_zh = zh_chars([segment.get("fallback_text_zh", "")])
    return ((not en_words(gating_en) or fallback_en <= en_words(allowed_en))
            and (not zh_chars(gating_zh) or fallback_zh <= zh_chars(allowed_zh)))
def _fact_map(facts):
    return {
        fact.get("fact_id"): fact for fact in facts
        if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
    }


def audit_commentary(
    segments,
    events,
    facts,
    enabled_concept_ids,
    duration_s,
):
    """Return mechanical event, fact, claim, and scheduling errors."""
    errors = []
    if not isinstance(segments, list):
        return ["commentary must be an array"]
    if not isinstance(enabled_concept_ids, set):
        return ["enabled_concept_ids must be a set"]
    event_by_id = _event_map(events if isinstance(events, list) else [])
    fact_list = facts if isinstance(facts, list) else []
    fact_ids = [
        fact.get("fact_id") for fact in fact_list
        if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
    ]
    fact_by_id = _fact_map(fact_list)
    if len(fact_ids) != len(set(fact_ids)):
        errors.append("fact ids must be unique")

    seen_events = set()
    fact_ref_counts = {}
    previous_end = 0.0
    for index, segment in enumerate(segments):
        label = f"segments[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{label} must be an object")
            continue
        if any(field in segment for field in (
            "tactical_proposals_referenced",
            "proposals_referenced",
        )):
            errors.append(f"{label} contains a raw proposal reference")

        start, end = segment.get("timestamp_s"), segment.get("end_s")
        if not _finite(start) or not _finite(end):
            errors.append(f"{label} times must be finite numbers")
        elif start < 0 or end < start or end > duration_s:
            errors.append(f"{label} times are outside clip bounds")
        elif index and start < previous_end:
            errors.append(f"{label} overlaps or is out of order")
        if _finite(end):
            previous_end = max(previous_end, end)
        if segment.get("kind") not in KINDS:
            errors.append(f"{label}.kind is invalid")
        if segment.get("energy") not in ENERGIES:
            errors.append(f"{label}.energy is invalid")
        for field in ("text_zh", "text_en", "fallback_text_zh", "fallback_text_en"):
            if not isinstance(segment.get(field), str) or not segment[field].strip():
                errors.append(f"{label}.{field} must be non-empty text")
        if (
            all(
                isinstance(segment.get(field), str)
                for field in (
                    "text_zh",
                    "text_en",
                    "fallback_text_zh",
                    "fallback_text_en",
                )
            )
            and (
                len(segment["fallback_text_zh"].strip())
                > len(segment["text_zh"].strip())
                or len(segment["fallback_text_en"].strip())
                > len(segment["text_en"].strip())
            )
        ):
            errors.append(f"{label} fallback must be concise")

        event_refs = segment.get("events_referenced")
        fact_refs = segment.get("tactical_facts_referenced")
        event_claims = segment.get("event_claims")
        tactical_claims = segment.get("tactical_claims")
        if not isinstance(event_refs, list) or any(
            not isinstance(ref, str) for ref in event_refs
        ):
            errors.append(f"{label}.events_referenced must be a text array")
            event_refs = []
        if not isinstance(fact_refs, list) or any(
            not isinstance(ref, str) for ref in fact_refs
        ):
            errors.append(f"{label}.tactical_facts_referenced must be a text array")
            fact_refs = []
        if not isinstance(event_claims, list):
            errors.append(f"{label}.event_claims must be an array")
            event_claims = []
        if not isinstance(tactical_claims, list):
            errors.append(f"{label}.tactical_claims must be an array")
            tactical_claims = []

        is_tactical = segment.get("kind") in {"hybrid", "tactical"}
        if is_tactical and not fact_refs:
            errors.append(f"{label} must reference a verified fact")
        if not is_tactical and fact_refs:
            errors.append(f"{label} event commentary cannot reference tactical facts")

        claims_by_event = {
            claim.get("event_id"): claim
            for claim in event_claims
            if isinstance(claim, dict)
        }
        for ref in event_refs:
            event = event_by_id.get(ref)
            if event is None:
                errors.append(f"{label} references unknown event {ref}")
                continue
            if event.get("confidence") == "low":
                errors.append(f"{label} references low-confidence event {ref}")
                continue
            seen_events.add(ref)
            claim = claims_by_event.get(ref)
            exact = (
                claim is not None
                and claim.get("event_code") == event.get("event_code")
                and claim.get("player_team") == event.get("player_team")
                and claim.get("outcome") == event.get("outcome")
                and claim.get("assertion_strength") in ASSERTION_STRENGTHS
            )
            if not _fallback_preserves_event(segment, event):
                errors.append(
                    f"{label} fallback must preserve referenced events "
                    "using structured event facts"
                )
            referenced_events = [
                event_by_id[item] for item in event_refs if item in event_by_id
            ]
            if not _fallback_event_only(segment, event, referenced_events):
                errors.append(f"{label} fallback must be event-only")
            if not exact:
                errors.append(f"{label} lacks an exact claim for event {ref}")

        claims_by_fact = {
            claim.get("fact_id"): claim
            for claim in tactical_claims
            if isinstance(claim, dict)
        }
        for ref in fact_refs:
            fact = fact_by_id.get(ref)
            if fact is None:
                errors.append(f"{label} references missing fact {ref}")
                continue
            concept_id = fact.get("concept_id")
            if concept_id not in enabled_concept_ids:
                errors.append(f"{label} references disabled concept {concept_id}")
            if fact.get("verified_claim_levels") != ["observation"]:
                errors.append(f"{label} fact is not observation-only")
            if _finite(start) and (
                not _finite(fact.get("decision_time_s"))
                or start < fact["decision_time_s"]
            ):
                errors.append(f"{label} starts before fact decision_time_s")
            claim = claims_by_fact.get(ref)
            if (
                not isinstance(claim, dict)
                or claim.get("concept_id") != concept_id
                or claim.get("claim_level") != "observation"
            ):
                errors.append(
                    f"{label} tactical claim must bind the fact and "
                    "use observation claim level"
                )
            fact_ref_counts[ref] = fact_ref_counts.get(ref, 0) + 1

        for claim in tactical_claims:
            if (
                isinstance(claim, dict)
                and claim.get("fact_id") not in fact_refs
            ):
                errors.append(f"{label} tactical claim references an unlisted fact")

    for event_id, event in event_by_id.items():
        if _required_event(event) and event_id not in seen_events:
            errors.append(f"required event {event_id} is absent")
        if event.get("confidence") == "low" and event_id in seen_events:
            errors.append(f"low-confidence event {event_id} must be absent")
    if any(count > 1 for count in fact_ref_counts.values()):
        errors.append("verified tactical fact is reused")
    return errors

def _parse_composition(raw):
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("commentary"), list):
        return payload["commentary"]
    raise ValueError("response must be a commentary array or object")


def _semantic_errors(segments, events, call):
    errors = []
    for event in events:
        if event.get("confidence") != "high":
            continue
        refs = [segment for segment in segments
                if event.get("event_id") in segment.get("events_referenced", [])]
        for segment in refs:
            original_zh = event.get("suggested_wording_zh")
            original_en = event.get("suggested_wording_en")
            if ((not original_zh or segment.get("text_zh") == original_zh)
                    and (not original_en or segment.get("text_en") == original_en)):
                continue
            prompt = "Return JSON only as {\"equivalent\": true|false}. Preserve event code, team, actor, and outcome.\n"
            prompt += json.dumps({"event": event, "segment": segment}, ensure_ascii=False)
            try:
                verdict = json.loads(call(prompt, temperature=0.1))
            except (TypeError, json.JSONDecodeError):
                verdict = {}
            if verdict.get("equivalent") is not True:
                errors.append(f"high-confidence rewording for {event.get('event_id')} is not equivalent")
    return errors


def compose_hybrid(
    events,
    direct_commentary,
    facts,
    enabled_concept_ids,
    windows,
    duration_s,
    call: Callable = ark_chat,
):
    """Compose twice at most, then return the accepted direct baseline unchanged."""
    prompt = "Compose event-first bilingual football commentary and return JSON only.\n"
    prompt += json.dumps({"events": events, "direct_commentary": direct_commentary,
                          "verified_tactical_facts": facts, "commentary_windows": windows,
                          "enabled_concept_ids": sorted(enabled_concept_ids),
                          "duration_s": duration_s}, ensure_ascii=False)
    errors = []
    for _ in range(2):
        request = prompt
        if errors:
            request += "\nPrevious response errors: " + json.dumps(errors, ensure_ascii=False)
        try:
            segments = _parse_composition(call(request, temperature=0.7))
            errors = audit_commentary(segments, events, facts, enabled_concept_ids, duration_s)
            if not errors:
                errors = _semantic_errors(segments, events, call)
            if not errors:
                return segments
        except ValueError as exc:
            errors = [str(exc)]
    return direct_commentary
