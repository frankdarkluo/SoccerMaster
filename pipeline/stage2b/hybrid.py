"""Verified tactical candidates and event-first hybrid commentary."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable

from pipeline.relations.query import predicate_passes, resolve_query
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


def _concept_ids():
    path = Path(__file__).with_name("concepts.yaml")
    return {
        line.split(":", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- id:")
    }


def verify_candidates(relations, proposed, approved_windows=None):
    """Resolve candidate evidence in code and retain only passing candidates."""
    verified = []
    id_counts = {}
    for source in proposed if isinstance(proposed, list) else []:
        if isinstance(source, dict) and isinstance(source.get("candidate_id"), str):
            candidate_id = source["candidate_id"]
            id_counts[candidate_id] = id_counts.get(candidate_id, 0) + 1
    concepts = _concept_ids()
    approved_by_id = {
        window.get("window_id"): window
        for window in approved_windows if isinstance(window, dict)
    } if isinstance(approved_windows, list) else {}
    for source in proposed if isinstance(proposed, list) else []:
        if (not isinstance(source, dict) or source.get("concept_id") not in concepts
                or id_counts.get(source.get("candidate_id"), 0) != 1):
            continue
        candidate = copy.deepcopy(source)
        window = candidate.get("window")
        approved = approved_by_id.get(candidate.get("window_id"))
        queries = candidate.get("evidence_queries")
        if (not isinstance(window, dict) or not isinstance(approved, dict)
                or not isinstance(queries, list) or not 1 <= len(queries) <= 3):
            continue
        start, end = window.get("start_s"), window.get("end_s")
        if (not _finite(start) or not _finite(end) or start > end
                or start < approved.get("start_s", math.inf)
                or end > approved.get("end_s", -math.inf)):
            continue
        valid = True
        for evidence in queries:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("query"), dict):
                valid = False
                break
            query = evidence["query"]
            if (not _finite(query.get("t0")) or not _finite(query.get("t1"))
                    or query["t0"] < start or query["t1"] > end
                    or query["t0"] > query["t1"]):
                valid = False
                break
            result = resolve_query(relations, query)
            passed = predicate_passes(result, evidence.get("predicate"))
            evidence["result"] = result
            evidence["predicate_passed"] = passed
            valid = valid and passed
        candidate["verified"] = valid
        if valid:
            verified.append(candidate)
    return verified


def _required_event(event):
    if event.get("confidence") == "high":
        return True
    reasons = event.get("confidence_reasons", [])
    verification = event.get("verification", {})
    return event.get("confidence") == "medium" and (
        "directly_visible" in reasons or verification.get("directly_visible") is True
    )


def _scope_name(phase_scope):
    if isinstance(phase_scope, str):
        return phase_scope
    if isinstance(phase_scope, dict):
        if phase_scope.get("fragmented"):
            return "fragmented"
        if phase_scope.get("complete") or phase_scope.get("completed"):
            return "complete_attack"
        return str(phase_scope.get("scope", "incomplete_attack"))
    return "incomplete_attack"


def _window_limit(duration_s, phase_scope):
    scope = _scope_name(phase_scope)
    if scope == "fragmented" and duration_s <= 30.0:
        return 1
    if scope in {"complete", "complete_attack", "completed_phase"}:
        return 3
    if scope in {"incomplete", "incomplete_attack", "local"}:
        return 1
    return 2


def candidate_windows(events, duration_s, phase_scope):
    """Return sparse event-free windows eligible for tactical composition."""
    if not _finite(duration_s) or duration_s <= 0:
        return []
    required = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or not _required_event(event):
            continue
        start, end = event.get("start_s"), event.get("end_s")
        if _finite(start) and _finite(end) and 0 <= start <= end <= duration_s:
            required.append((float(start), float(end), event.get("event_id")))
    required.sort()
    free = []
    cursor = 0.0
    for start, end, _ in required:
        if start - cursor >= 4.0:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if duration_s - cursor >= 4.0:
        free.append((cursor, float(duration_s)))

    windows = []
    # Semantic priority is deliberate: attach context to a required event first,
    # then summarize a completed phase, then use standalone event-free gaps.
    previous_required_end = 0.0
    for start, end, event_id in required:
        causal_start = max(0.0, start - 5.0, previous_required_end)
        if start > causal_start:
            windows.append({"kind": "causal_attachment", "start_s": causal_start,
                            "end_s": start, "event_id": event_id})
        previous_required_end = max(previous_required_end, end)
    scope = _scope_name(phase_scope)
    if scope in {"complete", "complete_attack", "completed_phase"} and required:
        last_end = required[-1][1]
        if duration_s - last_end >= 1.0:
            windows.append({"kind": "completed_phase", "start_s": last_end,
                            "end_s": min(float(duration_s), last_end + 5.0)})
    for start, end in free:
        windows.append({"kind": "event_free_gap", "start_s": start,
                        "end_s": min(end, start + 5.0)})
    selected = windows[:_window_limit(float(duration_s), phase_scope)]
    for index, window in enumerate(selected, 1):
        window["window_id"] = f"window_{index:03d}"
    return selected


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
def _candidate_map(candidates):
    return {
        candidate.get("candidate_id"): candidate for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
    }


def audit_commentary(segments, events, candidates, duration_s):
    """Return mechanical composition errors; an empty list accepts output."""
    errors = []
    if not isinstance(segments, list):
        return ["commentary must be an array"]
    event_by_id = _event_map(events if isinstance(events, list) else [])
    candidate_list = candidates if isinstance(candidates, list) else []
    candidate_ids = [candidate.get("candidate_id") for candidate in candidate_list
                     if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)]
    candidate_by_id = _candidate_map(candidate_list)
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate ids must be unique")
    seen_events = set()
    tactical_ref_counts = {}
    tactical_insertions = 0
    previous_end = 0.0
    for index, segment in enumerate(segments):
        label = f"segments[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{label} must be an object")
            continue
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
        if (all(isinstance(segment.get(field), str) for field in
                ("text_zh", "text_en", "fallback_text_zh", "fallback_text_en"))
                and (len(segment["fallback_text_zh"].strip()) > len(segment["text_zh"].strip())
                     or len(segment["fallback_text_en"].strip()) > len(segment["text_en"].strip()))):
            errors.append(f"{label} fallback must be concise")
        event_refs = segment.get("events_referenced")
        tactic_refs = segment.get("tactical_candidates_referenced")
        claims = segment.get("event_claims")
        if not isinstance(event_refs, list) or any(not isinstance(ref, str) for ref in event_refs):
            errors.append(f"{label}.events_referenced must be a text array")
            event_refs = []
        if not isinstance(tactic_refs, list) or any(not isinstance(ref, str) for ref in tactic_refs):
            errors.append(f"{label}.tactical_candidates_referenced must be a text array")
            tactic_refs = []
        if not isinstance(claims, list):
            errors.append(f"{label}.event_claims must be an array")
            claims = []
        is_tactical = segment.get("kind") in {"hybrid", "tactical"}
        if is_tactical:
            tactical_insertions += 1
            if not tactic_refs:
                errors.append(f"{label} must reference a verified candidate")
            if not event_refs:
                errors.append(f"{label} fallback must preserve referenced events using structured event facts")
        claims_by_id = {
            claim.get("event_id"): claim for claim in claims if isinstance(claim, dict)
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
            claim = claims_by_id.get(ref)
            exact = (claim is not None
                     and claim.get("event_code") == event.get("event_code")
                     and claim.get("player_team") == event.get("player_team")
                     and claim.get("outcome") == event.get("outcome")
                     and claim.get("assertion_strength") in ASSERTION_STRENGTHS)
            if not _fallback_preserves_event(segment, event):
                errors.append(f"{label} fallback must preserve referenced events using structured event facts")
            referenced_events = [
                event_by_id[item] for item in event_refs if item in event_by_id
            ]
            if not _fallback_event_only(segment, event, referenced_events):
                errors.append(f"{label} fallback must be event-only")
            if not exact:
                errors.append(f"{label} lacks an exact claim for event {ref}")
        for ref in tactic_refs:
            candidate = candidate_by_id.get(ref)
            if candidate is None or candidate.get("verified") is not True:
                errors.append(f"{label} references unverified tactic {ref}")
            else:
                tactical_ref_counts[ref] = tactical_ref_counts.get(ref, 0) + 1
    for event_id, event in event_by_id.items():
        if _required_event(event) and event_id not in seen_events:
            errors.append(f"required event {event_id} is absent")
        if event.get("confidence") == "low" and event_id in seen_events:
            errors.append(f"low-confidence event {event_id} must be absent")
    if any(count > 1 for count in tactical_ref_counts.values()):
        errors.append("verified tactical candidate is reused")
    scopes = [candidate.get("phase_scope") for candidate in candidate_by_id.values()]
    phase_scope = scopes[0] if scopes else "incomplete_attack"
    if tactical_insertions > _window_limit(float(duration_s), phase_scope):
        errors.append("tactical count exceeds the sparse scheduling limit")
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


def compose_hybrid(events, direct_commentary, candidates, windows, duration_s,
                   call: Callable = ark_chat):
    """Compose twice at most, then return the accepted direct baseline unchanged."""
    prompt = "Compose event-first bilingual football commentary and return JSON only.\n"
    prompt += json.dumps({"events": events, "direct_commentary": direct_commentary,
                          "verified_candidates": candidates, "candidate_windows": windows,
                          "duration_s": duration_s}, ensure_ascii=False)
    errors = []
    for _ in range(2):
        request = prompt
        if errors:
            request += "\nPrevious response errors: " + json.dumps(errors, ensure_ascii=False)
        try:
            segments = _parse_composition(call(request, temperature=0.7))
            errors = audit_commentary(segments, events, candidates, duration_s)
            if not errors:
                errors = _semantic_errors(segments, events, call)
            if not errors:
                return segments
        except ValueError as exc:
            errors = [str(exc)]
    return direct_commentary
