"""Strict tactical proposal validation and fixed-checker fact reconstruction."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Callable


Checker = Callable[[dict, dict, dict], dict]
CHECKERS: dict[str, Checker] = {}
PROPOSAL_FIELDS = frozenset(
    {"concept_id", "proposed_window", "team_id", "actors"}
)
CANONICAL_TEAM_IDS = frozenset({"team_0", "team_1"})


class ProposalError(ValueError):
    pass


class VerificationError(ValueError):
    pass


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _concept_by_id(catalog: dict, concept_id: str) -> dict | None:
    for concept in catalog.get("concepts", []):
        if concept.get("id") == concept_id:
            return concept
    return None


def _normalize_proposal(
    raw: object,
    *,
    catalog: dict,
    tactical_state: dict,
    duration_s: float,
    allowed_concept_ids: set[str],
) -> tuple[dict, dict]:
    if not isinstance(raw, dict):
        raise ProposalError("invalid_proposal_type")
    if set(raw) != PROPOSAL_FIELDS:
        raise ProposalError("unknown_fields")

    concept_id = raw.get("concept_id")
    if not isinstance(concept_id, str):
        raise ProposalError("unknown_concept")
    concept = _concept_by_id(catalog, concept_id)
    if concept is None:
        raise ProposalError("unknown_concept")
    if concept_id not in allowed_concept_ids:
        raise ProposalError("concept_not_allowed")

    team_id = raw.get("team_id")
    if team_id not in CANONICAL_TEAM_IDS:
        raise ProposalError("invalid_team_id")
    window = raw.get("proposed_window")
    if not isinstance(window, dict) or set(window) != {"start_s", "end_s"}:
        raise ProposalError("invalid_proposed_window")
    start_s = window.get("start_s")
    end_s = window.get("end_s")
    if (
        not _finite_number(start_s)
        or not _finite_number(end_s)
        or float(start_s) < 0.0
        or float(start_s) >= float(end_s)
        or float(end_s) > duration_s
    ):
        raise ProposalError("invalid_proposed_window")

    actors = raw.get("actors")
    required_roles = concept.get("required_actor_roles")
    constraints = concept.get("actor_team_constraints")
    if (
        not isinstance(actors, dict)
        or not isinstance(required_roles, list)
        or set(actors) != set(required_roles)
        or not isinstance(constraints, dict)
        or set(constraints) != set(required_roles)
    ):
        raise ProposalError("invalid_actor_roles")
    actor_ids = list(actors.values())
    if any(
        not isinstance(track_id, int) or isinstance(track_id, bool)
        for track_id in actor_ids
    ):
        raise ProposalError("unknown_actor_track")
    if len(actor_ids) != len(set(actor_ids)):
        raise ProposalError("actor_tracks_not_distinct")

    tracks = tactical_state.get("measurements", {}).get("tracks", {})
    for role, track_id in actors.items():
        track = tracks.get(str(track_id))
        if not isinstance(track, dict):
            raise ProposalError("unknown_actor_track")
        actor_team = track.get("team_id")
        relation = constraints.get(role)
        if relation == "same" and actor_team != team_id:
            raise ProposalError("actor_team_constraint_failed")
        if relation == "opponent" and (
            actor_team not in CANONICAL_TEAM_IDS or actor_team == team_id
        ):
            raise ProposalError("actor_team_constraint_failed")
        if relation not in {"same", "opponent"}:
            raise ProposalError("invalid_actor_roles")

    return (
        {
            "concept_id": concept_id,
            "proposed_window": {
                "start_s": float(start_s),
                "end_s": float(end_s),
            },
            "team_id": team_id,
            "actors": dict(actors),
        },
        concept,
    )


def _verified_window(
    result: dict,
    *,
    proposal: dict,
    duration_s: float,
    tolerance_s: float,
) -> tuple[dict, dict, float]:
    window = result.get("verified_window")
    if not isinstance(window, dict) or set(window) != {"start_s", "end_s"}:
        raise VerificationError("invalid_verified_window")
    start_s = window.get("start_s")
    end_s = window.get("end_s")
    if (
        not _finite_number(start_s)
        or not _finite_number(end_s)
        or float(start_s) < 0.0
        or float(start_s) >= float(end_s)
        or float(end_s) > duration_s
    ):
        raise VerificationError("invalid_verified_window")

    proposed = proposal["proposed_window"]
    if (
        abs(float(start_s) - proposed["start_s"]) > tolerance_s
        or abs(float(end_s) - proposed["end_s"]) > tolerance_s
    ):
        raise VerificationError("verified_window_outside_tolerance")

    resolution = result.get("window_resolution")
    if not isinstance(resolution, dict):
        raise VerificationError("invalid_window_resolution")
    if resolution.get("method") != "deterministic_anchor_snap":
        raise VerificationError("invalid_window_resolution")
    reported_tolerance = resolution.get("tolerance_s")
    if (
        not _finite_number(reported_tolerance)
        or not math.isclose(float(reported_tolerance), tolerance_s)
    ):
        raise VerificationError("checker_tolerance_mismatch")

    decision_time_s = result.get("decision_time_s")
    if (
        not _finite_number(decision_time_s)
        or float(decision_time_s) < float(end_s)
        or float(decision_time_s) > duration_s
    ):
        raise VerificationError("invalid_decision_time")
    return (
        {"start_s": float(start_s), "end_s": float(end_s)},
        {
            "method": "deterministic_anchor_snap",
            "tolerance_s": tolerance_s,
        },
        float(decision_time_s),
    )


def _build_fact(
    *,
    fact_id: str,
    proposal_id: str,
    proposal: dict,
    concept: dict,
    tactical_state: dict,
    result: dict,
    duration_s: float,
) -> dict:
    recipe = concept.get("recipe")
    if not isinstance(recipe, dict):
        raise VerificationError("recipe_not_available")
    parameters = recipe.get("parameters")
    tolerance_s = (
        parameters.get("window_tolerance_s")
        if isinstance(parameters, dict)
        else None
    )
    if (
        not _finite_number(tolerance_s)
        or float(tolerance_s) < 0.0
    ):
        raise VerificationError("invalid_recipe_tolerance")
    tolerance_s = float(tolerance_s)
    window, resolution, decision_time_s = _verified_window(
        result,
        proposal=proposal,
        duration_s=duration_s,
        tolerance_s=tolerance_s,
    )

    evidence = result.get("resolved_evidence")
    quality_flags = result.get("quality_flags")
    if not isinstance(evidence, list) or any(
        not isinstance(item, dict) for item in evidence
    ):
        raise VerificationError("invalid_resolved_evidence")
    if not isinstance(quality_flags, list) or any(
        not isinstance(flag, str) for flag in quality_flags
    ):
        raise VerificationError("invalid_quality_flags")
    if concept.get("allowed_claim_levels") != ["observation"]:
        raise VerificationError("non_observation_claim_policy")

    recipe_id = recipe.get("id")
    recipe_version = recipe.get("version")
    if (
        recipe_id != concept["id"]
        or not isinstance(recipe.get("checker"), str)
        or not recipe["checker"]
        or recipe_version is None
    ):
        raise VerificationError("invalid_recipe")
    team_id = proposal["team_id"]
    teams = tactical_state.get("teams", {})
    if team_id not in teams:
        raise VerificationError("missing_team_state")

    return {
        "fact_id": fact_id,
        "clip_id": tactical_state.get("clip_id"),
        "concept_id": concept["id"],
        "recipe": f"{recipe_id}@{recipe_version}",
        "state_source": tactical_state.get("state_source"),
        "proposed_window": deepcopy(proposal["proposed_window"]),
        "verified_window": window,
        "window_resolution": resolution,
        "decision_time_s": decision_time_s,
        "team_id": team_id,
        "attack_direction": teams[team_id].get("attack_direction", "unknown"),
        "actors": deepcopy(proposal["actors"]),
        "verified_claim_levels": ["observation"],
        "resolved_evidence": deepcopy(evidence),
        "quality_flags": list(quality_flags),
        "provenance": {
            "proposal_id": proposal_id,
            "state_schema_version": tactical_state.get("schema_version"),
            "recipe_id": recipe_id,
            "recipe_version": recipe_version,
            "state_provenance": deepcopy(tactical_state.get("provenance", {})),
        },
    }


def _audit_record(
    *,
    index: int,
    raw: object,
    tactical_state: dict,
) -> dict:
    return {
        "proposal_id": f"proposal_{index:03d}",
        "clip_id": tactical_state.get("clip_id"),
        "state_source": tactical_state.get("state_source"),
        "raw": deepcopy(raw),
        "normalized_proposal": None,
        "proposal_status": "rejected",
        "reasons": [],
        "verification_status": "not_run",
        "fact_id": None,
    }


def process_tactical_proposals(
    raw: object,
    *,
    catalog: dict,
    tactical_state: dict,
    duration_s: float,
    allowed_concept_ids: set[str],
    checkers: dict[str, Checker] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return one audit record per raw proposal and rebuilt verified facts."""
    if not _finite_number(duration_s) or float(duration_s) <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not isinstance(allowed_concept_ids, set):
        raise ValueError("allowed_concept_ids must be a set")
    items = raw if isinstance(raw, list) else [raw]
    registry = CHECKERS if checkers is None else checkers
    audits: list[dict] = []
    facts: list[dict] = []

    for index, item in enumerate(items, start=1):
        audit = _audit_record(
            index=index,
            raw=item,
            tactical_state=tactical_state,
        )
        audits.append(audit)
        try:
            proposal, concept = _normalize_proposal(
                item,
                catalog=catalog,
                tactical_state=tactical_state,
                duration_s=float(duration_s),
                allowed_concept_ids=allowed_concept_ids,
            )
        except ProposalError as error:
            audit["reasons"].append(str(error))
            continue

        audit["normalized_proposal"] = deepcopy(proposal)
        audit["proposal_status"] = "accepted"
        recipe = concept.get("recipe")
        checker_name = (
            recipe.get("checker") if isinstance(recipe, dict) else None
        )
        checker = registry.get(checker_name) if isinstance(checker_name, str) else None
        if checker is None:
            audit["reasons"].append("checker_not_registered")
            continue

        try:
            result = checker(proposal, concept, tactical_state)
        except Exception:
            audit["verification_status"] = "unsupported"
            audit["reasons"].append("checker_exception")
            continue
        if not isinstance(result, dict):
            audit["verification_status"] = "unsupported"
            audit["reasons"].append("invalid_checker_result")
            continue
        status = result.get("status")
        if status in {"failed", "unsupported"}:
            audit["verification_status"] = status
            reason = result.get("reason")
            audit["reasons"].append(
                reason if isinstance(reason, str) and reason else status
            )
            continue
        if status != "passed":
            audit["verification_status"] = "unsupported"
            audit["reasons"].append("invalid_checker_status")
            continue

        fact_id = f"fact_{len(facts) + 1:03d}"
        try:
            fact = _build_fact(
                fact_id=fact_id,
                proposal_id=audit["proposal_id"],
                proposal=proposal,
                concept=concept,
                tactical_state=tactical_state,
                result=result,
                duration_s=float(duration_s),
            )
        except VerificationError as error:
            audit["verification_status"] = "unsupported"
            audit["reasons"].append(str(error))
            continue
        facts.append(fact)
        audit["verification_status"] = "passed"
        audit["fact_id"] = fact_id
    return audits, facts
