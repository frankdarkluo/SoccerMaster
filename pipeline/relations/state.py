"""Concept-neutral 25 Hz tactical-state adapter."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from typing import Iterable

from pipeline.stage2b.digest import FrameData


CANONICAL_TEAM_IDS = frozenset({"team_0", "team_1"})
ATTACK_DIRECTIONS = frozenset({"left_to_right", "right_to_left", "unknown"})
CONTEXT_FIELDS = {
    "possession_phase": frozenset(
        {"build_up", "transition", "settled_attack", "settled_defense", "unknown"}
    ),
    "restart_state": frozenset(
        {"open_play", "set_piece", "interruption", "unknown"}
    ),
    "progression_state": frozenset(
        {"stable", "progressing", "penetrating", "unknown"}
    ),
    "danger_state": frozenset({"neutral", "threat", "unknown"}),
}
TEAM_RELATIVE_CONTEXT = frozenset(
    {"possession_phase", "progression_state", "danger_state"}
)
EVENT_TEAM_FIELDS = frozenset(
    {
        "team",
        "team_id",
        "player_team",
        "opponent_team",
        "possession_team",
        "defending_team",
    }
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _confidence(value: object, field: str) -> float:
    if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field}.confidence must be finite and within [0, 1]")
    return float(value)


def canonical_team_map(
    source_labels: set[str],
    *,
    recorded: dict[str, str] | None = None,
) -> dict[str, str]:
    """Validate a recorded bijection or deterministically map two source labels."""
    if (
        not isinstance(source_labels, set)
        or len(source_labels) != 2
        or any(not isinstance(label, str) or not label for label in source_labels)
    ):
        raise ValueError("exactly two non-empty source team labels are required")
    if recorded is None:
        return {
            source: canonical
            for source, canonical in zip(
                sorted(source_labels), sorted(CANONICAL_TEAM_IDS)
            )
        }
    if (
        not isinstance(recorded, dict)
        or set(recorded) != source_labels
        or set(recorded.values()) != CANONICAL_TEAM_IDS
        or len(set(recorded.values())) != 2
    ):
        raise ValueError("recorded source team map must be a complete bijection")
    return dict(recorded)


def _context_defaults() -> dict:
    return {
        "possession_phase": {"team_id": None, "value": "unknown"},
        "restart_state": {"value": "unknown"},
        "progression_state": {"team_id": None, "value": "unknown"},
        "danger_state": {"team_id": None, "value": "unknown"},
    }


def _normalize_context(value: dict | None) -> tuple[dict, dict]:
    if value is None:
        return _context_defaults(), {}
    if not isinstance(value, dict) or not set(value).issubset(CONTEXT_FIELDS):
        raise ValueError("normalized_context contains unknown fields")

    normalized = _context_defaults()
    provenance: dict[str, dict] = {}
    for field, record in value.items():
        if not isinstance(record, dict):
            raise ValueError(f"{field} must be an object")
        allowed_keys = {"value", "source", "confidence"}
        if field in TEAM_RELATIVE_CONTEXT:
            allowed_keys.add("team_id")
        if not set(record).issubset(allowed_keys):
            raise ValueError(f"{field} contains unknown fields")
        context_value = record.get("value", "unknown")
        if context_value not in CONTEXT_FIELDS[field]:
            raise ValueError(f"{field}.value is invalid")

        output = {"value": context_value}
        if field in TEAM_RELATIVE_CONTEXT:
            team_id = record.get("team_id")
            if context_value != "unknown" and team_id not in CANONICAL_TEAM_IDS:
                raise ValueError(f"{field}.team_id must be canonical")
            if context_value == "unknown" and team_id not in {None, *CANONICAL_TEAM_IDS}:
                raise ValueError(f"{field}.team_id must be canonical or null")
            output = {"team_id": team_id, "value": context_value}

        if context_value != "unknown":
            source = record.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{field}.source is required")
            confidence = _confidence(record.get("confidence"), field)
            provenance[field] = {
                "source": source,
                "confidence": confidence,
            }
        normalized[field] = output
    return normalized, provenance


def _attack_direction_records(
    value: dict[str, dict] | None,
) -> tuple[dict[str, str], dict[str, dict]]:
    directions = {team_id: "unknown" for team_id in CANONICAL_TEAM_IDS}
    provenance: dict[str, dict] = {}
    if value is None:
        return directions, provenance
    if not isinstance(value, dict) or not set(value).issubset(CANONICAL_TEAM_IDS):
        raise ValueError("attack_directions must use canonical team ids")
    for team_id, record in value.items():
        if not isinstance(record, dict):
            raise ValueError(f"attack_directions.{team_id} must be an object")
        direction = record.get("value")
        if direction not in ATTACK_DIRECTIONS:
            raise ValueError(f"attack_directions.{team_id}.value is invalid")
        source = record.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"attack_directions.{team_id}.source is required")
        confidence = _confidence(record.get("confidence"), f"attack_directions.{team_id}")
        directions[team_id] = direction
        provenance[team_id] = {
            "value": direction,
            "source": source,
            "confidence": confidence,
        }
    return directions, provenance


def _majority(values: Iterable[object], default: object = None):
    votes = Counter(value for value in values if value is not None)
    return votes.most_common(1)[0][0] if votes else default


def build_tactical_state(
    frames: list[FrameData],
    *,
    fps: float,
    clip_id: str,
    state_source: str,
    source_team_map: dict[str, str] | None = None,
    attack_directions: dict[str, dict] | None = None,
    source_labels: dict | None = None,
    normalized_context: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """Create serializable, concept-neutral raw measurements and quality."""
    if state_source not in {"ground_truth", "gsr_prediction"}:
        raise ValueError("state_source must be ground_truth or gsr_prediction")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("clip_id must be a non-empty string")
    if not _finite_number(fps) or float(fps) <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list")
    ordered = sorted(frames, key=lambda frame: frame.frame_id)
    frame_ids = [frame.frame_id for frame in ordered]
    if (
        any(not isinstance(frame_id, int) or frame_id <= 0 for frame_id in frame_ids)
        or len(frame_ids) != len(set(frame_ids))
    ):
        raise ValueError("frame ids must be unique positive integers")

    observed_source_teams = {
        player.get("team")
        for frame in ordered
        for player in frame.players
        if player.get("role") != "referee"
        and isinstance(player.get("team"), str)
        and player.get("team")
    }
    team_map = canonical_team_map(
        observed_source_teams,
        recorded=source_team_map,
    )
    inverse_team_map = {canonical: source for source, canonical in team_map.items()}
    directions, direction_provenance = _attack_direction_records(attack_directions)
    context, context_provenance = _normalize_context(normalized_context)

    if source_labels is None:
        copied_source_labels = {}
    elif isinstance(source_labels, dict):
        copied_source_labels = deepcopy(source_labels)
    else:
        raise ValueError("source_labels must be an object")
    if provenance is None:
        state_provenance = {}
    elif isinstance(provenance, dict):
        state_provenance = deepcopy(provenance)
    else:
        raise ValueError("provenance must be an object")
    state_provenance["team_mapping"] = {
        "method": (
            "recorded_source_team_map"
            if source_team_map is not None
            else "deterministic_source_label_sort"
        ),
        "source_to_canonical": team_map,
    }
    if direction_provenance:
        state_provenance["attack_directions"] = direction_provenance
    if context_provenance:
        state_provenance["normalized_context"] = context_provenance

    player_samples: dict[int, list[dict]] = defaultdict(list)
    player_teams: dict[int, list[str]] = defaultdict(list)
    player_roles: dict[int, list[str]] = defaultdict(list)
    player_jerseys: dict[int, list[str]] = defaultdict(list)
    frames_with_pitch = 0
    ball_samples: list[dict] = []
    for frame in ordered:
        valid_player_in_frame = False
        time_s = round((frame.frame_id - 1) / float(fps), 6)
        for player in frame.players:
            track_id = player.get("track_id")
            source_team = player.get("team")
            x = player.get("x")
            y = player.get("y")
            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or source_team not in team_map
                or not _finite_number(x)
                or not _finite_number(y)
            ):
                continue
            valid_player_in_frame = True
            player_samples[track_id].append(
                {
                    "frame_id": frame.frame_id,
                    "t": time_s,
                    "x": float(x),
                    "y": float(y),
                }
            )
            player_teams[track_id].append(source_team)
            player_roles[track_id].append(player.get("role") or "player")
            if player.get("jersey"):
                player_jerseys[track_id].append(str(player["jersey"]))
        if valid_player_in_frame:
            frames_with_pitch += 1

        if (
            isinstance(frame.ball_xy, tuple)
            and len(frame.ball_xy) == 2
            and _finite_number(frame.ball_xy[0])
            and _finite_number(frame.ball_xy[1])
        ):
            ball_samples.append(
                {
                    "frame_id": frame.frame_id,
                    "t": time_s,
                    "x": float(frame.ball_xy[0]),
                    "y": float(frame.ball_xy[1]),
                }
            )

    tracks = {}
    continuity = {}
    for track_id in sorted(player_samples):
        source_team = _majority(player_teams[track_id])
        samples = player_samples[track_id]
        tracks[str(track_id)] = {
            "track_id": track_id,
            "team_id": team_map[source_team],
            "role": _majority(player_roles[track_id], "player"),
            "jersey": _majority(player_jerseys[track_id], ""),
            "samples": samples,
        }
        continuity[str(track_id)] = round(len(samples) / len(ordered), 6)

    return {
        "schema_version": "tactical-state-v1",
        "clip_id": clip_id,
        "state_source": state_source,
        "fps": float(fps),
        "n_frames": len(ordered),
        "teams": {
            team_id: {
                "source_team_label": inverse_team_map[team_id],
                "attack_direction": directions[team_id],
            }
            for team_id in sorted(CANONICAL_TEAM_IDS)
        },
        "source_labels": copied_source_labels,
        "normalized_context": context,
        "measurements": {
            "tracks": tracks,
            "ball": ball_samples,
        },
        "quality": {
            "track_continuity": continuity,
            "pitch_coordinate_coverage": round(
                frames_with_pitch / len(ordered), 6
            ),
            "ball_coverage": round(len(ball_samples) / len(ordered), 6),
        },
        "provenance": state_provenance,
    }


def _sample_by_step(samples: list[dict], step: int) -> list[dict]:
    return [
        deepcopy(sample)
        for sample in samples
        if (sample["frame_id"] - 1) % step == 0
    ]


def project_proposal_state(state: dict, *, max_hz: float = 12.5) -> dict:
    """Return compact public measurements, context, and quality."""
    if not _finite_number(max_hz) or float(max_hz) <= 0:
        raise ValueError("max_hz must be finite and positive")
    fps = float(state["fps"])
    effective_max_hz = min(float(max_hz), 12.5)
    step = max(1, math.ceil(fps / effective_max_hz))
    tracks = {
        track_id: {
            key: deepcopy(track[key])
            for key in ("track_id", "team_id", "role", "jersey")
        }
        | {"samples": _sample_by_step(track["samples"], step)}
        for track_id, track in state["measurements"]["tracks"].items()
    }
    return {
        "schema_version": "tactical-proposal-state-v1",
        "clip_id": state["clip_id"],
        "state_source": state["state_source"],
        "fps": fps,
        "analysis_hz": fps / step,
        "n_frames": state["n_frames"],
        "teams": {
            team_id: {
                "attack_direction": team["attack_direction"],
            }
            for team_id, team in state["teams"].items()
        },
        "normalized_context": deepcopy(state["normalized_context"]),
        "measurements": {
            "tracks": tracks,
            "ball": _sample_by_step(state["measurements"]["ball"], step),
        },
        "quality": deepcopy(state["quality"]),
    }


def project_proposal_events(events: list[dict], *, state: dict) -> list[dict]:
    """Return copied events with canonical team IDs for the tactical prompt."""
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("events must be a list of objects")
    mapping = state["provenance"]["team_mapping"]["source_to_canonical"]
    projected = deepcopy(events)
    for event in projected:
        for field in EVENT_TEAM_FIELDS.intersection(event):
            value = event[field]
            if value is None or value in CANONICAL_TEAM_IDS:
                continue
            if value not in mapping:
                raise ValueError(f"unmapped event team label in {field}: {value}")
            event[field] = mapping[value]
    return projected
