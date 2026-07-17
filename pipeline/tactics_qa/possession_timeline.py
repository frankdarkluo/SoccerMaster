"""Merge holder segments into team runs and conservative transition events."""
from .evidence import PossessionEvent

STABLE_RUN_MIN_S = 2.0
CONTESTED_RUN_MAX_S = 0.9
MERGE_SAME_TEAM_GAP_S = 1.0


def team_runs(segments, fps: float) -> list:
    normalized = []
    for segment in segments:
        if isinstance(segment, tuple):
            _, team, start, end = segment
        else:
            team, start, end = segment.team, segment.start_fid, segment.end_fid
        if team is not None:
            normalized.append(
                {"team": team, "start_s": start / fps, "end_s": end / fps})
    normalized.sort(key=lambda run: run["start_s"])
    merged = []
    for run in normalized:
        if (merged and merged[-1]["team"] == run["team"]
                and run["start_s"] - merged[-1]["end_s"] <= MERGE_SAME_TEAM_GAP_S):
            merged[-1]["end_s"] = max(merged[-1]["end_s"], run["end_s"])
        else:
            merged.append(dict(run))
    return merged


def transition_events(runs: list, attacking_team: str) -> list:
    events = []
    for previous, current in zip(runs, runs[1:]):
        if previous["team"] == current["team"]:
            continue
        relative_team = (
            "attacking" if current["team"] == attacking_team else "defending")
        duration = current["end_s"] - current["start_s"]
        if duration >= STABLE_RUN_MIN_S and relative_team == "attacking":
            events.append(PossessionEvent(
                current["start_s"], "attacking", "controlled_recovery"))
        elif duration <= CONTESTED_RUN_MAX_S:
            events.append(PossessionEvent(
                current["start_s"], relative_team, "contested_touch"))
    return events
