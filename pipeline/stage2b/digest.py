"""Possession and team assignment helpers."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from pipeline.utils.labels import frame_index_from_labels
from collections import Counter, defaultdict
from typing import Dict, List, Optional

@dataclass
class FrameData:
    frame_id: int
    ball_xy: Optional[Tuple[float, float]] = None
    players: List[dict] = field(default_factory=list)


@dataclass
class PossessionSegment:
    track_id: int
    team: Optional[str]
    jersey: Optional[str]
    start_fid: int
    end_fid: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]

POSSESSION_RADIUS_M = 3.0
def load_frames(path: Path) -> List[FrameData]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_frame, _ = frame_index_from_labels(data)
    frames_dict: Dict[int, FrameData] = {
        frame_num: FrameData(frame_id=frame_num)
        for frame_num in sorted(set(image_id_to_frame.values()))
    }

    for ann in data.get("annotations", []):
        frame_num = image_id_to_frame.get(str(ann.get("image_id", "")))
        if frame_num is None:
            continue

        bp = ann.get("bbox_pitch")
        if not isinstance(bp, dict):
            continue

        x = bp.get("x_bottom_middle")
        y = bp.get("y_bottom_middle")
        if x is None or y is None:
            continue

        attrs = ann.get("attributes", {}) or {}
        if attrs.get("role") == "ball":
            frames_dict[frame_num].ball_xy = (float(x), float(y))
        else:
            frames_dict[frame_num].players.append({
                "track_id": ann.get("track_id"),
                "x": float(x),
                "y": float(y),
                "role": attrs.get("role", "other"),
                "team": attrs.get("team"),
                "jersey": attrs.get("jersey", ""),
            })

    return [frames_dict[frame_num] for frame_num in sorted(frames_dict.keys())]

POSSESSION_MIN_FRAMES = 3


def resolve_team_by_track(frames: List[FrameData]) -> Dict[int, Optional[str]]:
    votes: Dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        for player in frame.players:
            track_id = player.get("track_id")
            team = player.get("team")
            if track_id is None or team is None:
                continue
            votes[int(track_id)][team] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items()}


def resolve_role_by_track(frames: List[FrameData]) -> Dict[int, str]:
    """Majority-vote role per track (player / goalkeeper / referee / other)."""
    votes: Dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        for player in frame.players:
            track_id = player.get("track_id")
            role = player.get("role")
            if track_id is None or not role:
                continue
            votes[int(track_id)][role] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items()}


def _nearest_holder(frame: FrameData) -> Optional[dict]:
    if frame.ball_xy is None:
        return None
    bx, by = frame.ball_xy
    candidates = [player for player in frame.players if player.get("role") != "referee"]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda player: math.hypot(player["x"] - bx, player["y"] - by))
    if math.hypot(nearest["x"] - bx, nearest["y"] - by) > POSSESSION_RADIUS_M:
        return None
    return nearest


def possession_segments(
    frames: List[FrameData],
    team_by_track: Dict[int, Optional[str]],
    min_frames: int = POSSESSION_MIN_FRAMES,
) -> List[PossessionSegment]:
    ordered = sorted(frames, key=lambda frame: frame.frame_id)
    segments: List[PossessionSegment] = []

    current_track_id = None
    current_start_frame = None
    current_start_holder = None
    current_end_frame = None
    current_end_holder = None

    candidate_track_id = None
    candidate_start_idx = None
    candidate_count = 0

    for idx, frame in enumerate(ordered):
        holder = _nearest_holder(frame)
        track_id = int(holder["track_id"]) if holder and holder.get("track_id") is not None else None

        if track_id is None:
            if current_track_id is not None and current_start_frame is not None and current_end_frame is not None:
                segments.append(PossessionSegment(
                    track_id=current_track_id,
                    team=team_by_track.get(current_track_id),
                    jersey=current_start_holder.get("jersey") if current_start_holder else None,
                    start_fid=current_start_frame.frame_id,
                    end_fid=current_end_frame.frame_id,
                    start_xy=(current_start_holder["x"], current_start_holder["y"]),
                    end_xy=(current_end_holder["x"], current_end_holder["y"]),
                ))
            current_track_id = None
            current_start_frame = None
            current_start_holder = None
            current_end_frame = None
            current_end_holder = None
            candidate_track_id = None
            candidate_start_idx = None
            candidate_count = 0
            continue

        if current_track_id is None:
            if candidate_track_id == track_id:
                candidate_count += 1
            else:
                candidate_track_id = track_id
                candidate_start_idx = idx
                candidate_count = 1
            if candidate_count >= min_frames:
                current_track_id = track_id
                current_start_frame = frame
                current_start_holder = holder
                current_end_frame = frame
                current_end_holder = holder
                candidate_track_id = None
                candidate_start_idx = None
                candidate_count = 0
            continue

        if track_id == current_track_id:
            current_end_frame = frame
            current_end_holder = holder
            candidate_track_id = None
            candidate_start_idx = None
            candidate_count = 0
            continue

        if candidate_track_id == track_id:
            candidate_count += 1
        else:
            candidate_track_id = track_id
            candidate_start_idx = idx
            candidate_count = 1

        if candidate_count >= min_frames and candidate_start_idx is not None:
            if current_start_frame is not None and current_end_frame is not None and current_start_holder is not None and current_end_holder is not None:
                segments.append(PossessionSegment(
                    track_id=current_track_id,
                    team=team_by_track.get(current_track_id),
                    jersey=current_start_holder.get("jersey"),
                    start_fid=current_start_frame.frame_id,
                    end_fid=current_end_frame.frame_id,
                    start_xy=(current_start_holder["x"], current_start_holder["y"]),
                    end_xy=(current_end_holder["x"], current_end_holder["y"]),
                ))
            current_track_id = track_id
            current_start_frame = frame
            current_start_holder = holder
            current_end_frame = frame
            current_end_holder = holder
            candidate_track_id = None
            candidate_start_idx = None
            candidate_count = 0

    if current_track_id is not None and current_start_frame is not None and current_end_frame is not None and current_start_holder is not None and current_end_holder is not None:
        segments.append(PossessionSegment(
            track_id=current_track_id,
            team=team_by_track.get(current_track_id),
            jersey=current_start_holder.get("jersey"),
            start_fid=current_start_frame.frame_id,
            end_fid=current_end_frame.frame_id,
            start_xy=(current_start_holder["x"], current_start_holder["y"]),
            end_xy=(current_end_holder["x"], current_end_holder["y"]),
        ))

    return segments
def build_tracking_digest(path: Path, fps: float) -> str:
    frames = load_frames(path)
    teams = resolve_team_by_track(frames)
    roles = resolve_role_by_track(frames)
    possessions = possession_segments(frames, teams)
    seen = {}
    for frame in frames:
        for player in frame.players:
            track_id = player.get("track_id")
            if track_id is not None:
                seen[int(track_id)] = (
                    teams.get(int(track_id)),
                    roles.get(int(track_id), "player"),
                    str(player.get("jersey") or ""),
                )
    roster = [
        f"- track {track}: {team} {role} #{jersey or '?'}"
        for track, (team, role, jersey) in sorted(seen.items())
    ]
    windows = [
        f"- {(seg.start_fid - 1) / fps:.2f}-{(seg.end_fid - 1) / fps:.2f}s: "
        f"{seg.team} #{seg.jersey or '?'}"
        for seg in possessions
    ]
    return "\n".join(["[Tracked roster]", *roster, "[Possession windows]", *windows])
