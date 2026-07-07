"""Possession and team assignment helpers."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from pipeline.stage2_events.types import FrameData, PossessionSegment

POSSESSION_RADIUS_M = 3.0
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
