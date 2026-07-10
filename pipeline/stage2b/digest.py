"""Build a compact tracking digest directly from Stage 1 predictions."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.utils.labels import frame_index_from_labels

POSSESSION_RADIUS_M = 3.0
POSSESSION_MIN_FRAMES = 3


@dataclass
class FrameData:
    frame_id: int
    ball_xy: Optional[tuple[float, float]] = None
    players: list[dict] = field(default_factory=list)


@dataclass
class PossessionSegment:
    track_id: int
    team: Optional[str]
    jersey: Optional[str]
    start_fid: int
    end_fid: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


def load_frames(path: Path) -> list[FrameData]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    image_to_frame, _ = frame_index_from_labels(data)
    frames = {
        frame_id: FrameData(frame_id)
        for frame_id in sorted(set(image_to_frame.values()))
    }
    for ann in data.get("annotations", []):
        frame_id = image_to_frame.get(str(ann.get("image_id", "")))
        pitch = ann.get("bbox_pitch")
        if frame_id is None or not isinstance(pitch, dict):
            continue
        x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
        if x is None or y is None:
            continue
        attrs = ann.get("attributes") or {}
        if attrs.get("role") == "ball":
            frames[frame_id].ball_xy = (float(x), float(y))
        else:
            frames[frame_id].players.append({
                "track_id": ann.get("track_id"),
                "x": float(x),
                "y": float(y),
                "role": attrs.get("role", "other"),
                "team": attrs.get("team"),
                "jersey": attrs.get("jersey", ""),
            })
    return [frames[frame_id] for frame_id in sorted(frames)]


def _majority(frames: list[FrameData], key: str) -> dict[int, str]:
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        for player in frame.players:
            track_id, value = player.get("track_id"), player.get(key)
            if track_id is not None and value:
                votes[int(track_id)][value] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items()}


def _nearest_holder(frame: FrameData) -> Optional[dict]:
    if frame.ball_xy is None:
        return None
    bx, by = frame.ball_xy
    players = [p for p in frame.players if p.get("role") != "referee"]
    if not players:
        return None
    nearest = min(players, key=lambda p: math.hypot(p["x"] - bx, p["y"] - by))
    return nearest if math.hypot(nearest["x"] - bx, nearest["y"] - by) <= POSSESSION_RADIUS_M else None


def possession_segments(
    frames: list[FrameData],
    team_by_track: dict[int, str],
    min_frames: int = POSSESSION_MIN_FRAMES,
) -> list[PossessionSegment]:
    segments: list[PossessionSegment] = []
    current: Optional[dict] = None
    candidate: Optional[dict] = None

    def close() -> None:
        nonlocal current
        if current:
            segments.append(PossessionSegment(
                track_id=current["track_id"],
                team=team_by_track.get(current["track_id"]),
                jersey=current["start_holder"].get("jersey"),
                start_fid=current["start_frame"].frame_id,
                end_fid=current["end_frame"].frame_id,
                start_xy=(current["start_holder"]["x"], current["start_holder"]["y"]),
                end_xy=(current["end_holder"]["x"], current["end_holder"]["y"]),
            ))
        current = None

    for frame in sorted(frames, key=lambda item: item.frame_id):
        holder = _nearest_holder(frame)
        track_id = int(holder["track_id"]) if holder and holder.get("track_id") is not None else None
        if track_id is None:
            close()
            candidate = None
            continue
        if current and current["track_id"] == track_id:
            current["end_frame"], current["end_holder"] = frame, holder
            candidate = None
            continue
        if not candidate or candidate["track_id"] != track_id:
            candidate = {"track_id": track_id, "count": 1}
        else:
            candidate["count"] += 1
        if candidate["count"] < min_frames:
            continue
        close()
        current = {
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
            "start_holder": holder,
            "end_holder": holder,
        }
        candidate = None
    close()
    return segments


def _third(x: float) -> str:
    return "left third" if x < -17.5 else "right third" if x > 17.5 else "middle third"


def build_tracking_digest(predictions_json: Path, fps: int = 25) -> str:
    frames = load_frames(predictions_json)
    if not frames:
        return "[Tracking Digest]\n(no tracking data available)"

    team_by_track = _majority(frames, "team")
    role_by_track = _majority(frames, "role")
    segments = possession_segments(frames, team_by_track)
    parts = ["[Team Rosters - jersey numbers seen by the tracking system]"]
    jerseys: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        for player in frame.players:
            if player.get("team") and player.get("jersey"):
                jerseys[player["team"]].add(str(player["jersey"]))
    for team in sorted(jerseys):
        numbers = ", ".join(f"#{n}" for n in sorted(jerseys[team], key=lambda n: (len(n), n)))
        parts.append(f"{team} team: {numbers}")
    if not jerseys:
        parts.append("(no jersey numbers read)")

    parts.append("\n[Possession Timeline] (left/right thirds are screen-space pitch halves)")
    for segment in segments:
        start = max(0, segment.start_fid - POSSESSION_MIN_FRAMES) / fps
        end = segment.end_fid / fps
        jersey = f"#{segment.jersey}" if segment.jersey else "unknown number"
        parts.append(
            f"t={start:.1f}-{end:.1f}s: {jersey} "
            f"({segment.team or 'unknown team'}, {role_by_track.get(segment.track_id, 'player')}) "
            f"holds the ball in the {_third(segment.start_xy[0])}"
        )
    if not segments:
        parts.append("(no stable possession detected)")
    return "\n".join(parts)
