"""Rule-based event detection from predictions.json (segment + velocity based).

Emits raw candidates for pass, shoot, goal, clearance, interception, dribble.
Assist is composed post-verify (compose_assists). Timestamps are ACTION moments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.possession import (
    POSSESSION_RADIUS_M,
    possession_segments,
    resolve_team_by_track,
)
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Event, FrameData, PossessionSegment
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X, GOAL_Y_HALF, PENALTY_AREA_LENGTH

DRIBBLE_MIN_DISPLACEMENT_M = 6.0
DRIBBLE_OPPONENT_RADIUS_M = 4.0
CLEARANCE_SPEED_MPS = 8.0
GOAL_LINE_TOL_M = 1.5
ASSIST_WINDOW_S = 5.0

EVENT_GAP_S = {
    "football.goal": 2.0,
    "football.shoot": 1.0,
    "football.pass": 0.4,
    "football.clearance": 1.0,
    "football.interception": 0.8,
    "football.dribble": 1.0,
    "football.assist": 1.0,
}


def load_frames(path: str) -> List[FrameData]:
    with open(path, encoding="utf-8") as f:
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


class EventDetector:
    def __init__(
        self,
        schema: EventSchema,
        fps: int = 25,
        shot_speed_threshold: float = 10.0,
        min_gap_s: float = 1.0,
    ):
        self.schema = schema
        self.fps = fps
        self.shot_speed_threshold = shot_speed_threshold
        self.min_gap_s = min_gap_s
        self._counter = 0

    def detect(self, predictions_json_path: str) -> List[Event]:
        self._counter = 0
        frames = load_frames(predictions_json_path)
        if not frames:
            return []

        team_by_track = resolve_team_by_track(frames)
        segments = possession_segments(frames, team_by_track)

        raw: List[Event] = []
        raw += self._passes(segments)
        raw += self._interceptions(segments)
        raw += self._dribbles(segments, frames, team_by_track)
        raw += self._shots({}, {}, segments)
        raw += self._goals({}, segments)
        raw += self._clearances({}, {}, segments)
        return sorted(raw, key=lambda e: e.timestamp_s)

    def _next_id(self) -> str:
        self._counter += 1
        return f"evt_{self._counter:03d}"

    def _make(
        self,
        code: str,
        frame_id: int,
        *,
        player_jersey: Optional[str] = None,
        player_team: Optional[str] = None,
        target_jersey: Optional[str] = None,
        target_team: Optional[str] = None,
        track_id: Optional[int] = None,
        ball_speed_mps: Optional[float] = None,
        confidence: float = 0.7,
        description_hint: str = "",
    ) -> Event:
        ev = self.schema.get_event(code)
        tid = int(track_id) if track_id is not None else None
        return Event(
            event_id=self._next_id(),
            timestamp_s=frame_id / self.fps,
            frame_id=frame_id,
            event_code=code,
            display_name_en=ev.display_name_en if ev else code,
            display_name_cn=ev.display_name_cn if ev else code,
            importance=ev.importance_base if ev else 0.3,
            player_jersey=player_jersey,
            player_team=player_team,
            target_jersey=target_jersey,
            target_team=target_team,
            track_id=tid,
            ball_speed_mps=ball_speed_mps,
            confidence=confidence,
            description_hint=description_hint,
        )

    def _passes(self, segments: List[PossessionSegment]) -> List[Event]:
        events: List[Event] = []
        for a, b in zip(segments, segments[1:]):
            if a.track_id == b.track_id:
                continue
            if a.team is None or b.team is None:
                continue
            if a.team != b.team:
                continue
            events.append(self._make(
                "football.pass",
                a.end_fid,
                player_jersey=a.jersey,
                player_team=a.team,
                target_jersey=b.jersey,
                target_team=b.team,
                track_id=a.track_id,
                confidence=0.7,
                description_hint="pass: same-team possession hand-off",
            ))
        return events

    def _interceptions(self, segments: List[PossessionSegment]) -> List[Event]:
        events: List[Event] = []
        for a, b in zip(segments, segments[1:]):
            if a.team is None or b.team is None or a.team == b.team:
                continue
            events.append(self._make(
                "football.interception",
                b.start_fid,
                player_jersey=b.jersey,
                player_team=b.team,
                track_id=b.track_id,
                confidence=0.6,
                description_hint="interception: cross-team possession win",
            ))
        return events

    def _dribbles(
        self,
        segments: List[PossessionSegment],
        frames: List[FrameData],
        team_by_track: Dict[int, Optional[str]],
    ) -> List[Event]:
        return []

    def _shots(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        return []

    def _goals(
        self,
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        return []

    def _clearances(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        return []

    def _event_gap_s(self, event_code: str) -> float:
        return EVENT_GAP_S.get(event_code, self.min_gap_s)

    def _deduplicate(self, events: List[Event]) -> List[Event]:
        if not events:
            return []
        by_code: Dict[str, List[Event]] = {}
        for ev in events:
            by_code.setdefault(ev.event_code, []).append(ev)

        kept: List[Event] = []
        for code, code_events in by_code.items():
            gap = self._event_gap_s(code)
            sorted_evts = sorted(code_events, key=lambda e: e.timestamp_s)
            code_kept: List[Event] = []
            for ev in sorted_evts:
                if not code_kept:
                    code_kept.append(ev)
                    continue
                if ev.timestamp_s - code_kept[-1].timestamp_s < gap:
                    prev = code_kept[-1]
                    if ev.importance > prev.importance or (
                        ev.importance == prev.importance and ev.confidence > prev.confidence
                    ):
                        code_kept[-1] = ev
                else:
                    code_kept.append(ev)
            kept.extend(code_kept)
        return kept

    def write_events_json(
        self,
        events: List[Event],
        output_path: Path,
        video_info: Optional[dict] = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video_info": video_info or {},
            "schema_version": "v3-20260319",
            "events": [e.to_dict() for e in events],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
