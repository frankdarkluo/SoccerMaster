"""Rule-based event detection from predictions.json (segment + velocity based).

Emits raw candidates for pass, shoot, goal, clearance, interception, dribble.
Assist is composed post-verify (compose_assists). Timestamps are ACTION moments.
"""
from __future__ import annotations

import json
import math
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
        ball_pos = {
            frame.frame_id: frame.ball_xy
            for frame in frames
            if frame.ball_xy is not None
        }
        velocities = self._velocities(ball_pos)

        raw: List[Event] = []
        raw += self._passes(segments)
        raw += self._interceptions(segments)
        raw += self._dribbles(segments, frames, team_by_track)
        raw += self._shots(velocities, ball_pos, segments)
        raw += self._goals(ball_pos, segments)
        raw += self._clearances(velocities, ball_pos, segments)
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
        events: List[Event] = []
        frame_by_id = {frame.frame_id: frame for frame in frames}
        for segment in segments:
            displacement = math.hypot(
                segment.end_xy[0] - segment.start_xy[0],
                segment.end_xy[1] - segment.start_xy[1],
            )
            if displacement < DRIBBLE_MIN_DISPLACEMENT_M:
                continue

            engaged_fid = None
            min_d = None
            for fid in range(segment.start_fid, segment.end_fid + 1):
                frame = frame_by_id.get(fid)
                if frame is None or frame.ball_xy is None:
                    continue
                bx, by = frame.ball_xy
                for player in frame.players:
                    track_id = player.get("track_id")
                    if track_id is not None and team_by_track.get(int(track_id)) == segment.team:
                        continue
                    d = math.hypot(player["x"] - bx, player["y"] - by)
                    if d <= DRIBBLE_OPPONENT_RADIUS_M and (min_d is None or d < min_d):
                        min_d = d
                        engaged_fid = fid

            if engaged_fid is None:
                continue

            events.append(self._make(
                "football.dribble",
                engaged_fid,
                player_jersey=segment.jersey,
                player_team=segment.team,
                track_id=segment.track_id,
                confidence=0.6,
                description_hint=f"dribble: {displacement:.1f}m carry past opponent",
            ))
        return events

    def _velocities(self, ball_pos: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        velocities: Dict[int, float] = {}
        previous_fid = None
        previous_xy = None
        for fid in sorted(ball_pos):
            xy = ball_pos[fid]
            if previous_fid is not None and previous_xy is not None:
                velocities[fid] = math.hypot(xy[0] - previous_xy[0], xy[1] - previous_xy[1]) * self.fps
            previous_fid = fid
            previous_xy = xy
        return velocities

    def _last_holder_before(
        self,
        segments: List[PossessionSegment],
        frame_id: int,
    ) -> Optional[PossessionSegment]:
        best = None
        for segment in segments:
            if segment.end_fid <= frame_id and (best is None or segment.end_fid > best.end_fid):
                best = segment
        return best

    def _holder_at_frame(
        self,
        segments: List[PossessionSegment],
        frame_id: int,
    ) -> Optional[PossessionSegment]:
        for segment in segments:
            if segment.start_fid <= frame_id <= segment.end_fid:
                return segment
        return self._last_holder_before(segments, frame_id)

    def _shots(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        events: List[Event] = []
        for fid, speed in velocities.items():
            if speed < self.shot_speed_threshold or fid not in ball_pos:
                continue
            bx, _ = ball_pos[fid]
            if abs(bx) <= (GOAL_X - PENALTY_AREA_LENGTH):
                continue
            shooter = self._last_holder_before(segments, fid)
            events.append(self._make(
                "football.shoot",
                fid,
                player_jersey=shooter.jersey if shooter else None,
                player_team=shooter.team if shooter else None,
                track_id=shooter.track_id if shooter else None,
                ball_speed_mps=speed,
                confidence=min(speed / 20.0, 1.0),
                description_hint=f"shot: ball {speed:.1f} m/s near goal",
            ))
        return events

    def _goals(
        self,
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        events: List[Event] = []
        for fid, (bx, by) in ball_pos.items():
            if abs(abs(bx) - GOAL_X) > GOAL_LINE_TOL_M or abs(by) > GOAL_Y_HALF:
                continue
            scorer = self._last_holder_before(segments, fid)
            events.append(self._make(
                "football.goal",
                fid,
                player_jersey=scorer.jersey if scorer else None,
                player_team=scorer.team if scorer else None,
                track_id=scorer.track_id if scorer else None,
                confidence=0.85,
                description_hint="goal: ball crosses goal line",
            ))
        return events

    def _clearances(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
        segments: List[PossessionSegment],
    ) -> List[Event]:
        events: List[Event] = []
        for fid, speed in velocities.items():
            if speed < CLEARANCE_SPEED_MPS or fid not in ball_pos or (fid - 1) not in ball_pos:
                continue
            bx, _ = ball_pos[fid]
            prev_bx, _ = ball_pos[fid - 1]
            moving_away = abs(bx) < abs(prev_bx)
            in_own_half = abs(prev_bx) > 20
            if not (moving_away and in_own_half):
                continue
            clearer = self._holder_at_frame(segments, fid)
            events.append(self._make(
                "football.clearance",
                fid,
                player_jersey=clearer.jersey if clearer else None,
                player_team=clearer.team if clearer else None,
                track_id=clearer.track_id if clearer else None,
                ball_speed_mps=speed,
                confidence=0.6,
                description_hint="clearance: ball driven away from own goal",
            ))
        return events

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


def dedup_events(events: List[Event]) -> List[Event]:
    by_code: Dict[str, List[Event]] = {}
    for ev in events:
        by_code.setdefault(ev.event_code, []).append(ev)
    kept: List[Event] = []
    for code, group in by_code.items():
        gap = EVENT_GAP_S.get(code, 1.0)
        for ev in sorted(group, key=lambda e: e.timestamp_s):
            if kept and kept[-1].event_code == code and ev.timestamp_s - kept[-1].timestamp_s < gap:
                prev = kept[-1]
                better = ev.importance > prev.importance or (
                    ev.importance == prev.importance and ev.confidence > prev.confidence
                )
                if better:
                    kept[-1] = ev
            else:
                kept.append(ev)
    return sorted(kept, key=lambda e: e.timestamp_s)


def write_events_json(events: List[Event], output_path, video_info: Optional[dict] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info or {},
        "schema_version": "v3-20260319",
        "events": [e.to_dict() for e in events],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
