"""Rule-based event detection from predictions.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.enricher import enrich_events
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Event, FrameData
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X, GOAL_Y_HALF, PENALTY_AREA_LENGTH

EVENT_GAP_S = {
    "football.goal": 2.0,
    "football.shoot": 1.0,
    "football.pass": 0.4,
    "football.clearance": 1.0,
    "football.interception": 0.8,
    "football.assist": 1.0,
}
GOAL_SHOT_SUPPRESS_S = 1.5
PASS_POSSESSION_STABLE_FRAMES = 2


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
        self._event_counter = 0

    def detect(self, predictions_json_path: str) -> List[Event]:
        self._event_counter = 0
        frames = self._load_frames(predictions_json_path)
        if not frames:
            return []

        ball_positions = self._extract_ball_positions(frames)
        ball_velocities = self._compute_velocities(ball_positions)
        possession_chain = self._compute_possession(frames)

        raw_events: List[Event] = []
        raw_events += self._detect_passes(possession_chain)
        raw_events += self._detect_shots(ball_velocities, ball_positions, frames)
        raw_events += self._detect_goals(ball_positions, frames)
        raw_events += self._detect_clearances(ball_velocities, ball_positions)
        raw_events += self._detect_interceptions(possession_chain)
        raw_events += self._detect_assists(raw_events)

        deduped = self._deduplicate(raw_events)
        deduped = self._suppress_shots_near_goals(deduped)
        enriched = enrich_events(deduped, frames)
        return sorted(enriched, key=lambda e: e.timestamp_s)

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:03d}"

    def _load_frames(self, path: str) -> List[FrameData]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        image_id_to_frame, _ = frame_index_from_labels(data)
        frames_dict: Dict[int, FrameData] = {}

        for frame_num in sorted(set(image_id_to_frame.values())):
            frames_dict[frame_num] = FrameData(frame_id=frame_num)

        for ann in data.get("annotations", []):
            image_id = str(ann.get("image_id", ""))
            frame_num = image_id_to_frame.get(image_id)
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
            role = attrs.get("role", "other")
            if role == "ball":
                frames_dict[frame_num].ball_xy = (float(x), float(y))
            else:
                frames_dict[frame_num].players.append({
                    "track_id": ann.get("track_id"),
                    "x": float(x),
                    "y": float(y),
                    "role": role,
                    "team": attrs.get("team"),
                    "jersey": attrs.get("jersey", ""),
                })

        return [frames_dict[k] for k in sorted(frames_dict.keys())]

    def _extract_ball_positions(self, frames: List[FrameData]) -> Dict[int, Tuple[float, float]]:
        return {f.frame_id: f.ball_xy for f in frames if f.ball_xy is not None}

    def _compute_velocities(self, ball_pos: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        vels: Dict[int, float] = {}
        sorted_ids = sorted(ball_pos.keys())
        for i in range(1, len(sorted_ids)):
            f1, f2 = sorted_ids[i - 1], sorted_ids[i]
            dt = (f2 - f1) / self.fps
            if dt <= 0:
                continue
            dx = ball_pos[f2][0] - ball_pos[f1][0]
            dy = ball_pos[f2][1] - ball_pos[f1][1]
            vels[f2] = math.hypot(dx, dy) / dt
        return vels

    def _compute_possession(self, frames: List[FrameData]) -> Dict[int, Optional[dict]]:
        possession: Dict[int, Optional[dict]] = {}
        for f in frames:
            if f.ball_xy is None or not f.players:
                possession[f.frame_id] = None
                continue
            bx, by = f.ball_xy
            nearest = min(f.players, key=lambda p: math.hypot(p["x"] - bx, p["y"] - by))
            dist = math.hypot(nearest["x"] - bx, nearest["y"] - by)
            possession[f.frame_id] = nearest if dist < 5.0 else None
        return possession

    def _make_event(
        self,
        event_code: str,
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
        ev_def = self.schema.get_event(event_code)
        tid = int(track_id) if track_id is not None else None
        return Event(
            event_id=self._next_id(),
            timestamp_s=frame_id / self.fps,
            frame_id=frame_id,
            event_code=event_code,
            display_name_en=ev_def.display_name_en if ev_def else event_code,
            display_name_cn=ev_def.display_name_cn if ev_def else event_code,
            importance=ev_def.importance_base if ev_def else 0.3,
            player_jersey=player_jersey,
            player_team=player_team,
            target_jersey=target_jersey,
            target_team=target_team,
            track_id=tid,
            ball_speed_mps=ball_speed_mps,
            confidence=confidence,
            description_hint=description_hint,
        )

    def _detect_passes(self, possession: Dict[int, Optional[dict]]) -> List[Event]:
        events: List[Event] = []
        prev_player = None
        stable_player = None
        stable_frames = 0
        for fid in sorted(possession.keys()):
            curr = possession[fid]
            if curr is None:
                prev_player = None
                stable_player = None
                stable_frames = 0
                continue

            if stable_player is not None and curr["track_id"] == stable_player["track_id"]:
                stable_frames += 1
            else:
                stable_player = curr
                stable_frames = 1

            if (
                prev_player is not None
                and curr["track_id"] != prev_player["track_id"]
                and stable_frames >= PASS_POSSESSION_STABLE_FRAMES
                and curr["team"] == prev_player["team"]
                and curr["team"] is not None
            ):
                events.append(self._make_event(
                    "football.pass",
                    fid,
                    player_jersey=prev_player.get("jersey"),
                    player_team=prev_player.get("team"),
                    target_jersey=curr.get("jersey"),
                    target_team=curr.get("team"),
                    track_id=prev_player.get("track_id"),
                    confidence=0.7,
                    description_hint="pass detected via possession switch",
                ))
            prev_player = curr
        return events

    def _detect_shots(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
        frames: List[FrameData],
    ) -> List[Event]:
        events: List[Event] = []
        frame_by_id = {f.frame_id: f for f in frames}
        for fid, speed in velocities.items():
            if speed < self.shot_speed_threshold or fid not in ball_pos:
                continue
            bx, _ = ball_pos[fid]
            if abs(bx) <= (GOAL_X - PENALTY_AREA_LENGTH):
                continue
            frame_data = frame_by_id.get(fid)
            shooter = None
            if frame_data and frame_data.ball_xy:
                possession = self._compute_possession([frame_data])
                shooter = possession.get(fid)
            events.append(self._make_event(
                "football.shoot",
                fid,
                player_jersey=shooter.get("jersey") if shooter else None,
                player_team=shooter.get("team") if shooter else None,
                ball_speed_mps=speed,
                confidence=min(speed / 20.0, 1.0),
                description_hint=f"shot detected: ball speed {speed:.1f} m/s near goal",
            ))
        return events

    def _detect_goals(
        self,
        ball_pos: Dict[int, Tuple[float, float]],
        frames: List[FrameData],
    ) -> List[Event]:
        events: List[Event] = []
        frame_by_id = {f.frame_id: f for f in frames}
        for fid, (bx, by) in ball_pos.items():
            if abs(abs(bx) - GOAL_X) > 1.5 or abs(by) > GOAL_Y_HALF:
                continue
            frame_data = frame_by_id.get(fid)
            scorer = None
            if frame_data and frame_data.ball_xy:
                possession = self._compute_possession([frame_data])
                scorer = possession.get(fid)
            events.append(self._make_event(
                "football.goal",
                fid,
                player_jersey=scorer.get("jersey") if scorer else None,
                player_team=scorer.get("team") if scorer else None,
                confidence=0.85,
                description_hint="goal detected: ball crosses goal line",
            ))
        return events

    def _detect_clearances(
        self,
        velocities: Dict[int, float],
        ball_pos: Dict[int, Tuple[float, float]],
    ) -> List[Event]:
        events: List[Event] = []
        for fid, speed in velocities.items():
            if speed < 8.0 or fid not in ball_pos:
                continue
            bx, _ = ball_pos[fid]
            prev_fid = fid - 1
            if prev_fid not in ball_pos:
                continue
            prev_bx, _ = ball_pos[prev_fid]
            moving_away = abs(bx) < abs(prev_bx)
            in_own_half = abs(prev_bx) > 20
            if moving_away and in_own_half:
                events.append(self._make_event(
                    "football.clearance",
                    fid,
                    ball_speed_mps=speed,
                    confidence=0.6,
                    description_hint="clearance: ball moving away from goal at high speed",
                ))
        return events

    def _detect_interceptions(self, possession: Dict[int, Optional[dict]]) -> List[Event]:
        events: List[Event] = []
        prev_player = None
        for fid in sorted(possession.keys()):
            curr = possession[fid]
            if curr is None:
                prev_player = None
                continue
            if prev_player is not None and curr["track_id"] != prev_player["track_id"]:
                if (
                    curr["team"] != prev_player["team"]
                    and curr["team"] is not None
                    and prev_player["team"] is not None
                ):
                    events.append(self._make_event(
                        "football.interception",
                        fid,
                        player_jersey=curr.get("jersey"),
                        player_team=curr.get("team"),
                        track_id=curr.get("track_id"),
                        confidence=0.6,
                        description_hint="interception: possession switched between teams",
                    ))
            prev_player = curr
        return events

    def _detect_assists(self, existing_events: List[Event]) -> List[Event]:
        events: List[Event] = []
        goals = [e for e in existing_events if e.event_code == "football.goal"]
        passes = [e for e in existing_events if e.event_code == "football.pass"]
        for goal in goals:
            candidates = [
                p for p in passes
                if 0 < (goal.timestamp_s - p.timestamp_s) < 5.0
                and p.target_team == goal.player_team
            ]
            if not candidates:
                continue
            last_pass = max(candidates, key=lambda p: p.timestamp_s)
            events.append(self._make_event(
                "football.assist",
                last_pass.frame_id,
                player_jersey=last_pass.player_jersey,
                player_team=last_pass.player_team,
                confidence=0.8,
                description_hint=f"assist: pass by #{last_pass.player_jersey} before goal",
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

    def _suppress_shots_near_goals(self, events: List[Event]) -> List[Event]:
        goals = [e for e in events if e.event_code == "football.goal"]
        if not goals:
            return events
        filtered: List[Event] = []
        for ev in events:
            if ev.event_code != "football.shoot":
                filtered.append(ev)
                continue
            near_goal = any(
                0 <= goal.timestamp_s - ev.timestamp_s <= GOAL_SHOT_SUPPRESS_S
                for goal in goals
            )
            if not near_goal:
                filtered.append(ev)
        return filtered

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
