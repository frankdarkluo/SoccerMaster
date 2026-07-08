"""Untyped key-moment candidate detection from predictions.json.

The rule engine no longer names actions. It detects 5 structural signals -
kick (segment end + ball speed), possession change, carry-past-opponent,
sustained pressure, geometry (goal line / box entry) - merges them into
candidates (timestamp + actor + facts), and classify.py asks a VLM to name
the action. Typed helpers that run AFTER classification (dedup, assists,
buildup density filler, events.json writer) also live here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Candidate, Event, FrameData, PossessionSegment
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X, GOAL_Y_HALF, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH

KICK_SPEED_MPS = 8.0
PRESSURE_RADIUS_M = 4.0
PRESSURE_MIN_FRAMES = 10
CARRY_MIN_DISPLACEMENT_M = 6.0
CARRY_OPPONENT_RADIUS_M = 4.0
GOAL_LINE_TOL_M = 1.5
MERGE_WINDOW_S = 0.5
CANDIDATE_CAP = 20
BIN_S = 5.0
ASSIST_WINDOW_S = 5.0

SIGNAL_STRENGTH = {
    "goal_line": 1.0,
    "kick": 0.7,
    "possession_win": 0.6,
    "carry": 0.5,
    "box_entry": 0.5,
    "pressure": 0.4,
}

EVENT_GAP_S = {
    "football.goal": 2.0,
    "football.shoot": 1.0,
    "football.pass": 0.4,
    "football.clearance": 1.0,
    "football.interception": 0.8,
    "football.dribble": 1.0,
    "football.assist": 1.0,
    "football.tackle": 0.8,
    "football.pressing": 1.0,
    "football.save": 1.0,
    "football.goal_kick": 1.0,
    "football.buildup": 5.0,
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


def ball_positions(frames: List[FrameData]) -> Dict[int, Tuple[float, float]]:
    return {frame.frame_id: frame.ball_xy for frame in frames if frame.ball_xy is not None}


def ball_velocities(ball_pos: Dict[int, Tuple[float, float]], fps: int) -> Dict[int, float]:
    velocities: Dict[int, float] = {}
    fids = sorted(ball_pos)
    for i in range(1, len(fids)):
        f1, f2 = fids[i - 1], fids[i]
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        dx = ball_pos[f2][0] - ball_pos[f1][0]
        dy = ball_pos[f2][1] - ball_pos[f1][1]
        velocities[f2] = math.hypot(dx, dy) / dt
    return velocities


def _holder_info(segment: PossessionSegment, role_by_track: Dict[int, str], fid_key: str) -> dict:
    return {
        "track_id": segment.track_id,
        "jersey": segment.jersey,
        "team": segment.team,
        "role": role_by_track.get(segment.track_id, "player"),
        fid_key: segment.start_fid if fid_key == "start_fid" else segment.end_fid,
    }


def _max_speed_after(velocities: Dict[int, float], fid: int, horizon: int = 5) -> Optional[float]:
    speeds = [velocities[f] for f in range(fid, fid + horizon + 1) if f in velocities]
    return max(speeds) if speeds else None


def _ball_direction(
    ball_pos: Dict[int, Tuple[float, float]],
    fid: int,
    team: Optional[str],
    horizon: int = 5,
) -> Optional[str]:
    pts = [ball_pos[f] for f in range(fid, fid + horizon + 1) if f in ball_pos]
    if len(pts) < 2 or team not in ("left", "right"):
        return None
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    attack_sign = 1 if team != "right" else -1
    toward = dx * attack_sign
    if abs(toward) < abs(dy) * 0.5:
        return "lateral"
    return "toward_opponent_goal" if toward > 0 else "toward_own_goal"


def _raw(signal: str, frame_id: int, fps: int, **kw) -> Candidate:
    return Candidate(
        candidate_id="",
        frame_id=frame_id,
        timestamp_s=frame_id / fps,
        signals=[signal],
        strength=SIGNAL_STRENGTH[signal],
        **kw,
    )


def _kick_candidates(segments, velocities, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    for i, seg in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        speed = _max_speed_after(velocities, seg.end_fid)
        if nxt is None and (speed is None or speed < KICK_SPEED_MPS):
            continue
        out.append(_raw(
            "kick",
            seg.end_fid,
            fps,
            track_id=seg.track_id,
            jersey=seg.jersey,
            team=seg.team,
            role=role_by_track.get(seg.track_id, "player"),
            ball_speed_mps=speed,
            ball_xy=ball_pos.get(seg.end_fid),
            ball_direction=_ball_direction(ball_pos, seg.end_fid, seg.team),
            next_holder=_holder_info(nxt, role_by_track, "start_fid") if nxt else None,
        ))
    return out


def _possession_win_candidates(segments, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    for a, b in zip(segments, segments[1:]):
        if a.team is None or b.team is None or a.team == b.team:
            continue
        out.append(_raw(
            "possession_win",
            b.start_fid,
            fps,
            track_id=b.track_id,
            jersey=b.jersey,
            team=b.team,
            role=role_by_track.get(b.track_id, "player"),
            ball_xy=ball_pos.get(b.start_fid),
            prev_holder=_holder_info(a, role_by_track, "end_fid"),
        ))
    return out


def _carry_candidates(
    segments,
    frames,
    team_by_track,
    role_by_track,
    ball_pos,
    fps,
) -> List[Candidate]:
    out = []
    frame_by_id = {frame.frame_id: frame for frame in frames}
    for seg in segments:
        displacement = math.hypot(seg.end_xy[0] - seg.start_xy[0], seg.end_xy[1] - seg.start_xy[1])
        if displacement < CARRY_MIN_DISPLACEMENT_M:
            continue
        engaged_fid = None
        min_d = None
        for fid in range(seg.start_fid, seg.end_fid + 1):
            frame = frame_by_id.get(fid)
            if frame is None or frame.ball_xy is None:
                continue
            bx, by = frame.ball_xy
            for player in frame.players:
                tid = player.get("track_id")
                if tid is not None and team_by_track.get(int(tid)) == seg.team:
                    continue
                if player.get("role") == "referee":
                    continue
                d = math.hypot(player["x"] - bx, player["y"] - by)
                if d <= CARRY_OPPONENT_RADIUS_M and (min_d is None or d < min_d):
                    min_d, engaged_fid = d, fid
        if engaged_fid is None:
            continue
        out.append(_raw(
            "carry",
            engaged_fid,
            fps,
            track_id=seg.track_id,
            jersey=seg.jersey,
            team=seg.team,
            role=role_by_track.get(seg.track_id, "player"),
            ball_xy=ball_pos.get(engaged_fid),
        ))
    return out


def _pressure_candidates(
    segments,
    frames,
    team_by_track,
    role_by_track,
    ball_pos,
    fps,
) -> List[Candidate]:
    out = []
    frame_by_id = {frame.frame_id: frame for frame in frames}
    for seg in segments:
        run_start = None
        run_len = 0
        presser = None
        for fid in range(seg.start_fid, seg.end_fid + 1):
            frame = frame_by_id.get(fid)
            opp = None
            if frame is not None and frame.ball_xy is not None:
                bx, by = frame.ball_xy
                best = None
                for player in frame.players:
                    tid = player.get("track_id")
                    if tid is None or player.get("role") == "referee":
                        continue
                    if team_by_track.get(int(tid)) == seg.team:
                        continue
                    d = math.hypot(player["x"] - bx, player["y"] - by)
                    if d <= PRESSURE_RADIUS_M and (best is None or d < best[0]):
                        best = (d, player)
                opp = best[1] if best else None
            if opp is not None:
                if run_start is None:
                    run_start, presser = fid, opp
                run_len += 1
                if run_len == PRESSURE_MIN_FRAMES:
                    out.append(_raw(
                        "pressure",
                        run_start,
                        fps,
                        track_id=int(presser["track_id"]),
                        jersey=presser.get("jersey"),
                        team=team_by_track.get(int(presser["track_id"])),
                        role=role_by_track.get(int(presser["track_id"]), "player"),
                        ball_xy=ball_pos.get(run_start),
                        prev_holder=_holder_info(seg, role_by_track, "end_fid"),
                    ))
            else:
                run_start, run_len, presser = None, 0, None
    return out


def _in_penalty_area(bx: float, by: float) -> bool:
    return abs(bx) >= (GOAL_X - PENALTY_AREA_LENGTH) and abs(by) <= PENALTY_AREA_WIDTH / 2


def _last_holder_before(segments, frame_id) -> Optional[PossessionSegment]:
    best = None
    for seg in segments:
        if seg.end_fid <= frame_id and (best is None or seg.end_fid > best.end_fid):
            best = seg
    return best


def _geometry_candidates(segments, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    fids = sorted(ball_pos)
    seen_goal_line = False
    for i, fid in enumerate(fids):
        bx, by = ball_pos[fid]
        crossed = abs(abs(bx) - GOAL_X) <= GOAL_LINE_TOL_M and abs(by) <= GOAL_Y_HALF
        entered = i > 0 and _in_penalty_area(bx, by) and not _in_penalty_area(*ball_pos[fids[i - 1]])
        signal = "goal_line" if crossed and not seen_goal_line else "box_entry" if entered else None
        if crossed:
            seen_goal_line = True
        elif not entered:
            seen_goal_line = False
        if signal is None:
            continue
        holder = _last_holder_before(segments, fid)
        out.append(_raw(
            signal,
            fid,
            fps,
            track_id=holder.track_id if holder else None,
            jersey=holder.jersey if holder else None,
            team=holder.team if holder else None,
            role=role_by_track.get(holder.track_id, "player") if holder else "player",
            ball_xy=ball_pos.get(fid),
        ))
    return out


def detect_candidates(
    frames: List[FrameData],
    segments: List[PossessionSegment],
    team_by_track: Dict[int, Optional[str]],
    role_by_track: Dict[int, str],
    fps: int = 25,
) -> List[Candidate]:
    """All raw signal candidates, time-sorted, unmerged and without ids."""
    ball_pos = ball_positions(frames)
    velocities = ball_velocities(ball_pos, fps)
    raw: List[Candidate] = []
    raw += _kick_candidates(segments, velocities, ball_pos, role_by_track, fps)
    raw += _possession_win_candidates(segments, ball_pos, role_by_track, fps)
    raw += _carry_candidates(segments, frames, team_by_track, role_by_track, ball_pos, fps)
    raw += _pressure_candidates(segments, frames, team_by_track, role_by_track, ball_pos, fps)
    raw += _geometry_candidates(segments, ball_pos, role_by_track, fps)
    return sorted(raw, key=lambda c: c.timestamp_s)


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


def write_candidates_json(
    candidates: List[Candidate],
    output_path,
    video_info: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info or {},
        "schema_version": "v4-candidates-20260707",
        "candidates": [c.to_dict() for c in candidates],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compose_assists(events: List[Event]) -> List[Event]:
    """Append assist events for a pass whose receiver scores within ASSIST_WINDOW_S.

    Runs POST-classification so only surviving passes and goals qualify.
    """
    schema = EventSchema()
    goals = [e for e in events if e.event_code == "football.goal"]
    passes = [e for e in events if e.event_code == "football.pass"]
    ev_def = schema.get_event("football.assist")
    out = list(events)
    counter = len(events)
    for goal in goals:
        cands = [
            p for p in passes
            if 0 < (goal.timestamp_s - p.timestamp_s) < ASSIST_WINDOW_S
            and p.target_team == goal.player_team
            and (p.target_jersey == goal.player_jersey or goal.player_jersey is None)
        ]
        if not cands:
            continue
        last = max(cands, key=lambda p: p.timestamp_s)
        counter += 1
        out.append(Event(
            event_id=f"evt_{counter:03d}",
            timestamp_s=last.timestamp_s,
            frame_id=last.frame_id,
            event_code="football.assist",
            display_name_en=ev_def.display_name_en,
            display_name_cn=ev_def.display_name_cn,
            importance=ev_def.importance_base,
            player_jersey=last.player_jersey,
            player_team=last.player_team,
            target_jersey=last.target_jersey,
            target_team=last.target_team,
            confidence=0.8,
            description_hint=f"assist: #{last.player_jersey} before goal",
        ))
    return out
