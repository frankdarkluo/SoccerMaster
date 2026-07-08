"""Shared event data types for stage 2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Event:
    event_id: str
    timestamp_s: float
    frame_id: int
    event_code: str
    display_name_en: str
    display_name_cn: str
    importance: float
    player_jersey: Optional[str] = None
    player_team: Optional[str] = None
    target_jersey: Optional[str] = None
    target_team: Optional[str] = None
    track_id: Optional[int] = None
    ball_speed_mps: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    description_hint: str = ""

    def to_dict(self) -> dict:
        d = {
            "event_id": self.event_id,
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_id": self.frame_id,
            "event_code": self.event_code,
            "display_name_en": self.display_name_en,
            "display_name_cn": self.display_name_cn,
            "importance": self.importance,
            "player_jersey": self.player_jersey,
            "player_team": self.player_team,
            "tags": self.tags,
            "confidence": round(self.confidence, 2),
            "description_hint": self.description_hint,
        }
        if self.target_jersey:
            d["target_jersey"] = self.target_jersey
            d["target_team"] = self.target_team
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        if self.ball_speed_mps is not None:
            d["ball_speed_mps"] = round(self.ball_speed_mps, 1)
        return d


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


@dataclass
class Candidate:
    """An untyped key moment proposed by the rule engine. The VLM names the action."""
    candidate_id: str
    frame_id: int
    timestamp_s: float
    signals: List[str]
    strength: float
    track_id: Optional[int] = None
    jersey: Optional[str] = None
    team: Optional[str] = None
    role: str = "player"
    ball_speed_mps: Optional[float] = None
    ball_xy: Optional[Tuple[float, float]] = None
    ball_direction: Optional[str] = None  # toward_opponent_goal | toward_own_goal | lateral
    prev_holder: Optional[dict] = None    # {track_id, jersey, team, role, end_fid}
    next_holder: Optional[dict] = None    # {track_id, jersey, team, role, start_fid}

    def to_dict(self) -> dict:
        d = {
            "candidate_id": self.candidate_id,
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_id": self.frame_id,
            "signals": list(self.signals),
            "strength": round(self.strength, 2),
            "track_id": int(self.track_id) if self.track_id is not None else None,
            "jersey": self.jersey,
            "team": self.team,
            "role": self.role,
        }
        if self.ball_speed_mps is not None:
            d["ball_speed_mps"] = round(self.ball_speed_mps, 1)
        if self.ball_xy is not None:
            d["ball_xy"] = [round(self.ball_xy[0], 1), round(self.ball_xy[1], 1)]
        if self.ball_direction:
            d["ball_direction"] = self.ball_direction
        if self.prev_holder is not None:
            d["prev_holder"] = dict(self.prev_holder)
        if self.next_holder is not None:
            d["next_holder"] = dict(self.next_holder)
        return d


@dataclass
class Classification:
    """Parsed VLM answer for one candidate."""
    action: str = "none"                 # full code (football.pass) or "none"
    outcome: Optional[str] = None        # success | failure | None
    actor_jersey: Optional[str] = None
    actor_team: Optional[str] = None
    receiver_jersey: Optional[str] = None
    confidence: float = 0.5
    tags: Dict[str, str] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Verdict:
    verdict: str = "uncertain"
    outcome: Optional[str] = None
    actor_jersey: Optional[str] = None
    actor_team: Optional[str] = None
    receiver_jersey: Optional[str] = None
    corrected_event_code: Optional[str] = None
    reason: str = ""
