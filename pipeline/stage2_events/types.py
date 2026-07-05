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
class Verdict:
    verdict: str = "uncertain"
    outcome: Optional[str] = None
    actor_jersey: Optional[str] = None
    actor_team: Optional[str] = None
    receiver_jersey: Optional[str] = None
    corrected_event_code: Optional[str] = None
    reason: str = ""
