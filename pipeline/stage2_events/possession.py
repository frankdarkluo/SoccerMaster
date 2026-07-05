"""Possession and team assignment helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from pipeline.stage2_events.types import FrameData


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
