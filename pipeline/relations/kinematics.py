"""Per-track smoothed position/velocity series from FrameData.

Measurements only. Velocities are EMA-smoothed central differences; the EMA
horizon (smooth_s) suppresses GSR jitter without inventing anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pipeline.stage2b.digest import FrameData, resolve_role_by_track, resolve_team_by_track


@dataclass
class TrackSeries:
    track_id: int
    team: Optional[str]
    role: str
    jersey: str
    fids: List[int] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    coverage: float = 0.0  # fraction of frames this track appears in

    def at(self, fid: int) -> Optional[int]:
        """Index of the sample at frame id `fid`, or None."""
        try:
            return self.fids.index(fid)
        except ValueError:
            return None


def _majority_jersey(frames: List[FrameData], track_id: int) -> str:
    from collections import Counter

    votes = Counter()
    for fr in frames:
        for p in fr.players:
            if p.get("track_id") == track_id and p.get("jersey"):
                votes[p["jersey"]] += 1
    return votes.most_common(1)[0][0] if votes else ""


def build_tracks(
    frames: List[FrameData], fps: float, smooth_s: float = 0.5
) -> Dict[int, TrackSeries]:
    teams = resolve_team_by_track(frames)
    roles = resolve_role_by_track(frames)
    n_frames = len(frames)

    raw: Dict[int, List[tuple]] = {}
    for fr in frames:
        for p in fr.players:
            tid = p.get("track_id")
            if tid is None:
                continue
            raw.setdefault(int(tid), []).append((fr.frame_id, p["x"], p["y"]))

    alpha = 1.0 - pow(0.5, 1.0 / max(1.0, smooth_s * fps))
    tracks: Dict[int, TrackSeries] = {}
    for tid, samples in raw.items():
        samples.sort()
        tr = TrackSeries(
            track_id=tid,
            team=teams.get(tid),
            role=roles.get(tid, "player"),
            jersey=_majority_jersey(frames, tid),
        )
        ema_vx = ema_vy = 0.0
        for i, (fid, x, y) in enumerate(samples):
            tr.fids.append(fid)
            tr.x.append(x)
            tr.y.append(y)
            if i == 0:
                inst_vx = inst_vy = 0.0
            else:
                pf, px, py = samples[i - 1]
                dt = max(1, fid - pf) / fps
                inst_vx = (x - px) / dt
                inst_vy = (y - py) / dt
            ema_vx = alpha * inst_vx + (1 - alpha) * ema_vx
            ema_vy = alpha * inst_vy + (1 - alpha) * ema_vy
            tr.vx.append(ema_vx)
            tr.vy.append(ema_vy)
            tr.speed.append((ema_vx ** 2 + ema_vy ** 2) ** 0.5)
        tr.coverage = len(samples) / max(1, n_frames)
        tracks[tid] = tr
    return tracks


def ball_series(frames: List[FrameData], fps: float, smooth_s: float = 0.2):
    """(fids, x, y, speed) for the ball. Shorter smoothing: kicks are fast."""
    fids, xs, ys, speeds = [], [], [], []
    alpha = 1.0 - pow(0.5, 1.0 / max(1.0, smooth_s * fps))
    ema = 0.0
    prev = None
    for fr in frames:
        if fr.ball_xy is None:
            continue
        x, y = fr.ball_xy
        if prev is not None:
            pf, px, py = prev
            dt = max(1, fr.frame_id - pf) / fps
            inst = (((x - px) ** 2 + (y - py) ** 2) ** 0.5) / dt
        else:
            inst = 0.0
        ema = alpha * inst + (1 - alpha) * ema
        fids.append(fr.frame_id)
        xs.append(x)
        ys.append(y)
        speeds.append(ema)
        prev = (fr.frame_id, x, y)
    return fids, xs, ys, speeds
