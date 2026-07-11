"""Per-snapshot relational measurements. Measurements only — no tactic labels."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from pipeline.relations.kinematics import TrackSeries, ball_series
from pipeline.stage2b.digest import FrameData
from pipeline.topology.analysis import infer_attack_dirs
from pipeline.topology.io_gamestate import Detection

CARRIER_RADIUS_M = 3.0
LOCAL_RADIUS_M = 15.0
MIN_COVERAGE = 0.8


def _detections_for_attack_dir(frames: List[FrameData]) -> List[Detection]:
    dets = []
    for fr in frames:
        for p in fr.players:
            if p.get("team") in ("left", "right"):
                dets.append(Detection(
                    frame=fr.frame_id, track_id=p.get("track_id") or -1,
                    x=p["x"], y=p["y"], team=p["team"],
                    role=p.get("role", "player"),
                    jersey_number=(int(p["jersey"]) if p.get("jersey") else None),
                ))
    return dets


def _sample_track(tr: TrackSeries, fid: int) -> Optional[dict]:
    """Nearest stored sample within 3 frames of fid."""
    best, best_d = None, 4
    for i, f in enumerate(tr.fids):
        d = abs(f - fid)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    return {"x": tr.x[best], "y": tr.y[best], "speed": tr.speed[best]}


def build_snapshots(frames: List[FrameData], tracks: Dict[int, TrackSeries],
                    fps: float, hz: float = 2.0) -> List[dict]:
    attack_dirs = infer_attack_dirs(_detections_for_attack_dir(frames))
    b_fids, b_x, b_y, b_speed = ball_series(frames, fps)
    ball_by_fid = {f: (x, y, s) for f, x, y, s in zip(b_fids, b_x, b_y, b_speed)}

    qualifying = {
        tid: tr for tid, tr in tracks.items()
        if tr.coverage >= MIN_COVERAGE and tr.role in ("player", "goalkeeper")
        and tr.team in ("left", "right")
    }

    step = max(1, int(round(fps / hz)))
    snapshots = []
    for fid in range(1, frames[-1].frame_id + 1, step):
        ball = ball_by_fid.get(fid)
        if ball is None:  # tolerate short ball dropouts
            near = [f for f in ball_by_fid if abs(f - fid) <= 3]
            if not near:
                continue
            ball = ball_by_fid[min(near, key=lambda f: abs(f - fid))]
        bx, by, bs = ball

        samples = {}
        for tid, tr in qualifying.items():
            s = _sample_track(tr, fid)
            if s is not None:
                s.update(track_id=tid, team=tr.team, jersey=tr.jersey, role=tr.role)
                samples[tid] = s

        carrier = None
        best = CARRIER_RADIUS_M
        for s in samples.values():
            d = math.hypot(s["x"] - bx, s["y"] - by)
            if d < best:
                best, carrier = d, s

        players_out = []
        teams_out = {}
        for team in ("left", "right"):
            adir = attack_dirs.get(team, 1)
            own = [s for s in samples.values() if s["team"] == team]
            opp = [s for s in samples.values() if s["team"] != team]
            opp_outfield = [s for s in opp if s["role"] != "goalkeeper"]
            xs = sorted((s["x"] * adir for s in opp_outfield), reverse=True)
            opp_line_x_signed = xs[1] if len(xs) >= 2 else (xs[0] if xs else None)
            teams_out[team] = {
                "attack_dir": adir,
                "opp_line_x": (opp_line_x_signed * adir) if opp_line_x_signed is not None else None,
                "n_within_15m_of_ball": sum(
                    1 for s in own if math.hypot(s["x"] - bx, s["y"] - by) <= LOCAL_RADIUS_M),
            }
            for s in own:
                row = {
                    "track_id": s["track_id"], "team": team, "jersey": s["jersey"],
                    "role": s["role"],
                    "x": round(s["x"], 1), "y": round(s["y"], 1),
                    "speed": round(s["speed"], 1),
                    "dist_ball": round(math.hypot(s["x"] - bx, s["y"] - by), 1),
                }
                if carrier is not None and s["track_id"] != carrier["track_id"]:
                    row["rel_x"] = round((s["x"] - carrier["x"]) * adir, 1)
                    row["rel_y"] = round(s["y"] - carrier["y"], 1)
                if opp_outfield:
                    row["dist_nearest_opp"] = round(min(
                        math.hypot(s["x"] - o["x"], s["y"] - o["y"]) for o in opp_outfield), 1)
                if opp_line_x_signed is not None:
                    row["depth_vs_line"] = round(s["x"] * adir - opp_line_x_signed, 1)
                players_out.append(row)

        snapshots.append({
            "t": round((fid - 1) / fps, 2),
            "frame_id": fid,
            "ball": {"x": round(bx, 1), "y": round(by, 1), "speed": round(bs, 1)},
            "carrier": ({"track_id": carrier["track_id"], "jersey": carrier["jersey"],
                         "team": carrier["team"]} if carrier else None),
            "players": players_out,
            "teams": teams_out,
        })
    return snapshots
