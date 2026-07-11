"""Assemble relations.json: snapshots + kick anchors + meta. CLI included."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pipeline.relations.kinematics import ball_series, build_tracks
from pipeline.relations.snapshots import build_snapshots
from pipeline.stage2b.digest import FrameData, load_frames

KICK_SPEED_MPS = 8.0  # same threshold detector.py uses


def _kick_anchors(frames: List[FrameData], fps: float) -> List[dict]:
    """Ball-speed spikes: t where smoothed speed crosses KICK_SPEED_MPS upward."""
    fids, _, _, speeds = ball_series(frames, fps)
    anchors = []
    above = False
    for fid, speed in zip(fids, speeds):
        if speed >= KICK_SPEED_MPS and not above:
            anchors.append({"t": round((fid - 1) / fps, 2), "ball_speed": round(speed, 1)})
            above = True
        elif speed < KICK_SPEED_MPS * 0.6:
            above = False
    return anchors


def build_relations(frames: List[FrameData], fps: float, snapshot_hz: float = 2.0) -> dict:
    tracks = build_tracks(frames, fps)
    snapshots = build_snapshots(frames, tracks, fps, hz=snapshot_hz)
    return {
        "schema_version": "relations-v1-20260709",
        "video_info": {
            "duration_s": round(frames[-1].frame_id / fps, 2),
            "fps": fps,
            "n_frames": frames[-1].frame_id,
        },
        "conventions": (
            "Pitch meters, origin center. rel_x/rel_y are player position minus "
            "carrier position, rel_x signed so + means toward the goal this "
            "player's team attacks. depth_vs_line + means beyond the opponents' "
            "second-last outfield defender. Only players tracked >=80% of the "
            "clip appear; absence of a player means NOT OBSERVED, never absence "
            "from the pitch."
        ),
        "snapshots": snapshots,
        "kick_anchors": _kick_anchors(frames, fps),
    }


def write_relations_json(relations: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(relations, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return output_path


def main() -> None:
    import argparse


    ap = argparse.ArgumentParser(description="predictions.json -> relations.json")
    ap.add_argument("--predictions-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--snapshot-hz", type=float, default=2.0)
    args = ap.parse_args()

    frames = load_frames(args.predictions_json)
    relations = build_relations(frames, fps=args.fps, snapshot_hz=args.snapshot_hz)
    write_relations_json(relations, args.out)
    print(f"wrote {args.out}: {len(relations['snapshots'])} snapshots, "
          f"{len(relations['kick_anchors'])} kick anchors")


if __name__ == "__main__":
    main()
