from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from . import metrics
from .io_gamestate import Detection
from .lines import detect_lines
from .pitch import canonicalize
from .possession import possession_team
from .windows import sliding_windows

MIN_COVERAGE = 8


def infer_attack_dirs(detections: List[Detection]) -> Dict[str, int]:
    """Infer attack direction from goalkeeper side, falling back to side labels."""
    goalkeeper_x = defaultdict(list)
    for det in detections:
        if det.role == "goalkeeper" and det.team in ("left", "right"):
            goalkeeper_x[det.team].append(det.x)

    dirs = {}
    for team in ("left", "right"):
        if goalkeeper_x[team]:
            dirs[team] = -1 if float(np.mean(goalkeeper_x[team])) > 0 else 1
        else:
            dirs[team] = 1 if team == "left" else -1
    return dirs


def analyze(
    detections: List[Detection],
    fps: float,
    win_s: float = 3.0,
    stride_s: float = 1.0,
    min_coverage: int = MIN_COVERAGE,
    gap_delta: float = 7.0,
    in_play_frames: Optional[Set[int]] = None,
) -> List[dict]:
    dirs = infer_attack_dirs(detections)
    windows = sliding_windows(detections, fps, win_s, stride_s, in_play_frames)

    records: List[dict] = []
    for window in windows:
        poss = possession_team(window)
        for team, info in window["teams"].items():
            raw_points = info["players"]
            if len(raw_points) == 0:
                continue

            pts = canonicalize(raw_points, dirs[team])
            coverage_n = len(pts)
            possession_tag = "in" if poss == team else ("unknown" if poss is None else "out")
            record = {
                "t_start": window["t_start"],
                "t_end": window["t_end"],
                "team": team,
                "possession_tag": possession_tag,
                "block_height_m": round(metrics.block_height(pts), 2),
                "block_depth_m": round(metrics.block_depth(pts), 2),
                "block_width_m": round(metrics.block_width(pts), 2),
                "hull_area_m2": round(metrics.hull_area(pts), 1),
                "centroid_y_m": round(metrics.centroid_y(pts), 2),
                "lane_counts": metrics.lane_counts(pts),
                "band_counts": metrics.band_counts(pts),
                "side_overload": round(metrics.side_overload(pts), 3),
                "coverage_n": coverage_n,
            }

            if coverage_n >= min_coverage:
                line_count, gaps = detect_lines(pts[:, 0], gap_delta=gap_delta)
                record["line_count"] = line_count
                record["inter_line_gaps_m"] = gaps
                record["low_confidence"] = line_count < 2
            else:
                record["line_count"] = None
                record["inter_line_gaps_m"] = []
                record["low_confidence"] = True

            records.append(record)

    return records
