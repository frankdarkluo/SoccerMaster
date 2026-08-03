from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Set

import numpy as np

from . import metrics
from .io_gamestate import Detection, load_detections
from .lines import detect_lines
from .pitch import canonicalize
from .possession import possession_team
from .windows import sliding_windows

MIN_COVERAGE = 8


def infer_attack_dirs(detections: List[Detection]) -> Dict[str, int]:
    """Compatibility for the legacy metrics analyzer."""
    return infer_attack_dirs_with_confidence(detections)[0]


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

VISIBLE_TOPOLOGY_VERSION = "visible-topology-v1"
VISIBLE_TOPOLOGY_PARAMS = {
    "window_s": 1.0,
    "stride_s": 0.5,
    "support_threshold": 0.6,
    "depth_gap_m": 4.5,
    "minimum_depth_spread_m": 6.0,
    "minimum_lateral_span_m": 4.0,
    "maximum_ball_distance_m": 5.0,
}


def infer_attack_dirs_with_confidence(
    detections: List[Detection],
) -> tuple[Dict[str, int], Dict[str, str]]:
    """Infer directions from visible goalkeepers; label fallbacks are not trusted."""
    goalkeeper_x = defaultdict(list)
    for det in detections:
        if det.role == "goalkeeper" and det.team in ("left", "right"):
            goalkeeper_x[det.team].append(det.x)

    dirs: Dict[str, int] = {}
    provenance: Dict[str, str] = {}
    for team in ("left", "right"):
        if goalkeeper_x[team]:
            dirs[team] = -1 if float(np.median(goalkeeper_x[team])) > 0 else 1
            provenance[team] = "goalkeeper"
    if len(dirs) == 1:
        known = next(iter(dirs))
        other = "right" if known == "left" else "left"
        dirs[other] = -dirs[known]
        provenance[other] = "opponent_goalkeeper"
    for team, fallback in (("left", 1), ("right", -1)):
        if team not in dirs:
            dirs[team] = fallback
            provenance[team] = "team_label_fallback"
    return dirs, provenance


def resolve_position_source(
    repo_root: Path,
    clip_uid: str,
    *,
    labels_path: Path | None = None,
    preprocessing_root: Path | None = None,
) -> tuple[Path, str]:
    """Resolve GameState GT first, then SoccerMaster predictions."""
    repo_root = Path(repo_root).resolve()
    clip = clip_uid.split(":", 1)[-1]
    if labels_path is not None:
        candidate = Path(labels_path)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve(), (
            "gt" if candidate.name == "Labels-GameState.json" else "soccer_master"
        )

    gt_root = repo_root / "codes/sn-gamestate/datasets/SoccerNetGS"
    matches = sorted(gt_root.glob(f"*/{clip}/Labels-GameState.json"))
    if len(matches) > 1:
        raise ValueError(f"Multiple GameState GT files found for {clip}: {matches}")
    if matches:
        return matches[0].resolve(), "gt"

    prediction = (
        Path(preprocessing_root)
        if preprocessing_root is not None
        else repo_root / "outputs/preprocessing"
    ) / clip / "predictions.json"
    if prediction.is_file():
        return prediction.resolve(), "soccer_master"
    raise FileNotFoundError(f"No GameState GT or SoccerMaster predictions for {clip}")


def _frame_groups(
    detections: List[Detection],
) -> tuple[dict[int, list[Detection]], dict[int, str | None]]:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        by_frame[det.frame].append(det)

    possession: dict[int, str | None] = {}
    for frame, items in by_frame.items():
        ball = next((det for det in items if det.role == "ball"), None)
        best = (float("inf"), None)
        if ball is not None:
            for det in items:
                if det.role not in ("player", "goalkeeper") or det.team not in ("left", "right"):
                    continue
                distance = float(np.hypot(det.x - ball.x, det.y - ball.y))
                if distance < best[0]:
                    best = distance, det.team
        possession[frame] = (
            best[1]
            if best[0] <= VISIBLE_TOPOLOGY_PARAMS["maximum_ball_distance_m"]
            else None
        )
    return by_frame, possession


def _depth_groups(players: list[Detection], attack_dir: int) -> list[list[Detection]]:
    ordered = sorted(players, key=lambda det: attack_dir * det.x)
    if len(ordered) < 2:
        return []
    depths = np.asarray([attack_dir * det.x for det in ordered], dtype=float)
    gaps = np.diff(depths)
    candidates = [
        index
        for index in np.argsort(gaps)[::-1]
        if gaps[index] >= VISIBLE_TOPOLOGY_PARAMS["depth_gap_m"]
    ][:2]
    if (
        not candidates
        and float(depths[-1] - depths[0]) >= VISIBLE_TOPOLOGY_PARAMS["minimum_depth_spread_m"]
    ):
        candidates = [int(np.argmax(gaps))]
    split_indices = sorted(index + 1 for index in candidates)
    return [list(group) for group in np.split(np.asarray(ordered, dtype=object), split_indices)]


def _frame_roles(
    items: list[Detection],
    attacking_team: str,
    dirs: Dict[str, int],
) -> dict[str, list[Detection]]:
    defending_team = "right" if attacking_team == "left" else "left"
    attacking = _depth_groups(
        [det for det in items if det.role == "player" and det.team == attacking_team],
        dirs[attacking_team],
    )
    defending = _depth_groups(
        [det for det in items if det.role == "player" and det.team == defending_team],
        dirs[defending_team],
    )
    roles: dict[str, list[Detection]] = {}
    if len(attacking) >= 2:
        roles["attacking_forward"] = attacking[-1]
        roles["attacking_midfield"] = attacking[-2]
    if len(defending) >= 2:
        roles["defending_defensive"] = defending[0]
    return roles


def _line_record(
    window_id: str,
    role: str,
    team: str,
    groups: list[list[Detection]],
    valid_frame_count: int,
    direction_source: str,
) -> dict:
    role_support = len(groups) / valid_frame_count if valid_frame_count else 0.0
    by_track: dict[int, list[Detection]] = defaultdict(list)
    for group in groups:
        for det in group:
            by_track[det.track_id].append(det)
    members = []
    for track_id, samples in by_track.items():
        support = len(samples) / valid_frame_count if valid_frame_count else 0.0
        if support < VISIBLE_TOPOLOGY_PARAMS["support_threshold"]:
            continue
        members.append({
            "track_id": track_id,
            "jersey_number": next(
                (sample.jersey_number for sample in samples if sample.jersey_number is not None),
                None,
            ),
            "x_m": round(float(np.median([sample.x for sample in samples])), 2),
            "y_m": round(float(np.median([sample.y for sample in samples])), 2),
            "support_ratio": round(support, 3),
        })
    members.sort(key=lambda member: member["y_m"])
    lateral_span = (
        max(member["y_m"] for member in members)
        - min(member["y_m"] for member in members)
        if members else 0.0
    )
    gaps = []
    if direction_source == "team_label_fallback":
        gaps.append("attack direction inferred only from team label")
    if role_support < VISIBLE_TOPOLOGY_PARAMS["support_threshold"]:
        gaps.append("role is not stable in 60% of valid frames")
    if len(members) < 2:
        gaps.append("fewer than two stable visible members")
    if lateral_span < VISIBLE_TOPOLOGY_PARAMS["minimum_lateral_span_m"]:
        gaps.append("stable members have insufficient lateral span")
    return {
        "line_id": f"{window_id}:{role}",
        "team": team,
        "role": role,
        "status": "renderable" if not gaps else "candidate",
        "support_ratio": round(role_support, 3),
        "direction_source": direction_source,
        "members": members,
        "lateral_span_m": round(lateral_span, 2),
        "evidence_gaps": gaps,
    }


def _unit_band(line: dict) -> dict | None:
    if line["status"] != "renderable":
        return None
    xs = [member["x_m"] for member in line["members"]]
    ys = [member["y_m"] for member in line["members"]]
    polygon = [
        [min(xs) - 1.5, min(ys) - 1.5],
        [max(xs) + 1.5, min(ys) - 1.5],
        [max(xs) + 1.5, max(ys) + 1.5],
        [min(xs) - 1.5, max(ys) + 1.5],
    ]
    return {
        "zone_id": f"{line['line_id']}:unit_band",
        "type": "unit_band",
        "team": line["team"],
        "supporting_line_ids": [line["line_id"]],
        "polygon_pitch": [[round(x, 2), round(y, 2)] for x, y in polygon],
        "status": "renderable",
    }


def _inter_line_space(lines: list[dict]) -> dict | None:
    by_role = {line["role"]: line for line in lines if line["status"] == "renderable"}
    forward = by_role.get("attacking_forward")
    midfield = by_role.get("attacking_midfield")
    if not forward or not midfield or forward["team"] != midfield["team"]:
        return None
    f_y = [member["y_m"] for member in forward["members"]]
    m_y = [member["y_m"] for member in midfield["members"]]
    low, high = max(min(f_y), min(m_y)), min(max(f_y), max(m_y))
    if high - low < VISIBLE_TOPOLOGY_PARAMS["minimum_lateral_span_m"]:
        return None
    f_x = float(np.mean([member["x_m"] for member in forward["members"]]))
    m_x = float(np.mean([member["x_m"] for member in midfield["members"]]))
    return {
        "zone_id": f"{forward['line_id']}:inter_line_space",
        "type": "inter_line_space",
        "team": forward["team"],
        "supporting_line_ids": [midfield["line_id"], forward["line_id"]],
        "polygon_pitch": [
            [round(m_x, 2), round(low, 2)],
            [round(f_x, 2), round(low, 2)],
            [round(f_x, 2), round(high, 2)],
            [round(m_x, 2), round(high, 2)],
        ],
        "status": "renderable",
    }


def analyze_visible_topology(
    detections: List[Detection],
    fps: float,
    *,
    clip_uid: str,
    position_source: str,
    labels_path: str,
    win_s: float = 1.0,
    stride_s: float = 0.5,
) -> dict:
    """Return conservative visible tactical lines and selective zones."""
    dirs, direction_sources = infer_attack_dirs_with_confidence(detections)
    by_frame, possession = _frame_groups(detections)
    if not by_frame:
        raise ValueError("No pitch detections available")
    max_t = max(by_frame) / float(fps)
    windows = []
    start = 0.0
    while start <= max_t:
        frames = [
            frame for frame in sorted(by_frame)
            if start <= frame / float(fps) < start + win_s
        ]
        votes = defaultdict(int)
        for frame in frames:
            if possession.get(frame):
                votes[possession[frame]] += 1
        attacking_team = max(votes, key=votes.get) if votes else None
        valid_frames = [
            frame for frame in frames if possession.get(frame) == attacking_team
        ] if attacking_team else []
        if valid_frames:
            defending_team = "right" if attacking_team == "left" else "left"
            role_groups: dict[str, list[list[Detection]]] = defaultdict(list)
            for frame in valid_frames:
                for role, group in _frame_roles(
                    by_frame[frame], attacking_team, dirs,
                ).items():
                    role_groups[role].append(group)
            window_id = f"w{int(round(start * 1000)):06d}"
            roles = (
                ("attacking_forward", attacking_team),
                ("attacking_midfield", attacking_team),
                ("defending_defensive", defending_team),
            )
            lines = [
                _line_record(
                    window_id,
                    role,
                    team,
                    role_groups.get(role, []),
                    len(valid_frames),
                    direction_sources[team],
                )
                for role, team in roles
            ]
            zones = [
                zone for line in lines
                if line["role"] == "defending_defensive" and (zone := _unit_band(line))
            ]
            inter_line = _inter_line_space(lines)
            if inter_line:
                zones.append(inter_line)
            windows.append({
                "window_id": window_id,
                "start_s": round(start, 3),
                "end_s": round(min(start + win_s, max_t), 3),
                "attacking_team": attacking_team,
                "valid_frame_count": len(valid_frames),
                "lines": lines,
                "zones": zones,
            })
        start += stride_s

    parameters = {
        **VISIBLE_TOPOLOGY_PARAMS,
        "window_s": win_s,
        "stride_s": stride_s,
    }
    return {
        "schema_version": VISIBLE_TOPOLOGY_VERSION,
        "clip_uid": clip_uid,
        "fps": fps,
        "position_source": position_source,
        "labels_path": labels_path,
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(
            json.dumps(parameters, sort_keys=True).encode()
        ).hexdigest(),
        "windows": windows,
    }


def analyze_clip(
    repo_root: Path,
    clip_uid: str,
    output_path: Path,
    *,
    fps: float = 25.0,
    force: bool = False,
    labels_path: Path | None = None,
    preprocessing_root: Path | None = None,
) -> dict:
    """Analyze one clip and atomically persist its reusable topology facts."""
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    repo_root = Path(repo_root).resolve()
    source_path, source = resolve_position_source(
        repo_root,
        clip_uid,
        labels_path=labels_path,
        preprocessing_root=preprocessing_root,
    )
    try:
        stored_path = str(source_path.relative_to(repo_root))
    except ValueError:
        stored_path = str(source_path)
    result = analyze_visible_topology(
        load_detections(str(source_path)),
        fps,
        clip_uid=clip_uid,
        position_source=source,
        labels_path=stored_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return result
