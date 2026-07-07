"""Shared frame overlay logic for light beams and tactical lines."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pipeline.config import PipelineConfig
from pipeline.stage3_effects.beam_targets import foot_point_from_bbox, resolve_beam_points
from pipeline.stage3_effects.light_beam import compute_beam_alpha, draw_cone_beam, draw_foot_marker
from pipeline.stage3_effects.projection import get_h_inv_for_frame, pitch_to_image
from pipeline.stage3_effects.tactical_lines import (
    delaunay_adjacency,
    draw_formation_lines,
    draw_player_marker,
)


EVENT_COLORS = {
    "football.shoot": (100, 255, 200),
    "football.goal": (0, 215, 255),
    "football.pass": (150, 255, 150),
    "football.clearance": (255, 150, 50),
    "football.save": (0, 255, 100),
    "football.assist": (150, 255, 150),
}

TEAM_LINE_COLORS = {
    "left": (200, 170, 100),
    "right": (100, 170, 220),
}

TOPO_LEAD_S = 2.0
TOPO_TRAIL_S = 1.0
TOPO_FADE_S = 0.5


def _team_player_positions(
    annotations: List[dict],
    team: str,
    pitch_to_image_fn=None,
    include_goalkeeper: bool = True,
) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    for ann in annotations:
        attrs = ann.get("attributes") or {}
        if attrs.get("team") != team:
            continue
        role = attrs.get("role")
        if role not in ("player", "goalkeeper"):
            continue
        if not include_goalkeeper and role == "goalkeeper":
            continue

        point = foot_point_from_bbox(ann.get("bbox_image") or {})
        if point is None and pitch_to_image_fn is not None:
            bp = ann.get("bbox_pitch") or {}
            px = bp.get("x_bottom_middle")
            py = bp.get("y_bottom_middle")
            if px is not None and py is not None:
                point = pitch_to_image_fn((float(px), float(py)))
        if point is not None:
            positions.append(point)
    return positions


def _topo_alpha_for_frame(
    frame_num: int,
    events: list,
    fps: int,
    importance_threshold: float,
    alpha_max: float = 0.35,
) -> float:
    """Return tactical-line alpha: non-zero only near events, with fade."""
    best = 0.0
    for event in events:
        if event.get("importance", 0) < importance_threshold:
            continue
        event_frame = int(
            event.get("frame_id", round(event.get("timestamp_s", 0) * fps))
        )
        offset_s = (frame_num - event_frame) / fps
        if offset_s < -TOPO_LEAD_S or offset_s > TOPO_TRAIL_S:
            continue
        if offset_s < -TOPO_LEAD_S + TOPO_FADE_S:
            t = (offset_s + TOPO_LEAD_S) / TOPO_FADE_S
        elif offset_s > TOPO_TRAIL_S - TOPO_FADE_S:
            t = (TOPO_TRAIL_S - offset_s) / TOPO_FADE_S
        else:
            t = 1.0
        best = max(best, t * alpha_max)
    return best


def apply_tactical_lines(
    frame,
    frame_num: int,
    events: list,
    annotations: List[dict],
    homo_frames: Optional[dict],
    image_id: str,
    config: PipelineConfig,
) -> None:
    if not config.topology_lines_enabled:
        return

    alpha = _topo_alpha_for_frame(
        frame_num, events, config.fps, config.event_importance_threshold
    )
    if alpha < 0.01:
        return

    h_inv = get_h_inv_for_frame(homo_frames, image_id) if homo_frames else None

    def pitch_to_image_fn(point):
        if h_inv is None:
            return None
        return pitch_to_image(point, h_inv)

    for team, color in TEAM_LINE_COLORS.items():
        field_positions = _team_player_positions(
            annotations, team, pitch_to_image_fn, include_goalkeeper=False
        )
        gk_positions = _team_player_positions(
            annotations, team, pitch_to_image_fn, include_goalkeeper=True
        )
        all_markers = gk_positions

        if len(field_positions) >= 2:
            adjacency = delaunay_adjacency(field_positions)
            draw_formation_lines(
                frame, field_positions, adjacency, color=color, alpha=alpha
            )
        for pos in all_markers:
            draw_player_marker(frame, pos, color=color, alpha=alpha)


def apply_event_beams(
    frame,
    frame_num: int,
    events: list,
    frame_to_image_id: dict,
    anns_by_image: dict,
    homo_frames: Optional[dict],
    config: PipelineConfig,
) -> None:
    beam_half_frames = max(1, int(config.beam_duration_s * config.fps))
    image_id = frame_to_image_id.get(frame_num)
    if image_id is None:
        return

    annotations = anns_by_image.get(image_id, [])
    h_inv = get_h_inv_for_frame(homo_frames, image_id) if homo_frames else None

    def pitch_to_image_fn(point):
        if h_inv is None:
            return None
        return pitch_to_image(point, h_inv)

    for event in events:
        if event.get("importance", 0) < config.event_importance_threshold:
            continue
        event_frame = int(event.get("frame_id", round(event.get("timestamp_s", 0) * config.fps)))
        offset = frame_num - event_frame
        if abs(offset) > beam_half_frames:
            continue

        alpha = compute_beam_alpha(offset, beam_half_frames, config.beam_alpha_max)
        if alpha <= 0.01:
            continue

        points = resolve_beam_points(event, annotations, pitch_to_image_fn)
        if points is None:
            continue

        origin, target = points
        color = EVENT_COLORS.get(event.get("event_code"), (255, 255, 255))
        draw_foot_marker(frame, origin, color, alpha=alpha)
        draw_cone_beam(frame, origin, target, color, alpha=alpha)


def apply_frame_overlays(
    frame,
    frame_num: int,
    events: list,
    frame_to_image_id: Dict[int, str],
    anns_by_image: Dict[str, List[dict]],
    homo_frames: Optional[dict],
    config: PipelineConfig,
) -> None:
    image_id = frame_to_image_id.get(frame_num)
    if image_id is None:
        return

    annotations = anns_by_image.get(image_id, [])
    apply_tactical_lines(
        frame, frame_num, events, annotations, homo_frames, image_id, config
    )
    apply_event_beams(
        frame,
        frame_num,
        events,
        frame_to_image_id,
        anns_by_image,
        homo_frames,
        config,
    )
