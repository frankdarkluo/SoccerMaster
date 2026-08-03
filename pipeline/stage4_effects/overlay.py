"""Shared frame overlay logic for light beams and tactical lines."""
from __future__ import annotations

from typing import Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.stage4_effects.beam_targets import resolve_beam_origin
from pipeline.stage4_effects.light_beam import compute_beam_alpha, draw_vertical_beam
from pipeline.stage4_effects.projection import get_h_inv_for_frame, pitch_to_image
from pipeline.stage4_effects.tactical_lines import (
    draw_confirmed_topology,
    select_topology_window,
)


SPOTLIGHT_EVENT_CODES = {"football.pass", "football.assist"}

TOPOLOGY_STYLE = {
    "attacker_color_bgr": (70, 190, 70),
    "defender_color_bgr": (40, 40, 220),
    "inter_line_color_bgr": (220, 160, 60),
    "white_color_bgr": (245, 245, 245),
    "line_alpha": 0.55,
    "zone_alpha": 0.18,
    "stripe_spacing": 16,
}

TOPO_LEAD_S = 2.0
TOPO_TRAIL_S = 1.0
TOPO_FADE_S = 0.5


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
    topology: Optional[dict] = None,
) -> None:
    if not config.topology_lines_enabled or not topology:
        return
    alpha = _topo_alpha_for_frame(
        frame_num, events, config.fps, config.event_importance_threshold
    )
    if alpha < 0.01:
        return
    window = select_topology_window(topology, frame_num / float(config.fps))
    if not window:
        return
    h_inv = get_h_inv_for_frame(homo_frames, image_id) if homo_frames else None
    draw_confirmed_topology(
        frame,
        annotations,
        window,
        attacking_team=window["attacking_team"],
        homography_valid=h_inv is not None,
        pitch_to_image_fn=(
            (lambda point: pitch_to_image(point, h_inv))
            if h_inv is not None else None
        ),
        style=TOPOLOGY_STYLE,
        alpha=alpha,
    )


def apply_event_beams(
    frame,
    frame_num: int,
    events: list,
    frame_to_image_id: dict,
    anns_by_image: dict,
    config: PipelineConfig,
) -> None:
    beam_half_frames = max(1, int(config.beam_duration_s * config.fps))
    image_id = frame_to_image_id.get(frame_num)
    if image_id is None:
        return

    annotations = anns_by_image.get(image_id, [])

    for event in events:
        if (
            event.get("event_code") not in SPOTLIGHT_EVENT_CODES
            or event.get("importance", 0) < config.event_importance_threshold
        ):
            continue
        event_frame = int(event.get("frame_id", round(event.get("timestamp_s", 0) * config.fps)))
        offset = frame_num - event_frame
        if abs(offset) > beam_half_frames:
            continue

        alpha = compute_beam_alpha(offset, beam_half_frames, config.beam_alpha_max)
        if alpha <= 0.01:
            continue

        origin = resolve_beam_origin(event, annotations)
        if origin is not None:
            draw_vertical_beam(frame, origin, alpha=alpha)


def apply_frame_overlays(
    frame,
    frame_num: int,
    events: list,
    frame_to_image_id: Dict[int, str],
    anns_by_image: Dict[str, List[dict]],
    homo_frames: Optional[dict],
    config: PipelineConfig,
    topology: Optional[dict] = None,
) -> None:
    image_id = frame_to_image_id.get(frame_num)
    if image_id is None:
        return

    annotations = anns_by_image.get(image_id, [])
    apply_tactical_lines(
        frame, frame_num, events, annotations, homo_frames, image_id, config, topology
    )
    apply_event_beams(
        frame,
        frame_num,
        events,
        frame_to_image_id,
        anns_by_image,
        config,
    )
