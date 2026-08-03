"""Rule-based tactical preview rendering."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pipeline.stage4_effects.beam_targets import (
    _nearest_field_player_to_ball,
    foot_point_from_bbox,
    load_predictions_index,
)
from pipeline.stage4_effects.light_beam import (
    SPOTLIGHT_COLOR_BGR,
    draw_vertical_beam,
)
from pipeline.stage4_effects.projection import get_h_inv_for_frame, load_homography, pitch_to_image
from pipeline.stage4_effects.tactical_lines import (
    draw_confirmed_topology,
    select_topology_window,
)
from pipeline.utils.video import reencode_to_h264

DEFAULT_STYLE = {
    "attacker_color_bgr": (70, 190, 70),
    "defender_color_bgr": (40, 40, 220),
    "beam_color_bgr": SPOTLIGHT_COLOR_BGR,
    "white_color_bgr": (245, 245, 245),
    "inter_line_color_bgr": (220, 160, 60),
    "ring_radius": 18,
    "ring_thickness": 3,
    "line_alpha": 0.55,
    "zone_alpha": 0.18,
    "stripe_spacing": 16,
    "beam_alpha": 0.26,
}
_COLOR_KEYS = {
    "attacker_color_bgr",
    "defender_color_bgr",
    "beam_color_bgr",
    "white_color_bgr",
    "inter_line_color_bgr",
}


def load_preview_style(path: Path | None) -> dict:
    style = dict(DEFAULT_STYLE)
    if path is None:
        return style
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    source = payload.get("stage4", payload) if isinstance(payload, dict) else {}
    for key in style:
        value = source.get(key)
        if key in _COLOR_KEYS:
            if isinstance(value, list) and len(value) == 3:
                style[key] = tuple(int(max(0, min(255, item))) for item in value)
        elif isinstance(value, (int, float)):
            style[key] = value
    return style


def parse_window(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid window {value!r}; expected START:END") from exc
    if start < 0 or end <= start:
        raise ValueError(f"Invalid window {value!r}; require 0 <= START < END")
    return start, end


def _annotation_score(annotation: dict) -> tuple[int, float]:
    bbox = annotation.get("bbox_image") or {}
    area = float(bbox.get("w", 0) or 0) * float(bbox.get("h", 0) or 0)
    pitch = annotation.get("bbox_pitch") or {}
    has_pitch = (
        pitch.get("x_bottom_middle") is not None
        and pitch.get("y_bottom_middle") is not None
    )
    return int(has_pitch), area


def _deduplicate(annotations: Iterable[dict]) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}
    for index, annotation in enumerate(annotations):
        track_id = annotation.get("track_id")
        key = (
            ("track", str(track_id))
            if track_id is not None
            else ("annotation", str(annotation.get("id", index)))
        )
        previous = unique.get(key)
        if previous is None or _annotation_score(annotation) > _annotation_score(previous):
            unique[key] = annotation
    return list(unique.values())


def _role(annotation: dict) -> str | None:
    return (annotation.get("attributes") or {}).get("role")


def _team(annotation: dict) -> str | None:
    return (annotation.get("attributes") or {}).get("team")




def _nearest_player(
    annotations: list[dict],
    *,
    team: str | None = None,
) -> dict | None:
    ball = next((item for item in annotations if _role(item) == "ball"), None)
    players = [
        item
        for item in annotations
        if _role(item) in ("player", "goalkeeper")
        and (team is None or _team(item) == team)
    ]
    if not players or ball is None:
        return None
    return _nearest_field_player_to_ball(
        players,
        ball,
        max_dist_m=5.0,
        prefer_outfield=False,
    )


def _infer_attacking_team(frames: list[list[dict]]) -> str | None:
    votes = Counter()
    for annotations in frames:
        nearest = _nearest_player(_deduplicate(annotations))
        if nearest is not None and _team(nearest):
            votes[_team(nearest)] += 1
    return votes.most_common(1)[0][0] if votes else None


def _draw_segmented_ring(
    frame: np.ndarray,
    center: tuple[int, int],
    primary: tuple[int, int, int],
    white: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    for start in range(0, 360, 90):
        cv2.ellipse(
            frame,
            center,
            (radius, max(5, radius // 3)),
            0,
            start + 8,
            start + 42,
            primary,
            thickness,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            frame,
            center,
            (radius, max(5, radius // 3)),
            0,
            start + 48,
            start + 82,
            white,
            thickness,
            cv2.LINE_AA,
        )


def draw_preview_effects(
    frame: np.ndarray,
    annotations: list[dict],
    attacking_team: str,
    *,
    homography_valid: bool,
    focus_track_id: int | None = None,
    style: dict | None = None,
    topology_window: dict | None = None,
    pitch_to_image_fn=None,
) -> np.ndarray:
    """Draw one preview frame; pitch-dependent effects require homography."""
    style = {**DEFAULT_STYLE, **(style or {})}
    annotations = _deduplicate(annotations)
    players = [
        item for item in annotations if _role(item) in ("player", "goalkeeper")
    ]
    focal = (
        next((item for item in players if item.get("track_id") == focus_track_id), None)
        if focus_track_id is not None
        else _nearest_player(annotations, team=attacking_team)
    )

    for player in players:
        foot = foot_point_from_bbox(player.get("bbox_image") or {})
        if foot is None:
            continue
        color = (
            style["attacker_color_bgr"]
            if _team(player) == attacking_team
            else style["defender_color_bgr"]
        )
        _draw_segmented_ring(
            frame,
            foot,
            color,
            style["white_color_bgr"],
            int(style["ring_radius"]),
            int(style["ring_thickness"]),
        )

    if focal is not None:
        foot = foot_point_from_bbox(focal.get("bbox_image") or {})
        if foot is not None:
            draw_vertical_beam(
                frame,
                foot,
                style["beam_color_bgr"],
                float(style["beam_alpha"]),
            )

    draw_confirmed_topology(
        frame,
        annotations,
        topology_window,
        attacking_team=attacking_team,
        homography_valid=homography_valid,
        pitch_to_image_fn=pitch_to_image_fn,
        style=style,
    )

    return frame


def render_preview_video(
    frames_dir: Path,
    predictions_json_path: Path,
    homography_json_path: Path,
    output_path: Path,
    *,
    style_profile_path: Path | None = None,
    topology_json_path: Path | None = None,
    repo_root: Path | None = None,
    focus_track_id: int | None = None,
    window: str = "21:25",
    fps: float = 25,
    reencode_h264: bool = True,
) -> Path:
    """Render a full silent video, applying tactical graphics only in ``window``."""
    start_s, end_s = parse_window(window)
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    topology = None
    annotations_path = Path(predictions_json_path)
    if topology_json_path is not None:
        topology = json.loads(Path(topology_json_path).read_text(encoding="utf-8"))
        labels_path = Path(topology["labels_path"])
        if not labels_path.is_absolute():
            labels_path = (
                Path(repo_root) if repo_root is not None
                else Path(__file__).resolve().parents[2]
            ) / labels_path
        annotations_path = labels_path
    frame_to_image_id, anns_by_image = load_predictions_index(annotations_path)
    homo_frames = load_homography(homography_json_path)
    style = load_preview_style(style_profile_path)
    active = [
        anns_by_image.get(frame_to_image_id.get(int(path.stem), ""), [])
        for index, path in enumerate(frame_paths)
        if start_s <= index / float(fps) <= end_s
    ]
    attacking_team = _infer_attacking_team(active)

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise FileNotFoundError(f"Could not read frame: {frame_paths[0]}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temporary}")

    try:
        for index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {frame_path}")
            timestamp_s = index / float(fps)
            image_id = frame_to_image_id.get(int(frame_path.stem))
            if (
                start_s <= timestamp_s <= end_s
                and image_id is not None
                and attacking_team is not None
            ):
                h_inv = get_h_inv_for_frame(homo_frames, image_id)
                draw_preview_effects(
                    frame,
                    anns_by_image.get(image_id, []),
                    attacking_team,
                    homography_valid=h_inv is not None,
                    focus_track_id=focus_track_id,
                    style=style,
                    topology_window=select_topology_window(topology, timestamp_s),
                    pitch_to_image_fn=(
                        (lambda point: pitch_to_image(point, h_inv))
                        if h_inv is not None else None
                    ),
                )
            writer.write(frame)
        writer.release()
        if reencode_h264:
            reencode_to_h264(temporary, output_path)
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, output_path)
    finally:
        writer.release()
        temporary.unlink(missing_ok=True)
    return output_path
