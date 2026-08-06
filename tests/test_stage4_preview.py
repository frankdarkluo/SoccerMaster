from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.stage4_effects.light_beam import draw_vertical_beam
from pipeline.stage4_effects.overlay import apply_event_beams
from pipeline.stage4_effects.render import load_effect_events
from pipeline.stage4_effects.preview import (
    _infer_attacking_team,
    draw_preview_effects,
    render_preview_video,
)


def _player(track_id: int, team: str, x: int, y: int) -> dict:
    return {
        "id": str(track_id),
        "track_id": track_id,
        "bbox_image": {
            "x": x - 4,
            "y": y - 16,
            "w": 8,
            "h": 16,
            "x_center": x,
            "y_center": y - 8,
        },
        "bbox_pitch": {"x_bottom_middle": x / 4, "y_bottom_middle": y / 4},
        "attributes": {"role": "player", "team": team},
    }


def _annotations(image_id: str) -> list[dict]:
    players = [
        _player(1, "left", 35, 70),
        _player(2, "left", 55, 65),
        _player(3, "left", 70, 75),
        _player(4, "right", 105, 55),
        _player(5, "right", 125, 70),
        _player(6, "right", 145, 60),
    ]
    ball = {
        "id": "99",
        "track_id": 99,
        "bbox_image": {
            "x": 35,
            "y": 67,
            "w": 4,
            "h": 4,
            "x_center": 37,
            "y_center": 69,
        },
        "bbox_pitch": {"x_bottom_middle": 9.25, "y_bottom_middle": 17.25},
        "attributes": {"role": "ball"},
    }
    return [dict(item, image_id=image_id) for item in [*players, ball]]

def _topology_window() -> dict:
    return {
        "window_id": "w000000",
        "attacking_team": "left",
        "lines": [
            {
                "line_id": "attack",
                "team": "left",
                "role": "attacking_forward",
                "status": "renderable",
                "members": [
                    {"track_id": 1, "y_m": 17.5},
                    {"track_id": 2, "y_m": 16.25},
                ],
            },
            {
                "line_id": "defense",
                "team": "right",
                "role": "defending_defensive",
                "status": "renderable",
                "members": [
                    {"track_id": 4, "y_m": 13.75},
                    {"track_id": 5, "y_m": 17.5},
                ],
            },
        ],
        "zones": [{
            "zone_id": "defense:unit_band",
            "type": "unit_band",
            "team": "right",
            "supporting_line_ids": ["defense"],
            "status": "renderable",
            "polygon_pitch": [[25, 12], [35, 12], [35, 20], [25, 20]],
        }],
    }



def test_preview_effects_deduplicate_tracks_and_gate_pitch_effects():
    frame = np.full((100, 180, 3), 10, dtype=np.uint8)
    annotations = _annotations("1")
    topology = {
        "topology_window": _topology_window(),
        "pitch_to_image_fn": lambda point: tuple(round(value * 4) for value in point),
    }

    single = draw_preview_effects(
        frame.copy(), annotations, "left", homography_valid=True, **topology
    )
    duplicate = draw_preview_effects(
        frame.copy(),
        [*annotations, dict(annotations[0], id="duplicate")],
        "left",
        homography_valid=True,
        **topology,
    )
    no_homography = draw_preview_effects(
        frame.copy(), annotations, "left", homography_valid=False, **topology
    )

    assert np.array_equal(single, duplicate)
    assert not np.array_equal(single, no_homography)
    assert np.count_nonzero(no_homography != frame) > 0


def test_vertical_beam_has_warm_shaft_and_ground_spotlight():
    frame = np.full((100, 180, 3), 10, dtype=np.uint8)
    draw_vertical_beam(frame, (90, 80), alpha=0.4)

    assert frame[40, 90, 1] > frame[40, 90, 0]
    assert frame[80, 90].mean() > frame[40, 90].mean()
    assert frame[80, 55].mean() > 10


def test_preview_beam_can_lock_to_one_track():
    frame = np.full((100, 180, 3), 10, dtype=np.uint8)
    rendered = draw_preview_effects(
        frame,
        _annotations("1"),
        "right",
        homography_valid=False,
        focus_track_id=3,
    )

    assert rendered[8, 70].mean() > rendered[8, 35].mean() + 2


def test_event_beam_only_highlights_decisive_passer():
    frame = np.full((100, 180, 3), 10, dtype=np.uint8)
    annotations = _annotations("1")
    config = PipelineConfig(fps=25, beam_duration_s=1.5)

    goal = [{
        "event_code": "football.goal", "importance": 1.0, "frame_id": 25,
        "player_jersey": "1", "player_team": "left",
    }]
    apply_event_beams(frame, 25, goal, {25: "1"}, {"1": annotations}, config)
    assert np.all(frame == 10)

    key_pass = [{
        "event_code": "football.pass", "importance": 0.85, "frame_id": 25,
        "player_jersey": "1", "player_team": "left",
    }]
    apply_event_beams(frame, 25, key_pass, {25: "1"}, {"1": annotations}, config)
    assert np.any(frame != 10)


def test_effect_events_are_optional():
    assert load_effect_events(None) == []


def test_effect_events_keep_key_pass_importance_and_timestamp(tmp_path: Path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"events": [{
        "event_code": "football.pass",
        "importance": 0.85,
        "timestamp_s": 5.0,
    }]}), encoding="utf-8")

    event = load_effect_events(path)[0]

    assert event["importance"] == 0.85
    assert event["timestamp_s"] == 5.0


def test_preview_does_not_infer_possession_without_ball():
    players = [item for item in _annotations("1") if item["attributes"]["role"] != "ball"]

    assert _infer_attacking_team([players]) is None


def test_preview_rejects_far_ball_for_possession():
    far = _annotations("1")
    ball = next(item for item in far if item["attributes"]["role"] == "ball")
    ball["bbox_pitch"] = {"x_bottom_middle": 50, "y_bottom_middle": 50}
    assert _infer_attacking_team([far] * 5) is None


def test_render_preview_keeps_full_sequence_and_window(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    images = []
    annotations = []
    homography_frames = {}
    for frame_number in range(1, 5):
        cv2.imwrite(
            str(frames_dir / f"{frame_number:06d}.jpg"),
            np.full((100, 180, 3), 10, dtype=np.uint8),
        )
        image_id = str(100 + frame_number)
        images.append(
            {
                "image_id": image_id,
                "file_name": f"{frame_number:06d}.jpg",
                "width": 180,
                "height": 100,
            }
        )
        annotations.extend(_annotations(image_id))
        homography_frames[image_id] = {
            "valid": True,
            "H_inv": np.eye(3).tolist(),
        }

    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )
    homography = tmp_path / "homography.json"
    homography.write_text(
        json.dumps({"frames": homography_frames}), encoding="utf-8"
    )
    output = tmp_path / "preview.mp4"

    render_preview_video(
        frames_dir,
        predictions,
        homography,
        output,
        window="0.5:1.0",
        fps=2,
        reencode_h264=False,
    )

    capture = cv2.VideoCapture(str(output))
    rendered = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        rendered.append(image)
    capture.release()

    assert len(rendered) == 4
    changes = [
        np.mean(np.abs(image.astype(float) - 10.0))
        for image in rendered
    ]
    assert max(changes[1:3]) > max(changes[0], changes[3]) + 1.0
