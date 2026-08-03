from pathlib import Path

from pipeline.topology.analysis import (
    analyze_visible_topology,
    resolve_position_source,
)
from pipeline.topology.io_gamestate import Detection


def _det(frame, track_id, team, role, x, y, jersey=None):
    return Detection(frame, track_id, team, role, jersey, x, y)


def _visible_shape(drop_forward_member_after=None):
    detections = []
    for frame in range(25):
        detections.extend([
            _det(frame, 90, "left", "goalkeeper", -50, 0),
            _det(frame, 91, "right", "goalkeeper", 50, 0),
            _det(frame, 1, "left", "player", -5, -10, 6),
            _det(frame, 2, "left", "player", -5, 10, 8),
            _det(frame, 3, "left", "player", 10, -10, 9),
            _det(frame, 5, "right", "player", 20, -10, 4),
            _det(frame, 6, "right", "player", 20, 10, 5),
            _det(frame, 7, "right", "player", 5, -10, 7),
            _det(frame, 8, "right", "player", 5, 10, 10),
            _det(frame, 99, None, "ball", -5, -10),
        ])
        if drop_forward_member_after is None or frame < drop_forward_member_after:
            detections.append(_det(frame, 4, "left", "player", 10, 10, 11))
    return detections


def test_position_source_prefers_gamestate_gt(tmp_path: Path):
    gt = tmp_path / "codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-X/Labels-GameState.json"
    prediction = tmp_path / "outputs/preprocessing/SNGS-X/predictions.json"
    gt.parent.mkdir(parents=True)
    prediction.parent.mkdir(parents=True)
    gt.write_text("{}")
    prediction.write_text("{}")

    path, source = resolve_position_source(tmp_path, "SNGS-X")

    assert path == gt.resolve()
    assert source == "gt"


def test_visible_topology_returns_only_three_stable_target_lines():
    result = analyze_visible_topology(
        _visible_shape(),
        25,
        clip_uid="SNGS-X",
        position_source="gt",
        labels_path="Labels-GameState.json",
    )
    first = result["windows"][0]

    assert [(line["role"], line["status"]) for line in first["lines"]] == [
        ("attacking_forward", "renderable"),
        ("attacking_midfield", "renderable"),
        ("defending_defensive", "renderable"),
    ]
    assert {member["track_id"] for member in first["lines"][0]["members"]} == {3, 4}
    assert {zone["type"] for zone in first["zones"]} == {"unit_band", "inter_line_space"}


def test_member_below_sixty_percent_keeps_line_candidate_only():
    result = analyze_visible_topology(
        _visible_shape(drop_forward_member_after=14),
        25,
        clip_uid="SNGS-X",
        position_source="gt",
        labels_path="Labels-GameState.json",
    )
    forward = result["windows"][0]["lines"][0]

    assert forward["status"] == "candidate"
    assert [member["track_id"] for member in forward["members"]] == [3]
    assert all(
        forward["line_id"] not in zone["supporting_line_ids"]
        for zone in result["windows"][0]["zones"]
    )


def test_position_source_falls_back_to_soccer_master(tmp_path: Path):
    prediction = tmp_path / "outputs/preprocessing/SNGS-X/predictions.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}")

    path, source = resolve_position_source(tmp_path, "SNGS-X")

    assert path == prediction.resolve()
    assert source == "soccer_master"


def test_untrusted_attack_direction_never_renders_lines():
    no_goalkeepers = [
        detection for detection in _visible_shape()
        if detection.role != "goalkeeper"
    ]
    result = analyze_visible_topology(
        no_goalkeepers,
        25,
        clip_uid="SNGS-X",
        position_source="soccer_master",
        labels_path="predictions.json",
    )

    assert {line["status"] for line in result["windows"][0]["lines"]} == {"candidate"}
