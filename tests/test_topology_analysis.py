import json

from pipeline.stage3_effects.topology_analysis import run_topology_analysis


def _write_predictions(tmp_path, annotations, fps=25.0):
    path = tmp_path / "predictions.json"
    images = [
        {"image_id": str(3000000 + i), "file_name": f"{i:06d}.jpg"}
        for i in range(1, 200)
    ]
    path.write_text(
        json.dumps({
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": [],
        }),
        encoding="utf-8",
    )
    return path


def _player(image_id, track_id, team, role, x, y, jersey="1"):
    return {
        "id": str(track_id),
        "image_id": str(image_id),
        "track_id": track_id,
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
    }


def test_writes_topo_json_from_predictions(tmp_path):
    annotations = []
    for i in range(1, 100):
        image_id = 3000000 + i
        annotations.append(_player(image_id, 1, "left", "goalkeeper", -50.0, 0.0))
        annotations.append(_player(image_id, 2, "left", "player", -20.0, 5.0))
        annotations.append(_player(image_id, 3, "right", "goalkeeper", 50.0, 0.0))
        annotations.append(_player(image_id, 4, "right", "player", 20.0, -5.0))

    predictions_path = _write_predictions(tmp_path, annotations)
    output_path = tmp_path / "topo.json"

    result_path = run_topology_analysis(predictions_path, output_path, fps=25.0)

    assert result_path == output_path
    assert output_path.exists()
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert len(records) > 0
    assert {"team", "t_start", "t_end", "block_height_m"}.issubset(records[0].keys())


def test_missing_predictions_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    output_path = tmp_path / "topo.json"
    try:
        run_topology_analysis(missing, output_path, fps=25.0)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
