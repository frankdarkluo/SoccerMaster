import json
from pathlib import Path
import sys

from scripts.validate_soccernetgs import main, validate_dataset, validate_sequence


def _write_sequence(
    root: Path,
    *,
    split: str = "train",
    name: str = "SNGS-9001",
    sequence_id: str = "9001",
    version: str = "1.3",
    include_ball: bool = True,
) -> Path:
    sequence = root / split / name
    image_dir = sequence / "img1"
    image_dir.mkdir(parents=True)
    for file_name in ("000001.jpg", "000002.jpg"):
        (image_dir / file_name).write_bytes(b"image")

    annotations = [
        {
            "image_id": "1",
            "track_id": 10,
            "attributes": {"role": "player", "team": "left"},
            "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0},
        }
    ]
    if include_ball:
        annotations.append(
            {
                "image_id": "1",
                "track_id": 99,
                "attributes": {"role": "ball"},
                "bbox_pitch": {"x_bottom_middle": 1.5, "y_bottom_middle": 2.0},
            }
        )

    data = {
        "info": {
            "version": version,
            "game_id": "dev_game",
            "id": sequence_id,
            "name": name,
            "frame_rate": 25,
            "seq_length": 2,
            "im_dir": "img1",
            "im_ext": ".jpg",
        },
        "images": [
            {"image_id": "1", "file_name": "000001.jpg"},
            {"image_id": "2", "file_name": "000002.jpg"},
        ],
        "annotations": annotations,
        "categories": [],
    }
    (sequence / "Labels-GameState.json").write_text(json.dumps(data))
    return sequence


def test_valid_sequence_passes(tmp_path):
    report = validate_sequence(_write_sequence(tmp_path), split="train")

    assert report.ok
    assert report.errors == ()
    assert report.frame_count == 2
    assert report.pitch_annotation_count == 1
    assert report.ball_annotation_count == 1


def test_truncated_json_is_reported(tmp_path):
    sequence = _write_sequence(tmp_path)
    (sequence / "Labels-GameState.json").write_text('{"info":')

    report = validate_sequence(sequence, split="train")

    assert "invalid_json" in report.errors


def test_missing_frame_is_reported(tmp_path):
    sequence = _write_sequence(tmp_path)
    (sequence / "img1" / "000002.jpg").unlink()

    report = validate_sequence(sequence, split="train")

    assert "missing_frame" in report.errors
    assert "frame_count_mismatch" in report.errors


def test_annotation_image_reference_is_checked(tmp_path):
    sequence = _write_sequence(tmp_path)
    label_path = sequence / "Labels-GameState.json"
    data = json.loads(label_path.read_text())
    data["annotations"][0]["image_id"] = "missing"
    label_path.write_text(json.dumps(data))

    report = validate_sequence(sequence, split="train")

    assert "bad_annotation_image_ref" in report.errors


def test_dataset_detects_duplicate_sequence_id(tmp_path):
    _write_sequence(tmp_path, name="SNGS-9001", sequence_id="duplicate")
    _write_sequence(tmp_path, name="SNGS-9002", sequence_id="duplicate")

    reports = validate_dataset(tmp_path, splits=("train",))

    assert all("duplicate_sequence_id" in report.errors for report in reports)


def test_version_before_1_3_is_rejected(tmp_path):
    report = validate_sequence(
        _write_sequence(tmp_path, version="1.2"), split="train"
    )

    assert "dataset_version_before_1_3" in report.errors


def test_missing_ball_is_a_quality_warning_not_an_integrity_failure(tmp_path):
    report = validate_sequence(
        _write_sequence(tmp_path, include_ball=False), split="train"
    )

    assert report.ok
    assert "no_ball_coordinates" in report.warnings
    assert report.ball_annotation_count == 0


def test_repeated_expect_flags_are_accepted(tmp_path, monkeypatch):
    _write_sequence(tmp_path, split="train")
    _write_sequence(
        tmp_path,
        split="valid",
        name="SNGS-9002",
        sequence_id="9002",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_soccernetgs.py",
            "--root",
            str(tmp_path),
            "--splits",
            "train",
            "valid",
            "--expect",
            "train=1",
            "--expect",
            "valid=1",
        ],
    )

    assert main() == 0
