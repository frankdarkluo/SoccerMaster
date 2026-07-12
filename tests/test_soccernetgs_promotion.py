import json
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.promote_soccernetgs_splits as promotion
from scripts.validate_soccernetgs import validate_dataset


EXPECTED = {"train": 1, "valid": 1}


def _write_sequence(root: Path, split: str, sequence_id: str) -> None:
    name = f"SNGS-{sequence_id}"
    sequence = root / split / name
    image_dir = sequence / "img1"
    image_dir.mkdir(parents=True)
    (image_dir / "000001.jpg").write_bytes(b"image")
    label = {
        "info": {
            "version": "1.3",
            "game_id": f"game-{split}",
            "id": sequence_id,
            "name": name,
            "frame_rate": 25,
            "seq_length": 1,
            "im_dir": "img1",
            "im_ext": ".jpg",
        },
        "images": [{"image_id": "1", "file_name": "000001.jpg"}],
        "annotations": [
            {
                "image_id": "1",
                "track_id": 10,
                "attributes": {"role": "player", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0},
            },
            {
                "image_id": "1",
                "track_id": 99,
                "attributes": {"role": "ball"},
                "bbox_pitch": {"x_bottom_middle": 1.5, "y_bottom_middle": 2.0},
            },
        ],
        "categories": [],
    }
    (sequence / "Labels-GameState.json").write_text(json.dumps(label))


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    quarantine = canonical / "_quarantine" / "repair-v1.3-prepromotion"
    for split, old_id, new_id in (
        ("train", "1001", "2001"),
        ("valid", "1002", "2002"),
    ):
        _write_sequence(canonical, split, old_id)
        _write_sequence(staging, split, new_id)
    (canonical / "test").mkdir()
    (canonical / "test" / "untouched.bin").write_bytes(b"held-out")
    return canonical, staging, quarantine


def _promote(canonical: Path, staging: Path, quarantine: Path) -> dict:
    return promotion.promote_train_valid(
        canonical_root=canonical,
        staging_root=staging,
        quarantine_root=quarantine,
        expected_counts=EXPECTED,
        confirmation=promotion.CONFIRMATION,
    )


def _assert_rolled_back(canonical: Path, staging: Path) -> None:
    assert (canonical / "train" / "SNGS-1001").is_dir()
    assert (canonical / "valid" / "SNGS-1002").is_dir()
    assert (staging / "train" / "SNGS-2001").is_dir()
    assert (staging / "valid" / "SNGS-2002").is_dir()


def test_preflight_rejects_missing_or_invalid_staging_split(tmp_path):
    canonical, staging, quarantine = _roots(tmp_path)
    (staging / "valid" / "SNGS-2002" / "Labels-GameState.json").unlink()

    with pytest.raises(ValueError, match="staging"):
        _promote(canonical, staging, quarantine)

    _assert_rolled_back(canonical, staging)
    assert not quarantine.exists()


def test_preflight_rejects_existing_quarantine_destination(tmp_path):
    canonical, staging, quarantine = _roots(tmp_path)
    quarantine.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        _promote(canonical, staging, quarantine)

    _assert_rolled_back(canonical, staging)


def test_success_moves_exactly_train_and_valid_and_keeps_test_untouched(tmp_path):
    canonical, staging, quarantine = _roots(tmp_path)
    before = (canonical / "test" / "untouched.bin").read_bytes()

    result = _promote(canonical, staging, quarantine)

    assert result["status"] == "promoted"
    assert (canonical / "train" / "SNGS-2001").is_dir()
    assert (canonical / "valid" / "SNGS-2002").is_dir()
    assert (quarantine / "train" / "SNGS-1001").is_dir()
    assert (quarantine / "valid" / "SNGS-1002").is_dir()
    assert not (staging / "train").exists()
    assert not (staging / "valid").exists()
    assert (canonical / "test" / "untouched.bin").read_bytes() == before
    assert (quarantine / "promotion-journal.json").is_file()


@pytest.mark.parametrize("fail_on", [1, 2, 3, 4])
def test_failure_after_any_move_rolls_every_prior_move_back(
    tmp_path, monkeypatch, fail_on
):
    canonical, staging, quarantine = _roots(tmp_path)
    real_rename = promotion._rename
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise OSError("injected rename failure")
        real_rename(source, destination)

    monkeypatch.setattr(promotion, "_rename", fail_once)

    with pytest.raises(OSError, match="injected"):
        _promote(canonical, staging, quarantine)

    _assert_rolled_back(canonical, staging)
    assert (canonical / "test" / "untouched.bin").read_bytes() == b"held-out"


def test_post_move_validation_failure_rolls_back(tmp_path, monkeypatch):
    canonical, staging, quarantine = _roots(tmp_path)
    real_validate = validate_dataset

    def fail_canonical(root, *, splits, expected_counts=None):
        reports = real_validate(
            root, splits=splits, expected_counts=expected_counts
        )
        if root == canonical:
            reports[0] = replace(
                reports[0],
                errors=(*reports[0].errors, "injected_post_move_failure"),
            )
        return reports

    monkeypatch.setattr(promotion, "validate_dataset", fail_canonical)

    with pytest.raises(ValueError, match="post-move"):
        _promote(canonical, staging, quarantine)

    _assert_rolled_back(canonical, staging)


def test_confirmation_token_is_required(tmp_path):
    canonical, staging, quarantine = _roots(tmp_path)

    with pytest.raises(PermissionError, match=promotion.CONFIRMATION):
        promotion.promote_train_valid(
            canonical_root=canonical,
            staging_root=staging,
            quarantine_root=quarantine,
            expected_counts=EXPECTED,
            confirmation="yes",
        )

    _assert_rolled_back(canonical, staging)
    assert not quarantine.exists()
