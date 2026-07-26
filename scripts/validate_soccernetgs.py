"""Read-only integrity checks for SoccerNetGS train/valid splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable


LABEL_NAME = "Labels-GameState.json"
PITCH_ROLES = frozenset({"player", "goalkeeper"})


@dataclass(frozen=True)
class SequenceReport:
    split: str
    sequence: str
    sequence_id: str | None
    game_id: str | None
    version: str | None
    frame_count: int
    image_count: int
    annotation_count: int
    pitch_annotation_count: int
    ball_annotation_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _report(
    sequence_dir: Path,
    split: str,
    *,
    errors: Iterable[str],
    warnings: Iterable[str] = (),
    sequence_id: str | None = None,
    game_id: str | None = None,
    version: str | None = None,
    frame_count: int = 0,
    image_count: int = 0,
    annotation_count: int = 0,
    pitch_annotation_count: int = 0,
    ball_annotation_count: int = 0,
) -> SequenceReport:
    return SequenceReport(
        split=split,
        sequence=sequence_dir.name,
        sequence_id=sequence_id,
        game_id=game_id,
        version=version,
        frame_count=frame_count,
        image_count=image_count,
        annotation_count=annotation_count,
        pitch_annotation_count=pitch_annotation_count,
        ball_annotation_count=ball_annotation_count,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _version_before_1_3(value: object) -> bool:
    try:
        parts = tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return True
    return (parts + (0, 0))[:2] < (1, 3)


def _finite_pitch(annotation: object) -> bool:
    if not isinstance(annotation, dict):
        return False
    pitch = annotation.get("bbox_pitch")
    if not isinstance(pitch, dict):
        return False
    x = pitch.get("x_bottom_middle")
    y = pitch.get("y_bottom_middle")
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and isinstance(y, (int, float))
        and not isinstance(y, bool)
        and math.isfinite(x)
        and math.isfinite(y)
    )


def validate_sequence(sequence_dir: Path, *, split: str) -> SequenceReport:
    """Read and validate one sequence without writing to it."""
    label_path = sequence_dir / LABEL_NAME
    if not label_path.is_file():
        return _report(sequence_dir, split, errors=("missing_label",))

    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _report(sequence_dir, split, errors=("invalid_json",))

    if not isinstance(data, dict):
        return _report(sequence_dir, split, errors=("invalid_top_level",))

    info = data.get("info")
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    if (
        not isinstance(info, dict)
        or not isinstance(images, list)
        or not isinstance(annotations, list)
        or not isinstance(categories, list)
    ):
        return _report(sequence_dir, split, errors=("invalid_top_level",))

    errors: list[str] = []
    warnings: list[str] = []
    sequence_id = str(info["id"]) if info.get("id") is not None else None
    game_id = str(info["game_id"]) if info.get("game_id") is not None else None
    version = str(info["version"]) if info.get("version") is not None else None

    if _version_before_1_3(version):
        errors.append("dataset_version_before_1_3")
    if not game_id:
        errors.append("missing_game_id")
    if info.get("name") != sequence_dir.name:
        errors.append("name_mismatch")
    if sequence_id is None or sequence_dir.name.removeprefix("SNGS-") != sequence_id:
        errors.append("id_mismatch")
    frame_rate = info.get("frame_rate")
    if (
        not isinstance(frame_rate, (int, float))
        or isinstance(frame_rate, bool)
        or not math.isfinite(frame_rate)
        or frame_rate <= 0
    ):
        errors.append("bad_frame_rate")

    image_ids = [
        str(image.get("image_id"))
        for image in images
        if isinstance(image, dict) and image.get("image_id") is not None
    ]
    file_names = [
        image.get("file_name")
        for image in images
        if isinstance(image, dict) and isinstance(image.get("file_name"), str)
    ]
    if len(image_ids) != len(images) or len(set(image_ids)) != len(image_ids):
        errors.append("duplicate_image_id")
    if len(file_names) != len(images) or len(set(file_names)) != len(file_names):
        errors.append("duplicate_file_name")

    image_dir_name = info.get("im_dir", "img1")
    image_dir = sequence_dir / image_dir_name if isinstance(image_dir_name, str) else sequence_dir / "img1"
    disk_files = sorted(path.name for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else []
    seq_length = info.get("seq_length")
    if (
        not isinstance(seq_length, int)
        or isinstance(seq_length, bool)
        or seq_length != len(images)
        or seq_length != len(disk_files)
    ):
        errors.append("frame_count_mismatch")
    if any(not (image_dir / name).is_file() for name in file_names):
        errors.append("missing_frame")
    if any((image_dir / name).is_file() and (image_dir / name).stat().st_size == 0 for name in file_names):
        errors.append("empty_frame")

    image_id_set = set(image_ids)
    if any(
        not isinstance(annotation, dict)
        or str(annotation.get("image_id")) not in image_id_set
        for annotation in annotations
    ):
        errors.append("bad_annotation_image_ref")

    pitch_annotation_count = 0
    ball_annotation_count = 0
    for annotation in annotations:
        if not isinstance(annotation, dict) or not _finite_pitch(annotation):
            continue
        attributes = annotation.get("attributes")
        role = attributes.get("role") if isinstance(attributes, dict) else None
        if role in PITCH_ROLES:
            pitch_annotation_count += 1
        elif role == "ball":
            ball_annotation_count += 1
    if pitch_annotation_count == 0:
        errors.append("no_pitch_coordinates")
    if ball_annotation_count == 0:
        warnings.append("no_ball_coordinates")

    return _report(
        sequence_dir,
        split,
        errors=errors,
        warnings=warnings,
        sequence_id=sequence_id,
        game_id=game_id,
        version=version,
        frame_count=len(disk_files),
        image_count=len(images),
        annotation_count=len(annotations),
        pitch_annotation_count=pitch_annotation_count,
        ball_annotation_count=ball_annotation_count,
    )


def validate_dataset(
    root: Path,
    *,
    splits: tuple[str, ...],
    expected_counts: dict[str, int] | None = None,
) -> list[SequenceReport]:
    """Validate selected splits and detect duplicate info.id/name values."""
    reports: list[SequenceReport] = []
    for split in splits:
        start = len(reports)
        split_dir = root / split
        sequence_dirs = (
            sorted(path for path in split_dir.iterdir() if path.is_dir())
            if split_dir.is_dir()
            else []
        )
        reports.extend(validate_sequence(path, split=split) for path in sequence_dirs)
        if expected_counts is not None and len(sequence_dirs) != expected_counts.get(split):
            if start == len(reports):
                reports.append(
                    _report(split_dir, split, errors=("unexpected_sequence_count",))
                )
            else:
                for index in range(start, len(reports)):
                    reports[index] = replace(
                        reports[index],
                        errors=tuple(
                            dict.fromkeys(
                                (*reports[index].errors, "unexpected_sequence_count")
                            )
                        ),
                    )

    duplicates: set[int] = set()
    for key in ("sequence_id", "sequence"):
        by_value: dict[str, list[int]] = {}
        for index, report in enumerate(reports):
            value = getattr(report, key)
            if value:
                by_value.setdefault(value, []).append(index)
        for indexes in by_value.values():
            if len(indexes) > 1:
                duplicates.update(indexes)
    for index in duplicates:
        reports[index] = replace(
            reports[index],
            errors=tuple(
                dict.fromkeys((*reports[index].errors, "duplicate_sequence_id"))
            ),
        )
    return reports


def _parse_expected(values: list[str]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for value in values:
        split, separator, count = value.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"expected SPLIT=COUNT, got {value!r}")
        expected[split] = int(count)
    return expected


def _report_payload(
    root: Path,
    splits: tuple[str, ...],
    expected_counts: dict[str, int],
    reports: list[SequenceReport],
) -> dict:
    split_summaries = {}
    for split in splits:
        selected = [report for report in reports if report.split == split]
        split_summaries[split] = {
            "sequence_count": len(
                [report for report in selected if report.sequence != split]
            ),
            "ok_count": sum(report.ok for report in selected),
            "error_count": sum(not report.ok for report in selected),
            "warning_count": sum(bool(report.warnings) for report in selected),
        }
    return {
        "schema_version": "soccernetgs-integrity-v1",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "expected_counts": expected_counts,
        "split_summaries": split_summaries,
        "sequences": [
            {**asdict(report), "ok": report.ok}
            for report in reports
        ],
    }


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--expect", nargs="+", action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    splits = tuple(args.splits)
    expected_counts = _parse_expected(
        [value for group in args.expect for value in group]
    )
    reports = validate_dataset(
        args.root,
        splits=splits,
        expected_counts=expected_counts or None,
    )
    payload = _report_payload(args.root, splits, expected_counts, reports)
    if args.report:
        _write_atomic(args.report, payload)
    print(json.dumps(payload["split_summaries"], sort_keys=True))
    return 0 if reports and all(report.ok for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
