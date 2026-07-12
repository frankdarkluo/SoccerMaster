"""Transactionally replace SoccerNetGS train/valid from validated staging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from scripts.validate_soccernetgs import _parse_expected, validate_dataset


PROMOTION_SPLITS = ("train", "valid")
CONFIRMATION = "PROMOTE_TRAIN_VALID"


def _rename(source: Path, destination: Path) -> None:
    source.rename(destination)


def _require_roots_and_splits(canonical_root: Path, staging_root: Path) -> None:
    for root_name, root in (
        ("canonical", canonical_root),
        ("staging", staging_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{root_name} root does not exist: {root}")
        for split in PROMOTION_SPLITS:
            if not (root / split).is_dir():
                raise FileNotFoundError(
                    f"{root_name} split does not exist: {root / split}"
                )


def _require_valid(
    root: Path,
    *,
    expected_counts: dict[str, int],
    label: str,
) -> None:
    reports = validate_dataset(
        root,
        splits=PROMOTION_SPLITS,
        expected_counts=expected_counts,
    )
    failures = [report for report in reports if not report.ok]
    if failures or not reports:
        codes = sorted({code for report in failures for code in report.errors})
        raise ValueError(f"{label} validation failed: {codes}")


def _write_journal(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rollback(moves: Iterable[tuple[Path, Path]]) -> None:
    failures: list[str] = []
    for source, destination in reversed(tuple(moves)):
        try:
            _rename(destination, source)
        except Exception as error:
            failures.append(f"{destination} -> {source}: {error}")
    if failures:
        raise RuntimeError("promotion rollback failed: " + "; ".join(failures))


def promote_train_valid(
    *,
    canonical_root: Path,
    staging_root: Path,
    quarantine_root: Path,
    expected_counts: dict[str, int],
    confirmation: str,
) -> dict:
    """Preflight, validate, move both splits, or roll every move back."""
    if confirmation != CONFIRMATION:
        raise PermissionError(f"confirmation must be exactly {CONFIRMATION}")
    if set(expected_counts) != set(PROMOTION_SPLITS):
        raise ValueError("expected_counts must contain exactly train and valid")

    _require_roots_and_splits(canonical_root, staging_root)
    if quarantine_root.exists():
        raise FileExistsError(f"quarantine already exists: {quarantine_root}")
    _require_valid(
        staging_root,
        expected_counts=expected_counts,
        label="staging",
    )
    if canonical_root.stat().st_dev != staging_root.stat().st_dev:
        raise OSError("canonical and staging roots must be on the same filesystem")

    quarantine_root.mkdir(parents=True)
    planned_moves = [
        (canonical_root / "train", quarantine_root / "train"),
        (canonical_root / "valid", quarantine_root / "valid"),
        (staging_root / "train", canonical_root / "train"),
        (staging_root / "valid", canonical_root / "valid"),
    ]
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in planned_moves:
            _rename(source, destination)
            completed.append((source, destination))
        try:
            _require_valid(
                canonical_root,
                expected_counts=expected_counts,
                label="post-move",
            )
        except Exception as error:
            raise ValueError(f"post-move validation failed: {error}") from error

        result = {
            "status": "promoted",
            "splits": list(PROMOTION_SPLITS),
            "canonical_root": str(canonical_root),
            "staging_root": str(staging_root),
            "quarantine_root": str(quarantine_root),
            "moves": [
                {"source": str(source), "destination": str(destination)}
                for source, destination in completed
            ],
        }
        _write_journal(quarantine_root / "promotion-journal.json", result)
        return result
    except Exception:
        _rollback(completed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--expect", nargs="+", action="append", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    expected_counts = _parse_expected(
        [value for group in args.expect for value in group]
    )
    result = promote_train_valid(
        canonical_root=args.canonical_root,
        staging_root=args.staging_root,
        quarantine_root=args.quarantine_root,
        expected_counts=expected_counts,
        confirmation=args.confirm,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
