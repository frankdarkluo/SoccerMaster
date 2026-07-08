"""Validate detected events against clip_index.csv ground-truth rows."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pipeline.config import DATASET_ROOT
from pipeline.stage2_events.detector import detect_candidates, load_frames
from pipeline.stage2_events.possession import (
    possession_segments,
    resolve_role_by_track,
    resolve_team_by_track,
)
from pipeline.utils.clip_index_to_events import ACTION_TO_EVENT_CODE


@dataclass
class ValidationRow:
    sample_id: str
    expected_time_s: float
    normalized_action: str
    expected_event_code: Optional[str]
    matched: bool
    matched_time_s: Optional[float]
    time_error_s: Optional[float]
    note: str = ""


def _labels_path(dataset_root: Path, sample_id: str) -> Path:
    split = "test" if sample_id.startswith("SNGS-1") or sample_id.startswith("SNGS-2") else "train"
    return dataset_root / split / sample_id / "Labels-GameState.json"


def _best_match(
    candidates: list,
    expected_time_s: float,
    tolerance_s: float,
):
    matches = [
        c for c in candidates
        if abs(c.timestamp_s - expected_time_s) <= tolerance_s
    ]
    if not matches:
        return None
    return min(matches, key=lambda c: abs(c.timestamp_s - expected_time_s))


def _detect_candidates(labels_path: Path, fps: int):
    frames = load_frames(str(labels_path))
    team_by_track = resolve_team_by_track(frames)
    role_by_track = resolve_role_by_track(frames)
    segments = possession_segments(frames, team_by_track)
    return detect_candidates(frames, segments, team_by_track, role_by_track, fps=fps)


def validate_clip_index(
    clip_index_csv: Path,
    dataset_root: Path = DATASET_ROOT,
    tolerance_s: float = 1.5,
    fps: int = 25,
) -> List[ValidationRow]:
    """Run candidate detector on each clip_index row and compare rounded event time.

    Stage 2 now emits untyped candidates; this utility validates timestamp
    coverage only, not final action labels.
    """
    rows: List[ValidationRow] = []

    with open(clip_index_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"].strip()
            action = row["normalized_action"].strip()
            expected_time = round(float(row["event_time_sec"]))
            expected_code = ACTION_TO_EVENT_CODE.get(action)

            if expected_code is None:
                rows.append(ValidationRow(
                    sample_id=sample_id,
                    expected_time_s=expected_time,
                    normalized_action=action,
                    expected_event_code=None,
                    matched=False,
                    matched_time_s=None,
                    time_error_s=None,
                    note="action not in detector vocabulary",
                ))
                continue

            labels_path = _labels_path(dataset_root, sample_id)
            if not labels_path.exists():
                rows.append(ValidationRow(
                    sample_id=sample_id,
                    expected_time_s=expected_time,
                    normalized_action=action,
                    expected_event_code=expected_code,
                    matched=False,
                    matched_time_s=None,
                    time_error_s=None,
                    note=f"missing labels: {labels_path}",
                ))
                continue

            candidates = _detect_candidates(labels_path, fps)
            match = _best_match(candidates, expected_time, tolerance_s)
            if match is None:
                rows.append(ValidationRow(
                    sample_id=sample_id,
                    expected_time_s=expected_time,
                    normalized_action=action,
                    expected_event_code=expected_code,
                    matched=False,
                    matched_time_s=None,
                    time_error_s=None,
                    note="no matching detected candidate",
                ))
                continue

            rows.append(ValidationRow(
                sample_id=sample_id,
                expected_time_s=expected_time,
                normalized_action=action,
                expected_event_code=expected_code,
                matched=True,
                matched_time_s=match.timestamp_s,
                time_error_s=round(match.timestamp_s - expected_time, 2),
            ))

    return rows


def summarize_validation(rows: List[ValidationRow]) -> dict:
    evaluable = [r for r in rows if r.expected_event_code is not None]
    matched = [r for r in evaluable if r.matched]
    return {
        "total_rows": len(rows),
        "evaluable": len(evaluable),
        "matched": len(matched),
        "match_rate": round(len(matched) / len(evaluable), 3) if evaluable else 0.0,
        "skipped": len(rows) - len(evaluable),
    }


def print_validation_report(rows: List[ValidationRow]) -> None:
    summary = summarize_validation(rows)
    print(
        f"clip_index validation: {summary['matched']}/{summary['evaluable']} matched "
        f"({summary['match_rate']:.1%}), skipped {summary['skipped']}"
    )
    for row in rows:
        if row.expected_event_code is None:
            continue
        status = "OK" if row.matched else "MISS"
        detail = (
            f"t={row.matched_time_s}s err={row.time_error_s:+.2f}s"
            if row.matched
            else row.note
        )
        print(
            f"  [{status}] {row.sample_id} {row.normalized_action} "
            f"expected={row.expected_time_s}s -> {detail}"
        )
