#!/usr/bin/env python3
"""Export the provisional GroundedTactics workbook as four lean track files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa.grounded_assets import (
    CLIP_RE,
    load_criterion_registry,
    validate_question_bank,
)


DEFAULT_WORKBOOK = ROOT / "data/annotations/GroundedTactics_GSR2025_plus_9clips_merged_status_dropdowns.xlsx"
DEFAULT_OUT = ROOT / "data/grounded_tactics"
DEFAULT_REGISTRY = ROOT / "benchmark/grounded_tactics/criterion_registry.json"
TRACK_FILES = {
    "T1_TEMPORAL": "t1_temporal.json",
    "T2_OBSERVATION": "t2_observation.json",
    "T3_DIAGNOSIS": "t3_diagnosis.json",
    "T4_INTEGRATED": "t4_integrated.json",
}


def _option_id(prefix: str, answer: str | None) -> str | None:
    return f"{prefix}{ord(answer) - 64}" if answer else None


def _track_item(question: dict, index: int) -> dict:
    match = CLIP_RE.fullmatch(question["clip_id"])
    if not match:
        raise ValueError(f"invalid release-aware clip id: {question['clip_id']!r}")
    release, clip_id = match.groups()
    video = ROOT / f"data/GSR-{release or '2024'}/videos/{clip_id}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"missing video for {question['question_id']}: {video}")

    metadata = {
        "track": question["track"],
        "review_status": question["benchmark_status"],
        "annotation_status": question["gold_status"],
        "clip_uid": question["clip_uid"],
        "target_team": question["target_team"],
        "review_flags": question["review_flags"],
    }
    item = {
        "id": f"{index:03d}",
        "Q": question["question"],
        "materials": [video.relative_to(ROOT).as_posix()],
        **{f"O{number}": question["options"][letter] for number, letter in enumerate("ABCD", 1)},
        "answer": _option_id("O", question["provisional_answer"]),
        "metadata": metadata,
    }
    if question["track"] == "T3_DIAGNOSIS":
        metadata["criterion_states"] = (question["provisional_output_contract"] or {}).get("criterion_states", {})
    elif question["track"] == "T4_INTEGRATED":
        intervals = question["interval_options"]
        if [option["id"] for option in intervals] != ["I1", "I2", "I3", "I4"]:
            raise ValueError(f"{question['question_id']}: T4 evidence options must be I1-I4 in order")
        item.update({option["id"]: option["raw"] for option in intervals})
        item["evidence_answer"] = _option_id("I", question["provisional_evidence_option"])
    return item


def build_exports(workbook: Path) -> tuple[dict[str, list[dict]], dict]:
    questions, report = validate_question_bank(
        workbook, load_criterion_registry(DEFAULT_REGISTRY), root=ROOT
    )
    errors = [issue for issue in report["issues"] if issue["level"] == "error"]
    if errors:
        raise ValueError(f"question bank has {len(errors)} validation errors; first: {errors[0]}")

    grouped = {track: [] for track in TRACK_FILES}
    for question in questions:
        grouped[question["track"]].append(question)
    exports = {
        track: [
            _track_item(question, index)
            for index, question in enumerate(sorted(rows, key=lambda row: row["clip_uid"]), 1)
        ]
        for track, rows in grouped.items()
    }
    counts = {track: len(items) for track, items in exports.items()}
    if counts != {track: 135 for track in TRACK_FILES}:
        raise ValueError(f"expected 135 questions per track, got {counts}")
    if any(
        len({item["metadata"]["clip_uid"] for item in aligned}) != 1
        for aligned in zip(*(exports[track] for track in TRACK_FILES))
    ):
        raise ValueError("the same file-local id must identify the same clip across tracks")
    return exports, report


def write_exports(exports: dict[str, list[dict]], out: Path, *, force: bool = False) -> None:
    if out.exists() and not out.is_dir():
        raise ValueError(f"output path is not a directory: {out}")
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"output directory is not empty: {out}; pass --force to overwrite generated files")
    out.mkdir(parents=True, exist_ok=True)
    for track, filename in TRACK_FILES.items():
        path = out / filename
        path.write_text(json.dumps(exports[track], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if json.loads(path.read_text(encoding="utf-8")) != exports[track]:
            raise ValueError(f"post-write validation failed: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exports, report = build_exports(args.workbook)
    write_exports(exports, args.out, force=args.force)
    warning_counts = Counter(issue["code"] for issue in report["issues"] if issue["level"] == "warning")
    print(f"Wrote 4 JSON files / 540 provisional questions to {args.out}")
    print(f"Track counts: {dict(sorted(report['track_counts'].items()))}")
    print(f"Status counts: {dict(sorted(report['status_counts'].items()))}")
    print(f"Warnings preserved: {dict(sorted(warning_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
