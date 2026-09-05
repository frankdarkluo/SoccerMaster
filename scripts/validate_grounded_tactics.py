#!/usr/bin/env python3
"""Audit GroundedTactics sources and optionally export non-gold derived assets."""

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
    SCHEMA_VERSION,
    build_candidate_queues,
    crosscheck_manifest,
    issue_counts,
    load_candidate_csv,
    load_candidate_xlsx,
    load_criterion_registry,
    load_label_map,
    load_manifest,
    validate_question_bank,
    validate_structured_annotations,
    write_jsonl,
)


DEFAULT_HUMAN = ROOT / "data/annotations/多模态模型战术识别 - Human Labelling-GSR2024.csv"
DEFAULT_SOL = ROOT / "data/annotations/多模态模型战术识别 - 5.6 Sol Pro-GSR2024.csv"
DEFAULT_WORKBOOK = ROOT / "data/annotations/GroundedTactics_9clips_question_bank_provisional.xlsx"
DEFAULT_ONTOLOGY = ROOT / "data/足球战术数据库_词条表_Grid.csv"
DEFAULT_ASSETS = ROOT / "benchmark/grounded_tactics"


def load_candidate(path: Path, sheet: str | None, **kwargs) -> list[dict]:
    if path.suffix.casefold() == ".xlsx":
        if not sheet:
            raise ValueError(f"--human-sheet/--sol-sheet is required for XLSX input: {path}")
        return load_candidate_xlsx(path, sheet, **kwargs)
    return load_candidate_csv(path, **kwargs)


def candidate_report(claims: list[dict], manifest: dict[str, dict] | None) -> dict:
    human_source_ids = sorted({
        row["source"]["source_id"]
        for row in claims
        if row["source"]["source_type"] == "human_expert_seed"
    })
    human_clip_sets = [
        {row["clip_id"] for row in claims if row["source"]["source_id"] == source_id}
        for source_id in human_source_ids
    ]
    common_human_clips = set.intersection(*human_clip_sets) if human_clip_sets else set()
    human_readiness = {
        "rater_count": len(human_source_ids),
        "source_ids": human_source_ids,
        "common_clip_count": len(common_human_clips),
        "coverage_aligned": bool(human_clip_sets) and all(
            clips == human_clip_sets[0] for clips in human_clip_sets
        ),
        "three_rater_ready": len(human_source_ids) >= 3 and bool(human_clip_sets) and all(clips == human_clip_sets[0] for clips in human_clip_sets),
    }
    return {
        "human_rater_readiness": human_readiness,
        "claim_count": len(claims),
        "clip_count": len({row["clip_id"] for row in claims}),
        "source_counts": dict(Counter(row["source"]["source_id"] for row in claims)),
        "source_clip_counts": {
            source_id: len({row["clip_id"] for row in claims if row["source"]["source_id"] == source_id})
            for source_id in sorted({row["source"]["source_id"] for row in claims})
        },
        "label_kind_counts": dict(Counter(row["label_kind"] for row in claims)),
        "span_status_counts": dict(Counter(row["span_status"] for row in claims)),
        "review_flag_counts": dict(Counter(flag for row in claims for flag in row["review_flags"])),
        "manifest": crosscheck_manifest((row["clip_id"] for row in claims), manifest),
        "source_files": {
            row["source"]["source_id"]: {
                "path": row["source"]["path"],
                "sha256": row["source"]["file_sha256"],
            }
            for row in claims
        },
    }


def build_report(args: argparse.Namespace) -> tuple[dict, list[dict], dict, list[dict]]:
    for path in (
        args.human,
        *args.additional_human,
        args.sol,
        args.workbook,
        args.ontology,
        args.aliases,
        args.criterion_registry,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    manifest = load_manifest(args.manifest) if args.manifest else None
    label_map = load_label_map(args.ontology, args.aliases)
    registry = load_criterion_registry(args.criterion_registry)
    human = load_candidate(
        args.human, args.human_sheet,
        source_id="human_annotator_1",
        source_type="human_expert_seed",
        label_map=label_map,
        root=ROOT,
        release=args.human_release,
    )
    additional_humans = []
    for index, path in enumerate(args.additional_human, 2):
        additional_humans.extend(load_candidate_csv(
            path,
            source_id=f"human_annotator_{index}",
            source_type="human_expert_seed",
            label_map=label_map,
            root=ROOT,
        ))
    sol = load_candidate(
        args.sol, args.sol_sheet,
        source_id="gpt-5.6-sol-pro",
        source_type="model_weak_label",
        label_map=label_map,
        root=ROOT,
        release=args.sol_release,
    )
    claims = human + additional_humans + sol
    questions, question_report = validate_question_bank(
        args.workbook, registry, manifest, root=ROOT
    )
    queues = build_candidate_queues(claims, questions)

    issues = list(question_report["issues"])
    candidate_summary = candidate_report(claims, manifest)
    if candidate_summary["manifest"]["status"] == "failed":
        issues.append({
            "level": "error",
            "code": "manifest_missing_candidate_clips",
            "message": "candidate clips are absent from the manifest",
            "missing_clip_ids": candidate_summary["manifest"]["missing_clip_ids"],
        })

    structured_reports = [
        validate_structured_annotations(path, registry, manifest)
        for path in args.structured_annotations
    ]
    for structured in structured_reports:
        issues.extend(structured["issues"])
    registry_statuses = Counter(entry["status"] for entry in registry["criteria"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "release_status": "construction_audit_not_gold",
        "candidates": candidate_summary,
        "queues": {"counts": queues["counts"], "semantics": queues["queue_semantics"]},
        "provisional_question_bank": {
            key: value for key, value in question_report.items() if key != "issues"
        },
        "criterion_registry": {
            "criterion_count": len(registry["criteria"]),
            "status_counts": dict(registry_statuses),
            "incomplete_definition_count": sum(
                not entry.get("definition") for entry in registry["criteria"]
            ),
            "allowed_states": sorted(registry["allowed_states"]),
        },
        "structured_annotations": structured_reports,
        "issue_counts": issue_counts(issues),
        "issues": issues,
        "guardrails": [
            "Human and Sol rows remain separate candidate claims.",
            "Sol is a weak-label source and never counts as a human annotator.",
            "The provisional workbook is not benchmark gold.",
            "GSR game-state labels and preprocessing artifacts are not tactic gold.",
        ],
    }
    return report, claims, queues, questions


def write_outputs(out: Path, report: dict, claims: list[dict], queues: dict, questions: list[dict], *, force: bool = False) -> None:
    resolved = out.resolve()
    if resolved == (ROOT / "data").resolve() or (ROOT / "data").resolve() in resolved.parents:
        raise ValueError("derived outputs must not be written under data/")
    if out.exists() and not out.is_dir():
        raise ValueError(f"output path is not a directory: {out}")
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"output directory is not empty: {out}; pass --force to overwrite generated files")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "candidate_claims.jsonl", claims)
    (out / "candidate_queues.json").write_text(
        json.dumps(queues, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(out / "provisional_questions.jsonl", questions)
    (out / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--human-sheet", help="Sheet name when --human is XLSX.")
    parser.add_argument("--human-release", choices=("2024", "2025"))
    parser.add_argument(
        "--additional-human",
        type=Path,
        action="append",
        default=[],
        help="Additional blind Round-1 Human CSV; repeat for annotators 2 and 3.",
    )
    parser.add_argument("--sol", type=Path, default=DEFAULT_SOL)
    parser.add_argument("--sol-sheet", help="Sheet name when --sol is XLSX.")
    parser.add_argument("--sol-release", choices=("2024", "2025"))
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ASSETS / "label_aliases.json")
    parser.add_argument(
        "--criterion-registry",
        type=Path,
        default=DEFAULT_ASSETS / "criterion_registry.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional GSR-2024 clip_manifest.jsonl; a supplied missing/invalid manifest is fatal.",
    )
    parser.add_argument(
        "--structured-annotations",
        type=Path,
        action="append",
        default=[],
        help="Optional Stage-2 JSONL to validate; repeat for multiple annotators.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional derived-output directory (outputs/ recommended). No files are written if omitted.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing generated files in a nonempty --out directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report, claims, queues, questions = build_report(args)
        if args.out:
            write_outputs(args.out, report, claims, queues, questions, force=args.force)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "release_status": report["release_status"],
        "candidate_claims": report["candidates"]["claim_count"],
        "candidate_clips": report["candidates"]["clip_count"],
        "questions": report["provisional_question_bank"]["question_count"],
        "question_clips": report["provisional_question_bank"]["clip_count"],
        "queues": report["queues"]["counts"],
        "issue_counts": report["issue_counts"],
        "manifest": report["candidates"]["manifest"]["status"],
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report["issue_counts"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
