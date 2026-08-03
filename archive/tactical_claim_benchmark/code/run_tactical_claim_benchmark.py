#!/usr/bin/env python3
"""Run the isolated tactical evidence-chain benchmark and P0 scan."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa.claim_benchmark import (
    export_claim_judge_queue,
    generate_claim_report,
    generate_p0_report,
    import_judge_decisions,
    prepare_gt,
    run_claims,
    run_p0,
)

GT_CSV = ROOT / "data/足球战术识别.csv"
GLOSSARY = ROOT / "data/足球战术数据库_词条表_Grid.csv"
SOURCE_ROWS = ROOT / "benchmark/tactical_prototypes/source_rows.jsonl"
BENCHMARK_ROOT = ROOT / "outputs/tactical_claim_benchmark"
P0_ROOT = ROOT / "outputs/tactical_claim_benchmark/p0_nomination"
SMOKE_CLIPS = (
    "event_clips:0006",
    "soccernetgs:SNGS-101",
    "soccernetgs:SNGS-148",
)


def _paths(parser: argparse.ArgumentParser, *, p0: bool = False) -> None:
    parser.add_argument(
        "--output-root", type=Path, default=P0_ROOT if p0 else BENCHMARK_ROOT,
    )
    parser.add_argument("--glossary", type=Path, default=GLOSSARY)
    parser.add_argument("--source-rows", type=Path, default=SOURCE_ROWS)


def _run_args(parser: argparse.ArgumentParser, *, p0: bool = False) -> None:
    _paths(parser, p0=p0)
    parser.add_argument("--clip-uid", action="append", dest="clip_uids")
    parser.add_argument("--allow-external-upload", action="store_true")
    parser.add_argument("--force", action="store_true")
    if p0:
        parser.add_argument(
            "--provider", choices=("doubao", "gemini"), required=True,
        )
    else:
        parser.add_argument("--provider", choices=("doubao", "gemini"), action="append")
    parser.add_argument("--retry-failed", action="store_true")
    if p0:
        parser.add_argument(
            "--phase",
            choices=("phase1_direct", "phase2_observation_first"),
            default="phase1_direct",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-gt")
    _paths(prepare)
    prepare.add_argument("--csv", type=Path, default=GT_CSV)
    prepare.add_argument("--approve", action="store_true")

    claims = commands.add_parser("run-claims")
    _run_args(claims)

    smoke = commands.add_parser("smoke")
    _run_args(smoke)

    export = commands.add_parser("export-judge")
    export.add_argument("--mode", choices=("claims",), required=True)
    _paths(export)
    export.set_defaults(output_root=None)

    import_parser = commands.add_parser("import-judge")
    import_parser.add_argument("--mode", choices=("claims",), required=True)
    import_parser.add_argument("--reviews", type=Path, required=True)
    _paths(import_parser)
    import_parser.set_defaults(output_root=None)

    report = commands.add_parser("report")
    _paths(report)

    p0 = commands.add_parser("run-p0")
    _run_args(p0, p0=True)

    p0_report = commands.add_parser("p0-report")
    _paths(p0_report, p0=True)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if getattr(args, "output_root", None) is None:
        args.output_root = P0_ROOT if args.mode == "p0" else BENCHMARK_ROOT
    common = {
        "glossary_path": args.glossary,
        "source_rows": args.source_rows,
        "repo_root": ROOT,
    }
    if args.command == "prepare-gt":
        result = prepare_gt(
            args.csv, output_root=args.output_root, approved=args.approve, **common,
        )
    elif args.command in {"run-claims", "smoke"}:
        clips = list(SMOKE_CLIPS) if args.command == "smoke" else args.clip_uids
        result = run_claims(
            args.output_root,
            clip_uids=clips,
            allow_external_upload=args.allow_external_upload,
            force=args.force,
            providers=args.provider or ("doubao", "gemini"),
            retry_failed=args.retry_failed,
            **common,
        )
    elif args.command == "run-p0":
        result = run_p0(
            args.output_root,
            phase=args.phase,
            clip_uids=args.clip_uids,
            allow_external_upload=args.allow_external_upload,
            force=args.force,
            providers=(args.provider,),
            retry_failed=args.retry_failed,
            **common,
        )
    elif args.command == "export-judge":
        result = export_claim_judge_queue(args.output_root, **common)
    elif args.command == "import-judge":
        result = import_judge_decisions(args.output_root, args.reviews)
    elif args.command == "report":
        report = generate_claim_report(args.output_root, ROOT)
        result = {
            "execution": report["execution"],
            "systems": report["systems"],
            "report": (args.output_root / "reports/claim_metrics.json").as_posix(),
        }
    else:
        result = generate_p0_report(args.output_root, args.glossary)
    _print(result)


if __name__ == "__main__":
    main()
