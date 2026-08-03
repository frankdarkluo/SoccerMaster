#!/usr/bin/env python3
"""Run or score the frozen two-phase open tactical-tag benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa.opentac import (
    ONE_SHOT_TACTICS,
    ONE_SHOT_V2_TACTICS,
    PHASES,
    report,
    report_shot_comparison,
    report_shot_comparison_v2,
    run,
    run_one_shot,
    run_one_shot_v2,
)

OUTPUT = ROOT / "outputs/tactical_claim_benchmark/opentac"
GROUND_TRUTH = OUTPUT / "evaluation/ground_truth.jsonl"
GLOSSARY = ROOT / "data/足球战术数据库_词条表_Grid.csv"
SOURCE_ROWS = ROOT / "benchmark/tactical_prototypes/source_rows.jsonl"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("run")
    generate.add_argument("--phase", choices=PHASES, required=True)
    generate.add_argument("--provider", choices=("doubao", "gemini"), required=True)
    generate.add_argument("--clip-uid", action="append", dest="clip_uids")
    generate.add_argument("--allow-external-upload", action="store_true")
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--retry-failed", action="store_true")
    one_shot = commands.add_parser("run-one-shot")
    one_shot.add_argument("--provider", choices=("doubao", "gemini"), required=True)
    one_shot.add_argument("--tactic-id", choices=ONE_SHOT_TACTICS, action="append", dest="tactic_ids")
    one_shot.add_argument("--clip-uid", action="append", dest="clip_uids")
    one_shot.add_argument("--allow-external-upload", action="store_true")
    one_shot.add_argument("--force", action="store_true")
    one_shot.add_argument("--retry-failed", action="store_true")
    one_shot_v2 = commands.add_parser("run-one-shot-v2")
    one_shot_v2.add_argument("--provider", choices=("doubao", "gemini"), required=True)
    one_shot_v2.add_argument("--tactic-id", choices=ONE_SHOT_V2_TACTICS, action="append", dest="tactic_ids")
    one_shot_v2.add_argument("--clip-uid", action="append", dest="clip_uids")
    one_shot_v2.add_argument("--allow-external-upload", action="store_true")
    one_shot_v2.add_argument("--force", action="store_true")
    one_shot_v2.add_argument("--retry-failed", action="store_true")
    commands.add_parser("report")
    commands.add_parser("report-shot-comparison")
    commands.add_parser("report-shot-comparison-v2")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "report":
        value = report(OUTPUT, GLOSSARY, GROUND_TRUTH)
    elif args.command == "report-shot-comparison":
        value = report_shot_comparison(OUTPUT, GLOSSARY, GROUND_TRUTH)
    elif args.command == "report-shot-comparison-v2":
        value = report_shot_comparison_v2(OUTPUT, GLOSSARY, GROUND_TRUTH)
    elif args.command == "run-one-shot":
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run_one_shot(
            OUTPUT,
            GLOSSARY,
            SOURCE_ROWS,
            GROUND_TRUTH,
            ROOT,
            provider=args.provider,
            tactic_ids=args.tactic_ids,
            clip_uids=args.clip_uids,
            force=args.force,
            retry_failed=args.retry_failed,
        )
    elif args.command == "run-one-shot-v2":
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run_one_shot_v2(
            OUTPUT,
            GLOSSARY,
            SOURCE_ROWS,
            GROUND_TRUTH,
            ROOT,
            provider=args.provider,
            tactic_ids=args.tactic_ids,
            clip_uids=args.clip_uids,
            force=args.force,
            retry_failed=args.retry_failed,
        )
    else:
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run(
            OUTPUT,
            GLOSSARY,
            SOURCE_ROWS,
            GROUND_TRUTH,
            ROOT,
            phase=args.phase,
            provider=args.provider,
            clip_uids=args.clip_uids,
            force=args.force,
            retry_failed=args.retry_failed,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
