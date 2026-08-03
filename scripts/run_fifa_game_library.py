#!/usr/bin/env python3
"""Prepare and run the FIFA Game Library analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fifa_game_library import (
    apply_grid,
    migrate_results,
    prepare_index,
    report,
    run_library,
    visual_scan,
)

LIBRARY = ROOT / "data/FIFA Game Library"
INDEX = ROOT / "data/FIFA_Game_Library_examples.csv"
GLOSSARY = ROOT / "data/足球战术数据库_词条表_Grid.csv"
OUTPUT = ROOT / "outputs/fifa_game_library"
MAPPING_REVIEW = ROOT / "data/FIFA_Game_Library_mapping_review.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("visual-scan")
    run = subparsers.add_parser("run")
    run.add_argument("--allow-external-upload", action="store_true")
    run.add_argument("--clip-sha", action="append", default=[])
    run.add_argument("--force", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    subparsers.add_parser("report")
    subparsers.add_parser("apply-grid")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = {
            "index": prepare_index(LIBRARY, INDEX),
            "analysis": migrate_results(INDEX, OUTPUT),
        }
    elif args.command == "visual-scan":
        result = visual_scan(LIBRARY, OUTPUT)
    elif args.command == "run":
        if not args.allow_external_upload:
            raise SystemExit(
                "Refusing external video upload without --allow-external-upload"
            )
        result = run_library(
            LIBRARY,
            INDEX,
            GLOSSARY,
            OUTPUT,
            force=args.force,
            retry_failed=args.retry_failed,
            clip_shas=args.clip_sha,
        )
    elif args.command == "report":
        result = report(INDEX, OUTPUT, GLOSSARY)
    else:
        result = apply_grid(INDEX, GLOSSARY, OUTPUT, MAPPING_REVIEW)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
