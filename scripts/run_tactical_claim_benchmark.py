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

from pipeline.tactics_qa.opentac import PHASES, report, run

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
    generate.add_argument("--samples", type=int, dest="score_samples", help="odd score-pass sample count (default: Gemini 3, Doubao 1)")
    generate.add_argument("--temperature", type=float, help="score-pass temperature for providers that support it; Gemini 3.6 rejects this flag")
    generate.add_argument("--no-topology", action="store_true", help="write a geometry-free ablation under ablation_no_topology/")
    commands.add_parser("report")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "report":
        value = report(OUTPUT, GLOSSARY, GROUND_TRUTH)
    else:
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run(
            OUTPUT / "ablation_no_topology" if args.no_topology else OUTPUT,
            GLOSSARY,
            SOURCE_ROWS,
            GROUND_TRUTH,
            ROOT,
            phase=args.phase,
            provider=args.provider,
            clip_uids=args.clip_uids,
            force=args.force,
            retry_failed=args.retry_failed,
            score_samples=args.score_samples,
            temperature=args.temperature,
            use_topology=not args.no_topology,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
