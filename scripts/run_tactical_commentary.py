#!/usr/bin/env python3
"""Generate two-stage tactical commentary from reviewed facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.stage2b.tactical_commentary import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--provider", choices=("doubao", "gemini"), required=True)
    parser.add_argument("--clip-uid", action="append", dest="clip_uids", required=True)
    parser.add_argument("--allow-external-upload", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.allow_external_upload:
        raise SystemExit("Refusing external video upload without --allow-external-upload")
    outcomes = run(
        ROOT / "benchmark/tactical_prototypes/recognition_review.jsonl",
        ROOT / "outputs/preprocessing",
        ROOT / "outputs/tactical_commentary",
        ROOT / "benchmark/tactical_prototypes/deep_commentary_exemplar_v2.json",
        provider=args.provider,
        clip_uids=args.clip_uids,
        force=args.force,
        repo_root=ROOT,
        topology_root=ROOT / "outputs/tactical_topology",
    )
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
