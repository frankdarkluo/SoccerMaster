#!/usr/bin/env python3
"""Analyze stable, visible tactical lines from GameState positions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.topology.analysis import analyze_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze",))
    parser.add_argument("--clip-uid", action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tactical_topology")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.labels and len(args.clip_uid) != 1:
        raise SystemExit("--labels requires exactly one --clip-uid")
    outcomes = []
    for clip_uid in args.clip_uid:
        clip = clip_uid.split(":", 1)[-1]
        output = args.output_root / f"{clip}.json"
        result = analyze_clip(
            ROOT,
            clip_uid,
            output,
            fps=args.fps,
            force=args.force,
            labels_path=args.labels,
        )
        outcomes.append({
            "clip_uid": clip_uid,
            "position_source": result["position_source"],
            "output": str(output),
            "windows": len(result["windows"]),
        })
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
