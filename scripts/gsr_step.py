"""Run a single GSR step for one clip (honors step-level parallelism).

run.py only exposes the full pipeline; this thin driver lets us fan out
step1/step2/step3 across clips independently. Step 3 also converts the
final pklz to predictions.json (CPU, cheap) so success == predictions.json.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import (
    PipelineConfig,
    REPO_ROOT,
    default_gsr_frames_root,
    resolve_gsr_clip,
)
from pipeline.stage1_inference.run_gsr import run_step1, run_step2, run_step3
from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

ap = argparse.ArgumentParser()
ap.add_argument(
    "--frames-root",
    type=Path,
    help="flat images root or legacy SoccerNetGS root (default: auto-detect)",
)
ap.add_argument("--clip", required=True)          # e.g. SNGS-001
ap.add_argument("--step", required=True, choices=["1", "2", "3"])
ap.add_argument("--split", help="logical source split override")
ap.add_argument(
    "--output-root",
    type=Path,
    default=Path("outputs/preprocessing"),
    help="parent of <clip>/step{1,2,3} (default: outputs/preprocessing)",
)
args = ap.parse_args()

if args.frames_root:
    frames_root = args.frames_root
elif "GSR-2025" in args.output_root.parts:
    # ponytail: compatibility for launchers loaded before --frames-root existed.
    frames_root = REPO_ROOT / "data" / "GSR-2025" / "images"
else:
    frames_root = default_gsr_frames_root()
clip_dir = resolve_gsr_clip(frames_root, args.clip, args.split)
out = args.output_root / args.clip
cfg = PipelineConfig(
    clip_dir=clip_dir, output_dir=out, fps=25, gsr_split=args.split, gsr_frames_root=frames_root
)

if args.step == "1":
    run_step1(cfg)
elif args.step == "2":
    run_step2(cfg)
else:
    final_pklz = run_step3(cfg)
    seq_id = re.match(r"SNGS-(\d+)", args.clip, re.I).group(1)
    convert_pklz_to_json(final_pklz, seq_id, out, fps=25, sequence_name=args.clip)
    print(f"[gsr_step] {args.clip} predictions.json ->", out / "predictions.json")
