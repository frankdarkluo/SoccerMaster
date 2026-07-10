#!/usr/bin/env bash
# Pipeline Stage 3 only: visual effects overlay → annotated_video.mp4 for ONE clip.
# Reads:  <output_dir>/events.json (Stage 2B)
#         <output_dir>/predictions.json (Stage 1)
#         optional: <output_dir>/homography_per_frame.json
#         <clip_dir>/img1/*.jpg
# Writes: <output_dir>/annotated_video.mp4
#
# Does NOT touch Stage 1–2B JSON outputs or Stage 5 (commentary, TTS, final_video).
#
# Usage:
#   bash scripts/run_stage3.sh
#   bash scripts/run_stage3.sh outputs/SNGS-148
#   bash scripts/run_stage3.sh outputs/SNGS-148 codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148
#   NO_TOPOLOGY_LINES=1 bash scripts/run_stage3.sh outputs/SNGS-148
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-148}"
SEQ_NAME="$(basename "$OUTPUT_DIR")"

if [[ -n "${2:-}" ]]; then
  CLIP_DIR="$2"
else
  CLIP_DIR=""
  for split in test sn500 train valid; do
    cand="codes/sn-gamestate/datasets/SoccerNetGS/${split}/${SEQ_NAME}"
    if [[ -d "$cand/img1" ]]; then
      CLIP_DIR="$cand"
      break
    fi
  done
  if [[ -z "$CLIP_DIR" ]]; then
    echo "ERROR: cannot find dataset frames for ${SEQ_NAME} under codes/sn-gamestate/datasets/SoccerNetGS/{test,sn500,train,valid}/" >&2
    echo "       pass clip_dir explicitly: bash scripts/run_stage3.sh ${OUTPUT_DIR} <clip_dir>" >&2
    exit 1
  fi
fi

if [[ ! -f "$OUTPUT_DIR/events.json" ]]; then
  echo "ERROR: $OUTPUT_DIR/events.json not found — run Stage 2B first (scripts/run_stage2b.sh)." >&2
  exit 1
fi

if [[ ! -f "$OUTPUT_DIR/predictions.json" ]]; then
  echo "ERROR: $OUTPUT_DIR/predictions.json not found — run Stage 1 first (scripts/run_stage1.sh)." >&2
  exit 1
fi

if [[ ! -d "$CLIP_DIR/img1" ]]; then
  echo "ERROR: clip frames not found: $CLIP_DIR/img1" >&2
  exit 1
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  for _conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$_conda_sh" ]]; then
      # shellcheck source=/dev/null
      source "$_conda_sh"
      conda activate tracklab 2>/dev/null && break
    fi
  done
fi
PYTHON="${GSR_PYTHON:-python}"

FPS="${FPS:-25}"
BEAM_DURATION_S="${BEAM_DURATION_S:-0.5}"
BEAM_ALPHA_MAX="${BEAM_ALPHA_MAX:-0.3}"
NO_TOPOLOGY_LINES="${NO_TOPOLOGY_LINES:-0}"

echo "=== Stage 3: Visual Effects (single clip) ==="
echo "  output_dir:   $OUTPUT_DIR"
echo "  clip_dir:     $CLIP_DIR"
echo "  sequence:     $SEQ_NAME"
echo "  fps:          $FPS"
echo "  beam_duration:${BEAM_DURATION_S}s"
echo "  beam_alpha:   $BEAM_ALPHA_MAX"
echo "  topology:     $([[ "$NO_TOPOLOGY_LINES" == "1" ]] && echo off || echo on)"
if [[ -f "$OUTPUT_DIR/homography_per_frame.json" ]]; then
  echo "  homography:   $OUTPUT_DIR/homography_per_frame.json"
else
  echo "  homography:   (none)"
fi
echo

"$PYTHON" - <<PY
from pathlib import Path
import logging

from pipeline.config import PipelineConfig
from pipeline.run import run_stage3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

clip_dir = Path("${CLIP_DIR}")
output_dir = Path("${OUTPUT_DIR}")

config = PipelineConfig(
    clip_dir=clip_dir,
    output_dir=output_dir,
    gsr_split=clip_dir.parent.name,
    fps=${FPS},
    beam_duration_s=${BEAM_DURATION_S},
    beam_alpha_max=${BEAM_ALPHA_MAX},
    topology_lines_enabled=${NO_TOPOLOGY_LINES} != 1,
    force=True,
)

run_stage3(config)
print(f"Stage 3 complete: {config.annotated_video}")
PY
