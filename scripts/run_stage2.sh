#!/usr/bin/env bash
# Pipeline Stage 2 only: event detection (+ optional verify) + enrichment for ONE clip.
# Reads:  <output_dir>/predictions.json, <output_dir>/homography_per_frame.json (Stage 1 output)
# Writes: <output_dir>/events_detected.json, <output_dir>/events.json
#
# Does NOT touch Stage 1 (predictions.json) or downstream outputs
# (annotated_video.mp4, commentary, final_video).
#
# Usage:
#   bash scripts/run_stage2.sh
#   bash scripts/run_stage2.sh outputs/SNGS-148
#   VERIFY_EVENTS=1 VERIFY_BACKEND=doubao bash scripts/run_stage2.sh outputs/SNGS-148
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-148}"

if [[ ! -f "$OUTPUT_DIR/predictions.json" ]]; then
  echo "ERROR: $OUTPUT_DIR/predictions.json not found — run Stage 1 first (scripts/run_stage1.sh)." >&2
  exit 1
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

# nohup/non-interactive shells often leave conda on base (no torch). Activate tracklab.
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
VERIFY_EVENTS="${VERIFY_EVENTS:-0}"
VERIFY_BACKEND="${VERIFY_BACKEND:-doubao}"
VERIFY_WINDOW_S="${VERIFY_WINDOW_S:-0.5}"
MIN_EVENT_GAP_S="${MIN_EVENT_GAP_S:-1.0}"
BALL_SPEED_SHOT_THRESHOLD_MPS="${BALL_SPEED_SHOT_THRESHOLD_MPS:-10.0}"

SEQ_NAME="$(basename "$OUTPUT_DIR")"

echo "=== Stage 2: SoccerMaster Event Detection (single clip) ==="
echo "  output_dir:   $OUTPUT_DIR"
echo "  sequence:     $SEQ_NAME"
echo "  fps:          $FPS"
echo "  verify:       $VERIFY_EVENTS (backend=$VERIFY_BACKEND, window=${VERIFY_WINDOW_S}s)"
echo "  min_gap_s:    $MIN_EVENT_GAP_S"
echo "  shot_thresh:  ${BALL_SPEED_SHOT_THRESHOLD_MPS} m/s"
echo

"$PYTHON" - <<PY
from pathlib import Path
import logging

from pipeline.config import PipelineConfig
from pipeline.run import run_stage2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

output_dir = Path("${OUTPUT_DIR}")

config = PipelineConfig(
    clip_dir=output_dir,
    output_dir=output_dir,
    fps=${FPS},
    verify_events=${VERIFY_EVENTS} == 1,
    verify_backend="${VERIFY_BACKEND}",
    verify_window_s=${VERIFY_WINDOW_S},
    min_event_gap_s=${MIN_EVENT_GAP_S},
    ball_speed_shot_threshold_mps=${BALL_SPEED_SHOT_THRESHOLD_MPS},
    force=True,
)

n_events = run_stage2(config)
print(f"Stage 2 complete: {n_events} events -> {config.events_json}")
PY
