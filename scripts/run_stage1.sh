#!/usr/bin/env bash
# Pipeline Stage 1 only: SoccerMaster GSR inference for ONE clip (longest step).
# Detection + tracking → SAM2 → calibration / jersey / team → predictions.json
#
# Writes only: predictions.json, homography_per_frame.json, and step{1,2,3}/ GSR artifacts.
# Does NOT touch downstream outputs (events.json, annotated_video.mp4, commentary, final_video).
# Step 3 uses vLLM + Qwen2.5-VL-7B-Instruct-AWQ (gsr_step_3_example_accelerate_vllm).
# Run Stage 2B next; Stage 3 is optional and Stage 5 is driven by Stage 2B.
#
# The clip_dir path selects both the split and the sequence, e.g.:
#   .../test/SNGS-148  → only test/SNGS-148
#   .../sn500/SNGS-148 → only sn500/SNGS-148
#
# Usage:
#   bash scripts/run_stage1.sh
#   bash scripts/run_stage1.sh \
#     codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148 \
#     outputs/SNGS-148
#
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_CLIP="codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148"
if [[ ! -d "$DEFAULT_CLIP/img1" ]]; then
  DEFAULT_CLIP="codes/sn-gamestate/datasets/SoccerNetGS/sn500/SNGS-148"
fi

CLIP_DIR="${1:-$DEFAULT_CLIP}"
OUTPUT_DIR="${2:-outputs/SNGS-148}"
INPUT_VIDEO="${INPUT_VIDEO:-}"

if [[ ! -d "$CLIP_DIR" ]]; then
  echo "ERROR: clip dir not found: $CLIP_DIR" >&2
  exit 1
fi

N_FRAMES=0
if [[ -d "$CLIP_DIR/img1" ]]; then
  N_FRAMES=$(find "$CLIP_DIR/img1" -maxdepth 1 -name '*.jpg' | wc -l)
fi

if [[ "$N_FRAMES" -eq 0 ]]; then
  if [[ -z "$INPUT_VIDEO" ]]; then
    for cand in \
      "codes/gsr_tasks/doubao/clips_small/$(basename "$CLIP_DIR").mp4" \
      "codes/gsr_tasks/chatgpt_batch_clips/$(basename "$CLIP_DIR").mp4"
    do
      if [[ -f "$cand" && -s "$cand" ]]; then
        INPUT_VIDEO="$cand"
        break
      fi
    done
  fi
  if [[ -z "$INPUT_VIDEO" || ! -s "$INPUT_VIDEO" ]]; then
    echo "ERROR: no frames in $CLIP_DIR/img1 and no non-empty input video." >&2
    echo "Set INPUT_VIDEO=/path/to/clip.mp4" >&2
    exit 1
  fi
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
export GSR_PYTHON="$(command -v "$PYTHON")"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "ERROR: torch not found in Python ($("$PYTHON" --version 2>&1))." >&2
  echo "Activate the tracklab env first, e.g.:  conda activate tracklab" >&2
  echo "Or set GSR_PYTHON=/path/to/tracklab/bin/python" >&2
  exit 1
fi
echo "  python:      $("$PYTHON" --version 2>&1) ($GSR_PYTHON)"

# CPU limits — tracklab yaml defaults use 32–64 workers and can SIGKILL shared hosts.
# Raise only if the machine has headroom (e.g. GSR_NUM_CORES=8 GSR_NUM_THREADS=4).
export GSR_NUM_CORES="${GSR_NUM_CORES:-8}"
export GSR_NUM_THREADS="${GSR_NUM_THREADS:-4}"
export GPU_LIST="${GPU_LIST:-0}"
export MAX_PROCESSES_PER_GPU="${MAX_PROCESSES_PER_GPU:-1}"

SEQ_NAME="$(basename "$CLIP_DIR")"
SPLIT_NAME="$(basename "$(dirname "$CLIP_DIR")")"

echo "=== Stage 1: SoccerMaster Inference (single clip) ==="
echo "  clip_dir:    $CLIP_DIR"
echo "  split:       $SPLIT_NAME"
echo "  sequence:    $SEQ_NAME"
echo "  output_dir:  $OUTPUT_DIR"
echo "  gsr_out:     $OUTPUT_DIR/step{1,2,3}/"
echo "  cpu_cores:   $GSR_NUM_CORES (GSR_NUM_CORES)"
echo "  cpu_threads: $GSR_NUM_THREADS (GSR_NUM_THREADS)"
echo "  gpu:         $GPU_LIST (max $MAX_PROCESSES_PER_GPU proc/GPU)"
echo "  frames:      $N_FRAMES"
if [[ "$N_FRAMES" -gt 0 ]]; then
  echo "  preprocess:  skip (using existing img1/)"
else
  echo "  input_video: $INPUT_VIDEO"
fi
echo "  (longest step — detection / SAM2 / identity, ONLY this clip)"
echo

"$PYTHON" - <<PY
from pathlib import Path
import logging
import os

from pipeline.config import PipelineConfig
from pipeline.run import run_stage1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

clip_dir = Path("${CLIP_DIR}")
output_dir = Path("${OUTPUT_DIR}")
input_video_s = "${INPUT_VIDEO}"
input_video = Path(input_video_s) if input_video_s else None
if input_video is not None and (not input_video.exists() or input_video.stat().st_size == 0):
    input_video = None

config = PipelineConfig(
    clip_dir=clip_dir,
    output_dir=output_dir,
    input_video=input_video,
    sequence_prefix=clip_dir.name,
    gsr_split=clip_dir.parent.name,
    force=False,
)

run_stage1(config)
print(f"Stage 1 complete: {config.predictions_json}")
if config.homography_json.exists():
    print(f"Homography:      {config.homography_json}")
PY
