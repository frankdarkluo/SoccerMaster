#!/usr/bin/env bash
# Pipeline Stage 4 only: LLM commentary generation for ONE clip.
# Reads:  <output_dir>/events.json (Stage 2 output)
#         optional: <output_dir>/annotated_video.mp4 or topdown_video.mp4 (visual context)
# Writes: <output_dir>/commentary.json
#
# Does NOT touch Stage 1–3 outputs or Stage 5 (TTS / final_video).
#
# Usage:
#   bash scripts/run_stage4.sh
#   bash scripts/run_stage4.sh outputs/SNGS-148
#   bash scripts/run_stage4.sh outputs/SNGS-148 codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148
#   LLM_BACKEND=doubao LANG="en zh" bash scripts/run_stage4.sh outputs/SNGS-148
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
    echo "       pass clip_dir explicitly: bash scripts/run_stage4.sh ${OUTPUT_DIR} <clip_dir>" >&2
    exit 1
  fi
fi

if [[ ! -f "$OUTPUT_DIR/events.json" ]]; then
  echo "ERROR: $OUTPUT_DIR/events.json not found — run Stage 2 first (scripts/run_stage2.sh)." >&2
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
LLM_BACKEND="${LLM_BACKEND:-doubao}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
ROSTER_JSON="${ROSTER_JSON:-}"
LANG="${LANG:-en zh}"

echo "=== Stage 4: LLM Commentary (single clip) ==="
echo "  output_dir:   $OUTPUT_DIR"
echo "  clip_dir:     $CLIP_DIR"
echo "  sequence:     $SEQ_NAME"
echo "  llm_backend:  $LLM_BACKEND"
echo "  languages:    $LANG"
echo "  temperature:  $LLM_TEMPERATURE"
if [[ -n "$ROSTER_JSON" ]]; then
  echo "  roster:       $ROSTER_JSON"
fi
if [[ -f "$OUTPUT_DIR/annotated_video.mp4" ]]; then
  echo "  visual:       $OUTPUT_DIR/annotated_video.mp4"
elif [[ -f "$OUTPUT_DIR/topdown_video.mp4" ]]; then
  echo "  visual:       $OUTPUT_DIR/topdown_video.mp4"
else
  echo "  visual:       (none — text-only prompt)"
fi
echo

"$PYTHON" - <<PY
from pathlib import Path
import logging

from pipeline.config import PipelineConfig
from pipeline.run import run_stage4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

clip_dir = Path("${CLIP_DIR}")
output_dir = Path("${OUTPUT_DIR}")
roster = "${ROSTER_JSON}".strip() or None

config = PipelineConfig(
    clip_dir=clip_dir,
    output_dir=output_dir,
    gsr_split=clip_dir.parent.name,
    fps=${FPS},
    llm_backend="${LLM_BACKEND}",
    languages="${LANG}".split(),
    llm_temperature=${LLM_TEMPERATURE},
    max_tokens=${MAX_TOKENS},
    roster_json=Path(roster) if roster else None,
    force=True,
)

run_stage4(config)
print(f"Stage 4 complete: {config.commentary_json}")
PY
