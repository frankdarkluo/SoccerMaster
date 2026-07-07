#!/usr/bin/env bash
# Pipeline Stage 5 only: TTS voice synthesis + video mux for ONE clip.
# Reads:  <output_dir>/commentary.json (Stage 4)
#         <output_dir>/annotated_video.mp4 (Stage 3)
#         optional: <output_dir>/events.json (energy tiers)
# Writes: <output_dir>/commentary_{lang}_default.mp3, raw_final_video[_en].mp4
#         <output_dir>/commentary_{lang}.mp3, final_video.mp4
#
# Step 1: edge-tts default voice → raw_final_video (preview)
# Step 2: doubao_tts 王楚淇 clone → final_video.mp4
#
# Usage:
#   bash scripts/run_stage5.sh
#   bash scripts/run_stage5.sh outputs/SNGS-148
#   TTS_LANGUAGE=zh FORCE=1 bash scripts/run_stage5.sh outputs/SNGS-148
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-148}"
SEQ_NAME="$(basename "$OUTPUT_DIR")"

if [[ ! -f "$OUTPUT_DIR/commentary.json" ]]; then
  echo "ERROR: $OUTPUT_DIR/commentary.json not found — run Stage 4 first (scripts/run_stage4.sh)." >&2
  exit 1
fi

if [[ ! -f "$OUTPUT_DIR/annotated_video.mp4" ]]; then
  echo "ERROR: $OUTPUT_DIR/annotated_video.mp4 not found — run Stage 3 first." >&2
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

TTS_LANGUAGE="${TTS_LANGUAGE:-zh}"
FORCE="${FORCE:-0}"

FORCE_FLAG=()
if [[ "$FORCE" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

echo "=== Stage 5: TTS + Final Video (single clip) ==="
echo "  output_dir:   $OUTPUT_DIR"
echo "  sequence:     $SEQ_NAME"
echo "  language:     $TTS_LANGUAGE"
echo "  force:        $FORCE"
echo

"$PYTHON" -c "import edge_tts" 2>/dev/null || pip install edge-tts

echo "--- Step 1/2: default voice preview (edge-tts) ---"
"$PYTHON" -m pipeline.stage5_tts.make_raw_final_video \
  --output-dir "$OUTPUT_DIR" \
  --language "$TTS_LANGUAGE" \
  "${FORCE_FLAG[@]}"

echo
echo "--- Step 2/2: 王楚淇 clone voice (doubao_tts) ---"
"$PYTHON" -m pipeline.stage5_tts.make_final_video \
  --output-dir "$OUTPUT_DIR" \
  --language "$TTS_LANGUAGE" \
  "${FORCE_FLAG[@]}"

echo
echo "Stage 5 complete:"
echo "  preview:  $OUTPUT_DIR/raw_final_video.mp4"
echo "  final:    $OUTPUT_DIR/final_video.mp4"
