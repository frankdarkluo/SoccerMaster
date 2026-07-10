#!/usr/bin/env bash
# Stage 2B + Stage 5: direct video commentary and both voices in one output dir.
# Usage: bash scripts/run_stage2b.sh outputs/SNGS-116 [clip_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-116}"
SEQ_NAME="$(basename "$OUTPUT_DIR")"
CLIP_DIR="${2:-codes/sn-gamestate/datasets/SoccerNetGS/test/${SEQ_NAME}}"
PYTHON_BIN="${PYTHON:-python3}"
TTS_LANGUAGE="${TTS_LANGUAGE:-zh}"
FORCE="${FORCE:-0}"

for command in "$PYTHON_BIN" ffmpeg ffprobe; do
  command -v "$command" >/dev/null 2>&1 || { echo "ERROR: command not found: $command" >&2; exit 1; }
done
[[ -f "$OUTPUT_DIR/predictions.json" ]] || { echo "ERROR: Stage 1 predictions missing: $OUTPUT_DIR/predictions.json" >&2; exit 1; }
[[ -d "$CLIP_DIR/img1" ]] || { echo "ERROR: clip frames missing: $CLIP_DIR/img1" >&2; exit 1; }

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
FORCE_FLAG=()
[[ "$FORCE" == "1" ]] && FORCE_FLAG=(--force)

"$PYTHON_BIN" -m pipeline.stage2b.run \
  --clip-dir "$CLIP_DIR" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --output-dir "$OUTPUT_DIR" \
  "${FORCE_FLAG[@]}"

"$PYTHON_BIN" -m pipeline.stage5_tts.run \
  --output-dir "$OUTPUT_DIR" \
  --language "$TTS_LANGUAGE" \
  --voice both \
  "${FORCE_FLAG[@]}"
