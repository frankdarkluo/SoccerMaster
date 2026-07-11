#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-116}"

# Ponytail defaults. Override by env if needed.
STAGE3_LANGUAGE="${STAGE3_LANGUAGE:-zh}"
STAGE3_VOICE="${STAGE3_VOICE:-both}"
STAGE3_FORCE="${STAGE3_FORCE:-1}"
STAGE3_PROMPT_WAV="${STAGE3_PROMPT_WAV:-}"
STAGE3_PROMPT_TEXT="${STAGE3_PROMPT_TEXT:-}"

cmd=(
  python -m pipeline.stage3_tts.run "$OUTPUT_DIR"
  --language "$STAGE3_LANGUAGE"
  --voice "$STAGE3_VOICE"
)

if [[ "${STAGE3_FORCE,,}" == "1" || "${STAGE3_FORCE,,}" == "true" ]]; then
  cmd+=(--force)
fi

if [[ -n "$STAGE3_PROMPT_WAV" ]]; then
  cmd+=(--prompt-wav "$STAGE3_PROMPT_WAV")
fi

if [[ -n "$STAGE3_PROMPT_TEXT" ]]; then
  cmd+=(--prompt-text "$STAGE3_PROMPT_TEXT")
fi

exec "${cmd[@]}"
