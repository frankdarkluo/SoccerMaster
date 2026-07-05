#!/usr/bin/env bash
# Stage 5 TTS: default-voice preview → 王楚淇 clone
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import edge_tts" 2>/dev/null || pip install edge-tts

# Step 1：默认音色预览
python -m pipeline.stage5_tts.make_raw_final_video --output-dir outputs/SNGS-148

# Step 2：王楚淇克隆音
python -m pipeline.stage5_tts.make_final_video --output-dir outputs/SNGS-148
