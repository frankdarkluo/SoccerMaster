#!/usr/bin/env bash
# One-time CosyVoice3 setup: clone repo + download Fun-CosyVoice3-0.5B weights (HuggingFace).
# vLLM is NOT required — plain PyTorch inference is used.
#
# Usage: bash scripts/setup_cosyvoice.sh
# Env:   COSYVOICE_REPO       (default codes/CosyVoice)
#        COSYVOICE_MODEL_DIR  (default pretrained_models/Fun-CosyVoice3-0.5B)
#        PYTHON               (default python; use a Python 3.10+ env)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_DIR="${COSYVOICE_REPO:-codes/CosyVoice}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "--- Cloning CosyVoice repo -> $REPO_DIR"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO_DIR"
fi

echo "--- Installing CosyVoice requirements"
"$PYTHON_BIN" -m pip install "setuptools<81" wheel
"$PYTHON_BIN" -m pip install --no-build-isolation --extra-index-url https://pypi.nvidia.com/ -r "$REPO_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 "torchvision==0.18.1+cu121"

echo "--- Downloading Fun-CosyVoice3-0.5B-2512 weights -> $MODEL_DIR"
"$PYTHON_BIN" - "$MODEL_DIR" <<'PY_DL'
import sys
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=sys.argv[1])
PY_DL

echo "CosyVoice ready: repo=$REPO_DIR model=$MODEL_DIR"
