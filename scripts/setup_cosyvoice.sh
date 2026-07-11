#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f codes/CosyVoice/cosyvoice/cli/cosyvoice.py
test -f pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
test -f pretrained_models/Fun-CosyVoice3-0.5B/llm.pt
test -f pretrained_models/Fun-CosyVoice3-0.5B/flow.pt
test -f pretrained_models/Fun-CosyVoice3-0.5B/hift.pt
echo "CosyVoice3 source and model are ready."
