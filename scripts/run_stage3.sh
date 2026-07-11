#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m pipeline.stage3_tts.run "$@"
