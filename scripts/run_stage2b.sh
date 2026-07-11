#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_stage2b.sh <output_dir> [--mode direct|hybrid] [--force] [--clip-dir DIR]
  bash scripts/run_stage2b.sh --input-dir <output_dir> [--mode direct|hybrid] [--force] [--clip-dir DIR]

Examples:
  bash scripts/run_stage2b.sh outputs/SNGS-116 --mode hybrid --force
  bash scripts/run_stage2b.sh --input-dir outputs/SNGS-116 --mode direct --force
USAGE
}

cd "$(dirname "$0")/.."

if [[ ${1-} == --help || ${1-} == -h ]]; then
  usage
  exit 0
fi

args=()
input_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir=*)
      input_dir="${1#*=}"
      shift
      ;;
    --input-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --input-dir requires a value" >&2
        usage >&2
        exit 1
      fi
      input_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${args[*]-}" && -z "$input_dir" ]]; then
  echo "error: missing output_dir" >&2
  usage >&2
  exit 1
fi

if [[ -n "$input_dir" ]]; then
  args+=("$input_dir")
fi

exec python -m pipeline.stage2b.run "${args[@]}"
