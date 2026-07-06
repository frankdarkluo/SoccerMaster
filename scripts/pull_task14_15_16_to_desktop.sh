#!/usr/bin/env bash
# Pull Task 14–16 files from remote rocky10 to local Mac Desktop repo.
#
# Run on your Mac:
#   bash scripts/pull_task14_15_16_to_desktop.sh
#   bash scripts/pull_task14_15_16_to_desktop.sh --apply
#
set -euo pipefail

SRC_HOST="${SRC_HOST:-knq-rocky10-frp}"
SRC_USER="${SRC_USER:-luoguoqing}"
SRC_PATH="${SRC_PATH:-/home/luoguoqing/SoccerMaster}"
DEST="${DEST:-/Users/gluo/Desktop/SoccerMaster}"
MANIFEST="${MANIFEST:-scripts/rsync_task14_15_16_manifest.txt}"

APPLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --dry-run) APPLY=0; shift ;;
    -h|--help)
      echo "Usage: $0 [--apply] [--dry-run]"
      echo "  DEST=${DEST}"
      echo "  SRC=${SRC_USER}@${SRC_HOST}:${SRC_PATH}"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$DEST" ]]; then
  echo "ERROR: DEST not found: $DEST" >&2
  exit 1
fi

if [[ ! -f "$DEST/$MANIFEST" && ! -f "./$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  echo "Sync manifest from remote first, or run from repo root." >&2
  exit 1
fi

MF="$DEST/$MANIFEST"
if [[ ! -f "$MF" ]]; then
  MF="./$MANIFEST"
fi

# Build --files-from list (skip comments/blank lines)
FILES_FROM="$(mktemp)"
grep -v '^[[:space:]]*#' "$MF" | grep -v '^[[:space:]]*$' > "$FILES_FROM"

RSYNC_OPTS=(-avh --info=progress2 --partial --human-readable)
if [[ "$APPLY" -eq 0 ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

echo "=== pull Task 14–16 files ==="
echo "source: ${SRC_USER}@${SRC_HOST}:${SRC_PATH}"
echo "dest:   ${DEST}"
echo "apply:  $([[ $APPLY -eq 1 ]] && echo yes || echo dry-run)"
echo "files:  $(wc -l < "$FILES_FROM")"
echo

mkdir -p "$DEST"
rsync "${RSYNC_OPTS[@]}" --files-from="$FILES_FROM" \
  "${SRC_USER}@${SRC_HOST}:${SRC_PATH}/" "${DEST}/"

rm -f "$FILES_FROM"

echo
if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run done. Re-run with --apply to transfer."
else
  echo "Done. Files landed under ${DEST}"
fi
