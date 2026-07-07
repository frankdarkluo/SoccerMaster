#!/usr/bin/env bash
# Targeted calibration recompute for one sequence (diagnostic for H1' vs H2).
#
# Recomputes ONLY [pitch, calibration, apply_camera_params] from the Step-2 pklz,
# saving to a FRESH state path so TrackLab's video-level cache does not skip it.
# This reproduces exactly what the original Step-3 run computed for calibration,
# without paying for reid / legibility / jersey(vLLM) / team.
#
# After it finishes, compare frames 326-564 with check_recompute.py:
#   block fills with keypoints -> H1' (stale cache);  still empty -> H2 (genuine).
#
# Usage:  bash pipeline/stage1_inference/recompute_calibration.sh [SEQ_ID]
set -euo pipefail

SEQ="${1:-148}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GSR="$REPO/codes/sn-gamestate"
OUT="$REPO/outputs/SNGS-${SEQ}"
LOAD="$OUT/step2/refined_sn-gamestate.pklz"
RUNDIR="$OUT/step3_recompute"
SAVE="$RUNDIR/states/sn-gamestate.pklz"

if [[ ! -f "$LOAD" ]]; then
  echo "ERROR: Step-2 input not found: $LOAD" >&2
  exit 1
fi
if [[ -f "$SAVE" ]]; then
  echo "ERROR: $SAVE already exists — video-level cache would SKIP the recompute." >&2
  echo "       Remove it first:  rm -rf '$RUNDIR'" >&2
  exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$GSR"

# Replicates run_gsr.py::_run — patch torch.load (weights_only default) before tracklab.
python -c '
import sys
from pipeline.stage1_inference.torch_compat import patch_torch_load
patch_torch_load()
import runpy
sys.argv = [
    "tracklab.main", "-cn", "gsr_step_3_example_accelerate_vllm",
    "experiment_subname=step_3_recompute_SNGS-'"$SEQ"'",
    "dataset.eval_set=test",
    "dataset.start_vid='"$SEQ"'",
    "dataset.end_vid='"$SEQ"'",
    "pipeline=[pitch,calibration,apply_camera_params]",
    "state.load_file='"$LOAD"'",
    "state.save_file='"$SAVE"'",
    "hydra.run.dir='"$RUNDIR"'",
]
runpy.run_module("tracklab.main", run_name="__main__", alter_sys=True)
'

echo
echo "Done. Now compare against the original:"
echo "  python pipeline/stage1_inference/check_recompute.py $SEQ"
