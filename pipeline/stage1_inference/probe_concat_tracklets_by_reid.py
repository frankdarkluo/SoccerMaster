#!/usr/bin/env python3
"""Fast smoke test for concat_tracklets_by_reid (no vLLM, ~seconds).

Reproduces the Step-3 crash:
  ValueError: array at index 0 has size 256 and array at index N has size 1

Root cause: np.mean over tracklet embeddings that include NaN rows yields a
scalar, which cannot be vstack'd with 256-dim ReID vectors.

Usage:
    python pipeline/stage1_inference/probe_concat_tracklets_by_reid.py [SEQ_ID]

Exits 0 if concat_tracklets_by_reid completes; 1 on failure.
"""
from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GSR = REPO / "codes" / "sn-gamestate"


def main() -> int:
    seq = sys.argv[1] if len(sys.argv) > 1 else "116"
    pklz = REPO / f"outputs/SNGS-{seq}/step2/refined_sn-gamestate.pklz"
    if not pklz.is_file():
        print(f"ERROR: missing step2 input: {pklz}")
        return 1

    sys.path.insert(0, str(GSR))
    from sn_gamestate.concat_tracklets_by_jn.concat_tracklets_by_jn_api import ConcatTrackletsByJN
    from sn_gamestate.concat_tracklets_by_reid.concat_tracklets_by_reid_api import ConcatTrackletsByReid

    with zipfile.ZipFile(pklz) as z:
        det = pickle.load(z.open(f"{seq}.pkl"))
        img = pickle.load(z.open(f"{seq}_image.pkl"))

    # Fake jersey numbers so concat_by_jn runs (same position as full Step 3).
    det = det.copy()
    det["jersey_number"] = [str(int(tid) % 99 + 1) for tid in det["track_id"]]

    print(f"Testing SNGS-{seq} step2 pklz ({len(det)} detections, {det.track_id.nunique()} tracks)")

    det = ConcatTrackletsByJN().process(det, img)
    print(f"  after concat_by_jn: {det.track_id.nunique()} tracks")

    try:
        out = ConcatTrackletsByReid(threshold=0.1).process(det, img)
    except ValueError as e:
        print(f"  concat_by_reid: FAILED — {e}")
        return 1

    print(f"  concat_by_reid: OK ({out.track_id.nunique()} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
