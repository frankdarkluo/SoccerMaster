#!/usr/bin/env python3
"""Compare original vs recomputed calibration to decide H1' (stale cache) vs H2 (genuine).

Reads keypoint counts / homography validity per frame from both the original Step-3
pklz and the targeted-recompute pklz, and reports the invalid blocks side by side.

Run AFTER recompute_calibration.sh:
    python pipeline/stage1_inference/check_recompute.py [SEQ_ID]
"""
from __future__ import annotations

import sys
import zipfile
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_image_df(pklz: Path, seq: str):
    with zipfile.ZipFile(pklz) as z:
        with z.open(f"{seq}_image.pkl") as f:
            return pickle.load(f)


def h_valid(h) -> bool:
    if h is None:
        return False
    try:
        a = np.array(h, dtype=float)
    except (TypeError, ValueError):
        return False
    return a.shape == (3, 3) and np.isfinite(a).all()


def n_kp(k) -> int:
    return len(k) if isinstance(k, dict) else 0


def invalid_blocks(frames_valid):
    """frames_valid: list of (frame, is_valid) -> list of (start, end) invalid runs."""
    blocks, start = [], None
    for fr, ok in frames_valid:
        if not ok and start is None:
            start = fr
        elif ok and start is not None:
            blocks.append((start, prev))
            start = None
        prev = fr
    if start is not None:
        blocks.append((start, prev))
    return blocks


def summarize(pklz: Path, seq: str, label: str):
    df = load_image_df(pklz, seq)
    rows = [(int(r["frame"]), n_kp(r["keypoints"]), h_valid(r["h"])) for _, r in df.iterrows()]
    fv = [(fr, ok) for fr, _, ok in rows]
    blocks = invalid_blocks(fv)
    n_valid = sum(1 for _, _, ok in rows if ok)
    print(f"\n=== {label}: {pklz} ===")
    print(f"frames={len(rows)}  valid_H={n_valid}  invalid_H={len(rows) - n_valid}")
    print(f"invalid blocks (frame ranges): {blocks}")
    # focus window 326-564
    win = [(fr, kp) for fr, kp, _ in rows if 326 <= fr <= 564]
    kps = [kp for _, kp in win]
    if kps:
        print(f"block 326-564: mean_kp={np.mean(kps):.2f}  zero_kp={sum(1 for k in kps if k == 0)}/{len(kps)}")
    return {fr: kp for fr, kp, _ in rows}


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "148"
    orig = REPO / f"outputs/SNGS-{seq}/step3/states/sn-gamestate.pklz"
    recomp = REPO / f"outputs/SNGS-{seq}/step3_recompute/states/sn-gamestate.pklz"

    o = summarize(orig, seq, "ORIGINAL")
    if not recomp.exists():
        print(f"\nRecompute not found: {recomp}\nRun recompute_calibration.sh {seq} first.")
        return
    r = summarize(recomp, seq, "RECOMPUTE")

    # verdict on block 326-564
    win = range(326, 565)
    o_mean = np.mean([o.get(f, 0) for f in win])
    r_mean = np.mean([r.get(f, 0) for f in win])
    print("\n================ VERDICT ================")
    print(f"block 326-564 mean keypoints:  original={o_mean:.2f}   recompute={r_mean:.2f}")
    if r_mean >= 4.0:
        print("=> H1' CONFIRMED: recompute FILLS the block. Original data was a stale/frozen")
        print("   artifact. Root cause = pipeline serving unrevalidated calibration across resumes.")
    elif r_mean < 1.0:
        print("=> H2 CONFIRMED: recompute is STILL empty. The keypoint model is genuinely blind")
        print("   on this span. Next: probe raw heatmap peaks; wire SequentialCalib; image-space fallback.")
    else:
        print("=> PARTIAL: recompute recovered some but not all. Inspect per-frame; likely threshold-marginal (H2).")


if __name__ == "__main__":
    main()
