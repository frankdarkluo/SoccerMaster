"""Structural guard: detect anomalous calibration dropout in Step-3 pklz output.

Flags sustained blocks where homography is invalid AND keypoint counts collapse
relative to the video baseline. Prevents silent acceptance of bad calibration.
"""
from __future__ import annotations

import json
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Minimum consecutive invalid-h frames to report as a block.
MIN_BLOCK_FRAMES = 5
# Block mean keypoints must fall below this fraction of video baseline to flag.
DROPOUT_RATIO = 0.25
# Absolute ceiling: blocks with mean kp above this are not flagged as dropout.
MAX_DROPOUT_MEAN_KP = 2.0


@dataclass
class InvalidBlock:
    start_frame: int
    end_frame: int
    n_frames: int
    mean_keypoints: float
    zero_keypoint_frames: int
    invalid_homography_frames: int


@dataclass
class CalibrationReport:
    video_id: str
    n_frames: int
    valid_homography_frames: int
    baseline_mean_keypoints: float
    invalid_blocks: list[InvalidBlock]
    flagged: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["invalid_blocks"] = [asdict(b) for b in self.invalid_blocks]
        return d


def _h_valid(h) -> bool:
    if h is None:
        return False
    if isinstance(h, float) and np.isnan(h):
        return False
    try:
        arr = np.array(h, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.shape == (3, 3) and np.isfinite(arr).all()


def _n_keypoints(kp) -> int:
    return len(kp) if isinstance(kp, dict) else 0


def _invalid_h_blocks(frames: list[tuple[int, bool]]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    prev = 0
    for fr, invalid in frames:
        if invalid and start is None:
            start = fr
        elif not invalid and start is not None:
            blocks.append((start, prev))
            start = None
        prev = fr
    if start is not None:
        blocks.append((start, prev))
    return blocks


def analyze_image_df(image_df, video_id: str = "") -> CalibrationReport:
    rows = []
    for _, row in image_df.iterrows():
        fr = int(row["frame"])
        kp = _n_keypoints(row.get("keypoints"))
        h_ok = _h_valid(row.get("h"))
        rows.append((fr, kp, h_ok))

    if not rows:
        return CalibrationReport(
            video_id=video_id, n_frames=0, valid_homography_frames=0,
            baseline_mean_keypoints=0.0, invalid_blocks=[], flagged=False,
        )

    all_kps = [kp for _, kp, _ in rows]
    valid_kps = [kp for _, kp, h_ok in rows if h_ok]
    baseline = float(np.mean(valid_kps)) if valid_kps else float(np.mean(all_kps))

    invalid_frames = [(fr, not h_ok) for fr, _, h_ok in rows]
    blocks = _invalid_h_blocks(invalid_frames)

    invalid_blocks: list[InvalidBlock] = []
    for start, end in blocks:
        block_rows = [(fr, kp, h_ok) for fr, kp, h_ok in rows if start <= fr <= end]
        if len(block_rows) < MIN_BLOCK_FRAMES:
            continue
        kps = [kp for _, kp, _ in block_rows]
        mean_kp = float(np.mean(kps))
        if mean_kp > MAX_DROPOUT_MEAN_KP:
            continue
        if baseline > 0 and mean_kp / baseline > DROPOUT_RATIO:
            continue
        invalid_blocks.append(InvalidBlock(
            start_frame=start,
            end_frame=end,
            n_frames=len(block_rows),
            mean_keypoints=mean_kp,
            zero_keypoint_frames=sum(1 for k in kps if k == 0),
            invalid_homography_frames=len(block_rows),
        ))

    n_valid_h = sum(1 for _, _, h_ok in rows if h_ok)
    return CalibrationReport(
        video_id=video_id,
        n_frames=len(rows),
        valid_homography_frames=n_valid_h,
        baseline_mean_keypoints=baseline,
        invalid_blocks=invalid_blocks,
        flagged=len(invalid_blocks) > 0,
    )


def load_image_df(pklz_path: Path, video_id: str):
    with zipfile.ZipFile(pklz_path) as z:
        with z.open(f"{video_id}_image.pkl") as f:
            return pickle.load(f)


def analyze_pklz(pklz_path: Path, video_id: str) -> CalibrationReport:
    return analyze_image_df(load_image_df(pklz_path, video_id), video_id=video_id)


def write_report(report: CalibrationReport, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return output_path


def check_pklz(pklz_path: Path, video_id: str, report_path: Path | None = None) -> CalibrationReport:
    """Analyze pklz and optionally write calibration_report.json. Logs warnings if flagged."""
    import logging
    log = logging.getLogger(__name__)

    report = analyze_pklz(pklz_path, video_id)
    if report_path is not None:
        write_report(report, report_path)

    if report.flagged:
        for block in report.invalid_blocks:
            log.warning(
                "Calibration dropout block frames %d-%d: mean_kp=%.2f (baseline=%.2f), "
                "%d/%d zero-keypoint frames",
                block.start_frame, block.end_frame, block.mean_keypoints,
                report.baseline_mean_keypoints,
                block.zero_keypoint_frames, block.n_frames,
            )
    else:
        log.info(
            "Calibration guard OK: %d/%d valid-H frames, baseline mean_kp=%.2f",
            report.valid_homography_frames, report.n_frames, report.baseline_mean_keypoints,
        )
    return report
