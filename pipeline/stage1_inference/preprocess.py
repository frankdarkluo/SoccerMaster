"""Convert raw video to GSR directory format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pipeline.config import DATASET_ROOT
from pipeline.utils.video import extract_frames, get_video_info


def _update_sequences_info(seq_info_path: Path, split: str, sequence_name: str, n_frames: int) -> None:
    existing: dict = {}
    if seq_info_path.exists():
        with open(seq_info_path, encoding="utf-8") as f:
            existing = json.load(f)

    split_entries = list(existing.get(split, []))
    updated = False
    for entry in split_entries:
        if entry.get("name") == sequence_name:
            entry["n_frames"] = n_frames
            updated = True
            break
    if not updated:
        next_id = max((entry.get("id", -1) for entry in split_entries), default=-1) + 1
        split_entries.append({"id": next_id, "name": sequence_name, "n_frames": n_frames})

    existing[split] = split_entries
    with open(seq_info_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def preprocess_video(
    video_path: Path,
    sequence_name: str = "SNGS-10001",
    split: str = "sn500",
    dataset_root: Optional[Path] = None,
) -> dict:
    """
    Extract frames from video and create GSR directory structure.

    Returns dict with sequence info: name, n_frames, fps, seq_dir.
    """
    video_path = Path(video_path)
    root = Path(dataset_root) if dataset_root else DATASET_ROOT
    info = get_video_info(video_path)
    seq_dir = root / split / sequence_name / "img1"
    n_frames = extract_frames(video_path, seq_dir)

    seq_info_path = root / "sequences_info.json"
    _update_sequences_info(seq_info_path, split, sequence_name, n_frames)

    return {
        "name": sequence_name,
        "n_frames": n_frames,
        "fps": info["fps"],
        "seq_dir": str(seq_dir),
        "width": info["width"],
        "height": info["height"],
    }
