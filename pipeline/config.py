"""Global pipeline configuration with a unified input model.

One dataset layout, rooted at `gsr_dataset_root`:

    <root>/images/<clip>/img1/000001.jpg   frames
    <root>/videos/<clip>.mp4               source video
    <root>/sequences_info.json             split membership and frame counts
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
GSR_DATASET_ROOT = REPO_ROOT / "data" / "GSR-2025"

# sequences_info.json names the validation split "validation"; TrackLab calls it "valid".
SPLITS = ("train", "validation", "test", "challenge")


def tracklab_split(split: str) -> str:
    return "valid" if split == "validation" else split


def source_split_for_clip(dataset_root: Path, clip_name: str) -> str:
    """Look a clip's split up in sequences_info.json."""
    path = Path(dataset_root) / "sequences_info.json"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    for split in SPLITS:
        entries = metadata.get(split, [])
        if any(entry.get("name") == clip_name for entry in entries):
            return split
    raise ValueError(f"{clip_name} is not listed in any split of {path}")


@dataclass
class PipelineConfig:
    # --- Core input (required) ---
    clip_dir: Path = Path(".")
    output_dir: Path = Path("outputs/pipeline_run")

    # --- General ---
    fps: int = 25
    force: bool = False

    # --- Stage 1 ---
    sequence_prefix: str = "SNGS-10001"
    gsr_dataset_root: Path = GSR_DATASET_ROOT
    gsr_split: str | None = None
    step3_config: str = "gsr_step_3_example_accelerate"
    skip_sam2: bool = False

    # --- Effects ---
    topology_lines_enabled: bool = True

    @property
    def gsr_frames_root(self) -> Path:
        return Path(self.gsr_dataset_root) / "images"

    @property
    def gsr_video_root(self) -> Path:
        return Path(self.gsr_dataset_root) / "videos"

    @property
    def predictions_json(self) -> Path:
        return self.output_dir / "predictions.json"

    @property
    def homography_json(self) -> Path:
        return self.output_dir / "homography_per_frame.json"

    @property
    def voice_dir(self) -> Path:
        return self.output_dir / "voice"

    @property
    def annotated_video(self) -> Path:
        return self.output_dir / "annotated_video.mp4"

    def commentary_audio(self, language: str) -> Path:
        return self.voice_dir / f"commentary_{language}.mp3"

    def final_video(self, language: str) -> Path:
        suffix = "_en" if language == "en" else ""
        return self.voice_dir / f"final_video{suffix}.mp4"
