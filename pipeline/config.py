"""Global pipeline configuration with unified input model."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
LEGACY_DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"
GSR2024_ROOT = REPO_ROOT / "data" / "GSR-2024"
GSR2024_FRAMES_ROOT = GSR2024_ROOT / "images"
GSR2024_VIDEO_ROOT = GSR2024_ROOT / "videos"
LEGACY_VIDEO_ROOT = REPO_ROOT / "data" / "videos" / "SoccerNetGS_videos"


def default_gsr_frames_root() -> Path:
    """Prefer the flat GSR-2024 layout while it coexists with the legacy tree."""
    migrated = GSR2024_ROOT / "clip_manifest.jsonl"
    return (
        GSR2024_FRAMES_ROOT if migrated.is_file() else LEGACY_DATASET_ROOT
    )


def default_gsr_video_root() -> Path:
    migrated = GSR2024_ROOT / "clip_manifest.jsonl"
    return GSR2024_VIDEO_ROOT if migrated.is_file() else LEGACY_VIDEO_ROOT


def sequences_info_path(frames_root: Path) -> Path:
    """Return metadata beside either a dataset root or its flat images directory."""
    root = Path(frames_root)
    direct = root / "sequences_info.json"
    return direct if direct.is_file() else root.parent / "sequences_info.json"


def source_split_for_clip(frames_root: Path, clip_name: str) -> str | None:
    path = sequences_info_path(frames_root)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    preferred = ("train", "validation", "test", "challenge")
    keys = (*preferred, *(key for key in metadata if key not in preferred))
    for key in keys:
        entries = metadata.get(key, [])
        if isinstance(entries, list) and any(
            isinstance(entry, dict) and entry.get("name") == clip_name
            for entry in entries
        ):
            return key
    return None


def resolve_gsr_clip(frames_root: Path, clip_name: str, split: str | None = None) -> Path:
    """Resolve a clip from a flat root first, then from a legacy split tree."""
    root = Path(frames_root)
    direct = root / clip_name
    if direct.is_dir():
        return direct
    legacy_split = "valid" if split == "validation" else split
    if legacy_split and (root / legacy_split / clip_name).is_dir():
        return root / legacy_split / clip_name
    matches = sorted(root.glob(f"*/{clip_name}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple directories found for {clip_name}: {matches}")
    raise FileNotFoundError(f"No directory found for {clip_name} under {root}")


# Backward-compatible name used by video preprocessing.
DATASET_ROOT = default_gsr_frames_root()

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
    gsr_split: str | None = None
    gsr_frames_root: Path = field(default_factory=default_gsr_frames_root)
    gsr_video_root: Path = field(default_factory=default_gsr_video_root)
    step3_config: str = "gsr_step_3_example_accelerate"
    skip_sam2: bool = False

    # --- Effects ---
    event_importance_threshold: float = 0.5
    beam_duration_s: float = 1.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True

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

    def commentary_audio(self, language: str, default: bool = False) -> Path:
        marker = "_default" if default else ""
        return self.voice_dir / f"commentary_{language}{marker}.mp3"

    def raw_final_video(self, language: str) -> Path:
        suffix = "_en" if language == "en" else ""
        return self.voice_dir / f"raw_final_video{suffix}.mp4"

    def final_video(self, language: str) -> Path:
        suffix = "_en" if language == "en" else ""
        return self.voice_dir / f"final_video{suffix}.mp4"
