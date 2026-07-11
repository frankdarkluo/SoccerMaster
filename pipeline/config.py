"""Global pipeline configuration with unified input model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"
PRETRAINED_MODELS = GSR_ROOT / "pretrained_models"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_Y_HALF = GOAL_WIDTH / 2


@dataclass
class PipelineConfig:
    # --- Core input (required) ---
    clip_dir: Path = Path(".")
    output_dir: Path = Path("outputs/pipeline_run")

    # --- Optional Stage 1 inputs ---
    existing_predictions_json: Optional[Path] = None
    existing_homography_json: Optional[Path] = None
    existing_pklz_path: Optional[Path] = None

    # --- General ---
    fps: int = 25
    languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    force: bool = False

    # --- Stage 1 ---
    sequence_prefix: str = "SNGS-10001"
    gsr_split: str = "sn500"
    step3_config: str = "gsr_step_3_example_accelerate"
    input_video: Optional[Path] = None
    pklz_video_id: Optional[str] = None
    skip_sam2: bool = False
    sam2_propagation_margin: int = 50
    sam2_max_retries_per_segment: int = 2

    # --- Effects ---
    event_importance_threshold: float = 0.5
    beam_duration_s: float = 0.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True

    # --- Stage 2B ---
    commentary_mode: str = "hybrid"
    snapshot_hz: float = 2.0
    radar_hz: float = 1.0
    llm_max_images: int = 32

    @property
    def frames_dir(self) -> Path:
        return self.clip_dir / "img1"

    @property
    def predictions_json(self) -> Path:
        return self.existing_predictions_json or self.output_dir / "predictions.json"

    @property
    def homography_json(self) -> Path:
        return self.existing_homography_json or self.output_dir / "homography_per_frame.json"

    @property
    def comments_dir(self) -> Path:
        return self.output_dir / "comments"

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

    @property
    def events_json(self) -> Path:
        return self.comments_dir / "events.json"

    @property
    def event_spine_json(self) -> Path:
        return self.comments_dir / "event_spine.json"

    @property
    def commentary_direct_json(self) -> Path:
        return self.comments_dir / "commentary_direct.json"

    @property
    def tactical_candidates_json(self) -> Path:
        return self.comments_dir / "tactical_candidates.json"

    @property
    def commentary_json(self) -> Path:
        return self.comments_dir / "commentary.json"

    @property
    def relations_json(self) -> Path:
        return self.comments_dir / "relations.json"

    @property
    def radar_dir(self) -> Path:
        return self.comments_dir / "radar"

    def should_run_stage1(self) -> bool:
        if self.existing_predictions_json and Path(self.existing_predictions_json).exists():
            return False
        if self.existing_pklz_path and Path(self.existing_pklz_path).exists():
            return True
        return self.force or not (self.output_dir / "predictions.json").exists()
