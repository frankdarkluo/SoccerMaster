"""Global pipeline configuration with unified input model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"

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
    beam_duration_s: float = 1.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True

    @property
    def predictions_json(self) -> Path:
        return self.existing_predictions_json or self.output_dir / "predictions.json"

    @property
    def homography_json(self) -> Path:
        return self.existing_homography_json or self.output_dir / "homography_per_frame.json"

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

    def should_run_stage1(self) -> bool:
        if self.existing_predictions_json and Path(self.existing_predictions_json).exists():
            return False
        if self.existing_pklz_path and Path(self.existing_pklz_path).exists():
            return True
        return self.force or not (self.output_dir / "predictions.json").exists()
