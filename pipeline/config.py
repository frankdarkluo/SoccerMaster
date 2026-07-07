"""Global pipeline configuration with unified input model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"
PRETRAINED_MODELS = GSR_ROOT / "pretrained_models"
REFERENCE_CSV = REPO_ROOT / "reference_football.csv"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_Y_HALF = GOAL_WIDTH / 2


@dataclass
class PipelineConfig:
    # --- Core input (required) ---
    clip_dir: Path = Path(".")
    output_dir: Path = Path("outputs/pipeline_run")

    # --- Optional: pre-existing intermediate artifacts (skip corresponding stages) ---
    existing_predictions_json: Optional[Path] = None
    existing_homography_json: Optional[Path] = None
    existing_pklz_path: Optional[Path] = None
    existing_events_json: Optional[Path] = None

    # --- General ---
    fps: int = 25
    llm_backend: str = "doubao"
    languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    roster_json: Optional[Path] = None
    force: bool = False

    # --- Stage 1 ---
    sequence_prefix: str = "SNGS-10001"
    gsr_split: str = "sn500"
    step3_config: str = "gsr_step_3_example_accelerate_vllm"
    input_video: Optional[Path] = None
    pklz_video_id: Optional[str] = None
    skip_sam2: bool = False
    sam2_propagation_margin: int = 50
    sam2_max_retries_per_segment: int = 2

    # --- Stage 2 ---
    event_importance_threshold: float = 0.5
    min_event_gap_s: float = 1.0
    ball_speed_shot_threshold_mps: float = 10.0
    verify_events: bool = False
    verify_backend: str = "doubao"
    verify_window_s: float = 0.5
    verify_model_path: str = str(
        PRETRAINED_MODELS / "jn" / "Qwen2.5-VL-7B-Instruct"
    )

    # --- Stage 3 ---
    beam_duration_s: float = 0.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True

    # --- Stage 4 ---
    llm_temperature: float = 0.7
    max_tokens: int = 4096

    # --- Stage 5 ---
    tts_backend: str = "doubao_tts"
    tts_language: str = "zh"

    @property
    def frames_dir(self) -> Path:
        return self.clip_dir / "img1"

    @property
    def predictions_json(self) -> Path:
        if self.existing_predictions_json:
            return self.existing_predictions_json
        return self.output_dir / "predictions.json"

    @property
    def homography_json(self) -> Path:
        if self.existing_homography_json:
            return self.existing_homography_json
        return self.output_dir / "homography_per_frame.json"

    @property
    def events_json(self) -> Path:
        if self.existing_events_json:
            return self.existing_events_json
        return self.output_dir / "events.json"

    @property
    def events_verification_json(self) -> Path:
        return self.output_dir / "events_verification.json"

    @property
    def topology_json(self) -> Path:
        return self.output_dir / "topo.json"

    @property
    def annotated_video(self) -> Path:
        return self.output_dir / "annotated_video.mp4"

    @property
    def commentary_json(self) -> Path:
        return self.output_dir / "commentary.json"

    @property
    def topdown_video(self) -> Path:
        return self.output_dir / "topdown_video.mp4"

    @property
    def commentary_audio(self) -> Path:
        return self.output_dir / f"commentary_{self.tts_language}.mp3"

    @property
    def raw_final_video(self) -> Path:
        """Annotated video + default-voice commentary (Stage 5 step 1)."""
        return self.output_dir / "raw_final_video.mp4"

    @property
    def final_video(self) -> Path:
        return self.output_dir / "final_video.mp4"

    def should_run_stage1(self) -> bool:
        if self.existing_predictions_json and Path(self.existing_predictions_json).exists():
            return False
        if self.existing_pklz_path and Path(self.existing_pklz_path).exists():
            return True
        if self.force:
            return True
        return not (self.output_dir / "predictions.json").exists()

    def should_run_stage2(self) -> bool:
        if self.existing_events_json and Path(self.existing_events_json).exists():
            return False
        if self.force:
            return True
        return not (self.output_dir / "events.json").exists()

    def should_run_stage3(self) -> bool:
        if self.force:
            return True
        return not self.annotated_video.exists()

    def should_run_stage4(self) -> bool:
        if self.force:
            return True
        return not self.commentary_json.exists()

    def should_run_stage5(self) -> bool:
        if self.force:
            return True
        return not self.final_video.exists()
