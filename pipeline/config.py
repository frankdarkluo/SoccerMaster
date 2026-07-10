"""Configuration shared by retained pipeline stages."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GSR_ROOT = REPO_ROOT / "codes" / "sn-gamestate"
DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE pairs without overriding the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key := key.strip():
            os.environ.setdefault(key, value.strip().strip("'\""))


@dataclass
class PipelineConfig:
    clip_dir: Path = Path(".")
    output_dir: Path = Path("outputs/pipeline_run")
    existing_predictions_json: Optional[Path] = None
    existing_homography_json: Optional[Path] = None
    existing_pklz_path: Optional[Path] = None
    fps: int = 25
    force: bool = False

    sequence_prefix: str = "SNGS-10001"
    gsr_split: str = "sn500"
    step3_config: str = "gsr_step_3_example_accelerate_vllm"
    input_video: Optional[Path] = None
    pklz_video_id: Optional[str] = None
    skip_sam2: bool = False
    sam2_propagation_margin: int = 50
    sam2_max_retries_per_segment: int = 2

    beam_duration_s: float = 0.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True
    event_importance_threshold: float = 0.5

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
    def events_json(self) -> Path:
        return self.output_dir / "events.json"

    @property
    def topology_json(self) -> Path:
        return self.output_dir / "topo.json"

    @property
    def annotated_video(self) -> Path:
        return self.output_dir / "annotated_video.mp4"

    def should_run_stage1(self) -> bool:
        if self.existing_predictions_json and self.existing_predictions_json.exists():
            return False
        if self.existing_pklz_path and self.existing_pklz_path.exists():
            return True
        return self.force or not (self.output_dir / "predictions.json").exists()

    def should_run_stage3(self) -> bool:
        return self.force or not self.annotated_video.exists()
