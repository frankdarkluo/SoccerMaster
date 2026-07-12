# AI Football Commentary Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged pipeline that takes a 30s raw soccer video through SoccerMaster inference → event detection → visual effects → LLM commentary generation.

**Architecture:** Four decoupled stages with persistent JSON intermediates. Rule engine detects events and fills computable tags; LLM fills visual tags, generates commentary, and performs tactical reasoning. Visual effects (light beams + topology lines) are rendered on the original video via homography back-projection.

**Tech Stack:** Python 3.10, OpenCV, NumPy, SciPy, matplotlib, ffmpeg, Qwen2.5-VL (local), OpenAI GPT-5 API, Doubao API. Reuses SoccerMaster/TrackLab/formation_topology.

**Spec:** `docs/superpowers/specs/2026-07-01-ai-commentary-pipeline-design.md`

---

## File Map

### New files to create

```
pipeline/
├── __init__.py
├── config.py
├── run.py
├── stage1_inference/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── run_gsr.py
│   └── pklz_to_json.py
├── stage2_events/
│   ├── __init__.py
│   ├── schema.py
│   ├── detector.py
│   └── enricher.py
├── stage3_effects/
│   ├── __init__.py
│   ├── projection.py
│   ├── light_beam.py
│   ├── tactical_lines.py
│   └── render.py
├── stage4_commentary/
│   ├── __init__.py
│   ├── prompt_builder.py
│   ├── llm_adapter.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── qwen_local.py
│   │   ├── doubao_api.py
│   │   └── openai_api.py
│   └── postprocess.py
└── utils/
    ├── __init__.py
    ├── pitch.py
    ├── video.py
    └── clip_index_to_events.py    # Convert clip_index.csv row → events.json fixture
```

### Test files

```
tests/pipeline/
├── test_config.py
├── test_pklz_to_json.py
├── test_schema.py
├── test_detector.py
├── test_enricher.py
├── test_projection.py
├── test_light_beam.py
├── test_tactical_lines.py
├── test_prompt_builder.py
├── test_llm_adapter.py
└── test_postprocess.py
```

### Existing files to reference (read-only, not modify)

- `compare_step3_pred_gt.py` — pklz loading logic to reuse
- `formation_topology/possession.py` — nearest-player-to-ball
- `formation_topology/pitch.py` — pitch constants
- `formation_topology/lines.py` — depth-line analysis
- `formation_topology/pipeline.py` — `analyze()` orchestration
- `formation_topology/io_gamestate.py` — Labels-GameState JSON parsing
- `reference_football.csv` — event ontology
- `codes/sn-gamestate/run_train_SNGS-061.sh` — GSR invocation pattern
- `codes/sn-gamestate/datasets/SoccerNetGS/test/clip_index.csv` — event timestamps for test clips (validation + fixture)
- `codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148/Labels-GameState.json` — primary dev target GT labels

### Primary development clips

- **SNGS-148** (test set, goal at t≈25s): Stage 3/4 development — has GT labels (zero-conversion predictions) + clip_index.csv event
- **SNGS-061** (train set): Stage 1 development — has sn-gamestate.pklz for pklz→JSON testing

---

## Day 1 Tasks (6h): Scaffold + pklz→JSON + clip_index→events Fixture

### Parallel tracks start here

- **Track A**: `pklz_to_json.py` (for SNGS-061, Stage 1 foundation)
- **Track B**: `clip_index_to_events.py` (events fixture for SNGS-148, enables Stage 3/4 development)

### Task 1: Pipeline scaffold + config

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `pipeline/utils/__init__.py`
- Create: `pipeline/utils/pitch.py`
- Create: `pipeline/utils/video.py`
- Create: `pipeline/utils/clip_index_to_events.py`
- Create: `pipeline/stage1_inference/__init__.py`
- Create: `pipeline/stage2_events/__init__.py`
- Create: `pipeline/stage3_effects/__init__.py`
- Create: `pipeline/stage4_commentary/__init__.py`
- Create: `pipeline/stage4_commentary/adapters/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p pipeline/{stage1_inference,stage2_events,stage3_effects,stage4_commentary/adapters,utils}
touch pipeline/__init__.py
touch pipeline/stage1_inference/__init__.py
touch pipeline/stage2_events/__init__.py
touch pipeline/stage3_effects/__init__.py
touch pipeline/stage4_commentary/__init__.py
touch pipeline/stage4_commentary/adapters/__init__.py
touch pipeline/utils/__init__.py
```

- [ ] **Step 2: Write `pipeline/config.py`**

```python
"""Global pipeline configuration with unified input model."""
from pathlib import Path
from dataclasses import dataclass, field
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
    clip_dir: Path = Path(".")           # Parent dir containing img1/ (e.g., test/SNGS-148/)
    output_dir: Path = Path("outputs/pipeline_run")

    # --- Optional: pre-existing intermediate artifacts (skip corresponding stages) ---
    existing_predictions_json: Optional[Path] = None   # Skip Stage 1 entirely
    existing_homography_json: Optional[Path] = None     # Skip Stage 1 homography step
    existing_pklz_path: Optional[Path] = None           # Skip GSR inference, only convert pklz→JSON
    existing_events_json: Optional[Path] = None         # Skip Stage 2

    # --- General ---
    fps: int = 25
    llm_backend: str = "openai"          # "qwen_local" | "doubao" | "openai"
    languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    roster_json: Optional[Path] = None
    force: bool = False                  # Re-run all stages even if outputs exist

    # --- Stage 1 ---
    sequence_prefix: str = "SNGS-10001"
    gsr_split: str = "sn500"
    step3_config: str = "gsr_step_3_example_accelerate"

    # --- Stage 2 ---
    event_importance_threshold: float = 0.5
    min_event_gap_s: float = 1.0
    ball_speed_shot_threshold_mps: float = 10.0

    # --- Stage 3 ---
    beam_duration_s: float = 0.5
    beam_alpha_max: float = 0.3
    topology_lines_enabled: bool = True

    # --- Stage 4 ---
    llm_temperature: float = 0.7
    max_tokens: int = 4096

    @property
    def frames_dir(self) -> Path:
        """Raw frames directory (clip_dir/img1/)."""
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
    def annotated_video(self) -> Path:
        return self.output_dir / "annotated_video.mp4"

    @property
    def commentary_json(self) -> Path:
        return self.output_dir / "commentary.json"

    @property
    def topdown_video(self) -> Path:
        return self.output_dir / "topdown_video.mp4"

    def should_run_stage1(self) -> bool:
        """Whether Stage 1 needs to run (no pre-existing predictions)."""
        if self.force:
            return True
        if self.existing_predictions_json and Path(self.existing_predictions_json).exists():
            return False
        return not (self.output_dir / "predictions.json").exists()

    def should_run_stage2(self) -> bool:
        """Whether Stage 2 needs to run (no pre-existing events)."""
        if self.force:
            return True
        if self.existing_events_json and Path(self.existing_events_json).exists():
            return False
        return not (self.output_dir / "events.json").exists()
```

- [ ] **Step 3: Write `pipeline/utils/pitch.py`**

```python
"""Re-export pitch constants from formation_topology."""
from formation_topology.pitch import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    canonicalize,
)

GOAL_WIDTH = 7.32
GOAL_Y_HALF = GOAL_WIDTH / 2
GOAL_X = PITCH_LENGTH / 2          # 52.5m from center
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
SIX_YARD_LENGTH = 5.5
SIX_YARD_WIDTH = 18.32
```

- [ ] **Step 4: Write `pipeline/utils/video.py`**

```python
"""Video I/O helpers."""
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, quality: int = 2) -> int:
    """Extract all frames from video as JPEG. Returns frame count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-q:v", str(quality),
        str(output_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(output_dir.glob("*.jpg")))


def get_video_info(video_path: Path) -> dict:
    """Return fps, duration_s, width, height, total_frames."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    import json
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps_parts = vs["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    duration = float(data["format"]["duration"])
    return {
        "fps": fps,
        "duration_s": duration,
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "total_frames": int(duration * fps),
    }


def encode_video(frame_dir: Path, output_path: Path, fps: float = 25.0):
    """Encode directory of numbered JPEGs to MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

- [ ] **Step 5: Write `pipeline/utils/clip_index_to_events.py`**

```python
"""Convert clip_index.csv rows to events.json fixture format."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

ACTION_TO_EVENT_CODE = {
    "shoot": "football.shoot",
    "goal": "football.goal",
    "clearance": "football.clearance",
    "foul": "football.foul",
    "corner": "football.set_piece",
    "free_kick": "football.set_piece",
    "kick_off": "football.set_piece",
    "penalty": "football.set_piece",
    "offside": None,           # Not supported in V1
    "yellow_card": None,       # Not supported in V1
    "substitution": None,      # Not supported in V1
}

ACTION_IMPORTANCE = {
    "goal": 1.0,
    "shoot": 0.55,
    "penalty": 0.9,
    "clearance": 0.4,
    "foul": 0.35,
    "corner": 0.2,
    "free_kick": 0.2,
    "kick_off": 0.1,
}


def clip_index_row_to_event(
    sample_id: str,
    event_time_sec: float,
    normalized_action: str,
    fps: int = 25,
) -> Optional[dict]:
    """Convert one clip_index.csv row to an events.json event dict."""
    event_code = ACTION_TO_EVENT_CODE.get(normalized_action)
    if event_code is None:
        return None

    timestamp_s = round(event_time_sec)
    frame_id = int(timestamp_s * fps)

    return {
        "event_id": "evt_001",
        "timestamp_s": float(timestamp_s),
        "frame_id": frame_id,
        "event_code": event_code,
        "display_name_en": normalized_action.replace("_", " ").title(),
        "display_name_cn": _action_cn(normalized_action),
        "importance": ACTION_IMPORTANCE.get(normalized_action, 0.3),
        "player_jersey": None,
        "player_team": None,
        "tags": {},
        "confidence": 1.0,
        "description_hint": f"from clip_index.csv: {sample_id}",
    }


def _action_cn(action: str) -> str:
    mapping = {
        "shoot": "射门", "goal": "进球", "clearance": "解围",
        "foul": "犯规", "corner": "角球", "free_kick": "任意球",
        "kick_off": "开球", "penalty": "点球",
    }
    return mapping.get(action, action)


def generate_events_fixture(
    clip_index_csv: Path,
    sample_id: str,
    output_path: Path,
    fps: int = 25,
) -> Path:
    """Generate events.json fixture for a specific test clip from clip_index.csv."""
    events = []
    with open(clip_index_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample_id"].strip() == sample_id:
                ev = clip_index_row_to_event(
                    sample_id=row["sample_id"].strip(),
                    event_time_sec=float(row["event_time_sec"]),
                    normalized_action=row["normalized_action"].strip(),
                    fps=fps,
                )
                if ev:
                    events.append(ev)

    data = {
        "video_info": {
            "source": f"{sample_id}/img1/",
            "fps": fps,
            "duration_s": 30.0,
            "total_frames": fps * 30,
        },
        "schema_version": "v3-20260319",
        "events": events,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path
```

- [ ] **Step 6: Verify scaffold imports work**

```bash
cd /home/luoguoqing/SoccerMaster
python -c "from pipeline.config import PipelineConfig; c = PipelineConfig(); print(c.output_dir, c.should_run_stage1())"
```

Expected: `outputs/pipeline_run True`

- [ ] **Step 7: Generate SNGS-148 events fixture**

```bash
python -c "
from pipeline.utils.clip_index_to_events import generate_events_fixture
from pathlib import Path
p = generate_events_fixture(
    Path('codes/sn-gamestate/datasets/SoccerNetGS/test/clip_index.csv'),
    'SNGS-148',
    Path('outputs/SNGS-148/events.json'),
)
print('Generated:', p)
import json
with open(p) as f:
    print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
"
```

Expected: events.json with one `football.goal` event at t=25s

- [ ] **Step 8: Commit scaffold**

```bash
git add pipeline/ tests/
git commit -m "feat(pipeline): scaffold, unified config, clip_index→events fixture"
```

---

### Task 2: pklz → JSON converter (Track A)

**Files:**
- Create: `pipeline/stage1_inference/pklz_to_json.py`
- Create: `tests/pipeline/test_pklz_to_json.py`
- Reference: `compare_step3_pred_gt.py` (lines 152-179 for pklz loading)

- [ ] **Step 1: Write the test**

```python
# tests/pipeline/test_pklz_to_json.py
import json
import pytest
from pathlib import Path

SAMPLE_PKLZ = Path("codes/sn-gamestate/outputs/gsr/step_3_train_SNGS-061/states/sn-gamestate.pklz")


@pytest.mark.skipif(not SAMPLE_PKLZ.exists(), reason="No pklz file available")
class TestPklzToJson:
    def test_predictions_json_has_required_keys(self, tmp_path):
        from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json
        pred_path, homo_path = convert_pklz_to_json(
            pklz_path=SAMPLE_PKLZ,
            video_id="001",
            output_dir=tmp_path,
        )
        with open(pred_path) as f:
            data = json.load(f)
        assert "info" in data
        assert "images" in data
        assert "annotations" in data
        assert len(data["images"]) > 0
        assert len(data["annotations"]) > 0

    def test_annotations_have_bbox_pitch(self, tmp_path):
        from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json
        pred_path, _ = convert_pklz_to_json(
            pklz_path=SAMPLE_PKLZ,
            video_id="001",
            output_dir=tmp_path,
        )
        with open(pred_path) as f:
            data = json.load(f)
        ann = data["annotations"][0]
        assert "bbox_pitch" in ann
        assert "x_bottom_middle" in ann["bbox_pitch"]
        assert "y_bottom_middle" in ann["bbox_pitch"]
        assert "attributes" in ann
        assert "role" in ann["attributes"]

    def test_homography_json_has_frames(self, tmp_path):
        from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json
        _, homo_path = convert_pklz_to_json(
            pklz_path=SAMPLE_PKLZ,
            video_id="001",
            output_dir=tmp_path,
        )
        with open(homo_path) as f:
            data = json.load(f)
        assert "frames" in data
        first_key = list(data["frames"].keys())[0]
        assert "H" in data["frames"][first_key]
        assert "H_inv" in data["frames"][first_key]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_pklz_to_json.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.stage1_inference.pklz_to_json'`

- [ ] **Step 3: Implement `pklz_to_json.py`**

```python
# pipeline/stage1_inference/pklz_to_json.py
"""Convert TrackLab pklz state to Labels-GameState-compatible JSON + homography export."""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np


def _pitch_xy(bbox_pitch) -> dict | None:
    if bbox_pitch is None:
        return None
    if isinstance(bbox_pitch, float) and np.isnan(bbox_pitch):
        return None
    if not isinstance(bbox_pitch, dict):
        return None
    x = bbox_pitch.get("x_bottom_middle")
    y = bbox_pitch.get("y_bottom_middle")
    if x is None or y is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return {"x_bottom_middle": float(x), "y_bottom_middle": float(y)}


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def convert_pklz_to_json(
    pklz_path: Path,
    video_id: str,
    output_dir: Path,
    fps: int = 25,
) -> Tuple[Path, Path]:
    """
    Convert pklz tracker state to predictions.json + homography_per_frame.json.

    Returns (predictions_json_path, homography_json_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pklz_path = Path(pklz_path)

    with zipfile.ZipFile(pklz_path) as z:
        with z.open(f"{video_id}.pkl") as f:
            detections_df = pickle.load(f)
        with z.open(f"{video_id}_image.pkl") as f:
            image_df = pickle.load(f)

    # Build images list
    images = []
    for idx, row in image_df.iterrows():
        image_id = str(idx)
        file_path = row.get("file_path", "")
        file_name = Path(file_path).name if file_path else f"{int(idx):06d}.jpg"
        images.append({"image_id": image_id, "file_name": file_name})

    # Build annotations list — Labels-GameState-compatible format
    annotations = []
    ann_id = 0
    for _, row in detections_df.iterrows():
        bp = _pitch_xy(row.get("bbox_pitch"))
        if bp is None:
            continue
        # bbox_image: dict format matching Labels-GameState.json
        bbox_ltwh = row.get("bbox_ltwh")
        bbox_image = None
        if bbox_ltwh is not None:
            l, t, w, h = [_to_serializable(v) for v in bbox_ltwh]
            bbox_image = {
                "x": l, "y": t, "w": w, "h": h,
                "x_center": l + w / 2, "y_center": t + h / 2,
            }
        # Determine category
        role = row.get("role")
        cat_id = {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4}.get(role, 7)
        supercategory = "object"
        ann_id += 1
        annotations.append({
            "id": str(ann_id),
            "image_id": str(row["image_id"]),
            "track_id": int(row["track_id"]),
            "supercategory": supercategory,
            "category_id": cat_id,
            "bbox_image": bbox_image,
            "bbox_pitch": bp,
            "attributes": {
                "role": role,
                "team": row.get("team"),
                "jersey": str(row.get("jersey_number", "")) if row.get("jersey_number") is not None else "",
            },
        })

    predictions = {
        "info": {
            "name": f"SNGS-{video_id}",
            "n_frames": len(images),
            "fps": fps,
        },
        "images": images,
        "annotations": annotations,
    }

    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, default=_to_serializable)

    # Build homography per frame
    homography_data = {"frames": {}}
    for idx, row in image_df.iterrows():
        image_id = str(idx)
        h = row.get("h")
        if h is not None and not (isinstance(h, float) and np.isnan(h)):
            h_arr = np.array(h, dtype=float)
            try:
                h_inv = np.linalg.inv(h_arr)
                homography_data["frames"][image_id] = {
                    "H": h_arr.tolist(),
                    "H_inv": h_inv.tolist(),
                    "valid": True,
                }
            except np.linalg.LinAlgError:
                homography_data["frames"][image_id] = {"H": None, "H_inv": None, "valid": False}
        else:
            homography_data["frames"][image_id] = {"H": None, "H_inv": None, "valid": False}

    homo_path = output_dir / "homography_per_frame.json"
    with open(homo_path, "w", encoding="utf-8") as f:
        json.dump(homography_data, f, indent=2)

    return pred_path, homo_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/pipeline/test_pklz_to_json.py -v
```

Expected: PASS (or SKIP if no pklz available)

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage1_inference/pklz_to_json.py tests/pipeline/test_pklz_to_json.py
git commit -m "feat(pipeline): pklz to Labels-GameState JSON + homography converter"
```

---

## Day 2 Tasks (6h): Track A — Stage 2 Core | Track B — Stage 3 Light Beam

### Task 3: Video preprocessor (moved to Day 4)

**Files:**
- Create: `pipeline/stage1_inference/preprocess.py`

- [ ] **Step 1: Implement `preprocess.py`**

```python
# pipeline/stage1_inference/preprocess.py
"""Convert raw video to GSR directory format."""
import json
from pathlib import Path

from pipeline.config import DATASET_ROOT
from pipeline.utils.video import extract_frames, get_video_info


def preprocess_video(
    video_path: Path,
    sequence_name: str = "SNGS-10001",
    split: str = "sn500",
) -> dict:
    """
    Extract frames from video and create GSR directory structure.

    Returns dict with sequence info: name, n_frames, fps, seq_dir.
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)
    seq_dir = DATASET_ROOT / split / sequence_name / "img1"
    n_frames = extract_frames(video_path, seq_dir)

    # Write sequences_info.json
    seq_info = {
        split: [{"id": 0, "name": sequence_name, "n_frames": n_frames}]
    }
    seq_info_path = DATASET_ROOT / "sequences_info.json"
    with open(seq_info_path, "w") as f:
        json.dump(seq_info, f, indent=2)

    return {
        "name": sequence_name,
        "n_frames": n_frames,
        "fps": info["fps"],
        "seq_dir": str(seq_dir),
        "width": info["width"],
        "height": info["height"],
    }
```

- [ ] **Step 2: Quick smoke test**

```bash
python -c "
from pipeline.stage1_inference.preprocess import preprocess_video
from pathlib import Path
# Just verify the function is importable and has correct signature
import inspect
sig = inspect.signature(preprocess_video)
print('Parameters:', list(sig.parameters.keys()))
"
```

Expected: `Parameters: ['video_path', 'sequence_name', 'split']`

- [ ] **Step 3: Commit**

```bash
git add pipeline/stage1_inference/preprocess.py
git commit -m "feat(pipeline): video preprocessor for GSR format conversion"
```

### Task 4: GSR pipeline wrapper

**Files:**
- Create: `pipeline/stage1_inference/run_gsr.py`

- [ ] **Step 1: Implement `run_gsr.py`**

```python
# pipeline/stage1_inference/run_gsr.py
"""Wrap SoccerMaster Steps 1-3 as subprocess calls."""
import logging
import subprocess
from pathlib import Path

from pipeline.config import GSR_ROOT, PipelineConfig

log = logging.getLogger(__name__)


def run_step1(config: PipelineConfig) -> Path:
    """Run YOLO detection + tracking. Returns pklz directory."""
    log.info("Running GSR Step 1: Detection & Tracking")
    cmd = ["python", "-m", "tracklab.main", "-cn", "gsr_step_1_example"]
    subprocess.run(cmd, check=True, cwd=str(GSR_ROOT))
    return GSR_ROOT / "outputs" / "gsr" / "step_1_sn500" / "states"


def run_step2(config: PipelineConfig) -> Path:
    """Run SAM2 segmentation refinement. Returns refined pklz directory."""
    log.info("Running GSR Step 2: SAM2 Segmentation")
    sam2_dir = GSR_ROOT.parent / "sam2" / "step_2"
    subprocess.run(["bash", "gsr_step2_example.sh"], check=True, cwd=str(sam2_dir))
    subprocess.run(["bash", "merge_example.sh"], check=True, cwd=str(sam2_dir))
    return sam2_dir.parent / "outputs"


def run_step3(config: PipelineConfig) -> Path:
    """Run calibration + jersey + team. Returns final pklz path."""
    log.info("Running GSR Step 3: Calibration & Identity (%s)", config.step3_config)
    cmd = ["python", "-m", "tracklab.main", "-cn", config.step3_config]
    subprocess.run(cmd, check=True, cwd=str(GSR_ROOT))
    return GSR_ROOT / "outputs" / "gsr" / f"step_3_{config.gsr_split}" / "states" / "sn-gamestate.pklz"


def run_full_gsr(config: PipelineConfig) -> Path:
    """Run all 3 GSR steps sequentially. Returns final pklz path."""
    run_step1(config)
    run_step2(config)
    return run_step3(config)
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/stage1_inference/run_gsr.py
git commit -m "feat(pipeline): GSR Steps 1-3 subprocess wrapper"
```

---

## Day 3 Tasks (6h): Track A — Stage 2 Tuning | Track B — Stage 3 Full Render

### Task 5: Event schema from CSV (moved to Day 2 Track A)

**Files:**
- Create: `pipeline/stage2_events/schema.py`
- Create: `tests/pipeline/test_schema.py`

- [ ] **Step 1: Write the test**

```python
# tests/pipeline/test_schema.py
import pytest
from pipeline.stage2_events.schema import EventSchema


@pytest.fixture
def schema():
    return EventSchema()


class TestEventSchema:
    def test_loads_core_events(self, schema):
        core = schema.core_events()
        codes = [e.code for e in core]
        assert "football.goal" in codes
        assert "football.shoot" in codes
        assert "football.pass" in codes

    def test_get_event_returns_fields(self, schema):
        ev = schema.get_event("football.goal")
        assert ev is not None
        assert ev.display_name_cn == "进球"
        assert ev.display_name_en == "Goal"
        assert ev.importance_base == 1.0
        assert ev.source_type == "model_direct"

    def test_events_by_source_type(self, schema):
        direct = schema.events_by_source_type("model_direct")
        composed = schema.events_by_source_type("rule_composed")
        assert len(direct) > len(composed)

    def test_computable_tag_groups(self, schema):
        groups = schema.computable_tag_groups()
        assert "pitch_zone" in groups
        assert "shot_distance" in groups

    def test_visual_tag_groups(self, schema):
        groups = schema.visual_tag_groups()
        assert "body_part" in groups
        assert "foot_technique" in groups
        assert "pitch_zone" not in groups

    def test_tag_vocabulary_for_prompt(self, schema):
        vocab = schema.tag_vocabulary_for_prompt("football.shoot")
        assert "body_part" in vocab
        assert "shot_posture" in vocab
        assert "右脚" in vocab

    def test_event_definitions_for_prompt(self, schema):
        defs = schema.event_definitions_for_prompt()
        assert "football.goal" in defs
        assert "进球" in defs
        assert "Goal" in defs
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_schema.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement `schema.py`**

```python
# pipeline/stage2_events/schema.py
"""Load reference_football.csv into a structured event ontology."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import REFERENCE_CSV


@dataclass
class TagValue:
    code: str
    display_name_cn: str
    display_name_en: str
    importance_modifier: float = 0.0


@dataclass
class TagGroup:
    code: str
    display_name_cn: str
    display_name_en: str
    applies_to: List[str] = field(default_factory=list)
    exclusive: bool = True
    values: List[TagValue] = field(default_factory=list)


@dataclass
class EventDef:
    event_id: str
    code: str
    family: str
    display_name_cn: str
    display_name_en: str
    description: str
    level_hint: str
    source_type: str
    importance_base: float
    tags: List[str]
    trigger_notes: str
    negative_flag: bool


COMPUTABLE_TAGS = {"pitch_zone", "shot_distance", "pass_distance", "pass_direction", "pattern_of_play"}


class EventSchema:
    def __init__(self, csv_path: Path = REFERENCE_CSV):
        self._events: Dict[str, EventDef] = {}
        self._tag_groups: Dict[str, TagGroup] = {}
        self._load(csv_path)

    def _load(self, csv_path: Path):
        current_tag_group: Optional[str] = None
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = (row.get("event_id") or "").strip()
                if not eid or eid.startswith("===") or eid.startswith("---"):
                    continue
                if eid in ("Events", "Sports", "Broadcast"):
                    continue

                if eid == "_TAG_GROUP":
                    code = (row.get("code") or "").strip()
                    applies_raw = (row.get("description") or "")
                    applies_to = [a.strip() for a in applies_raw.replace("applies_to:", "").split(",") if a.strip()]
                    exclusive = (row.get("level_hint") or "").strip() == "exclusive"
                    self._tag_groups[code] = TagGroup(
                        code=code,
                        display_name_cn=(row.get("display_name_cn") or "").strip(),
                        display_name_en=(row.get("display_name_en") or "").strip(),
                        applies_to=applies_to,
                        exclusive=exclusive,
                    )
                    current_tag_group = code
                    continue

                if eid == "_TAG_VALUE" and current_tag_group:
                    code = (row.get("code") or "").strip()
                    imp = row.get("importance_base") or "0"
                    try:
                        imp_f = float(imp)
                    except ValueError:
                        imp_f = 0.0
                    self._tag_groups[current_tag_group].values.append(TagValue(
                        code=code,
                        display_name_cn=(row.get("display_name_cn") or "").strip(),
                        display_name_en=(row.get("display_name_en") or "").strip(),
                        importance_modifier=imp_f,
                    ))
                    continue

                if eid.startswith("football."):
                    current_tag_group = None
                    tags_raw = (row.get("tags") or "")
                    tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
                    imp = row.get("importance_base") or "0"
                    try:
                        imp_f = float(imp)
                    except ValueError:
                        imp_f = 0.0
                    neg = bool(row.get("negative_flag") or "")
                    self._events[eid] = EventDef(
                        event_id=eid,
                        code=(row.get("code") or "").strip(),
                        family=(row.get("family") or "").strip(),
                        display_name_cn=(row.get("display_name_cn") or "").strip(),
                        display_name_en=(row.get("display_name_en") or "").strip(),
                        description=(row.get("description") or "").strip(),
                        level_hint=(row.get("level_hint") or "").strip(),
                        source_type=(row.get("source_type") or "").strip(),
                        importance_base=imp_f,
                        tags=tags_list,
                        trigger_notes=(row.get("trigger_notes") or "").strip(),
                        negative_flag=neg,
                    )

    def get_event(self, event_id: str) -> Optional[EventDef]:
        return self._events.get(event_id)

    def core_events(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "core"]

    def narrative_events(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "narrative"]

    def event_qualifiers(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "" and e.importance_base > 0]

    def events_by_source_type(self, source_type: str) -> List[EventDef]:
        return [e for e in self._events.values() if e.source_type == source_type]

    def computable_tag_groups(self) -> List[str]:
        return sorted(COMPUTABLE_TAGS & set(self._tag_groups.keys()))

    def visual_tag_groups(self) -> List[str]:
        return sorted(set(self._tag_groups.keys()) - COMPUTABLE_TAGS)

    def get_tag_group(self, group_code: str) -> Optional[TagGroup]:
        return self._tag_groups.get(group_code)

    def tag_vocabulary_for_prompt(self, event_id: str) -> str:
        ev = self.get_event(event_id)
        if ev is None:
            return ""
        base_code = ev.code
        lines = []
        for tg_code, tg in self._tag_groups.items():
            if base_code in tg.applies_to or any(base_code.startswith(a) for a in tg.applies_to):
                pick = "pick one" if tg.exclusive else "pick any"
                vals = ", ".join(f"{v.code} ({v.display_name_cn})" for v in tg.values)
                lines.append(f"{tg_code} ({pick}): {vals}")
        return "\n".join(lines)

    def event_definitions_for_prompt(self) -> str:
        lines = []
        for eid, ev in sorted(self._events.items()):
            lines.append(f"{eid}: {ev.description} ({ev.display_name_cn} / {ev.display_name_en})")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/pipeline/test_schema.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/schema.py tests/pipeline/test_schema.py
git commit -m "feat(pipeline): event schema loading from reference_football.csv"
```

### Task 6: Core event detector

**Files:**
- Create: `pipeline/stage2_events/detector.py`
- Create: `tests/pipeline/test_detector.py`

- [ ] **Step 1: Write the test**

```python
# tests/pipeline/test_detector.py
import json
import pytest
from pathlib import Path
from pipeline.stage2_events.detector import EventDetector
from pipeline.stage2_events.schema import EventSchema

SAMPLE_PREDICTIONS = Path("outputs/pipeline_run/predictions.json")


@pytest.fixture
def detector():
    return EventDetector(EventSchema(), fps=25)


class TestEventDetector:
    def test_detect_returns_list(self, detector, tmp_path):
        """Minimal smoke test with synthetic data."""
        pred = {
            "info": {"fps": 25, "n_frames": 50},
            "images": [{"image_id": str(i), "file_name": f"{i:06d}.jpg"} for i in range(1, 51)],
            "annotations": [],
        }
        pred_path = tmp_path / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(pred, f)
        events = detector.detect(str(pred_path))
        assert isinstance(events, list)

    def test_event_has_required_fields(self, detector, tmp_path):
        """Synthetic data with a ball near goal to trigger shot detection."""
        annotations = []
        ann_id = 0
        for frame in range(1, 51):
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": str(frame),
                "track_id": 99,
                "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
                "attributes": {"role": "ball", "team": None, "jersey": ""},
            })
        pred = {
            "info": {"fps": 25, "n_frames": 50},
            "images": [{"image_id": str(i), "file_name": f"{i:06d}.jpg"} for i in range(1, 51)],
            "annotations": annotations,
        }
        pred_path = tmp_path / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(pred, f)
        events = detector.detect(str(pred_path))
        for ev in events:
            assert hasattr(ev, "timestamp_s")
            assert hasattr(ev, "frame_id")
            assert hasattr(ev, "event_code")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_detector.py -v
```

- [ ] **Step 3: Implement `detector.py`**

This is a large file. Core structure:

```python
# pipeline/stage2_events/detector.py
"""Rule-based event detection from predictions.json."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.schema import EventSchema
from pipeline.utils.pitch import GOAL_X, GOAL_Y_HALF, PENALTY_AREA_LENGTH

@dataclass
class Event:
    event_id: str
    timestamp_s: float
    frame_id: int
    event_code: str
    display_name_en: str
    display_name_cn: str
    importance: float
    player_jersey: Optional[str] = None
    player_team: Optional[str] = None
    target_jersey: Optional[str] = None
    target_team: Optional[str] = None
    ball_speed_mps: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    description_hint: str = ""

    def to_dict(self) -> dict:
        d = {
            "event_id": self.event_id,
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_id": self.frame_id,
            "event_code": self.event_code,
            "display_name_en": self.display_name_en,
            "display_name_cn": self.display_name_cn,
            "importance": self.importance,
            "player_jersey": self.player_jersey,
            "player_team": self.player_team,
            "tags": self.tags,
            "confidence": round(self.confidence, 2),
            "description_hint": self.description_hint,
        }
        if self.target_jersey:
            d["target_jersey"] = self.target_jersey
            d["target_team"] = self.target_team
        if self.ball_speed_mps is not None:
            d["ball_speed_mps"] = round(self.ball_speed_mps, 1)
        return d


@dataclass
class FrameData:
    frame_id: int
    ball_xy: Optional[Tuple[float, float]] = None
    players: List[dict] = field(default_factory=list)


class EventDetector:
    def __init__(self, schema: EventSchema, fps: int = 25,
                 shot_speed_threshold: float = 10.0, min_gap_s: float = 1.0):
        self.schema = schema
        self.fps = fps
        self.shot_speed_threshold = shot_speed_threshold
        self.min_gap_s = min_gap_s
        self._event_counter = 0

    def detect(self, predictions_json_path: str) -> List[Event]:
        frames = self._load_frames(predictions_json_path)
        if not frames:
            return []

        ball_positions = self._extract_ball_positions(frames)
        ball_velocities = self._compute_velocities(ball_positions)
        possession_chain = self._compute_possession(frames, ball_positions)

        raw_events: List[Event] = []
        raw_events += self._detect_passes(possession_chain, frames)
        raw_events += self._detect_shots(ball_velocities, ball_positions, frames)
        raw_events += self._detect_clearances(ball_velocities, ball_positions, possession_chain, frames)
        raw_events += self._detect_interceptions(possession_chain, frames)

        # rule_composed events
        raw_events += self._detect_assists(raw_events)

        deduped = self._deduplicate(raw_events)
        return sorted(deduped, key=lambda e: e.timestamp_s)

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:03d}"

    def _load_frames(self, path: str) -> List[FrameData]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        frames_dict: Dict[int, FrameData] = {}
        for img in data.get("images", []):
            fid = int(img["image_id"])
            frames_dict[fid] = FrameData(frame_id=fid)
        for ann in data.get("annotations", []):
            fid = int(ann["image_id"])
            if fid not in frames_dict:
                frames_dict[fid] = FrameData(frame_id=fid)
            bp = ann.get("bbox_pitch")
            if not isinstance(bp, dict):
                continue
            x = bp.get("x_bottom_middle")
            y = bp.get("y_bottom_middle")
            if x is None or y is None:
                continue
            attrs = ann.get("attributes", {}) or {}
            role = attrs.get("role", "other")
            if role == "ball":
                frames_dict[fid].ball_xy = (float(x), float(y))
            else:
                frames_dict[fid].players.append({
                    "track_id": ann.get("track_id"),
                    "x": float(x), "y": float(y),
                    "role": role,
                    "team": attrs.get("team"),
                    "jersey": attrs.get("jersey", ""),
                })
        return [frames_dict[k] for k in sorted(frames_dict.keys())]

    def _extract_ball_positions(self, frames: List[FrameData]) -> Dict[int, Tuple[float, float]]:
        return {f.frame_id: f.ball_xy for f in frames if f.ball_xy is not None}

    def _compute_velocities(self, ball_pos: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        vels = {}
        sorted_ids = sorted(ball_pos.keys())
        for i in range(1, len(sorted_ids)):
            f1, f2 = sorted_ids[i - 1], sorted_ids[i]
            dt = (f2 - f1) / self.fps
            if dt <= 0:
                continue
            dx = ball_pos[f2][0] - ball_pos[f1][0]
            dy = ball_pos[f2][1] - ball_pos[f1][1]
            vels[f2] = math.hypot(dx, dy) / dt
        return vels

    def _compute_possession(self, frames: List[FrameData],
                            ball_pos: Dict[int, Tuple[float, float]]) -> Dict[int, Optional[dict]]:
        possession = {}
        for f in frames:
            if f.ball_xy is None or not f.players:
                possession[f.frame_id] = None
                continue
            bx, by = f.ball_xy
            nearest = min(f.players, key=lambda p: math.hypot(p["x"] - bx, p["y"] - by))
            dist = math.hypot(nearest["x"] - bx, nearest["y"] - by)
            possession[f.frame_id] = nearest if dist < 5.0 else None
        return possession

    def _detect_passes(self, possession: Dict[int, Optional[dict]],
                       frames: List[FrameData]) -> List[Event]:
        events = []
        sorted_ids = sorted(possession.keys())
        prev_player = None
        for fid in sorted_ids:
            curr = possession[fid]
            if curr is None:
                prev_player = None
                continue
            if prev_player is not None and curr["track_id"] != prev_player["track_id"]:
                if curr["team"] == prev_player["team"] and curr["team"] is not None:
                    ev_def = self.schema.get_event("football.pass")
                    events.append(Event(
                        event_id=self._next_id(),
                        timestamp_s=fid / self.fps,
                        frame_id=fid,
                        event_code="football.pass",
                        display_name_en=ev_def.display_name_en if ev_def else "Pass",
                        display_name_cn=ev_def.display_name_cn if ev_def else "传球",
                        importance=ev_def.importance_base if ev_def else 0.15,
                        player_jersey=prev_player.get("jersey"),
                        player_team=prev_player.get("team"),
                        target_jersey=curr.get("jersey"),
                        target_team=curr.get("team"),
                        confidence=0.7,
                        description_hint="pass detected via possession switch",
                    ))
            prev_player = curr
        return events

    def _detect_shots(self, velocities: Dict[int, float],
                      ball_pos: Dict[int, Tuple[float, float]],
                      frames: List[FrameData]) -> List[Event]:
        events = []
        for fid, speed in velocities.items():
            if speed < self.shot_speed_threshold:
                continue
            if fid not in ball_pos:
                continue
            bx, by = ball_pos[fid]
            toward_goal = abs(bx) > (GOAL_X - PENALTY_AREA_LENGTH)
            if not toward_goal:
                continue
            ev_def = self.schema.get_event("football.shoot")
            frame_data = next((f for f in frames if f.frame_id == fid), None)
            shooter = None
            if frame_data:
                possession = self._compute_possession([frame_data], {fid: (bx, by)})
                shooter = possession.get(fid)
            events.append(Event(
                event_id=self._next_id(),
                timestamp_s=fid / self.fps,
                frame_id=fid,
                event_code="football.shoot",
                display_name_en=ev_def.display_name_en if ev_def else "Shot",
                display_name_cn=ev_def.display_name_cn if ev_def else "射门",
                importance=ev_def.importance_base if ev_def else 0.55,
                player_jersey=shooter.get("jersey") if shooter else None,
                player_team=shooter.get("team") if shooter else None,
                ball_speed_mps=speed,
                confidence=min(speed / 20.0, 1.0),
                description_hint=f"shot detected: ball speed {speed:.1f} m/s near goal",
            ))
        return events

    def _detect_clearances(self, velocities, ball_pos, possession, frames):
        events = []
        for fid, speed in velocities.items():
            if speed < 8.0 or fid not in ball_pos:
                continue
            bx, _ = ball_pos[fid]
            prev_fid = fid - 1
            if prev_fid in ball_pos:
                prev_bx, _ = ball_pos[prev_fid]
                moving_away = abs(bx) < abs(prev_bx)
                in_own_half = abs(prev_bx) > 20
                if moving_away and in_own_half:
                    ev_def = self.schema.get_event("football.clearance")
                    events.append(Event(
                        event_id=self._next_id(),
                        timestamp_s=fid / self.fps,
                        frame_id=fid,
                        event_code="football.clearance",
                        display_name_en=ev_def.display_name_en if ev_def else "Clearance",
                        display_name_cn=ev_def.display_name_cn if ev_def else "解围",
                        importance=ev_def.importance_base if ev_def else 0.4,
                        ball_speed_mps=speed,
                        confidence=0.6,
                        description_hint="clearance: ball moving away from goal at high speed",
                    ))
        return events

    def _detect_interceptions(self, possession, frames):
        events = []
        sorted_ids = sorted(possession.keys())
        prev_player = None
        for fid in sorted_ids:
            curr = possession[fid]
            if curr is None:
                prev_player = None
                continue
            if prev_player is not None and curr["track_id"] != prev_player["track_id"]:
                if curr["team"] != prev_player["team"] and curr["team"] is not None and prev_player["team"] is not None:
                    ev_def = self.schema.get_event("football.interception")
                    events.append(Event(
                        event_id=self._next_id(),
                        timestamp_s=fid / self.fps,
                        frame_id=fid,
                        event_code="football.interception",
                        display_name_en=ev_def.display_name_en if ev_def else "Interception",
                        display_name_cn=ev_def.display_name_cn if ev_def else "拦截",
                        importance=ev_def.importance_base if ev_def else 0.45,
                        player_jersey=curr.get("jersey"),
                        player_team=curr.get("team"),
                        confidence=0.6,
                        description_hint="interception: possession switched between teams",
                    ))
            prev_player = curr
        return events

    def _detect_assists(self, existing_events: List[Event]) -> List[Event]:
        """rule_composed: assist = pass immediately before a goal."""
        events = []
        goals = [e for e in existing_events if e.event_code == "football.goal"]
        passes = [e for e in existing_events if e.event_code == "football.pass"]
        for goal in goals:
            candidates = [p for p in passes
                          if 0 < (goal.timestamp_s - p.timestamp_s) < 5.0
                          and p.target_team == goal.player_team]
            if candidates:
                last_pass = max(candidates, key=lambda p: p.timestamp_s)
                ev_def = self.schema.get_event("football.assist")
                events.append(Event(
                    event_id=self._next_id(),
                    timestamp_s=last_pass.timestamp_s,
                    frame_id=last_pass.frame_id,
                    event_code="football.assist",
                    display_name_en=ev_def.display_name_en if ev_def else "Assist",
                    display_name_cn=ev_def.display_name_cn if ev_def else "助攻",
                    importance=ev_def.importance_base if ev_def else 0.85,
                    player_jersey=last_pass.player_jersey,
                    player_team=last_pass.player_team,
                    confidence=0.8,
                    description_hint=f"assist: pass by #{last_pass.player_jersey} before goal",
                ))
        return events

    def _deduplicate(self, events: List[Event]) -> List[Event]:
        if not events:
            return []
        sorted_evts = sorted(events, key=lambda e: (e.timestamp_s, -e.importance))
        kept = [sorted_evts[0]]
        for ev in sorted_evts[1:]:
            if ev.timestamp_s - kept[-1].timestamp_s >= self.min_gap_s or ev.event_code != kept[-1].event_code:
                kept.append(ev)
        return kept

    def write_events_json(self, events: List[Event], output_path: Path,
                          video_info: Optional[dict] = None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video_info": video_info or {},
            "schema_version": "v3-20260319",
            "events": [e.to_dict() for e in events],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/pipeline/test_detector.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/detector.py tests/pipeline/test_detector.py
git commit -m "feat(pipeline): rule-based event detector with pass/shot/clearance/interception"
```

---

## Day 4 Tasks (6h): Track A — Stage 1 Full Pipeline | Track B — Stage 4 LLM

### Task 7: Tag enricher (moved to Day 3 Track A)

**Files:**
- Create: `pipeline/stage2_events/enricher.py`
- Create: `tests/pipeline/test_enricher.py`

- [ ] **Step 1: Write the test**

```python
# tests/pipeline/test_enricher.py
import pytest
from pipeline.stage2_events.enricher import enrich_tags
from pipeline.stage2_events.detector import Event


class TestEnricher:
    def test_shot_gets_pitch_zone(self):
        ev = Event(
            event_id="evt_001", timestamp_s=5.0, frame_id=125,
            event_code="football.shoot", display_name_en="Shot",
            display_name_cn="射门", importance=0.55,
        )
        enriched = enrich_tags(ev, ball_x=45.0, ball_y=5.0, goal_x=52.5)
        assert "pitch_zone" in enriched.tags
        assert enriched.tags["pitch_zone"] == "outside_box"

    def test_shot_inside_box(self):
        ev = Event(
            event_id="evt_002", timestamp_s=5.0, frame_id=125,
            event_code="football.shoot", display_name_en="Shot",
            display_name_cn="射门", importance=0.55,
        )
        enriched = enrich_tags(ev, ball_x=48.0, ball_y=5.0, goal_x=52.5)
        assert enriched.tags["pitch_zone"] == "inside_box"

    def test_shot_distance_computed(self):
        ev = Event(
            event_id="evt_003", timestamp_s=5.0, frame_id=125,
            event_code="football.shoot", display_name_en="Shot",
            display_name_cn="射门", importance=0.55,
        )
        enriched = enrich_tags(ev, ball_x=30.0, ball_y=0.0, goal_x=52.5)
        assert enriched.tags["shot_distance"] == "long_range"

    def test_pass_direction_forward(self):
        ev = Event(
            event_id="evt_004", timestamp_s=5.0, frame_id=125,
            event_code="football.pass", display_name_en="Pass",
            display_name_cn="传球", importance=0.15,
        )
        enriched = enrich_tags(ev, passer_x=10.0, passer_y=0.0,
                               receiver_x=25.0, receiver_y=2.0, attack_dir=1)
        assert enriched.tags["pass_direction"] == "forward"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_enricher.py -v
```

- [ ] **Step 3: Implement `enricher.py`**

```python
# pipeline/stage2_events/enricher.py
"""Enrich events with computable tag dimensions."""
from __future__ import annotations

import math
from typing import Optional

from pipeline.stage2_events.detector import Event
from pipeline.utils.pitch import (
    GOAL_X, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH,
    SIX_YARD_LENGTH, SIX_YARD_WIDTH, PITCH_LENGTH,
)


def _pitch_zone(ball_x: float, ball_y: float, goal_x: float) -> str:
    dx = abs(abs(ball_x) - goal_x)
    dy = abs(ball_y)
    if dx <= SIX_YARD_LENGTH and dy <= SIX_YARD_WIDTH / 2:
        return "six_yard_box"
    if dx <= PENALTY_AREA_LENGTH and dy <= PENALTY_AREA_WIDTH / 2:
        return "inside_box"
    if abs(ball_x) < PITCH_LENGTH / 4:
        return "halfway_line"
    return "outside_box"


def _shot_distance(ball_x: float, ball_y: float, goal_x: float) -> str:
    dist = math.hypot(abs(ball_x) - goal_x, ball_y)
    if dist <= 8:
        return "close_range"
    if dist <= 16:
        return "mid_range"
    if dist <= 30:
        return "long_range"
    return "half_way"


def _pass_distance(passer_x, passer_y, receiver_x, receiver_y) -> str:
    dist = math.hypot(receiver_x - passer_x, receiver_y - passer_y)
    if dist < 15:
        return "short"
    if dist < 25:
        return "medium"
    return "long"


def _pass_direction(passer_x, passer_y, receiver_x, receiver_y, attack_dir: int) -> str:
    dx = (receiver_x - passer_x) * attack_dir
    dy = receiver_y - passer_y
    if dx < -5:
        return "back_pass"
    if abs(dy) > abs(dx) * 2:
        return "lateral"
    if dx > 5:
        return "forward"
    return "lateral"


def enrich_tags(
    event: Event,
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    goal_x: float = GOAL_X,
    passer_x: Optional[float] = None,
    passer_y: Optional[float] = None,
    receiver_x: Optional[float] = None,
    receiver_y: Optional[float] = None,
    attack_dir: int = 1,
) -> Event:
    """Add computable tags to an event. Returns the same event, mutated."""
    code = event.event_code

    if code in ("football.shoot", "football.goal"):
        event.tags["pitch_zone"] = _pitch_zone(ball_x, ball_y, goal_x)
        event.tags["shot_distance"] = _shot_distance(ball_x, ball_y, goal_x)
        event.tags["pattern_of_play"] = event.tags.get("pattern_of_play", "open_play")

    if code == "football.pass" and passer_x is not None and receiver_x is not None:
        event.tags["pass_distance"] = _pass_distance(passer_x, passer_y, receiver_x, receiver_y)
        event.tags["pass_direction"] = _pass_direction(passer_x, passer_y, receiver_x, receiver_y, attack_dir)

    return event
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/pipeline/test_enricher.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/enricher.py tests/pipeline/test_enricher.py
git commit -m "feat(pipeline): tag enricher for pitch_zone, shot_distance, pass_direction"
```

---

## Day 5 Tasks (6h): Track A — Full Stage 1→2 on SNGS-061 | Track B — Stage 4 Remaining

### Task 8: LLM adapter interface + prompt builder (moved to Day 4 Track B)

**Files:**
- Create: `pipeline/stage4_commentary/llm_adapter.py`
- Create: `pipeline/stage4_commentary/prompt_builder.py`
- Create: `pipeline/stage4_commentary/postprocess.py`

_(Full code in each step — see spec Section 7.2-7.5 for prompt structure)_

- [ ] **Step 1: Implement `llm_adapter.py`**

```python
# pipeline/stage4_commentary/llm_adapter.py
"""Abstract LLM adapter interface."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union


class LLMAdapter(ABC):
    @abstractmethod
    def supports_video(self) -> bool:
        """Whether this backend accepts video input directly."""

    @abstractmethod
    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        """Generate text from prompt + optional visual input."""

    def prepare_visual_input(self, visual_input: Union[Path, List[Path], None]) -> Union[Path, List[Path], None]:
        """If video given but not supported, extract frames."""
        if visual_input is None:
            return None
        if isinstance(visual_input, Path) and visual_input.suffix == ".mp4" and not self.supports_video():
            from pipeline.utils.video import extract_frames
            frame_dir = visual_input.parent / f"{visual_input.stem}_frames_for_llm"
            extract_frames(visual_input, frame_dir)
            return sorted(frame_dir.glob("*.jpg"))
        return visual_input
```

- [ ] **Step 2: Implement `prompt_builder.py`**

```python
# pipeline/stage4_commentary/prompt_builder.py
"""Build LLM prompts from events.json + topo.json + schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.stage2_events.schema import EventSchema


def build_commentary_prompt(
    events_json_path: Path,
    schema: EventSchema,
    languages: List[str],
    topo_json_path: Optional[Path] = None,
    roster: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    with open(events_json_path, encoding="utf-8") as f:
        events_data = json.load(f)

    parts = []

    # System
    lang_str = " and ".join(languages).upper()
    parts.append(f"""You are a professional football commentator. Generate second-by-second commentary for this 30-second match clip.

RULES:
1. Use ONLY the timestamps from the event list. Never invent timestamps.
2. Use the EXACT terminology from the tag display names shown in parentheses.
3. For gaps between events, describe formations, positioning, build-up play.
4. Refer to players by jersey number (or name if roster provided).
5. Generate BOTH {lang_str} commentary for each segment.
6. For events marked HIGHLIGHT, use more excited/vivid language.
7. Output valid JSON array of commentary segments.""")

    # Event definitions
    parts.append("\n[Event Definitions]")
    parts.append(schema.event_definitions_for_prompt())

    # Event timeline
    parts.append("\n[Event Timeline]")
    for ev in events_data.get("events", []):
        line = f"t={ev['timestamp_s']}s: [{ev['event_code']}]"
        if ev.get("player_jersey"):
            line += f" #{ev['player_jersey']}"
        if ev.get("player_team"):
            line += f" ({ev['player_team']})"
        if ev.get("target_jersey"):
            line += f" → #{ev['target_jersey']} ({ev.get('target_team', '')})"
        tags = ev.get("tags", {})
        if tags:
            tag_parts = []
            for k, v in tags.items():
                tg = schema.get_tag_group(k)
                if tg:
                    tv = next((tv for tv in tg.values if tv.code == v), None)
                    if tv:
                        tag_parts.append(f"{k}={v}({tv.display_name_cn})")
                    else:
                        tag_parts.append(f"{k}={v}")
                else:
                    tag_parts.append(f"{k}={v}")
            line += f"\n  Tags: {', '.join(tag_parts)}"
        if ev.get("importance", 0) >= 0.5:
            line += "  ⚡ HIGHLIGHT"
        parts.append(line)

    # Topology context
    if topo_json_path and topo_json_path.exists():
        parts.append("\n[Formation Context]")
        with open(topo_json_path, encoding="utf-8") as f:
            topo = json.load(f)
        for record in topo[:6]:
            t = record.get("window_start_s", "?")
            team = record.get("team", "?")
            height = record.get("block_height_m")
            depth = record.get("block_depth_m")
            parts.append(f"t~{t}s {team}: height={height}m, depth={depth}m")

    # Roster
    if roster:
        parts.append("\n[Player Roster]")
        for team, players in roster.items():
            parts.append(f"{team}: {json.dumps(players, ensure_ascii=False)}")

    return "\n".join(parts)


def build_visual_tag_prompt(event: dict, schema: EventSchema) -> str:
    """Build prompt for Step 1: LLM fills visual tags from video."""
    event_code = event["event_code"]
    vocab = schema.tag_vocabulary_for_prompt(event_code)
    if not vocab:
        return ""

    return f"""You are a football video analyst. For the event below, watch the video clip and fill in the visual tags. Use ONLY values from the provided vocabulary. Output JSON only, no explanation.

[Event]
t={event['timestamp_s']}s: {event_code} by #{event.get('player_jersey', '?')} ({event.get('player_team', '?')})

[Tag Vocabulary]
{vocab}"""


def build_tactical_reasoning_prompt(event: dict, topo_before: dict, topo_at: dict) -> str:
    """Build prompt for Step 3: tactical reasoning for goals/shots."""
    return f"""You are a football tactical analyst. Explain WHY this event succeeded or was dangerous.

Analyze:
1. What defensive weakness was exploited?
2. What attacking movement created the opportunity?
3. Which players' positioning was critical?
4. Could the defense have prevented it?

Use the topology metrics with specific numbers to support your analysis.

[Formation Data: Before event]
{json.dumps(topo_before, indent=2, ensure_ascii=False)}

[Formation Data: At event]
{json.dumps(topo_at, indent=2, ensure_ascii=False)}

[Event Details]
{json.dumps(event, indent=2, ensure_ascii=False)}

Output JSON with keys: text_en, text_zh, key_factors (list of strings)."""
```

- [ ] **Step 3: Implement `postprocess.py`**

```python
# pipeline/stage4_commentary/postprocess.py
"""Parse and validate LLM output into commentary.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_commentary_output(raw_text: str) -> List[dict]:
    """Extract JSON array of commentary segments from LLM output."""
    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return [{"timestamp_s": 0, "end_s": 30, "text_en": raw_text, "text_zh": raw_text, "events_referenced": []}]


def parse_visual_tags(raw_text: str) -> dict:
    """Extract JSON dict of visual tags from LLM output."""
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def parse_tactical_analysis(raw_text: str) -> dict:
    """Extract tactical analysis JSON from LLM output."""
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"text_en": raw_text, "text_zh": raw_text, "key_factors": []}


def write_commentary_json(
    commentary_segments: List[dict],
    output_path: Path,
    video_info: dict,
    model_info: dict,
    languages: List[str],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info,
        "model_info": model_info,
        "language": languages,
        "commentary": commentary_segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/stage4_commentary/llm_adapter.py pipeline/stage4_commentary/prompt_builder.py pipeline/stage4_commentary/postprocess.py
git commit -m "feat(pipeline): LLM adapter interface, prompt builder (3-step flow), postprocessor"
```

### Task 9: Three LLM adapter implementations

**Files:**
- Create: `pipeline/stage4_commentary/adapters/qwen_local.py`
- Create: `pipeline/stage4_commentary/adapters/doubao_api.py`
- Create: `pipeline/stage4_commentary/adapters/openai_api.py`

- [ ] **Step 1: Implement `qwen_local.py`**

```python
# pipeline/stage4_commentary/adapters/qwen_local.py
"""Local Qwen2.5-7B-VL-Instruct adapter."""
from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter


class QwenLocalAdapter(LLMAdapter):
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def supports_video(self) -> bool:
        return True

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        self._load_model()
        visual_input = self.prepare_visual_input(visual_input)

        messages = [{"role": "user", "content": []}]
        if visual_input is not None:
            if isinstance(visual_input, Path):
                messages[0]["content"].append({"type": "video", "video": str(visual_input)})
            elif isinstance(visual_input, list):
                for img_path in visual_input[:30]:
                    messages[0]["content"].append({"type": "image", "image": str(img_path)})
        messages[0]["content"].append({"type": "text", "text": prompt})

        from qwen_vl_utils import process_vision_info
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs,
                                  padding=True, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=4096)
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        return self._processor.decode(trimmed, skip_special_tokens=True)
```

- [ ] **Step 2: Implement `openai_api.py`**

```python
# pipeline/stage4_commentary/adapters/openai_api.py
"""GPT-5 API adapter."""
import base64
import os
from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter


class OpenAIAPIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-5", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def supports_video(self) -> bool:
        return False

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        visual_input = self.prepare_visual_input(visual_input)

        content = []
        if isinstance(visual_input, list):
            for img_path in visual_input[:30]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
```

- [ ] **Step 3: Implement `doubao_api.py`**

```python
# pipeline/stage4_commentary/adapters/doubao_api.py
"""Doubao (ByteDance) Vision API adapter."""
import base64
import os
from pathlib import Path
from typing import List, Union

from pipeline.stage4_commentary.llm_adapter import LLMAdapter


class DoubaoAPIAdapter(LLMAdapter):
    def __init__(self, model: str = "doubao-vision-pro", api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.base_url = base_url or os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    def supports_video(self) -> bool:
        return True

    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        visual_input = self.prepare_visual_input(visual_input)

        content = []
        if isinstance(visual_input, Path) and visual_input.suffix == ".mp4":
            with open(visual_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
            })
        elif isinstance(visual_input, list):
            for img_path in visual_input[:30]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/stage4_commentary/adapters/
git commit -m "feat(pipeline): Qwen local + GPT-5 API + Doubao API adapters"
```

---

## Day 6 Tasks (6h): Orchestrator + End-to-End Integration

### Task 10: Projection utilities + light beam (moved to Day 2 Track B)

**Files:**
- Create: `pipeline/stage3_effects/projection.py`
- Create: `pipeline/stage3_effects/light_beam.py`

- [ ] **Step 1: Implement `projection.py`**

```python
# pipeline/stage3_effects/projection.py
"""Homography projection utilities for pitch↔image coordinate mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_homography(homo_json_path: Path) -> Dict[str, dict]:
    with open(homo_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", {})


def pitch_to_image(point_pitch: Tuple[float, float], H_inv: np.ndarray) -> Optional[Tuple[int, int]]:
    """Project a pitch-coordinate point to image pixel coordinates."""
    px, py = point_pitch
    p = H_inv @ np.array([px, py, 1.0], dtype=float)
    if abs(p[2]) < 1e-9:
        return None
    ix, iy = p[0] / p[2], p[1] / p[2]
    return int(round(ix)), int(round(iy))


def get_h_inv_for_frame(homo_frames: dict, frame_id: int) -> Optional[np.ndarray]:
    key = str(frame_id)
    entry = homo_frames.get(key)
    if entry is None or not entry.get("valid"):
        return None
    h_inv = entry.get("H_inv")
    if h_inv is None:
        return None
    return np.array(h_inv, dtype=float)
```

- [ ] **Step 2: Implement `light_beam.py`**

```python
# pipeline/stage3_effects/light_beam.py
"""Perspective-correct cone light beam + foot marker rendering."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline.stage3_effects.projection import pitch_to_image


def draw_foot_marker(frame: np.ndarray, center: Tuple[int, int],
                     color: Tuple[int, int, int], radius: int = 20, alpha: float = 0.4):
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius * 1.5), color, 1, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(overlay, (0, 0), sigmaX=5)
    cv2.addWeighted(blurred, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_cone_beam(frame: np.ndarray,
                   origin: Tuple[int, int],
                   target: Tuple[int, int],
                   color: Tuple[int, int, int],
                   alpha: float = 0.3,
                   width_base: int = 40,
                   spread: float = 0.3):
    """Draw perspective cone beam from origin toward target."""
    ox, oy = origin
    tx, ty = target

    dx, dy = tx - ox, ty - oy
    length = max(1, int(np.hypot(dx, dy)))
    nx, ny = -dy / length, dx / length

    w0 = width_base / 2
    w1 = w0 + length * spread

    pts = np.array([
        [ox + nx * w0, oy + ny * w0],
        [ox - nx * w0, oy - ny * w0],
        [tx - nx * w1, ty - ny * w1],
        [tx + nx * w1, ty + ny * w1],
    ], dtype=np.int32)

    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay, [pts], color)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(15, length // 10))

    mask = overlay.astype(float) / 255.0
    frame[:] = (frame.astype(float) * (1 - mask * alpha) + overlay.astype(float) * alpha).clip(0, 255).astype(np.uint8)


def compute_beam_alpha(frame_offset: int, half_duration_frames: int, alpha_max: float) -> float:
    """Fade in/out alpha based on frame distance from event center."""
    if abs(frame_offset) >= half_duration_frames:
        return 0.0
    return alpha_max * (1 - abs(frame_offset) / half_duration_frames)
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/stage3_effects/
git commit -m "feat(pipeline): projection utils + cone light beam rendering"
```

### Task 11: Tactical lines + video renderer

**Files:**
- Create: `pipeline/stage3_effects/tactical_lines.py`
- Create: `pipeline/stage3_effects/render.py`

- [ ] **Step 1: Implement `tactical_lines.py`**

```python
# pipeline/stage3_effects/tactical_lines.py
"""Tactical topology line rendering on original video frames."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def draw_player_marker(frame: np.ndarray, center: Tuple[int, int],
                       color: Tuple[int, int, int] = (255, 255, 255),
                       radius: int = 12):
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)


def draw_formation_lines(frame: np.ndarray,
                         positions: List[Tuple[int, int]],
                         adjacency: List[Tuple[int, int]],
                         color: Tuple[int, int, int] = (255, 150, 50),
                         thickness: int = 2):
    for i, j in adjacency:
        if i < len(positions) and j < len(positions):
            cv2.line(frame, positions[i], positions[j], color, thickness, cv2.LINE_AA)


def draw_dashed_line(frame: np.ndarray,
                     pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, int, int],
                     thickness: int = 2, dash_length: int = 15):
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = max(1, int(np.hypot(dx, dy)))
    for i in range(0, dist, dash_length * 2):
        t1 = i / dist
        t2 = min((i + dash_length) / dist, 1.0)
        x1 = int(pt1[0] + dx * t1)
        y1 = int(pt1[1] + dy * t1)
        x2 = int(pt1[0] + dx * t2)
        y2 = int(pt1[1] + dy * t2)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_arrow(frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
               color: Tuple[int, int, int], thickness: int = 2):
    cv2.arrowedLine(frame, start, end, color, thickness, cv2.LINE_AA, tipLength=0.15)


def draw_running_path(frame: np.ndarray,
                      current: Tuple[int, int],
                      predicted: Tuple[int, int],
                      color: Tuple[int, int, int]):
    draw_dashed_line(frame, current, predicted, color, thickness=2)
    draw_arrow(frame, current, predicted, color, thickness=2)


def draw_pressing_line(frame: np.ndarray,
                       points: List[Tuple[int, int]],
                       color: Tuple[int, int, int] = (255, 200, 100)):
    """Draw horizontal dashed pressing line through projected pitch points."""
    if len(points) < 2:
        return
    sorted_pts = sorted(points, key=lambda p: p[0])
    draw_dashed_line(frame, sorted_pts[0], sorted_pts[-1], color, thickness=2, dash_length=20)
```

- [ ] **Step 2: Implement `render.py`**

```python
# pipeline/stage3_effects/render.py
"""Orchestrate all visual effects onto original video."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.stage3_effects.projection import load_homography, get_h_inv_for_frame, pitch_to_image
from pipeline.stage3_effects.light_beam import draw_cone_beam, draw_foot_marker, compute_beam_alpha
from pipeline.config import PipelineConfig
from pipeline.utils.pitch import GOAL_X


EVENT_COLORS = {
    "football.shoot": (100, 255, 200),
    "football.goal": (0, 215, 255),
    "football.pass": (150, 255, 150),
    "football.clearance": (255, 150, 50),
    "football.save": (0, 255, 100),
}


def render_annotated_video(
    raw_video_path: Path,
    events_json_path: Path,
    homography_json_path: Path,
    output_path: Path,
    config: PipelineConfig,
):
    with open(events_json_path, encoding="utf-8") as f:
        events_data = json.load(f)
    homo_frames = load_homography(homography_json_path)

    cap = cv2.VideoCapture(str(raw_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    beam_half_frames = int(config.beam_duration_s * fps)

    high_events = [e for e in events_data.get("events", []) if e.get("importance", 0) >= config.event_importance_threshold]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps

        for ev in high_events:
            ev_frame = ev.get("frame_id", int(ev["timestamp_s"] * fps))
            offset = frame_idx - ev_frame
            if abs(offset) > beam_half_frames:
                continue

            alpha = compute_beam_alpha(offset, beam_half_frames, config.beam_alpha_max)
            if alpha <= 0.01:
                continue

            h_inv = get_h_inv_for_frame(homo_frames, frame_idx)
            if h_inv is None:
                continue

            color = EVENT_COLORS.get(ev["event_code"], (255, 255, 255))

            # Goal target for beam direction
            attack_sign = 1 if ev.get("player_team") == "left" else -1
            goal_center = (GOAL_X * attack_sign, 0.0)
            goal_img = pitch_to_image(goal_center, h_inv)

            # Player foot position (approximate from ball position if no bbox)
            # In full implementation, look up player bbox from predictions.json
            # For now, use ball position as proxy
            ball_x = ev.get("tags", {}).get("_ball_x", 0)
            ball_y = ev.get("tags", {}).get("_ball_y", 0)
            player_img = pitch_to_image((ball_x, ball_y), h_inv)

            if player_img and goal_img:
                draw_foot_marker(frame, player_img, color, alpha=alpha)
                draw_cone_beam(frame, player_img, goal_img, color, alpha=alpha)

        writer.write(frame)

    cap.release()
    writer.release()

    # Re-encode with ffmpeg for H.264
    import subprocess
    tmp_path = output_path.with_suffix(".tmp.mp4")
    output_path.rename(tmp_path)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)
    tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/stage3_effects/
git commit -m "feat(pipeline): tactical lines + full video renderer with light beams"
```

### Task 12: Orchestrator `run.py`

**Files:**
- Create: `pipeline/run.py`

- [ ] **Step 1: Implement `run.py`**

```python
# pipeline/run.py
"""End-to-end pipeline orchestrator with flexible entry points."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Produce predictions.json + homography_per_frame.json
    if config.should_run_stage1():
        log.info("=== Stage 1: SoccerMaster Inference ===")
        if config.existing_pklz_path and Path(config.existing_pklz_path).exists():
            log.info("Using existing pklz: %s (skipping GSR inference)", config.existing_pklz_path)
            from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json
            convert_pklz_to_json(
                config.existing_pklz_path, "061", config.output_dir, fps=config.fps
            )
        else:
            from pipeline.stage1_inference.preprocess import preprocess_video
            from pipeline.stage1_inference.run_gsr import run_full_gsr
            from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

            seq_info = preprocess_video(config.clip_dir, config.sequence_prefix, config.gsr_split)
            pklz_path = run_full_gsr(config)
            convert_pklz_to_json(pklz_path, "001", config.output_dir, fps=int(seq_info["fps"]))
        log.info("Stage 1 complete: %s", config.predictions_json)
    else:
        log.info("Stage 1 skipped: predictions at %s", config.predictions_json)

    # Stage 2: Produce events.json
    if config.should_run_stage2():
        log.info("=== Stage 2: Event Detection ===")
        from pipeline.stage2_events.schema import EventSchema
        from pipeline.stage2_events.detector import EventDetector

        schema = EventSchema()
        detector = EventDetector(schema, fps=config.fps)
        events = detector.detect(str(config.predictions_json))

        video_info = {"source": str(config.clip_dir), "fps": config.fps,
                      "duration_s": 30.0, "total_frames": config.fps * 30}
        detector.write_events_json(events, config.output_dir / "events.json", video_info)
        log.info("Stage 2 complete: %d events → %s", len(events), config.events_json)
    else:
        log.info("Stage 2 skipped: events at %s", config.events_json)

    # Stage 3 and Stage 4 can run in parallel (sequential here for simplicity)

    # Stage 3: Produce annotated_video.mp4
    if config.force or not config.annotated_video.exists():
        log.info("=== Stage 3: Visual Effects ===")
        from pipeline.stage3_effects.render import render_annotated_video
        render_annotated_video(
            config.frames_dir, config.events_json,
            config.predictions_json, config.homography_json,
            config.annotated_video, config,
        )
        log.info("Stage 3 complete: %s", config.annotated_video)
    else:
        log.info("Stage 3 skipped: %s exists", config.annotated_video)

    # Stage 4: Produce commentary.json
    if config.force or not config.commentary_json.exists():
        log.info("=== Stage 4: LLM Commentary ===")
        from pipeline.stage2_events.schema import EventSchema
        from pipeline.stage4_commentary.prompt_builder import build_commentary_prompt
        from pipeline.stage4_commentary.postprocess import parse_commentary_output, write_commentary_json

        schema = EventSchema()
        roster = None
        if config.roster_json and config.roster_json.exists():
            with open(config.roster_json) as f:
                roster = json.load(f)

        prompt = build_commentary_prompt(
            config.events_json, schema, config.languages, roster=roster,
        )

        adapter = _get_adapter(config.llm_backend)
        visual = config.topdown_video if config.topdown_video.exists() else None
        raw_output = adapter.generate(prompt, visual)

        segments = parse_commentary_output(raw_output)
        video_info = {"source": str(config.clip_dir), "duration_s": 30.0}
        write_commentary_json(
            segments, config.commentary_json, video_info,
            {"name": config.llm_backend, "backend": config.llm_backend},
            config.languages,
        )
        log.info("Stage 4 complete: %s", config.commentary_json)
    else:
        log.info("Stage 4 skipped: %s exists", config.commentary_json)

    log.info("Pipeline complete. Outputs in %s", config.output_dir)


def _get_adapter(backend: str):
    if backend == "qwen_local":
        from pipeline.stage4_commentary.adapters.qwen_local import QwenLocalAdapter
        return QwenLocalAdapter()
    elif backend == "doubao":
        from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter
        return DoubaoAPIAdapter()
    elif backend == "openai":
        from pipeline.stage4_commentary.adapters.openai_api import OpenAIAPIAdapter
        return OpenAIAPIAdapter()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="AI Football Commentary Pipeline")
    parser.add_argument("--clip-dir", type=Path, required=True, help="Clip directory containing img1/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline_run"))
    parser.add_argument("--existing-predictions-json", type=Path, default=None, help="Skip Stage 1")
    parser.add_argument("--existing-homography-json", type=Path, default=None)
    parser.add_argument("--existing-pklz-path", type=Path, default=None, help="Skip GSR inference")
    parser.add_argument("--existing-events-json", type=Path, default=None, help="Skip Stage 2")
    parser.add_argument("--llm-backend", default="openai", choices=["qwen_local", "doubao", "openai"])
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--lang", nargs="+", default=["en", "zh"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        clip_dir=args.clip_dir,
        output_dir=args.output_dir,
        existing_predictions_json=args.existing_predictions_json,
        existing_homography_json=args.existing_homography_json,
        existing_pklz_path=args.existing_pklz_path,
        existing_events_json=args.existing_events_json,
        llm_backend=args.llm_backend,
        roster_json=args.roster,
        languages=args.lang,
        fps=args.fps,
        force=args.force,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI works**

```bash
python -m pipeline.run --help
```

Expected: Shows argument help without errors.

- [ ] **Step 3: Commit**

```bash
git add pipeline/run.py
git commit -m "feat(pipeline): end-to-end orchestrator with CLI interface"
```

---

## Day 7-8 Tasks (12h): Speed Optimization + Polish

### Task 13: Speed profiling and optimization

**Files:**
- Modify: `pipeline/stage1_inference/run_gsr.py` (add frame skipping option)
- Modify: `pipeline/config.py` (add speed-related config)

- [ ] **Step 1: Add frame-skip config**

Add to `PipelineConfig`:
```python
    frame_skip: int = 1               # Process every Nth frame in GSR
    step3_batch_size: int = 8          # Batch size for Qwen VL in Step 3
    skip_sam2: bool = False            # Skip Step 2 SAM2 for speed
```

- [ ] **Step 2: Update `run_gsr.py` with skip options**

Add conditional Step 2 skip and frame-stride option to Hydra config overrides.

- [ ] **Step 3: Benchmark before/after**

```bash
time python -m pipeline.run --input test_video.mp4 --output-dir outputs/bench_before
# Apply optimizations
time python -m pipeline.run --input test_video.mp4 --output-dir outputs/bench_after
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/
git commit -m "perf(pipeline): add frame skipping and batch size tuning for speed"
```

### Task 14: End-to-end testing + bug fixes

- [ ] **Step 1: Run pipeline on SNGS-148 (Track 1: test clip with GT labels)**

```bash
python -m pipeline.run \
    --clip-dir codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148/ \
    --output-dir outputs/SNGS-148/ \
    --existing-predictions-json codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148/Labels-GameState.json \
    --llm-backend openai \
    --lang en zh
```

- [ ] **Step 2: Run pipeline on SNGS-061 (Track 2: processed clip with pklz)**

```bash
python -m pipeline.run \
    --clip-dir codes/sn-gamestate/datasets/SoccerNetGS/train/SNGS-061/ \
    --output-dir outputs/SNGS-061/ \
    --existing-pklz-path codes/sn-gamestate/outputs/gsr/step_3_train_SNGS-061/states/sn-gamestate.pklz \
    --llm-backend openai \
    --lang en zh
```

- [ ] **Step 3: Validate all outputs exist**

```bash
ls -la outputs/SNGS-148/
ls -la outputs/SNGS-061/
# Both should contain: events.json, annotated_video.mp4, commentary.json
# SNGS-061 also: predictions.json, homography_per_frame.json
```

- [ ] **Step 3: Manual review of commentary quality**

Open `commentary.json` and verify:
- Timestamps match video events
- Chinese and English both present
- Event terminology matches reference_football.csv

- [ ] **Step 4: Fix any bugs found, run again**

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(pipeline): end-to-end AI football commentary pipeline v1.0"
```
