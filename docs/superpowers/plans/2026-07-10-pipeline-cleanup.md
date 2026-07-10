# Pipeline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce SoccerMaster to the active Stage 1 → Stage 2B → Stage 5 workflow, while retaining optional Stage 3 visual effects and topology in one canonical package.

**Architecture:** Canonical code lives only below `pipeline/` and canonical shell launchers live only below `scripts/`. Stage 2B absorbs its few live Stage 2/4 dependencies directly into `digest.py` and `generate_direct.py`; Stage 5 uses one runner for Edge preview and CosyVoice clone output. Every sequence uses one output directory.

**Tech Stack:** Python 3.10+, standard library, OpenCV, NumPy, FFmpeg/FFprobe, OpenAI-compatible Volcengine ARK client, Edge TTS, CosyVoice3, Bash.

**Spec:** `docs/superpowers/specs/2026-07-10-pipeline-cleanup-design.md`

**Verification policy:** The user explicitly requested removal of `tests/` and all test-only self-checks. This plan uses compilation, shell parsing, CLI loading, import searches, and optional live hardware/API checks only.

---

## File Map

### Create

- `pipeline/stage5_tts/run.py` — the only Stage 5 CLI; selects Edge preview, CosyVoice clone, or both.
- `scripts/run_stage2b.sh` — canonical single-output Stage 2B + Stage 5 launcher, moved from the root conceptually but rewritten for the replacement workflow.

### Modify

- `.gitignore` — ignore the removed root `tests/` directory without keeping exceptions.
- `pipeline/config.py` — retain Stage 1/3 configuration and add a neutral `.env` loader.
- `pipeline/run.py` — retain only `run_stage1` and `run_stage3` orchestration.
- `pipeline/stage1_inference/run_gsr.py` — preserve the newer root-copy timing and batch behavior.
- `pipeline/stage2b/digest.py` — absorb predictions loading and possession helpers.
- `pipeline/stage2b/generate_direct.py` — absorb ARK video calling and JSON normalization/writing.
- `pipeline/stage2b/run.py` — remove Stage 4 adapter imports and use the direct Stage 2B API function.
- `pipeline/stage3_effects/render.py` — preserve the newer root-copy FFmpeg fallback.
- `scripts/run_stage1.sh` — update downstream-stage wording.
- `scripts/run_stage3.sh` — refer to Stage 2B and the single output directory.
- `docs/superpowers/specs/2026-07-10-pipeline-cleanup-design.md` — include the orphaned ontology CSV in the deletion inventory.

### Delete: canonical obsolete code

- `pipeline/stage2_events/`
- `pipeline/stage4_commentary/`
- `pipeline/stage3_effects/render_preview.py`
- `pipeline/stage5_tts/adapters/doubao_tts.py`
- `pipeline/stage5_tts/make_final_video.py`
- `pipeline/stage5_tts/make_raw_final_video.py`
- `pipeline/stage5_tts/preview.py`
- `pipeline/stage1_inference/check_recompute.py`
- `pipeline/stage1_inference/probe_concat_tracklets_by_reid.py`
- `pipeline/stage1_inference/probe_heatmap_peaks.py`
- `pipeline/stage1_inference/recompute_calibration.sh`
- `pipeline/utils/clip_index_to_events.py`
- `pipeline/utils/validate_clip_index.py`
- `scripts/run_caption_verify.sh`
- `scripts/run_stage2.sh`
- `scripts/run_stage4.sh`
- `scripts/run_stage5.sh`
- `scripts/run_tts.sh`
- `scripts/smoke_doubao_tts.py`
- `reference_football.csv`
- `tests/`

### Delete: generated or mirrored root content

- `pipeline/SNGS-117-2b/`
- Root `__init__.py`, `config.py`, `run.py`, `video.py`
- Root `stage1_inference/`, `stage2_events/`, `stage2b/`, `stage3_effects/`, `stage4_commentary/`, `stage5_tts/`, `topology/`, `utils/`
- Root `probe_ark_video.py`, `run_caption_verify.sh`, `run_stage1.sh`, `run_stage2.sh`, `run_stage2b.sh`, `run_stage3.sh`, `run_stage4.sh`, `run_stage5.sh`, `run_tts.sh`, `setup_cosyvoice.sh`, `smoke_doubao_tts.py`

---

### Task 1: Record the dirty-tree preservation boundary

**Files:**
- Read: all currently modified, staged, deleted, and untracked paths
- Modify: none

- [ ] **Step 1: Capture status and staged-path baselines**

Run:

```bash
git status --short --branch
git diff --cached --name-only
git diff --name-only
```

Expected: the current Stage 2/test edits remain visible. Do not stash, reset, restore, or clean them.

- [ ] **Step 2: Reconfirm every divergent root/canonical source pair**

Run:

```bash
diff -u pipeline/stage1_inference/run_gsr.py stage1_inference/run_gsr.py || true
diff -u pipeline/stage3_effects/render.py stage3_effects/render.py || true
diff -u pipeline/stage4_commentary/postprocess.py stage4_commentary/postprocess.py || true
diff -u pipeline/stage4_commentary/adapters/doubao_api.py stage4_commentary/adapters/doubao_api.py || true
diff -u pipeline/stage5_tts/make_final_video.py stage5_tts/make_final_video.py || true
diff -u pipeline/stage5_tts/make_raw_final_video.py stage5_tts/make_raw_final_video.py || true
```

Expected: differences match the approved spec—Stage 1 timing/batching, Stage 3 FFmpeg fallback, Stage 2B ARK/normalization support, and Stage 5 base-video/language behavior.

---

### Task 2: Preserve retained Stage 1 and Stage 3 behavior canonically

**Files:**
- Modify: `pipeline/stage1_inference/run_gsr.py`
- Modify: `pipeline/stage3_effects/render.py`

- [ ] **Step 1: Apply the retained Stage 1 changes**

Apply these exact additions to `pipeline/stage1_inference/run_gsr.py`:

```diff
@@
 import subprocess
 import sys
+import time
 import zipfile
@@
 def _rel_to_gsr(path: Path) -> str:
     return os.path.relpath(path.resolve(), GSR_ROOT.resolve())
 
+def _format_duration(seconds: float) -> str:
+    seconds_i = int(round(seconds))
+    hours, rem = divmod(seconds_i, 3600)
+    minutes, secs = divmod(rem, 60)
+    if hours:
+        return f"{hours}h {minutes:02d}m {secs:02d}s"
+    if minutes:
+        return f"{minutes}m {secs:02d}s"
+    return f"{secs}s"
+
+
+def _timed_step(name: str, func, *args, **kwargs):
+    start = time.perf_counter()
+    log.info("Timer start: %s", name)
+    try:
+        return func(*args, **kwargs)
+    finally:
+        log.info("Timer done:  %s took %s", name, _format_duration(time.perf_counter() - start))
+
+
+def _env_int(name: str) -> str | None:
+    value = os.environ.get(name)
+    if value in (None, ""):
+        return None
+    try:
+        valid = int(value) >= 1
+    except ValueError:
+        valid = False
+    if not valid:
+        raise ValueError(f"{name} must be a positive integer, got {value!r}")
+    return value
+
+
+def gsr_batch_overrides(step: int) -> list[str]:
+    overrides: list[str] = []
+    if step == 1:
+        shared = _env_int("GSR_STEP1_BATCH")
+        for module, env_name in (
+            ("bbox_detector", "GSR_BBOX_BATCH"),
+            ("pose_bottomup", "GSR_POSE_BATCH"),
+            ("reid", "GSR_REID_BATCH"),
+            ("track", "GSR_TRACK_BATCH"),
+            ("legibility", "GSR_LEGIBILITY_BATCH"),
+            ("jersey_number_detect", "GSR_JERSEY_BATCH"),
+            ("role", "GSR_ROLE_BATCH"),
+        ):
+            value = _env_int(env_name) or shared
+            if value:
+                overrides.append(f"modules.{module}.batch_size={value}")
+    elif step == 3:
+        for module, env_name in (
+            ("legibility", "GSR_STEP3_LEGIBILITY_BATCH"),
+            ("jersey_and_role", "GSR_VLLM_BATCH"),
+        ):
+            value = _env_int(env_name)
+            if value:
+                overrides.append(f"modules.{module}.batch_size={value}")
+        if (use_vllm := os.environ.get("GSR_USE_VLLM")) is not None:
+            overrides.append(f"modules.jersey_and_role.cfg.use_vllm={use_vllm.lower()}")
+        for env_name, key in (
+            ("GSR_HF_MODEL", "model_path"),
+            ("GSR_VLLM_MODEL", "vllm_model_path"),
+        ):
+            if value := os.environ.get(env_name):
+                model_path = Path(value)
+                if not model_path.is_absolute():
+                    model_path = GSR_ROOT / "pretrained_models" / "jn" / value
+                overrides.append(f"modules.jersey_and_role.cfg.{key}={model_path}")
+        if gpu_mem := os.environ.get("GSR_VLLM_GPU_MEMORY_UTILIZATION"):
+            overrides.append(f"modules.jersey_and_role.cfg.vllm_gpu_memory_utilization={gpu_mem}")
+    return overrides
@@
 def _subprocess_env() -> dict[str, str]:
@@
     env["PYTHONPATH"] = repo + (os.pathsep + prev if prev else "")
+    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("GPU_LIST", "0"))
     return env
@@
         f"use_rich={os.environ.get('GSR_USE_RICH', 'false')}",
         *hydra_cpu_overrides(),
+        *gsr_batch_overrides(step=1),
@@
         f"state.load_file={load_file}",
+        "data_dir=${project_dir}/datasets",
+        "model_dir=${project_dir}/pretrained_models",
+        "dataset.dataset_path=${data_dir}/SoccerNetGS",
         f"hydra.run.dir={_rel_to_gsr(out_dir)}",
         f"use_rich={os.environ.get('GSR_USE_RICH', 'false')}",
+        f"visualization.cfg.save_videos={os.environ.get('GSR_SAVE_VIDEOS', 'false')}",
+        f"visualization.cfg.save_images={os.environ.get('GSR_SAVE_IMAGES', 'false')}",
         *hydra_cpu_overrides(),
+        *gsr_batch_overrides(step=3),
@@
 def run_full_gsr(config: PipelineConfig, dry_run: bool = False) -> Path:
     """Run all 3 GSR steps for the sequence in config.clip_dir. Returns final pklz."""
-    run_step1(config, dry_run=dry_run)
-    if not config.skip_sam2:
-        run_step2(config, dry_run=dry_run)
-    return run_step3(config, dry_run=dry_run)
+    total_start = time.perf_counter()
+    try:
+        _timed_step("GSR Step 1 detection/tracking", run_step1, config, dry_run=dry_run)
+        if config.skip_sam2:
+            log.info("Timer skip:  GSR Step 2 SAM2 refinement")
+        else:
+            _timed_step("GSR Step 2 SAM2 refinement", run_step2, config, dry_run=dry_run)
+        return _timed_step("GSR Step 3 calibration/identity", run_step3, config, dry_run=dry_run)
+    finally:
+        log.info("Timer total: GSR Stage 1 took %s", _format_duration(time.perf_counter() - total_start))
```

- [ ] **Step 2: Apply the retained Stage 3 fallback**

Apply:

```diff
@@
 import json
+import logging
 import shutil
@@
 from pipeline.stage3_effects.projection import load_homography
 
+log = logging.getLogger(__name__)
@@
-    if reencode_h264:
-        if not shutil.which("ffmpeg"):
-            tmp_path.unlink(missing_ok=True)
-            raise RuntimeError(
-                "ffmpeg is required to produce H.264 annotated_video.mp4 for IDE playback"
-            )
+    if reencode_h264 and shutil.which("ffmpeg"):
         reencode_to_h264(tmp_path, output_path)
         tmp_path.unlink(missing_ok=True)
     else:
-        tmp_path.rename(output_path)
+        if reencode_h264:
+            log.warning("ffmpeg not found; keeping OpenCV mp4v output at %s", output_path)
+        tmp_path.replace(output_path)
```

- [ ] **Step 3: Compile the retained modules**

Run:

```bash
python -m py_compile pipeline/stage1_inference/run_gsr.py pipeline/stage3_effects/render.py
```

Expected: exit 0 with no output.

- [ ] **Step 4: Commit only the preserved behavior**

```bash
git add pipeline/stage1_inference/run_gsr.py pipeline/stage3_effects/render.py
git commit --only pipeline/stage1_inference/run_gsr.py pipeline/stage3_effects/render.py -m "refactor: preserve active stage behavior"
```

---

### Task 3: Shrink configuration and orchestration to Stage 1 and optional Stage 3

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/run.py`

- [ ] **Step 1: Replace `pipeline/config.py` with the retained configuration**

```python
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
```

- [ ] **Step 2: Replace `pipeline/run.py` with only retained orchestration**

```python
"""Orchestration for Stage 1 inference and optional Stage 3 effects."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig

log = logging.getLogger(__name__)


def infer_video_id(clip_dir: Path, override: Optional[str] = None) -> str:
    if override:
        return override
    match = re.match(r"SNGS-(\d+)", Path(clip_dir).name, re.IGNORECASE)
    return match.group(1) if match else "001"


def resolve_input_video(config: PipelineConfig) -> Path:
    if config.input_video and Path(config.input_video).exists():
        return Path(config.input_video)
    clip_dir = Path(config.clip_dir)
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        if matches := sorted(clip_dir.glob(pattern)):
            return matches[0]
    if matches := sorted(clip_dir.parent.glob(f"{clip_dir.name}.mp4")):
        return matches[0]
    raise FileNotFoundError(
        f"No input video found for Stage 1. Pass --input-video or place an mp4 in {clip_dir}"
    )


def run_stage1(config: PipelineConfig) -> None:
    from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

    video_id = infer_video_id(config.clip_dir, config.pklz_video_id)
    output_dir = Path(config.output_dir)
    sequence_name = Path(config.clip_dir).name

    if config.existing_pklz_path and Path(config.existing_pklz_path).exists():
        convert_pklz_to_json(
            Path(config.existing_pklz_path),
            video_id,
            output_dir,
            fps=config.fps,
            sequence_name=sequence_name,
            ball_labels_path=(Path(config.clip_dir) / "Labels-GameState.json")
            if (Path(config.clip_dir) / "Labels-GameState.json").is_file()
            else None,
        )
        return

    from pipeline.stage1_inference.run_gsr import run_full_gsr

    frames_dir = Path(config.clip_dir) / "img1"
    n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.is_dir() else 0
    if n_frames:
        log.info("Using existing frames in %s (%d frames)", frames_dir, n_frames)
        fps = config.fps
    else:
        from pipeline.stage1_inference.preprocess import preprocess_video

        video_path = resolve_input_video(config)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Input video is missing or empty: {video_path}")
        sequence = preprocess_video(
            video_path,
            sequence_name=config.sequence_prefix,
            split=config.gsr_split,
        )
        fps = int(sequence["fps"])
        sequence_name = config.sequence_prefix

    labels_path = Path(config.clip_dir) / "Labels-GameState.json"
    convert_pklz_to_json(
        run_full_gsr(config),
        video_id,
        output_dir,
        fps=fps,
        sequence_name=sequence_name,
        ball_labels_path=labels_path if labels_path.is_file() else None,
    )


def run_stage3(config: PipelineConfig) -> None:
    from pipeline.stage3_effects.render import render_annotated_video
    from pipeline.stage3_effects.topology_analysis import run_topology_analysis

    if not config.events_json.is_file():
        raise FileNotFoundError(f"Stage 2B events not found: {config.events_json}")
    if not config.predictions_json.is_file():
        raise FileNotFoundError(f"Stage 1 predictions not found: {config.predictions_json}")

    render_annotated_video(
        frames_dir=config.frames_dir,
        events_json_path=config.events_json,
        predictions_json_path=config.predictions_json,
        output_path=config.annotated_video,
        config=config,
        homography_json_path=config.homography_json if config.homography_json.exists() else None,
    )
    if config.force or not config.topology_json.exists():
        run_topology_analysis(config.predictions_json, config.topology_json, fps=config.fps)
```

- [ ] **Step 3: Compile configuration and orchestration**

Run:

```bash
python -m py_compile pipeline/config.py pipeline/run.py
```

Expected: exit 0.

- [ ] **Step 4: Commit the orchestrator reduction**

```bash
git add pipeline/config.py pipeline/run.py
git commit --only pipeline/config.py pipeline/run.py -m "refactor: narrow pipeline orchestration"
```

---

### Task 4: Make the Stage 2B digest independent of Stage 2

**Files:**
- Modify: `pipeline/stage2b/digest.py`

- [ ] **Step 1: Replace `pipeline/stage2b/digest.py` with a self-contained digest**

```python
"""Build a compact tracking digest directly from Stage 1 predictions."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.utils.labels import frame_index_from_labels

POSSESSION_RADIUS_M = 3.0
POSSESSION_MIN_FRAMES = 3


@dataclass
class FrameData:
    frame_id: int
    ball_xy: Optional[tuple[float, float]] = None
    players: list[dict] = field(default_factory=list)


@dataclass
class PossessionSegment:
    track_id: int
    team: Optional[str]
    jersey: Optional[str]
    start_fid: int
    end_fid: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


def load_frames(path: Path) -> list[FrameData]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    image_to_frame, _ = frame_index_from_labels(data)
    frames = {
        frame_id: FrameData(frame_id)
        for frame_id in sorted(set(image_to_frame.values()))
    }
    for ann in data.get("annotations", []):
        frame_id = image_to_frame.get(str(ann.get("image_id", "")))
        pitch = ann.get("bbox_pitch")
        if frame_id is None or not isinstance(pitch, dict):
            continue
        x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
        if x is None or y is None:
            continue
        attrs = ann.get("attributes") or {}
        if attrs.get("role") == "ball":
            frames[frame_id].ball_xy = (float(x), float(y))
        else:
            frames[frame_id].players.append({
                "track_id": ann.get("track_id"),
                "x": float(x),
                "y": float(y),
                "role": attrs.get("role", "other"),
                "team": attrs.get("team"),
                "jersey": attrs.get("jersey", ""),
            })
    return [frames[frame_id] for frame_id in sorted(frames)]


def _majority(frames: list[FrameData], key: str) -> dict[int, str]:
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        for player in frame.players:
            track_id, value = player.get("track_id"), player.get(key)
            if track_id is not None and value:
                votes[int(track_id)][value] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items()}


def _nearest_holder(frame: FrameData) -> Optional[dict]:
    if frame.ball_xy is None:
        return None
    bx, by = frame.ball_xy
    players = [p for p in frame.players if p.get("role") != "referee"]
    if not players:
        return None
    nearest = min(players, key=lambda p: math.hypot(p["x"] - bx, p["y"] - by))
    return nearest if math.hypot(nearest["x"] - bx, nearest["y"] - by) <= POSSESSION_RADIUS_M else None


def possession_segments(
    frames: list[FrameData],
    team_by_track: dict[int, str],
    min_frames: int = POSSESSION_MIN_FRAMES,
) -> list[PossessionSegment]:
    segments: list[PossessionSegment] = []
    current: Optional[dict] = None
    candidate: Optional[dict] = None

    def close() -> None:
        nonlocal current
        if current:
            segments.append(PossessionSegment(
                track_id=current["track_id"],
                team=team_by_track.get(current["track_id"]),
                jersey=current["start_holder"].get("jersey"),
                start_fid=current["start_frame"].frame_id,
                end_fid=current["end_frame"].frame_id,
                start_xy=(current["start_holder"]["x"], current["start_holder"]["y"]),
                end_xy=(current["end_holder"]["x"], current["end_holder"]["y"]),
            ))
        current = None

    for frame in sorted(frames, key=lambda item: item.frame_id):
        holder = _nearest_holder(frame)
        track_id = int(holder["track_id"]) if holder and holder.get("track_id") is not None else None
        if track_id is None:
            close()
            candidate = None
            continue
        if current and current["track_id"] == track_id:
            current["end_frame"], current["end_holder"] = frame, holder
            candidate = None
            continue
        if not candidate or candidate["track_id"] != track_id:
            candidate = {"track_id": track_id, "count": 1}
        else:
            candidate["count"] += 1
        if candidate["count"] < min_frames:
            continue
        close()
        current = {
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
            "start_holder": holder,
            "end_holder": holder,
        }
        candidate = None
    close()
    return segments


def _third(x: float) -> str:
    return "left third" if x < -17.5 else "right third" if x > 17.5 else "middle third"


def build_tracking_digest(predictions_json: Path, fps: int = 25) -> str:
    frames = load_frames(predictions_json)
    if not frames:
        return "[Tracking Digest]\n(no tracking data available)"

    team_by_track = _majority(frames, "team")
    role_by_track = _majority(frames, "role")
    segments = possession_segments(frames, team_by_track)
    parts = ["[Team Rosters - jersey numbers seen by the tracking system]"]
    jerseys: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        for player in frame.players:
            if player.get("team") and player.get("jersey"):
                jerseys[player["team"]].add(str(player["jersey"]))
    for team in sorted(jerseys):
        numbers = ", ".join(f"#{n}" for n in sorted(jerseys[team], key=lambda n: (len(n), n)))
        parts.append(f"{team} team: {numbers}")
    if not jerseys:
        parts.append("(no jersey numbers read)")

    parts.append("\n[Possession Timeline] (left/right thirds are screen-space pitch halves)")
    for segment in segments:
        start = max(0, segment.start_fid - POSSESSION_MIN_FRAMES) / fps
        end = segment.end_fid / fps
        jersey = f"#{segment.jersey}" if segment.jersey else "unknown number"
        parts.append(
            f"t={start:.1f}-{end:.1f}s: {jersey} "
            f"({segment.team or 'unknown team'}, {role_by_track.get(segment.track_id, 'player')}) "
            f"holds the ball in the {_third(segment.start_xy[0])}"
        )
    if not segments:
        parts.append("(no stable possession detected)")
    return "\n".join(parts)
```

- [ ] **Step 2: Compile the self-contained digest**

Run:

```bash
python -m py_compile pipeline/stage2b/digest.py
```

Expected: exit 0.

---

### Task 5: Make Stage 2B generation independent of Stage 4

**Files:**
- Modify: `pipeline/stage2b/generate_direct.py`
- Modify: `pipeline/stage2b/run.py`

- [ ] **Step 1: Replace `pipeline/stage2b/generate_direct.py`**

```python
"""Direct ARK video-to-events/commentary generation for Stage 2B."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

from pipeline.config import load_env

log = logging.getLogger(__name__)

EVENTS = {
    "football.pass": "pass between teammates",
    "football.shoot": "shot toward goal",
    "football.goal": "goal scored",
    "football.clearance": "defensive clearance",
    "football.interception": "pass intercepted",
    "football.dribble": "controlled carry past pressure",
    "football.tackle": "challenge for the ball",
    "football.pressing": "active pressure on the ball carrier",
    "football.save": "goalkeeper save",
    "football.goal_kick": "goalkeeper restart or long kick",
    "football.buildup": "routine possession buildup",
}
ENERGY_LEVELS = {"calm", "engaged", "excited", "explosive"}
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260428"


def ark_model() -> str:
    load_env()
    return (
        os.environ.get("ARK_VIDEO_MODEL")
        or os.environ.get("ARK_RESPONSES_MODEL")
        or os.environ.get("DOUBAO_MODEL")
        or DEFAULT_ARK_MODEL
    )


def ark_generate(prompt: str, clip_mp4: Path) -> str:
    load_env()
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY or DOUBAO_API_KEY in the environment or .env")
    if not clip_mp4.is_file() or clip_mp4.suffix.lower() != ".mp4":
        raise FileNotFoundError(f"Stage 2B video not found or not mp4: {clip_mp4}")
    from openai import OpenAI

    encoded = base64.b64encode(clip_mp4.read_bytes()).decode("ascii")
    response = OpenAI(
        api_key=api_key,
        base_url=(
            os.environ.get("ARK_BASE_URL")
            or os.environ.get("DOUBAO_BASE_URL")
            or DEFAULT_ARK_BASE_URL
        ),
    ).chat.completions.create(
        model=ark_model(),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
        ]}],
        max_tokens=4096,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def _normalize_segment(segment: dict, index: int) -> dict:
    timestamp = segment.get("timestamp_s", segment.get("start_s", segment.get("t", index * 1.5)))
    refs = segment.get("events_referenced") or segment.get("event_ids") or []
    if isinstance(refs, str):
        refs = [refs]
    normalized = {
        "timestamp_s": float(timestamp),
        "end_s": float(segment.get("end_s", float(timestamp) + 1.5)),
        "text_en": str(segment.get("text_en") or segment.get("en_commentary") or segment.get("en") or ""),
        "text_zh": str(segment.get("text_zh") or segment.get("zh_commentary") or segment.get("zh") or ""),
        "events_referenced": list(refs),
    }
    if str(segment.get("energy", "")).lower() in ENERGY_LEVELS:
        normalized["energy"] = str(segment["energy"]).lower()
    return normalized


def validate_pacing(segments: list[dict], duration_s: float) -> list[str]:
    problems = []
    minimum = max(1, int(duration_s // 5))
    if len(segments) < minimum:
        problems.append(f"only {len(segments)} segments; need at least {minimum}")
    for segment in segments:
        window = segment["end_s"] - segment["timestamp_s"]
        if window > 7:
            problems.append(f"segment {segment['timestamp_s']}-{segment['end_s']}s spans {window:.1f}s; max 7s")
        elif window < 2:
            problems.append(f"segment {segment['timestamp_s']}-{segment['end_s']}s spans {window:.1f}s; min 2s")
    return problems


def build_direct_prompt(digest: str, duration_s: float, languages: list[str]) -> str:
    menu = "\n".join(f"- {code}: {description}" for code, description in EVENTS.items())
    minimum = max(1, int(duration_s // 5))
    return f"""You are a professional football commentator and match analyst.
Watch the attached {duration_s:.0f}-second clip from start to end.

[Reliable Stage 1 tracking digest]
{digest}

Choose event_code only from:
{menu}

Generate {' and '.join(languages).upper()} commentary. Use jersey numbers only
when visible or confirmed by the digest. energy must be calm, engaged, excited,
or explosive. Produce at least {minimum} segments, no segment longer than 7
seconds or shorter than 2 seconds, and cover the clip without dead air.

Return one JSON object only:
{{"events": [{{"event_id": "evt_001", "timestamp_s": 0.0,
"event_code": "football.buildup", "player_jersey": "", "player_team": "left",
"description": "short factual description"}}],
"commentary": [{{"timestamp_s": 0.0, "end_s": 5.0, "text_en": "...",
"text_zh": "...", "energy": "calm", "events_referenced": ["evt_001"]}}]}}
Every referenced event id must exist."""


def parse_direct_output(raw_text: str) -> tuple[list[dict], list[dict]]:
    match = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
    if not match:
        raise ValueError("LLM output contains no JSON object")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output JSON is malformed: {exc}") from exc
    events = [event for event in data.get("events", []) if isinstance(event, dict) and event.get("event_id")]
    invalid_codes = [event.get("event_code") for event in events if event.get("event_code") not in EVENTS]
    if invalid_codes:
        raise ValueError(f"events contain unsupported event codes: {invalid_codes}")
    raw_segments = data.get("commentary")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError('LLM output has no "commentary" array')
    segments = [_normalize_segment(segment, index) for index, segment in enumerate(raw_segments) if isinstance(segment, dict)]
    if not segments:
        raise ValueError("LLM output has no usable commentary entries")
    known_ids = {event["event_id"] for event in events}
    for segment in segments:
        unknown = [ref for ref in segment["events_referenced"] if ref not in known_ids]
        if unknown:
            raise ValueError(f"commentary references unknown event ids: {unknown}")
    return events, segments


def generate_direct(
    clip_mp4: Path,
    digest: str,
    duration_s: float,
    fps: int,
    output_dir: Path,
    languages: list[str],
    generate: Callable[[str, Path], str] = ark_generate,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_direct_prompt(digest, duration_s, languages)

    def attempt(text: str):
        try:
            events, segments = parse_direct_output(generate(text, clip_mp4))
        except ValueError as exc:
            return None, None, [str(exc)]
        return events, segments, validate_pacing(segments, duration_s)

    events, segments, problems = attempt(prompt)
    if problems:
        log.warning("Stage 2B attempt rejected: %s; retrying once", problems)
        events, segments, problems = attempt(
            prompt + "\nFix these previous-response problems and return the full JSON again:\n- " + "\n- ".join(problems)
        )
        if segments is None or problems:
            raise RuntimeError(f"Stage 2B generation failed after retry: {problems}")

    video_info = {"source": str(clip_mp4), "duration_s": duration_s, "fps": fps}
    (output_dir / "events.json").write_text(
        json.dumps({"video_info": video_info, "events": events}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commentary_path = output_dir / "commentary.json"
    commentary_path.write_text(json.dumps({
        "video_info": video_info,
        "model_info": {"name": ark_model(), "backend": "doubao-direct-2b"},
        "language": languages,
        "commentary": segments,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return commentary_path
```

- [ ] **Step 2: Replace `pipeline/stage2b/run.py`**

```python
#!/usr/bin/env python3
"""Stage 2B CLI: Stage 1 tracking + clip video -> events/commentary JSON."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from pipeline.stage2b.digest import build_tracking_digest
    from pipeline.stage2b.generate_direct import ark_model, generate_direct
    from pipeline.stage2b.video import build_clip_mp4, video_duration_s

    if not args.predictions.is_file():
        raise FileNotFoundError(f"Stage 1 predictions not found: {args.predictions}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_mp4 = args.output_dir / "clip.mp4"
    if args.force or not clip_mp4.exists():
        build_clip_mp4(args.clip_dir / "img1", clip_mp4, fps=args.fps)
    if not args.force and (args.output_dir / "commentary.json").exists():
        log.info("Reusing %s; use --force to regenerate", args.output_dir / "commentary.json")
        return

    duration_s = video_duration_s(clip_mp4)
    log.info("Calling %s with %.1fs video", ark_model(), duration_s)
    output = generate_direct(
        clip_mp4=clip_mp4,
        digest=build_tracking_digest(args.predictions, fps=args.fps),
        duration_s=duration_s,
        fps=args.fps,
        output_dir=args.output_dir,
        languages=["en", "zh"],
    )
    log.info("Stage 2B complete: %s", output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Confirm Stage 2B no longer imports removed stages**

Run:

```bash
python -m py_compile pipeline/stage2b/digest.py pipeline/stage2b/generate_direct.py pipeline/stage2b/run.py
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary" pipeline/stage2b
```

Expected: compilation exits 0 and `rg` prints nothing.

- [ ] **Step 4: Commit the self-contained Stage 2B package**

```bash
git add pipeline/stage2b
git commit --only pipeline/stage2b -m "refactor: make stage2b self contained"
```

---

### Task 6: Consolidate Stage 5 into one runner

**Files:**
- Create: `pipeline/stage5_tts/run.py`
- Preserve current retained implementation: `pipeline/stage5_tts/synthesize.py`
- Preserve current retained implementation: `pipeline/stage5_tts/mux.py`
- Preserve current retained implementation: `pipeline/stage5_tts/pace_filter.py`
- Preserve current retained implementation: `pipeline/stage5_tts/tts_adapter.py`
- Preserve current retained implementation: `pipeline/stage5_tts/adapters/edge_tts_adapter.py`
- Add currently untracked retained implementation: `pipeline/stage5_tts/adapters/cosyvoice_adapter.py`

- [ ] **Step 1: Create `pipeline/stage5_tts/run.py`**

```python
#!/usr/bin/env python3
"""Stage 5 CLI: synthesize Edge preview, CosyVoice clone, or both."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from pipeline.config import load_env

log = logging.getLogger(__name__)


def select_video(output_dir: Path, explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else [output_dir / "annotated_video.mp4", output_dir / "clip.mp4"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No base video found in {output_dir}; run Stage 2B or pass --video")


def _paths(output_dir: Path, language: str, voice: str) -> tuple[Path, Path, str]:
    suffix = "_en" if language == "en" else ""
    if voice == "preview":
        return (
            output_dir / f"commentary_{language}_default.mp3",
            output_dir / f"raw_final_video{suffix}.mp4",
            f"default_{language}",
        )
    return output_dir / f"commentary_{language}.mp3", output_dir / f"final_video{suffix}.mp4", "wang"


def render_voice(
    output_dir: Path,
    video: Path,
    language: str,
    voice: str,
    force: bool,
) -> Path:
    from pipeline.stage5_tts.mux import mux_audio_video
    from pipeline.stage5_tts.synthesize import synthesize_commentary

    commentary = output_dir / "commentary.json"
    if not commentary.is_file():
        raise FileNotFoundError(f"Stage 2B commentary not found: {commentary}")
    audio, final_video, voice_tag = _paths(output_dir, language, voice)
    if force:
        shutil.rmtree(output_dir / "tts_segments" / language / voice_tag, ignore_errors=True)
        audio.unlink(missing_ok=True)
    if not audio.exists():
        if voice == "preview":
            from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
            adapter = EdgeTTSAdapter(language=language)
        else:
            from pipeline.stage5_tts.adapters.cosyvoice_adapter import CosyVoiceAdapter
            load_env()
            adapter = CosyVoiceAdapter(language=language)
        synthesize_commentary(
            commentary,
            output_dir,
            language=language,
            adapter=adapter,
            events_json=output_dir / "events.json",
            voice_tag=voice_tag,
            audio_path=audio,
        )
    else:
        log.info("Reusing %s", audio)
    mux_audio_video(video, audio, final_video)
    print(final_video)
    return final_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--voice", choices=["preview", "clone", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    video = select_video(args.output_dir, args.video)
    voices = ("preview", "clone") if args.voice == "both" else (args.voice,)
    for voice in voices:
        render_voice(args.output_dir, video, args.language, voice, args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the Stage 5 CLI loads without loading a TTS model**

Run:

```bash
python -m pipeline.stage5_tts.run --help
python -m py_compile pipeline/stage5_tts/run.py
```

Expected: help lists `--voice {preview,clone,both}` and compilation exits 0.

- [ ] **Step 3: Commit the consolidated runner**

```bash
git add \
  pipeline/stage5_tts/run.py \
  pipeline/stage5_tts/synthesize.py \
  pipeline/stage5_tts/mux.py \
  pipeline/stage5_tts/pace_filter.py \
  pipeline/stage5_tts/tts_adapter.py \
  pipeline/stage5_tts/adapters/edge_tts_adapter.py \
  pipeline/stage5_tts/adapters/cosyvoice_adapter.py
git commit --only \
  pipeline/stage5_tts/run.py \
  pipeline/stage5_tts/synthesize.py \
  pipeline/stage5_tts/mux.py \
  pipeline/stage5_tts/pace_filter.py \
  pipeline/stage5_tts/tts_adapter.py \
  pipeline/stage5_tts/adapters/edge_tts_adapter.py \
  pipeline/stage5_tts/adapters/cosyvoice_adapter.py \
  -m "refactor: consolidate stage5 runner"
```

---

### Task 7: Canonicalize shell launchers and the single output directory

**Files:**
- Create: `scripts/run_stage2b.sh`
- Modify: `scripts/run_stage1.sh`
- Modify: `scripts/run_stage3.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Create the canonical `scripts/run_stage2b.sh`**

```bash
#!/usr/bin/env bash
# Stage 2B + Stage 5: direct video commentary and both voices in one output dir.
# Usage: bash scripts/run_stage2b.sh outputs/SNGS-116 [clip_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/SNGS-116}"
SEQ_NAME="$(basename "$OUTPUT_DIR")"
CLIP_DIR="${2:-codes/sn-gamestate/datasets/SoccerNetGS/test/${SEQ_NAME}}"
PYTHON_BIN="${PYTHON:-python3}"
TTS_LANGUAGE="${TTS_LANGUAGE:-zh}"
FORCE="${FORCE:-0}"

for command in "$PYTHON_BIN" ffmpeg ffprobe; do
  command -v "$command" >/dev/null 2>&1 || { echo "ERROR: command not found: $command" >&2; exit 1; }
done
[[ -f "$OUTPUT_DIR/predictions.json" ]] || { echo "ERROR: Stage 1 predictions missing: $OUTPUT_DIR/predictions.json" >&2; exit 1; }
[[ -d "$CLIP_DIR/img1" ]] || { echo "ERROR: clip frames missing: $CLIP_DIR/img1" >&2; exit 1; }

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
FORCE_FLAG=()
[[ "$FORCE" == "1" ]] && FORCE_FLAG=(--force)

"$PYTHON_BIN" -m pipeline.stage2b.run \
  --clip-dir "$CLIP_DIR" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --output-dir "$OUTPUT_DIR" \
  "${FORCE_FLAG[@]}"

"$PYTHON_BIN" -m pipeline.stage5_tts.run \
  --output-dir "$OUTPUT_DIR" \
  --language "$TTS_LANGUAGE" \
  --voice both \
  "${FORCE_FLAG[@]}"
```

- [ ] **Step 2: Update Stage 1 and Stage 3 wording**

Apply:

```diff
--- a/scripts/run_stage1.sh
+++ b/scripts/run_stage1.sh
@@
-# Re-run Stage 2–5 manually when you want to refresh those.
+# Run Stage 2B next; Stage 3 is optional and Stage 5 is driven by Stage 2B.
--- a/scripts/run_stage3.sh
+++ b/scripts/run_stage3.sh
@@
-# Reads:  <output_dir>/events.json (Stage 2)
+# Reads:  <output_dir>/events.json (Stage 2B)
@@
-  echo "ERROR: $OUTPUT_DIR/events.json not found — run Stage 2 first (scripts/run_stage2.sh)." >&2
+  echo "ERROR: $OUTPUT_DIR/events.json not found — run Stage 2B first (scripts/run_stage2b.sh)." >&2
```

- [ ] **Step 3: Restore a simple root-tests ignore rule**

Apply:

```diff
@@
-tests/*
-!tests/test_stage2b.py
+/tests/
```

- [ ] **Step 4: Parse every retained shell script**

Run:

```bash
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/setup_cosyvoice.sh
```

Expected: exit 0.

- [ ] **Step 5: Commit canonical launchers**

```bash
git add .gitignore scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/setup_cosyvoice.sh
git commit --only \
  .gitignore scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/setup_cosyvoice.sh \
  -m "refactor: canonicalize pipeline launchers"
```

---

### Task 8: Delete obsolete canonical stages, utilities, adapters, scripts, data, and tests

**Files:**
- Delete: every canonical obsolete path listed in the File Map

- [ ] **Step 1: Confirm retained code has no old-stage imports before deletion**

Run:

```bash
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary" \
  pipeline/config.py pipeline/run.py pipeline/stage1_inference pipeline/stage2b \
  pipeline/stage3_effects pipeline/stage5_tts pipeline/topology pipeline/utils scripts
```

Expected: any remaining matches occur only inside paths scheduled for deletion. If a match occurs in a retained file, stop and remove that dependency first.

- [ ] **Step 2: Delete tracked obsolete paths**

Run:

```bash
git rm -r -f \
  pipeline/stage2_events \
  pipeline/stage4_commentary \
  pipeline/stage3_effects/render_preview.py \
  pipeline/stage5_tts/adapters/doubao_tts.py \
  pipeline/stage5_tts/make_final_video.py \
  pipeline/stage5_tts/make_raw_final_video.py \
  pipeline/stage5_tts/preview.py \
  pipeline/stage1_inference/check_recompute.py \
  pipeline/stage1_inference/probe_concat_tracklets_by_reid.py \
  pipeline/stage1_inference/probe_heatmap_peaks.py \
  pipeline/stage1_inference/recompute_calibration.sh \
  pipeline/utils/clip_index_to_events.py \
  pipeline/utils/validate_clip_index.py \
  scripts/run_stage2.sh \
  scripts/run_stage4.sh \
  scripts/run_stage5.sh \
  scripts/run_tts.sh \
  scripts/smoke_doubao_tts.py \
  reference_football.csv \
  tests
```

Then remove the untracked canonical caption script if present:

```bash
rm -f scripts/run_caption_verify.sh
rm -rf pipeline/stage2_events pipeline/stage4_commentary
```

Expected: `git status --short` shows deletions only for the approved obsolete paths.

- [ ] **Step 3: Compile before committing deletion**

Run:

```bash
python -m compileall -q pipeline
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary" pipeline scripts
```

Expected: compilation exits 0 and `rg` returns no matches.

- [ ] **Step 4: Commit canonical deletion only**

```bash
git commit --only \
  pipeline/stage2_events pipeline/stage4_commentary \
  pipeline/stage3_effects/render_preview.py pipeline/stage5_tts \
  pipeline/stage1_inference pipeline/utils scripts reference_football.csv tests \
  -m "refactor: remove replaced pipeline stages"
```

---

### Task 9: Delete root mirrors and misplaced generated artifacts

**Files:**
- Delete: every untracked root mirror and `pipeline/SNGS-117-2b/`

- [ ] **Step 1: Reconfirm canonical files exist before deleting mirrors**

Run:

```bash
test -f pipeline/stage1_inference/run_gsr.py
test -f pipeline/stage2b/run.py
test -f pipeline/stage3_effects/render.py
test -f pipeline/stage5_tts/run.py
test -f pipeline/topology/analysis.py
test -f scripts/run_stage2b.sh
```

Expected: exit 0.

- [ ] **Step 2: Delete root mirrors and generated output**

Run exactly:

```bash
rm -rf \
  __init__.py config.py run.py video.py \
  stage1_inference stage2_events stage2b stage3_effects \
  stage4_commentary stage5_tts topology utils \
  probe_ark_video.py run_caption_verify.sh run_stage1.sh run_stage2.sh \
  run_stage2b.sh run_stage3.sh run_stage4.sh run_stage5.sh run_tts.sh \
  setup_cosyvoice.sh smoke_doubao_tts.py \
  pipeline/SNGS-117-2b
```

Expected: canonical `pipeline/` and `scripts/` files remain; root mirrors are absent.

- [ ] **Step 3: Confirm the repository root is clean of mirrored application entrypoints**

Run:

```bash
find . -maxdepth 1 -type f \( -name 'run*.py' -o -name 'run*.sh' -o -name 'config.py' -o -name 'video.py' \) -print
find . -maxdepth 1 -type d \( -name 'stage*' -o -name 'topology' -o -name 'utils' \) -print
```

Expected: both commands print nothing.

---

### Task 10: Final verification and scope audit

**Files:**
- Verify: all retained production files
- Modify: none unless a verification failure directly identifies an incomplete cleanup

- [ ] **Step 1: Run all approved non-test verification**

```bash
python -m compileall -q pipeline
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/setup_cosyvoice.sh
python -m pipeline.stage2b.run --help
python -m pipeline.stage5_tts.run --help
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary" pipeline scripts
git diff --check
```

Expected:

- Compilation and shell parsing exit 0.
- Both help commands show their arguments and exit 0.
- The old-stage import search prints nothing.
- `git diff --check` prints nothing.

- [ ] **Step 2: Confirm retained visual effects and topology**

Run:

```bash
test -f pipeline/stage3_effects/render.py
test -f pipeline/stage3_effects/overlay.py
test -f pipeline/stage3_effects/topology_analysis.py
test -f pipeline/topology/analysis.py
test -f pipeline/topology/io_gamestate.py
test -f pipeline/topology/lines.py
```

Expected: exit 0.

- [ ] **Step 3: Review final scope and line reduction**

Run:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
```

Expected: changes are limited to the approved cleanup, the previously approved obsolete deletions, and the two cleanup documents. No `codes/`, formation-topology asset, model, voice sample, or output-directory data is removed.

- [ ] **Step 4: Perform optional live checks only where prerequisites exist**

Stage 1 GPU:

```bash
bash scripts/run_stage1.sh \
  codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116 \
  outputs/SNGS-116
```

Stage 2B + Stage 5:

```bash
bash scripts/run_stage2b.sh \
  outputs/SNGS-116 \
  codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116
```

Optional Stage 3, followed by re-muxing Stage 5 onto the annotated video:

```bash
bash scripts/run_stage3.sh \
  outputs/SNGS-116 \
  codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116
python -m pipeline.stage5_tts.run --output-dir outputs/SNGS-116 --voice both --force
```

Expected when models, credentials, hardware, and network are available: Stage 1 writes predictions/homography; Stage 2B writes clip/events/commentary into the same directory; Stage 3 writes `annotated_video.mp4`; Stage 5 writes preview and clone final videos.

---

## Final acceptance checklist

- [ ] Only `pipeline/` contains application packages.
- [ ] Only `scripts/` contains shell launchers.
- [ ] Stage 2B imports no Stage 2 or Stage 4 code.
- [ ] Stage 5 has one runner and retains Edge + CosyVoice.
- [ ] Stage 3 effects and topology remain present.
- [ ] One output directory carries Stage 1 through Stage 5 artifacts.
- [ ] `tests/` and all test-only code are absent.
- [ ] No unrelated user-owned changes were discarded or swept into a cleanup commit.
