# Stage 2b: Direct Video→Commentary A/B Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An A/B alternative pipeline ("stage 2b") that skips Stages 2/3/4 entirely: Doubao watches the raw clip video plus a tracking digest and produces commentary.json + events.json directly, then the existing Stage-5 TTS (edge-tts preview + CosyVoice3 clone) speaks it.

**Architecture:** One new package `pipeline/stage2b/` (clip builder, tracking digest, direct generation) plus video support in the existing Doubao adapter. Arm B writes to a sibling directory `outputs/<SEQ>-2b/` so both arms coexist per clip. Everything downstream of commentary.json is shared with arm A — energy tiers, pacing validation + single retry, no-truncation speech, CosyVoice clone — so the A/B isolates exactly one variable: how commentary gets produced. The error-prone Stage-2 detector/classifier and Stage-3 renderer are not imported; only pure-geometry possession helpers are reused for the digest.

**Tech Stack:** Python 3.10, pytest, ffmpeg/ffprobe, Volcengine ARK chat-completions with `video_url` content (base64 mp4), existing Stage-5 TTS stack.

**Decisions locked in the grill-me session (2026-07-09):**
- Video pathway A: `img1/` frames → `clip.mp4` → base64 `video_url`. NO frame-sampling fallback — if the model rejects video, that is a result, not a bug to engineer around. `LLMAdapter.prepare_visual_input`'s silent frame-extraction degradation must NOT trigger on the 2b path.
- Probe first: `doubao-seed-2-0-lite-260428` may or may not accept video; a 1-minute probe script is the go/no-go gate. `ARK_VIDEO_MODEL` env var overrides the model for video calls if the user enables a vision model in the ARK console.
- Digest = roster (jerseys per team) + possession timeline only, from [possession.py](pipeline/stage2_events/possession.py) pure geometry. No detector candidates, no classifier.
- Doubao's single response is one JSON object `{"events": [...], "commentary": [...]}` → split into `events.json` + `commentary.json` (same schemas as arm A; `events_referenced` links to the new event ids).
- Output layout: sibling dir `outputs/<SEQ>-2b/` containing `clip.mp4`, `events.json`, `commentary.json`, `raw_final_video.mp4`, `final_video.mp4`, `tts_segments/`.
- `make_raw_final_video.py` and `make_final_video.py` gain a `--video` flag (default `annotated_video.mp4` — arm A untouched).
- `run_stage2b.sh` drives the full chain including both TTS steps; on a machine without CosyVoice the clone step fails with its existing clear setup error.
- Reuse from the 2026-07-08 plan (already executed): `postprocess.parse` conventions, `validate_pacing` + single retry, `energy` enum, `synthesize_commentary`, `CosyVoiceAdapter`.
- A/B evaluation is manual (watch both videos, read both commentary.json) — no eval tooling.

**Commit policy (user instruction, 2026-07-09):** make NO intermediate commits. All tasks run against the working tree; exactly ONE commit happens at the end (Task 8 Step 4), after the full test suite and the E2E run pass. Note the working tree also contains the executed-but-uncommitted 2026-07-08 plan (rsync'd from the GPU server) — the final commit sweeps that in too.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/probe_ark_video.py` | Create | go/no-go: does the ARK model accept `video_url`? |
| `pipeline/stage2b/__init__.py` | Create | package marker |
| `pipeline/stage2b/video.py` | Create | `img1/` frames → H.264 `clip.mp4` (≤720p) + duration probe |
| `pipeline/stage2b/digest.py` | Create | predictions.json → roster + possession-timeline text |
| `pipeline/stage2b/generate_direct.py` | Create | prompt build, ARK call, parse/split, validate+retry, write both JSONs |
| `pipeline/stage2b/run.py` | Create | CLI entry (`python -m pipeline.stage2b.run`) |
| `pipeline/stage4_commentary/adapters/doubao_api.py` | Modify | opt-in `video_url` support |
| `pipeline/stage4_commentary/postprocess.py` | Modify | expose public `normalize_segment` alias |
| `pipeline/stage5_tts/make_raw_final_video.py` | Modify | `--video` flag |
| `pipeline/stage5_tts/make_final_video.py` | Modify | `--video` flag |
| `scripts/run_stage2b.sh` | Create | full-chain driver for arm B |
| `tests/test_stage2b.py` | Create | digest, adapter content, parse/split, retry, CLI plumbing |

---

### Task 1: ARK video probe (go/no-go gate)

**Files:**
- Create: `scripts/probe_ark_video.py`

No pytest — this task's product IS a manual check against the live API.

- [ ] **Step 1: Write the probe script**

Create `scripts/probe_ark_video.py`:

```python
#!/usr/bin/env python3
"""Go/no-go probe: can our ARK model ingest a video_url content part?

Builds a 2-second test mp4 from the first 50 frames of a clip's img1/ dir,
sends it base64-embedded to the ARK chat-completions endpoint, and asks a
question only answerable by actually watching the video.

Usage:
    python scripts/probe_ark_video.py [clip_dir]
    # default clip_dir: codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116

Model: $ARK_VIDEO_MODEL if set, else $ARK_RESPONSES_MODEL (.env is loaded).
Exit 0 + "PROBE OK" when the model answers about video content.
Exit 1 + the API error when the model/endpoint rejects video input.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.stage4_commentary.generate import load_ark_env  # noqa: E402


def build_test_clip(frames_dir: Path, out_path: Path, n_frames: int = 50, fps: int = 25) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))[:n_frames]
    if not frames:
        raise SystemExit(f"No jpg frames in {frames_dir}")
    list_file = out_path.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{f.resolve()}'\nduration {1 / fps}\n" for f in frames),
        encoding="utf-8",
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "28",
         "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True,
    )
    list_file.unlink()


def main() -> None:
    load_ark_env()
    clip_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116"
    )
    model = os.environ.get("ARK_VIDEO_MODEL") or os.environ.get("ARK_RESPONSES_MODEL", "")
    api_key = os.environ.get("ARK_API_KEY", "")
    base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    if not (model and api_key):
        raise SystemExit("Need ARK_RESPONSES_MODEL/ARK_VIDEO_MODEL and ARK_API_KEY (.env)")

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "probe.mp4"
        build_test_clip(clip_dir / "img1", clip)
        b64 = base64.b64encode(clip.read_bytes()).decode("utf-8")
        print(f"model={model}  clip={clip.stat().st_size} bytes")

        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text":
                     "This is a short football clip. In one sentence: what sport is "
                     "shown and roughly how many players are visible?"},
                    {"type": "video_url",
                     "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                ]}],
                max_tokens=200,
            )
        except Exception as exc:  # deliberate: report ANY rejection verbatim
            print(f"PROBE FAILED — model rejected video input:\n{exc}")
            raise SystemExit(1)

    print("Model answer:", resp.choices[0].message.content)
    print("PROBE OK — video input accepted")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

Run: `python scripts/probe_ark_video.py`
Expected either:
- `PROBE OK — video input accepted` → proceed with the plan as-is, or
- `PROBE FAILED — model rejected video input: ...` → **STOP and report to the user**: they must enable a video-capable model in the ARK console (e.g. a doubao-seed vision variant) and set `ARK_VIDEO_MODEL=<id>` in `.env`, then re-run the probe. Per the locked decision there is no frame-sampling fallback.

(No commit — single commit at the end per the commit policy.)

---

### Task 2: Clip builder — frames → clip.mp4

**Files:**
- Create: `pipeline/stage2b/__init__.py`, `pipeline/stage2b/video.py`
- Test: `tests/test_stage2b.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage2b.py`:

```python
"""Stage 2b tests: clip builder, tracking digest, direct generation."""
import json
import subprocess
from pathlib import Path

import pytest

from pipeline.stage2b.video import build_clip_mp4, video_duration_s


def _make_test_frames(frames_dir: Path, n: int = 10) -> None:
    """Generate n numbered jpg frames with ffmpeg's testsrc."""
    frames_dir.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25",
         "-frames:v", str(n), str(frames_dir / "%06d.jpg")],
        check=True, capture_output=True,
    )


def test_build_clip_mp4_encodes_frames(tmp_path):
    frames = tmp_path / "img1"
    _make_test_frames(frames, n=25)
    out = build_clip_mp4(frames, tmp_path / "clip.mp4", fps=25)
    assert out.exists() and out.stat().st_size > 0
    assert video_duration_s(out) == pytest.approx(1.0, abs=0.2)


def test_build_clip_mp4_fails_on_empty_dir(tmp_path):
    empty = tmp_path / "img1"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No frames"):
        build_clip_mp4(empty, tmp_path / "clip.mp4", fps=25)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: FAIL — `pipeline.stage2b` module not found.

- [ ] **Step 3: Implement**

Create `pipeline/stage2b/__init__.py` (empty file) and `pipeline/stage2b/video.py`:

```python
"""Build the raw clip mp4 that Doubao watches (and that final videos mux onto)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def build_clip_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: int = 25,
    max_height: int = 720,
) -> Path:
    """Encode img1/ jpg frames into an H.264 mp4, downscaled to ≤max_height.

    720p CRF 26 keeps a 30s clip well under ARK's base64 request budget while
    staying legible for jersey numbers.
    """
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames (*.jpg) found in {frames_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = output_path.with_suffix(".frames.txt")
    list_file.write_text(
        "".join(f"file '{f.resolve()}'\nduration {1 / fps}\n" for f in frames),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vf", f"scale=-2:'min({max_height},ih)'",
        "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink()
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip encoding failed:\n{result.stderr[-800:]}")
    return output_path


def video_duration_s(path: Path) -> float:
    """Duration in seconds via ffprobe. Raises on failure — 2b needs it exact."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: 2 PASS.

(No commit — single commit at the end per the commit policy.)

---

### Task 3: Tracking digest — roster + possession timeline

**Files:**
- Create: `pipeline/stage2b/digest.py`
- Test: `tests/test_stage2b.py` (append)

Reuses ONLY pure-geometry helpers from `pipeline/stage2_events/possession.py` (`resolve_team_by_track`, `resolve_role_by_track`, `possession_segments`) and the predictions loader `pipeline/stage2_events/detector.load_frames`. The detector's candidate machinery is not touched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage2b.py`:

```python
from pipeline.stage2b.digest import build_tracking_digest


def _predictions_fixture(tmp_path: Path) -> Path:
    """Minimal GameState predictions.json: 30 frames, keeper #1 (left) holds
    the ball frames 1-15, then #9 (right) holds frames 16-30."""
    def ann(aid, image_id, track_id, role, team, jersey, px, py):
        return {
            "id": str(aid), "image_id": image_id, "track_id": track_id,
            "supercategory": "object", "category_id": 1,
            "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 20,
                           "x_center": 5, "y_center": 10},
            "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                           "x_bottom_left": px - 0.3, "y_bottom_left": py,
                           "x_bottom_right": px + 0.3, "y_bottom_right": py},
            "attributes": {"role": role, "team": team, "jersey": jersey},
        }

    def ball(aid, image_id, px, py):
        return {
            "id": str(aid), "image_id": image_id, "track_id": 99,
            "supercategory": "object", "category_id": 4,
            "bbox_image": {"x": 0, "y": 0, "w": 4, "h": 4,
                           "x_center": 2, "y_center": 2},
            "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                           "x_bottom_left": px, "y_bottom_left": py,
                           "x_bottom_right": px, "y_bottom_right": py},
            "attributes": {"role": "ball"},
        }

    images, annotations, aid = [], [], 1
    for f in range(1, 31):
        image_id = f"1{f:06d}"
        images.append({"image_id": image_id, "file_name": f"{f:06d}.jpg",
                       "width": 1920, "height": 1080})
        bx = -44.0 if f <= 15 else 20.0
        annotations.append(ball(aid, image_id, bx, 0.0)); aid += 1
        annotations.append(ann(aid, image_id, 1, "goalkeeper", "left", "1", -44.0, 0.0)); aid += 1
        annotations.append(ann(aid, image_id, 2, "player", "right", "9", 20.0, 0.0)); aid += 1
    data = {"info": {"fps": 25}, "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "player"}, {"id": 4, "name": "ball"}]}
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_digest_contains_roster_and_possession(tmp_path):
    digest = build_tracking_digest(_predictions_fixture(tmp_path), fps=25)
    assert "[Team Rosters" in digest
    assert "left" in digest and "right" in digest
    assert "#1" in digest and "#9" in digest
    assert "[Possession Timeline]" in digest
    assert "goalkeeper" in digest
    # timeline lines carry second-resolution timestamps
    assert "t=0.0" in digest


def test_digest_empty_predictions_says_so(tmp_path):
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")
    digest = build_tracking_digest(p, fps=25)
    assert "no tracking data" in digest.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v -k digest`
Expected: FAIL — `pipeline.stage2b.digest` not found.

- [ ] **Step 3: Implement**

Create `pipeline/stage2b/digest.py`:

```python
"""Compact text digest of Stage-1 tracking for the direct-VLM prompt.

Only pure-geometry facts: which jersey numbers exist per team, and who held
the ball when. Deliberately NOT the Stage-2 detector/classifier — 2b exists to
A/B against that machinery.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pipeline.stage2_events.detector import load_frames
from pipeline.stage2_events.possession import (
    possession_segments,
    resolve_role_by_track,
    resolve_team_by_track,
)


def _third(x: float) -> str:
    if x < -17.5:
        return "left third"
    if x > 17.5:
        return "right third"
    return "middle third"


def build_tracking_digest(predictions_json: Path, fps: int = 25) -> str:
    frames = load_frames(str(predictions_json))
    if not frames:
        return "[Tracking Digest]\n(no tracking data available)"

    team_by_track = resolve_team_by_track(frames)
    role_by_track = resolve_role_by_track(frames)
    segments = possession_segments(frames, team_by_track)

    parts = ["[Team Rosters — jersey numbers seen by the tracking system]"]
    jerseys: dict = defaultdict(set)
    for frame in frames:
        for player in frame.players:
            team, jersey = player.get("team"), player.get("jersey")
            if team and jersey:
                jerseys[team].add(str(jersey))
    for team in sorted(jerseys):
        nums = ", ".join(f"#{j}" for j in sorted(jerseys[team], key=lambda s: (len(s), s)))
        parts.append(f"{team} team: {nums}")
    if not jerseys:
        parts.append("(no jersey numbers read)")

    parts.append("\n[Possession Timeline] (left/right thirds are screen-space pitch halves)")
    for seg in segments:
        t0, t1 = seg.start_fid / fps, seg.end_fid / fps
        role = role_by_track.get(seg.track_id, "player")
        jersey = f"#{seg.jersey}" if seg.jersey else "unknown number"
        team = seg.team or "unknown team"
        parts.append(
            f"t={t0:.1f}-{t1:.1f}s: {jersey} ({team}, {role}) holds the ball "
            f"in the {_third(seg.start_xy[0])}"
        )
    if not segments:
        parts.append("(no stable possession detected)")

    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: all PASS.

(No commit — single commit at the end per the commit policy.)

---

### Task 4: Doubao adapter — opt-in video_url support

**Files:**
- Modify: `pipeline/stage4_commentary/adapters/doubao_api.py`
- Test: `tests/test_stage2b.py` (append)

Design: `DoubaoAPIAdapter(enable_video=True)` switches `supports_video()` to True (which also disarms `LLMAdapter.prepare_visual_input`'s silent frame extraction), uses `$ARK_VIDEO_MODEL` when set, and embeds a single `.mp4` `visual_input` as a base64 `video_url` content part. Default construction is unchanged — arm A is untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage2b.py`:

```python
from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter


def test_video_disabled_by_default():
    assert DoubaoAPIAdapter().supports_video() is False


def test_video_content_part(tmp_path, monkeypatch):
    monkeypatch.delenv("ARK_VIDEO_MODEL", raising=False)
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42fakebytes")
    adapter = DoubaoAPIAdapter(enable_video=True)
    assert adapter.supports_video() is True
    content = adapter._build_content("watch this", mp4)
    assert content[0] == {"type": "text", "text": "watch this"}
    assert content[1]["type"] == "video_url"
    assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_video_model_env_override(monkeypatch):
    monkeypatch.setenv("ARK_VIDEO_MODEL", "doubao-vision-test")
    adapter = DoubaoAPIAdapter(enable_video=True)
    assert adapter.model == "doubao-vision-test"


def test_image_content_still_works(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    adapter = DoubaoAPIAdapter()
    content = adapter._build_content("look", [img])
    assert content[1]["type"] == "image_url"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v -k "video or image_content"`
Expected: FAIL — no `enable_video` kwarg, no `_build_content` method.

- [ ] **Step 3: Implement**

In `pipeline/stage4_commentary/adapters/doubao_api.py`:

Replace `__init__` and `supports_video`, and refactor `generate` to use a testable `_build_content`:

```python
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_video: bool = False,
    ):
        self.enable_video = enable_video
        env_model = os.environ.get(
            "ARK_RESPONSES_MODEL",
            os.environ.get("DOUBAO_MODEL", DEFAULT_ARK_MODEL),
        )
        if enable_video:
            env_model = os.environ.get("ARK_VIDEO_MODEL", env_model)
        self.model = model or env_model
        self.api_key = api_key or os.environ.get(
            "ARK_API_KEY",
            os.environ.get("DOUBAO_API_KEY", ""),
        )
        self.base_url = (
            base_url
            or os.environ.get("ARK_BASE_URL")
            or os.environ.get("DOUBAO_BASE_URL")
            or DEFAULT_ARK_BASE_URL
        )

    def supports_video(self) -> bool:
        # Opt-in: stage 2b passes enable_video=True after the ARK probe
        # confirmed the account's model accepts video_url input.
        return self.enable_video

    def _build_content(
        self, prompt: str, visual_input: Union[Path, List[Path], None]
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if (
            self.enable_video
            and isinstance(visual_input, Path)
            and visual_input.suffix == ".mp4"
        ):
            b64 = base64.b64encode(visual_input.read_bytes()).decode("utf-8")
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
            })
        elif isinstance(visual_input, list):
            for img_path in visual_input[:12]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        return content
```

and slim `generate` to:

```python
    def generate(self, prompt: str, visual_input: Union[Path, List[Path], None] = None) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Missing ARK_API_KEY (or DOUBAO_API_KEY). "
                "Set it in the environment or a local .env file."
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        visual_input = self.prepare_visual_input(visual_input)
        content = self._build_content(prompt, visual_input)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: all PASS.

(No commit — single commit at the end per the commit policy.)

---

### Task 5: Direct generation — prompt, parse/split, validate+retry, write both JSONs

**Files:**
- Create: `pipeline/stage2b/generate_direct.py`
- Modify: `pipeline/stage4_commentary/postprocess.py` (public alias, 2 lines)
- Test: `tests/test_stage2b.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage2b.py`:

```python
from pipeline.stage2b.generate_direct import (
    build_direct_prompt,
    generate_direct,
    parse_direct_output,
)


def test_direct_prompt_carries_rules_digest_and_menu():
    prompt = build_direct_prompt("DIGEST-SENTINEL", duration_s=30.0,
                                 languages=["en", "zh"])
    assert "DIGEST-SENTINEL" in prompt
    assert "at least 6" in prompt                      # 30s // 5 cadence
    assert "7 characters per second" in prompt         # 1.5x budget
    assert '"calm"' in prompt and '"explosive"' in prompt
    assert "football.pass" in prompt and "football.buildup" in prompt
    assert '"events"' in prompt and '"commentary"' in prompt


def _direct_ok_response():
    return json.dumps({
        "events": [
            {"event_id": "evt_001", "timestamp_s": 3.0,
             "event_code": "football.pass", "player_jersey": "10",
             "player_team": "left", "description": "short pass"},
        ],
        "commentary": [
            {"timestamp_s": i * 5.0, "end_s": (i + 1) * 5.0, "text_en": "e",
             "text_zh": "z", "energy": "calm",
             "events_referenced": (["evt_001"] if i == 0 else [])}
            for i in range(6)
        ],
    })


def test_parse_direct_output_splits_and_normalizes():
    events, segments = parse_direct_output(_direct_ok_response())
    assert events[0]["event_id"] == "evt_001"
    assert len(segments) == 6
    assert segments[0]["energy"] == "calm"
    assert segments[0]["events_referenced"] == ["evt_001"]


def test_parse_direct_output_rejects_garbage_and_bad_refs():
    with pytest.raises(ValueError):
        parse_direct_output("I cannot watch videos, sorry.")
    bad = json.loads(_direct_ok_response())
    bad["commentary"][0]["events_referenced"] = ["evt_999"]
    with pytest.raises(ValueError, match="evt_999"):
        parse_direct_output(json.dumps(bad))


class FakeVideoAdapter:
    model = "fake-video"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.visuals = []

    def generate(self, prompt, visual_input=None):
        self.calls += 1
        self.visuals.append(visual_input)
        return self.responses.pop(0)


def test_generate_direct_writes_both_files_and_passes_video(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    adapter = FakeVideoAdapter([_direct_ok_response()])
    generate_direct(
        clip_mp4=clip, digest="D", duration_s=30.0, fps=25,
        output_dir=tmp_path, adapter=adapter, languages=["en", "zh"],
    )
    assert adapter.visuals == [clip]
    events = json.loads((tmp_path / "events.json").read_text())
    commentary = json.loads((tmp_path / "commentary.json").read_text())
    assert events["events"][0]["event_code"] == "football.pass"
    assert len(commentary["commentary"]) == 6
    assert commentary["video_info"]["duration_s"] == 30.0
    assert commentary["model_info"]["backend"] == "doubao-direct-2b"


def test_generate_direct_retries_once_then_fails(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    adapter = FakeVideoAdapter(["garbage", "still garbage"])
    with pytest.raises(RuntimeError):
        generate_direct(
            clip_mp4=clip, digest="D", duration_s=30.0, fps=25,
            output_dir=tmp_path, adapter=adapter, languages=["en", "zh"],
        )
    assert adapter.calls == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v -k direct`
Expected: FAIL — module not found.

- [ ] **Step 3: Add the public alias in postprocess**

In `pipeline/stage4_commentary/postprocess.py`, add directly below the `_normalize_segment` function definition:

```python
# Public alias — stage 2b normalizes segments from a {"events","commentary"}
# object and cannot go through parse_commentary_output's array extraction.
normalize_segment = _normalize_segment
```

- [ ] **Step 4: Implement generate_direct**

Create `pipeline/stage2b/generate_direct.py`:

```python
"""Stage 2b: Doubao watches the clip video + tracking digest and emits
{"events": [...], "commentary": [...]} directly — no detector, no classifier,
no per-event prompt building. A/B arm against stages 2/3/4.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage4_commentary.postprocess import (
    normalize_segment,
    validate_pacing,
    write_commentary_json,
)

log = logging.getLogger(__name__)

# Same action vocabulary the rule-engine arm uses, so A/B diffs align.
EVENT_MENU = [
    "football.pass", "football.shoot", "football.goal", "football.clearance",
    "football.interception", "football.dribble", "football.tackle",
    "football.pressing", "football.save", "football.goal_kick",
    "football.buildup",
]


def build_direct_prompt(digest: str, duration_s: float, languages: List[str]) -> str:
    min_segments = max(1, int(duration_s // 5))
    lang_str = " and ".join(languages).upper()
    schema = EventSchema()
    menu_lines = []
    for code in EVENT_MENU:
        ev = schema.get_event(code)
        menu_lines.append(f"- {code}: {ev.description} ({ev.display_name_cn})")
    menu = "\n".join(menu_lines)

    return f"""You are a professional football commentator AND match analyst.
Watch the attached {duration_s:.0f}-second clip carefully, from start to end.

First identify the notable moments (events), then write vivid spoken
commentary that covers the whole clip.

[Tracking Digest — reliable facts from a computer-vision tracking system]
{digest}

EVENT RULES:
1. List every clear action you see, choosing event_code ONLY from this menu:
{menu}
2. Timestamps must come from the video timeline (seconds from clip start).
3. Use jersey numbers ONLY when you can clearly read them or the digest
   confirms them; otherwise leave player_jersey empty and describe by role.
4. player_team is "left" or "right" as listed in the digest rosters.

COMMENTARY RULES:
5. Generate BOTH {lang_str} commentary for each segment.
6. energy is the voice actor's emotional intensity — exactly one of:
   "calm" (routine possession, quiet build-up),
   "engaged" (something developing: a forward pass, a press),
   "excited" (shot, dangerous attack, successful tackle/interception),
   "explosive" (goal).
7. Be vivid and specific: ball trajectory, player movement, spatial detail.
   Vary sentence structure. Avoid bland filler.

PACING (critical for TTS):
8. Budget: Chinese ≤ 7 characters per second, English ≤ 4.5 words per second.
9. At least {min_segments} segments; NO segment may span more than 7 seconds;
   none shorter than 2 seconds. Cover the full clip with no dead air.
10. Lines are spoken back-to-back; fill each window fully.

OUTPUT — a single JSON object, nothing else:
{{"events": [{{"event_id": "evt_001", "timestamp_s": 0.0,
  "event_code": "<menu code>", "player_jersey": "", "player_team": "left",
  "description": "<short factual description>"}}],
 "commentary": [{{"timestamp_s": 0.0, "end_s": 5.0, "text_en": "...",
  "text_zh": "...", "energy": "calm", "events_referenced": ["evt_001"]}}]}}
Event ids must be evt_001, evt_002, ... in timestamp order. Every id in
events_referenced must exist in "events"."""


def parse_direct_output(raw_text: str) -> Tuple[List[dict], List[dict]]:
    """Split model output into (events, normalized commentary segments).

    Raises ValueError on anything unusable — caller retries once, then fails.
    """
    match = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
    if not match:
        raise ValueError("LLM output contains no JSON object")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output JSON is malformed: {exc}") from exc

    events = [e for e in data.get("events", []) if isinstance(e, dict) and e.get("event_id")]
    raw_segments = data.get("commentary", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError('LLM output has no "commentary" array')
    segments = [
        normalize_segment(seg, i)
        for i, seg in enumerate(raw_segments)
        if isinstance(seg, dict)
    ]

    known_ids = {e["event_id"] for e in events}
    for seg in segments:
        unknown = [r for r in seg.get("events_referenced", []) if r not in known_ids]
        if unknown:
            raise ValueError(f"commentary references unknown event ids: {unknown}")
    return events, segments


def generate_direct(
    clip_mp4: Path,
    digest: str,
    duration_s: float,
    fps: int,
    output_dir: Path,
    adapter,
    languages: List[str],
) -> Path:
    """Prompt → one video call (single retry on problems) → events.json + commentary.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_direct_prompt(digest, duration_s, languages)

    def _attempt(p: str):
        raw = adapter.generate(p, clip_mp4)
        try:
            events, segs = parse_direct_output(raw)
        except ValueError as exc:
            return None, None, [str(exc)]
        return events, segs, validate_pacing(segs, duration_s)

    events, segments, problems = _attempt(prompt)
    if problems:
        log.warning("2b attempt 1 rejected: %s — retrying once", problems)
        retry_prompt = (
            prompt
            + "\n\nYour previous answer had problems:\n- "
            + "\n- ".join(problems)
            + "\nRegenerate the FULL JSON object fixing every problem."
        )
        events, segments, problems = _attempt(retry_prompt)
        if segments is None:
            raise RuntimeError(f"Stage 2b generation failed after retry: {problems}")
        if problems:
            log.warning("Problems remain after retry (accepted): %s", problems)

    video_info = {"source": str(clip_mp4), "duration_s": duration_s, "fps": fps}
    (output_dir / "events.json").write_text(
        json.dumps({"video_info": video_info, "events": events},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    model_info = {
        "name": getattr(adapter, "model", "unknown"),
        "backend": "doubao-direct-2b",
    }
    return write_commentary_json(
        segments, output_dir / "commentary.json", video_info, model_info, languages,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: all PASS.

(No commit — single commit at the end per the commit policy.)

---

### Task 6: `--video` flag for the two final-video makers

**Files:**
- Modify: `pipeline/stage5_tts/make_raw_final_video.py`
- Modify: `pipeline/stage5_tts/make_final_video.py`
- Test: `tests/test_stage2b.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage2b.py`:

```python
def test_final_video_makers_accept_video_override(tmp_path):
    from pipeline.stage5_tts.make_final_video import make_final_video
    from pipeline.stage5_tts.make_raw_final_video import make_raw_final_video
    missing = tmp_path / "clip.mp4"
    with pytest.raises(FileNotFoundError, match="clip.mp4"):
        make_final_video(tmp_path, video=missing)
    with pytest.raises(FileNotFoundError, match="clip.mp4"):
        make_raw_final_video(tmp_path, video=missing)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v -k override`
Expected: FAIL — unexpected keyword argument `video`.

- [ ] **Step 3: Implement in make_final_video.py**

Change the signature and video resolution:

```python
def make_final_video(
    output_dir: Path,
    language: str = "zh",
    force: bool = False,
    video: Path | None = None,
) -> Path:
    """<video or annotated_video> + CosyVoice3 (王楚淇) → final_video.mp4."""
```

and replace

```python
    annotated = output_dir / "annotated_video.mp4"
```
with
```python
    annotated = Path(video) if video else (output_dir / "annotated_video.mp4")
```
and the existence error message with:
```python
    if not annotated.exists():
        raise FileNotFoundError(f"base video not found: {annotated}")
```

In `main()`, add the argument and pass it through:

```python
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Base video to mux onto (default: <output-dir>/annotated_video.mp4)",
    )
```
```python
    make_final_video(args.output_dir, language=args.language, force=args.force,
                     video=args.video)
```

- [ ] **Step 4: Implement the same change in make_raw_final_video.py**

Same three edits: `video: Path | None = None` parameter on `make_raw_final_video`, `annotated = Path(video) if video else (output_dir / "annotated_video.mp4")`, error message `f"base video not found: {annotated}"`, plus the same `--video` argparse option passed through in its `main()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage2b.py -v`
Expected: all PASS.

(No commit — single commit at the end per the commit policy.)

---

### Task 7: CLI entry + full-chain driver script

**Files:**
- Create: `pipeline/stage2b/run.py`
- Create: `scripts/run_stage2b.sh`

- [ ] **Step 1: Create the CLI entry**

Create `pipeline/stage2b/run.py`:

```python
#!/usr/bin/env python3
"""Stage 2b CLI: clip video + tracking digest → Doubao → events/commentary JSONs.

Usage:
    python -m pipeline.stage2b.run \
        --clip-dir codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-116 \
        --predictions outputs/SNGS-116/predictions.json \
        --output-dir outputs/SNGS-116-2b [--fps 25] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True,
                        help="Clip dir containing img1/ frames")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Stage-1 predictions.json for the digest")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="2b output dir (e.g. outputs/SNGS-116-2b)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild clip.mp4 and regenerate JSONs")
    args = parser.parse_args()

    from pipeline.stage2b.digest import build_tracking_digest
    from pipeline.stage2b.generate_direct import generate_direct
    from pipeline.stage2b.video import build_clip_mp4, video_duration_s
    from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter
    from pipeline.stage4_commentary.generate import load_ark_env

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_mp4 = args.output_dir / "clip.mp4"
    if args.force or not clip_mp4.exists():
        log.info("Building %s from %s ...", clip_mp4, args.clip_dir / "img1")
        build_clip_mp4(args.clip_dir / "img1", clip_mp4, fps=args.fps)
    duration_s = video_duration_s(clip_mp4)

    commentary_json = args.output_dir / "commentary.json"
    if not args.force and commentary_json.exists():
        log.info("Reusing existing %s (use --force to regenerate)", commentary_json)
        return

    log.info("Building tracking digest from %s ...", args.predictions)
    digest = build_tracking_digest(args.predictions, fps=args.fps)

    load_ark_env()
    adapter = DoubaoAPIAdapter(enable_video=True)
    log.info("Calling %s with %.0fs video (%d bytes) ...",
             adapter.model, duration_s, clip_mp4.stat().st_size)
    out = generate_direct(
        clip_mp4=clip_mp4, digest=digest, duration_s=duration_s, fps=args.fps,
        output_dir=args.output_dir, adapter=adapter, languages=["en", "zh"],
    )
    log.info("Stage 2b complete: %s", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the driver script**

Create `scripts/run_stage2b.sh` (then `chmod +x scripts/run_stage2b.sh`):

```bash
#!/usr/bin/env bash
# Stage 2b (A/B arm): direct video→commentary, skipping stages 2/3/4.
# Reads:  <clip_dir>/img1/            (raw frames)
#         outputs/<SEQ>/predictions.json  (Stage 1, for the tracking digest)
# Writes: outputs/<SEQ>-2b/clip.mp4, events.json, commentary.json,
#         raw_final_video.mp4 (edge-tts preview), final_video.mp4 (CosyVoice clone)
#
# Usage:
#   bash scripts/run_stage2b.sh outputs/SNGS-116 [clip_dir]
#   TTS_LANGUAGE=zh FORCE=1 bash scripts/run_stage2b.sh outputs/SNGS-116
set -euo pipefail
cd "$(dirname "$0")/.."

A_DIR="${1:-outputs/SNGS-116}"
SEQ_NAME="$(basename "$A_DIR")"
CLIP_DIR="${2:-codes/sn-gamestate/datasets/SoccerNetGS/test/${SEQ_NAME}}"
B_DIR="outputs/${SEQ_NAME}-2b"
TTS_LANGUAGE="${TTS_LANGUAGE:-zh}"
FORCE="${FORCE:-0}"

if [[ ! -f "$A_DIR/predictions.json" ]]; then
  echo "ERROR: $A_DIR/predictions.json not found — run Stage 1 first." >&2
  exit 1
fi
if [[ ! -d "$CLIP_DIR/img1" ]]; then
  echo "ERROR: $CLIP_DIR/img1 not found — pass clip_dir explicitly." >&2
  exit 1
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
FORCE_FLAG=()
[[ "$FORCE" == "1" ]] && FORCE_FLAG=(--force)

echo "=== Stage 2b: direct VLM commentary (A/B arm) ==="
echo "  arm A dir:  $A_DIR"
echo "  arm B dir:  $B_DIR"
echo "  clip dir:   $CLIP_DIR"
echo

echo "--- Step 1/3: clip + digest → Doubao → events/commentary ---"
python -m pipeline.stage2b.run \
  --clip-dir "$CLIP_DIR" \
  --predictions "$A_DIR/predictions.json" \
  --output-dir "$B_DIR" \
  "${FORCE_FLAG[@]}"

echo
echo "--- Step 2/3: default voice preview (edge-tts) ---"
python -m pipeline.stage5_tts.make_raw_final_video \
  --output-dir "$B_DIR" --video "$B_DIR/clip.mp4" \
  --language "$TTS_LANGUAGE" "${FORCE_FLAG[@]}"

echo
echo "--- Step 3/3: 王楚淇 clone voice (CosyVoice3) ---"
python -m pipeline.stage5_tts.make_final_video \
  --output-dir "$B_DIR" --video "$B_DIR/clip.mp4" \
  --language "$TTS_LANGUAGE" "${FORCE_FLAG[@]}"

echo
echo "Stage 2b complete:"
echo "  A/B compare: $A_DIR/final_video.mp4  vs  $B_DIR/final_video.mp4"
```

- [ ] **Step 3: Verify script hygiene**

Run: `bash -n scripts/run_stage2b.sh && PYTHONPATH=. python -m pipeline.stage2b.run --help`
Expected: no syntax errors; help text prints.

- [ ] **Step 4: Make the driver executable**

```bash
chmod +x scripts/run_stage2b.sh
```

(No commit — single commit at the end per the commit policy.)

---

### Task 8: End-to-end A/B run on SNGS-116

No code — evidence gathering. Requires ARK key; Step 3 requires the GPU server for CosyVoice.

- [ ] **Step 1: Run arm B through commentary**

Run:
```bash
bash scripts/run_stage2b.sh outputs/SNGS-116
```
(If the machine lacks CosyVoice, steps 1–2 must succeed and step 3 must fail with the adapter's setup error — that is expected off-server.)

Then inspect:
```bash
PYTHONPATH=. python -c "
import json
d = json.load(open('outputs/SNGS-116-2b/commentary.json'))
e = json.load(open('outputs/SNGS-116-2b/events.json'))
segs, evs = d['commentary'], e['events']
print('events:', len(evs), '| segments:', len(segs))
ids = {x['event_id'] for x in evs}
for s in segs:
    w = s['end_s'] - s['timestamp_s']
    assert w <= 7.0 + 1e-6, f'window {w}'
    assert all(r in ids for r in s['events_referenced'])
    print(f\"{s['timestamp_s']:6.2f}-{s['end_s']:6.2f} energy={s.get('energy','-'):9s} {s['text_zh'][:28]}\")
assert len(segs) >= 6
print('2B FORMAT OK')
"
```
Expected: `2B FORMAT OK`; events carry menu codes with plausible timestamps.

- [ ] **Step 2: Watch the A/B pair**

Open `outputs/SNGS-116/raw_final_video.mp4` (arm A) and `outputs/SNGS-116-2b/raw_final_video.mp4` (arm B) side by side. Judgment is manual per the locked decision — check: do B's timestamps match on-screen action? Are B's events real? Jersey numbers sane (digest-grounded)?

- [ ] **Step 3: Clone pass on the GPU server**

Run: `bash scripts/run_stage2b.sh outputs/SNGS-116` (server has CosyVoice set up from the 2026-07-08 plan).
Expected: `outputs/SNGS-116-2b/final_video.mp4` with the 王楚淇 voice.

- [ ] **Step 4: The single commit**

Only after the full suite is green and the E2E run passed:

```bash
PYTHONPATH=. pytest tests/ -v     # full suite green
git status                        # review everything that will be swept in
git add -A
git commit -m "feat(stage2b): direct video->commentary A/B pipeline + 2026-07-08 density/CosyVoice execution

- stage2b: clip builder, tracking digest, direct Doubao video generation
- Doubao adapter: opt-in video_url support (ARK_VIDEO_MODEL override)
- stage5: --video flag on final-video makers; run_stage2b.sh driver
- includes previously uncommitted 2026-07-08 plan execution (rsync'd from server)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
Then use superpowers:finishing-a-development-branch.

---

## Self-Review Notes

- **Spec coverage:** 跳过2/3/4 → Task 5/7 import nothing from detector candidates/classify/stage3/stage4 prompt_builder (digest uses only possession geometry + loader); 豆包读完整视频 → Tasks 1 (probe), 2 (clip), 4 (video_url); 格式和commentary.json一致 → Task 5 reuses `normalize_segment`/`write_commentary_json`/`validate_pacing`; 带时间戳和事件 → events.json split with `events_referenced` linking (Q4=B); 合成语音和克隆 → Tasks 6/7 reuse stage 5 with `--video`.
- **Placeholder scan:** clean — every code step shows complete code; Task 6 Step 4 repeats the exact three edits rather than "similar to".
- **Type consistency:** `build_clip_mp4(frames_dir, output_path, fps)` / `video_duration_s(path)` (Task 2) match Task 7 usage; `build_tracking_digest(predictions_json, fps)` (Task 3) matches Task 7; `generate_direct(clip_mp4, digest, duration_s, fps, output_dir, adapter, languages)` (Task 5) matches Tasks 5-tests and 7; `make_final_video(..., video=)` (Task 6) matches Task 7 script's `--video`.
- **Known risk, contained:** ARK `video_url` content shape is probed live in Task 1 before anything depends on it; request-size headroom comes from 720p CRF 26 (~5–10MB for 30s). If the probe fails, the plan stops by design — no fallback arm.
