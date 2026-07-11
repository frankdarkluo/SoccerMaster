# Pipeline Cleanup, Stage Renaming, and Output Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the canonical Stage 1 → Stage 2B → Stage 3 TTS → optional Stage 4 Effects pipeline, delete replaced code, migrate outputs, and create one verified final commit.

**Architecture:** Execute this plan after the hybrid Stage 2B plan. Rename retained TTS/effects packages, replace all TTS backends with one direct Fun-CosyVoice3 integration, make Stage 4 safely upgrade the Stage 3 final video, then delete old packages/tests and migrate existing outputs. The sequence root contains shared artifacts, `comments/` owns analysis, and `voice/` owns speech and final video.

**Tech Stack:** Python 3.10+, Bash, FFmpeg/FFprobe, OpenCV, Fun-CosyVoice3/PyTorch/torchaudio, pytest.

## Global Constraints

- The hybrid plan must pass before this plan starts.
- Work in the current `main` checkout; do not create a worktree.
- Preserve user changes and never reset, restore, or stash.
- Final packages are exactly `stage1_inference`, `stage2b`, `stage3_tts`, and `stage4_effects`.
- Final launchers are exactly `run_stage1.sh`, `run_stage2b.sh`, `run_stage3.sh`, `run_stage4.sh`, and `setup_cosyvoice.sh`.
- Do not read or stage `AccessKey.txt`.
- Preserve `codes/`, `pretrained_models/`, `.omc/`, model weights, datasets, and `voice_sample.wav`.
- Only `test_calibration_guard.py` and `test_hybrid_smoke.py` may remain under `tests/`.
- Existing output migration is one-time; production code has no legacy path fallback.
- Do not commit after tasks. Create exactly one commit after both plans and the real SNGS-116 gate pass.

## File Map

**Create:**
- `pipeline/stage3_tts/cosyvoice.py`
- `pipeline/stage3_tts/run.py`
- `pipeline/stage4_effects/run.py`
- `scripts/setup_cosyvoice.sh`

**Move/Modify:**
- retained `pipeline/stage5_tts/{mux.py,synthesize.py}` → `pipeline/stage3_tts/`
- `pipeline/stage3_effects/` → `pipeline/stage4_effects/`
- `pipeline/config.py`
- `pipeline/run.py`
- `scripts/run_stage1.sh`
- `scripts/run_stage3.sh`
- `scripts/run_stage4.sh`
- `tests/test_hybrid_smoke.py`

**Delete:** old Stage 2/4 commentary/tactics/Stage 5 packages, obsolete launchers/probes/utilities, all other repository tests, `reference_football.csv`, generated source artifacts, and the already-approved `pitch_distances.py`.

---

### Task 1: Replace Stage 5 with one direct CosyVoice Stage 3 package

**Files:**
- Create: `pipeline/stage3_tts/__init__.py`
- Move: `pipeline/stage5_tts/mux.py` → `pipeline/stage3_tts/mux.py`
- Create: `pipeline/stage3_tts/cosyvoice.py`
- Create: `pipeline/stage3_tts/synthesize.py`
- Create: `pipeline/stage3_tts/run.py`
- Create: `scripts/setup_cosyvoice.sh`
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:**
- `CosyVoiceSynthesizer(model_dir: Path)`.
- `synthesize(text, output_path, *, voice, prompt_wav, prompt_text) -> Path`.
- `synthesize_fitting_segment(segment, language, output_path, slot_s, synthesizer, probe=audio_duration_s) -> Path`.
- CLI: `python -m pipeline.stage3_tts.run OUTPUT_DIR [--language zh|en] [--voice default|clone|both] [--prompt-wav PATH] [--prompt-text TEXT] [--force]`.

- [ ] **Step 1: Add a failing TTS fallback smoke test**

Append to `tests/test_hybrid_smoke.py`:

```python
from pipeline.stage3_tts.synthesize import synthesize_fitting_segment


class FakeSynthesizer:
    def __init__(self):
        self.texts = []

    def synthesize(self, text, output_path, **kwargs):
        self.texts.append(text)
        output_path.write_bytes(b"audio")
        return output_path


def test_tts_retries_once_with_event_fallback(tmp_path):
    segment = {
        "text_zh": "右路迅速转移，蓝队的横向移动被彻底拉开。",
        "fallback_text_zh": "右路把球转向远端。",
    }
    durations = iter([6.2, 2.1])
    synth = FakeSynthesizer()
    out = synthesize_fitting_segment(
        segment, "zh", tmp_path / "segment.wav", 3.0, synth,
        probe=lambda path: next(durations),
        voice="default",
        prompt_wav=Path("prompt.wav"),
        prompt_text=None,
    )
    assert out.is_file()
    assert synth.texts == [segment["text_zh"], segment["fallback_text_zh"]]
```

- [ ] **Step 2: Verify the missing Stage 3 package failure**

Run: `pytest -q tests/test_hybrid_smoke.py::test_tts_retries_once_with_event_fallback`

Expected: `ModuleNotFoundError: No module named 'pipeline.stage3_tts'`.

- [ ] **Step 3: Implement the direct CosyVoice wrapper**

Create `pipeline/stage3_tts/cosyvoice.py` with constants:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
COSYVOICE_ROOT = REPO_ROOT / "codes" / "CosyVoice"
MODEL_DIR = REPO_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
DEFAULT_PROMPT_WAV = COSYVOICE_ROOT / "asset" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "希望你以后能够做的比我还好呦。"
)
CLONE_PROMPT_WAV = REPO_ROOT / "voice_sample.wav"
```

The class loads `AutoModel(model_dir=str(model_dir))` once. Before import, add `codes/CosyVoice` and `codes/CosyVoice/third_party/Matcha-TTS` to `sys.path`.

Use:

```python
def synthesize(
    self,
    text: str,
    output_path: Path,
    *,
    voice: str,
    prompt_wav: Path | None = None,
    prompt_text: str | None = None,
) -> Path:
    reference = prompt_wav or (
        DEFAULT_PROMPT_WAV if voice == "default" else CLONE_PROMPT_WAV
    )
    if not reference.is_file():
        raise FileNotFoundError(f"CosyVoice reference audio not found: {reference}")
    if prompt_text:
        chunks = self.model.inference_zero_shot(
            text, prompt_text, str(reference), stream=False
        )
    else:
        chunks = self.model.inference_cross_lingual(
            f"You are a helpful assistant.<|endofprompt|>{text}",
            str(reference), stream=False,
        )
    tensors = [chunk["tts_speech"] for chunk in chunks]
    if not tensors:
        raise RuntimeError("CosyVoice returned no audio")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), torch.cat(tensors, dim=1), self.model.sample_rate)
    return output_path
```

Default mode always supplies `DEFAULT_PROMPT_TEXT`; clone mode uses transcript-free cross-lingual inference unless `--prompt-text` is supplied.

- [ ] **Step 4: Implement measured segment fitting**

`audio_duration_s` invokes FFprobe and rejects non-finite/non-positive values.

`synthesize_fitting_segment` selects `text_<language>`, synthesizes, and accepts when duration ≤ `slot_s + 0.05`. On overflow it selects `fallback_text_<language>`, overwrites the segment cache, and measures once more. A second overflow deletes that cache and raises:

```python
raise ValueError(
    f"TTS still exceeds slot after fallback: {duration:.2f}s > {slot_s:.2f}s"
)
```

Do not call `atrim`, `atempo`, or substring truncation.

- [ ] **Step 5: Implement Stage 3 orchestration**

`run.py` reads `comments/commentary.json`, validates `clip.mp4`, chooses language and voice mode, synthesizes every segment, assembles timed audio, and muxes the shared clip into both baseline and final video.

File contract:

```python
suffix = "_en" if language == "en" else ""
default_audio = voice_dir / f"commentary_{language}_default.mp3"
clone_audio = voice_dir / f"commentary_{language}.mp3"
baseline = voice_dir / f"raw_final_video{suffix}.mp4"
final = voice_dir / f"final_video{suffix}.mp4"
```

- `default`: default audio owns baseline and final.
- `clone`: clone audio owns baseline and final.
- `both`: generate default first, then clone; clone owns baseline and final.

Use a temporary mux target followed by `os.replace` for Stage 3 final creation as well.

- [ ] **Step 6: Create the setup check**

`scripts/setup_cosyvoice.sh` must not clone or download silently. It verifies:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f codes/CosyVoice/cosyvoice/cli/cosyvoice.py
test -f pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
test -f pretrained_models/Fun-CosyVoice3-0.5B/llm.pt
test -f pretrained_models/Fun-CosyVoice3-0.5B/flow.pt
test -f pretrained_models/Fun-CosyVoice3-0.5B/hift.pt
echo "CosyVoice3 source and model are ready."
```

- [ ] **Step 7: Verify Stage 3 offline behavior**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py::test_tts_retries_once_with_event_fallback
python -m pipeline.stage3_tts.run --help
bash -n scripts/setup_cosyvoice.sh
bash scripts/setup_cosyvoice.sh
```

Expected: smoke passes, CLI help exits 0, setup reports ready. Do not commit.

---

### Task 2: Rename visual effects to Stage 4 and safely upgrade final video

**Files:**
- Move: `pipeline/stage3_effects/` → `pipeline/stage4_effects/`
- Delete during move: `pipeline/stage3_effects/render_preview.py`
- Create: `pipeline/stage4_effects/run.py`
- Modify: all imports inside the moved package
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:**
- `replace_final_video(annotated_video, audio, final_video, mux=mux_audio_video) -> Path`.
- CLI: `python -m pipeline.stage4_effects.run OUTPUT_DIR [--clip-dir PATH] [--language zh|en] [--force]`.

- [ ] **Step 1: Add atomic replacement assertions**

```python
import pytest
from pipeline.stage4_effects.run import replace_final_video


def test_stage4_keeps_old_final_when_mux_fails(tmp_path):
    annotated = tmp_path / "annotated.mp4"
    audio = tmp_path / "commentary.mp3"
    final = tmp_path / "final_video.mp4"
    annotated.write_bytes(b"video")
    audio.write_bytes(b"audio")
    final.write_bytes(b"old")

    def fail_mux(video, sound, target):
        target.write_bytes(b"partial")
        raise RuntimeError("mux failed")

    with pytest.raises(RuntimeError, match="mux failed"):
        replace_final_video(annotated, audio, final, mux=fail_mux)
    assert final.read_bytes() == b"old"
    assert not final.with_name(".final_video.mp4.tmp").exists()
```

Run and expect a missing Stage 4 package.

- [ ] **Step 2: Move the package and imports**

Move all retained effects files, then replace every `pipeline.stage3_effects` import with `pipeline.stage4_effects`. Delete `render_preview.py` and its imports.

- [ ] **Step 3: Implement safe final replacement**

```python
def replace_final_video(
    annotated_video: Path,
    audio: Path,
    final_video: Path,
    *,
    mux=mux_audio_video,
) -> Path:
    temporary = final_video.with_name(f".{final_video.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        mux(annotated_video, audio, temporary)
        os.replace(temporary, final_video)
    finally:
        temporary.unlink(missing_ok=True)
    return final_video
```

- [ ] **Step 4: Implement Stage 4 run**

Validate root `predictions.json`, `comments/events.json`, clip frames, and Stage 3 selected audio. Render root `annotated_video.mp4`, run topology when enabled, then call `replace_final_video`. English selects `voice/commentary_en.mp3` or `commentary_en_default.mp3`; Chinese uses unsuffixed equivalents.

- [ ] **Step 5: Verify**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py::test_stage4_keeps_old_final_when_mux_fails
python -m pipeline.stage4_effects.run --help
python -m compileall -q pipeline/stage4_effects
```

Expected: test and commands pass. Do not commit.

---

### Task 3: Finalize configuration, orchestration, and thin launchers

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/run.py`
- Modify: `scripts/run_stage1.sh`
- Replace: `scripts/run_stage3.sh`
- Replace: `scripts/run_stage4.sh`

**Interfaces:**
- Root orchestrator order: Stage 1, Stage 2B, Stage 3 TTS, optional Stage 4 Effects.
- Every Python CLI accepts the sequence root.
- Shell launchers forward arguments only.

- [ ] **Step 1: Complete config paths**

Remove `REFERENCE_CSV`, old detector/verifier/LLM backend settings, Stage 5 properties, and Stage 5 predicates. Keep Stage 1/effects settings and the Stage 2B comments paths from the hybrid plan.

Add:

```python
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
```

- [ ] **Step 2: Finish the root orchestrator**

`pipeline.run.run_pipeline` calls retained stage functions in order and does not define `run_stage2`, old `run_stage3`, commentary `run_stage4`, or `run_stage5`. Optional effects are controlled by one explicit flag; there are no aliases for old stage numbers.

- [ ] **Step 3: Replace Stage 3 launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m pipeline.stage3_tts.run "$@"
```

- [ ] **Step 4: Replace Stage 4 launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m pipeline.stage4_effects.run "$@"
```

Keep `run_stage1.sh` production behavior but update comments to name Stage 2B, Stage 3 TTS, and optional Stage 4 Effects.

- [ ] **Step 5: Verify launchers and imports**

Run:

```bash
python -m compileall -q pipeline
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/run_stage4.sh scripts/setup_cosyvoice.sh
python -m pipeline.stage2b.run --help
python -m pipeline.stage3_tts.run --help
python -m pipeline.stage4_effects.run --help
```

Expected: exit 0. Do not commit.

---

### Task 4: Delete replaced packages, utilities, scripts, and tests

**Files:** Delete only the approved inventory.

**Interfaces:**
- Consumes: replacements verified by the hybrid plan and Tasks 1-3.
- Produces: no imports or files from obsolete stages, adapters, probes, or tests.

- [ ] **Step 1: Delete old production packages**

Delete:

```text
pipeline/stage2_events/
pipeline/stage4_commentary/
pipeline/tactics/
pipeline/stage5_tts/
reference_football.csv
pitch_distances.py
```

The retained TTS files must already exist under `stage3_tts/` before deleting Stage 5.

- [ ] **Step 2: Delete obsolete probes and utilities**

Delete:

```text
pipeline/stage1_inference/probe_concat_tracklets_by_reid.py
pipeline/stage1_inference/probe_heatmap_peaks.py
pipeline/stage1_inference/check_recompute.py
pipeline/stage1_inference/recompute_calibration.sh
pipeline/utils/clip_index_to_events.py
pipeline/utils/validate_clip_index.py
```

Delete `__pycache__` and generated runtime artifacts under `pipeline/` and `scripts/`.

- [ ] **Step 3: Delete obsolete launchers**

Delete all scripts except the five approved launchers. Specifically remove `run_stage2.sh`, `run_stage5.sh`, `run_tts.sh`, `smoke_doubao_tts.py`, former caption verification, and one-off probes.

- [ ] **Step 4: Reduce tests to two files**

Delete every file/directory below `tests/` except:

```text
tests/test_calibration_guard.py
tests/test_hybrid_smoke.py
```

This removes `tests/conftest.py`, package markers, old detector/tactics/relations tests, and fixtures. Ensure the two retained tests contain all local helpers they use.

- [ ] **Step 5: Verify no deleted-package references remain**

Run:

```bash
rg -n "pipeline\.(stage2_events|stage3_effects|stage4_commentary|stage5_tts|tactics)" pipeline scripts tests
rg -n "edge_tts|DoubaoTTS|doubao_tts|REFERENCE_CSV|reference_football" pipeline scripts tests
find tests -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: both scans print nothing; test listing contains only `test_calibration_guard.py` and `test_hybrid_smoke.py`. Do not commit.

---

### Task 5: Migrate all existing outputs once

**Files:** Modify ignored output directories only; no Git staging.

**Interfaces:**
- Consumes: five legacy output directories.
- Produces: three base sequence directories using root/comments/voice ownership.

- [ ] **Step 1: Create owned directories**

```bash
for root in outputs/SNGS-116 outputs/SNGS-117 outputs/SNGS-148; do
  mkdir -p "$root/comments" "$root/voice"
done
```

- [ ] **Step 2: Merge direct sibling outputs**

For `SNGS-116` and `SNGS-117`, copy first, verify, then remove siblings:

```bash
for seq in SNGS-116 SNGS-117; do
  base="outputs/$seq"
  sibling="outputs/$seq-2b"
  cp "$sibling/events.json" "$base/comments/events.json"
  cp "$sibling/commentary.json" "$base/comments/commentary_direct.json"
  cp "$sibling/commentary.json" "$base/comments/commentary.json"
  test -f "$base/clip.mp4" || cp "$sibling/clip.mp4" "$base/clip.mp4"
  find "$sibling" -maxdepth 1 -type f \( -name '*.mp3' -o -name '*final_video*.mp4' \)     -exec cp {} "$base/voice/" \;
  test -s "$base/comments/events.json"
  test -s "$base/comments/commentary.json"
  test -s "$base/clip.mp4"
done
```

- [ ] **Step 3: Move flat Stage 2B/voice artifacts**

Move existing `relations.json`, radar directory, draft/direct commentary artifacts, and final Stage 2B JSON below `comments/`. Move root MP3, TTS caches, and final videos below `voice/`. Shared Stage 1 files, `clip.mp4`, and `annotated_video.mp4` stay at root.

For `SNGS-148`, move flat `events.json` and `commentary.json` to `comments/` and audio/final video to `voice/`.

- [ ] **Step 4: Validate and remove sibling directories**

```bash
for seq in SNGS-116 SNGS-117 SNGS-148; do
  test -s "outputs/$seq/comments/events.json"
  test -s "outputs/$seq/comments/commentary.json"
  test -d "outputs/$seq/voice"
done
rm -rf outputs/SNGS-116-2b outputs/SNGS-117-2b
test ! -e outputs/SNGS-116-2b
test ! -e outputs/SNGS-117-2b
```

Expected: base outputs are complete and sibling directories are absent. Do not stage outputs.

---

### Task 6: Run full verification and SNGS-116 hard gate

**Files:** Verify all implementation files and SNGS-116 outputs.

**Interfaces:**
- Consumes: completed hybrid and cleanup tasks.
- Produces: static, offline, and real-video evidence required before staging.

- [ ] **Step 1: Run static and offline checks**

```bash
python -m compileall -q pipeline
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/run_stage4.sh scripts/setup_cosyvoice.sh
pytest -q tests/test_calibration_guard.py tests/test_hybrid_smoke.py
python -m pipeline.stage2b.run --help
python -m pipeline.stage3_tts.run --help
python -m pipeline.stage4_effects.run --help
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run real SNGS-116 Stage 2B**

```bash
bash scripts/run_stage2b.sh outputs/SNGS-116 --mode hybrid --force
```

Expected: opening corner and required later events pass the hybrid-plan assertions.

- [ ] **Step 3: Run CosyVoice default and clone**

```bash
bash scripts/run_stage3.sh outputs/SNGS-116 --language zh --voice both --force
```

Expected: default and clone audio are non-empty; `voice/raw_final_video.mp4` and `voice/final_video.mp4` are playable and non-empty; logs contain no truncation.

- [ ] **Step 4: Run Stage 4 Effects**

```bash
bash scripts/run_stage4.sh outputs/SNGS-116 --language zh --force
```

Expected: root `annotated_video.mp4` and updated `voice/final_video.mp4` are non-empty; `raw_final_video.mp4` remains unchanged.

- [ ] **Step 5: Probe final media**

```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1   outputs/SNGS-116/voice/raw_final_video.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1   outputs/SNGS-116/voice/final_video.mp4
```

Expected: both commands report positive durations.

- [ ] **Step 6: Confirm scope before staging**

```bash
git status --short
git diff --stat
git diff --name-status
```

Expected: no `.omc/`, `codes/CosyVoice/`, `pretrained_models/`, `AccessKey.txt`, `voice_sample.wav`, or `outputs/` path is staged.

---

### Task 7: Create the single final commit

**Files:** Stage only approved source, scripts, two tests, two specs, and two plans.

**Interfaces:**
- Consumes: the fully verified working tree.
- Produces: exactly one commit with no user assets, credentials, models, or outputs.

- [ ] **Step 1: Stage tracked changes and approved new source**

```bash
git add -u pipeline scripts tests reference_football.csv pitch_distances.py
git add pipeline/atomic.py
git add pipeline/stage2b pipeline/stage3_tts pipeline/stage4_effects pipeline/relations
git add scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/run_stage4.sh scripts/setup_cosyvoice.sh
git add -f tests/test_calibration_guard.py tests/test_hybrid_smoke.py
git add -f   docs/superpowers/specs/2026-07-10-pipeline-cleanup-design.md   docs/superpowers/specs/2026-07-10-hybrid-event-tactical-commentary-design.md   docs/superpowers/plans/2026-07-10-pipeline-cleanup.md   docs/superpowers/plans/2026-07-10-hybrid-event-tactical-commentary.md
```

- [ ] **Step 2: Audit staged scope**

```bash
git diff --cached --check
git diff --cached --name-status
git status --short
```

If any user-owned path is staged, unstage that exact path before continuing. Do not use reset/checkout to alter its working-tree contents.

- [ ] **Step 3: Commit once**

```bash
git commit -m "refactor: simplify pipeline and add hybrid commentary"
```

Expected: exactly one new commit containing both completed plans.

## Acceptance Checklist

- [ ] Hybrid plan passed before cleanup deletion.
- [ ] Only four numbered stage packages remain.
- [ ] Stage 3 uses only Fun-CosyVoice3.
- [ ] Stage 4 safely replaces final video.
- [ ] Only five launchers remain.
- [ ] Only two tests remain.
- [ ] Old Stage 2/4 commentary/tactics/Stage 5 references are absent.
- [ ] All existing outputs use root/comments/voice layout.
- [ ] SNGS-116 passes direct events, hybrid, CosyVoice, and effects gates.
- [ ] User assets and credentials are unstaged.
- [ ] Exactly one implementation commit was created.
