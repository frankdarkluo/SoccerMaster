# Pipeline Cleanup Design

**Date:** 2026-07-10
**Status:** Revised after joint cleanup/hybrid design review

## Goal

Reduce the repository to one canonical Stage 1 → Stage 2B → Stage 3 TTS → optional Stage 4 Effects workflow, while preserving the minimum relations/topology capabilities required by hybrid commentary.

## Final Decisions

- Canonical numbered packages are `pipeline/stage1_inference`, `pipeline/stage2b`, `pipeline/stage3_tts`, and `pipeline/stage4_effects`.
- There is no Stage 5 package or launcher.
- Stage 2B supports only `direct` and `hybrid`; hybrid is the default.
- Sequence artifacts are separated into root, `comments/`, and `voice/`.
- Stage 3 always produces a playable `voice/final_video*.mp4`.
- Optional Stage 4 replaces that final video safely after rendering effects.
- Only CosyVoice is retained for TTS. Edge and Doubao TTS are removed.
- Only two repository tests remain: the Stage 1 calibration guard and one offline hybrid smoke test.
- Existing outputs are migrated once; production code has no legacy path fallback.
- All implementation work is committed once, after the two implementation plans and all verification gates pass.

## Canonical Structure

```text
pipeline/
├── config.py
├── run.py
├── stage1_inference/
├── stage2b/
│   ├── __init__.py
│   ├── concepts.yaml
│   ├── digest.py
│   ├── events.py
│   ├── generate.py
│   ├── hybrid.py
│   ├── run.py
│   └── video.py
├── stage3_tts/
│   ├── __init__.py
│   ├── cosyvoice.py
│   ├── mux.py
│   ├── run.py
│   └── synthesize.py
├── stage4_effects/
├── relations/
│   ├── build.py
│   ├── kinematics.py
│   ├── query.py
│   ├── radar.py
│   └── snapshots.py
├── topology/
└── utils/
    ├── labels.py
    ├── pitch.py
    └── video.py

scripts/
├── run_stage1.sh
├── run_stage2b.sh
├── run_stage3.sh
├── run_stage4.sh
└── setup_cosyvoice.sh

tests/
├── test_calibration_guard.py
└── test_hybrid_smoke.py
```

No shared-helper or adapter-factory package is introduced. Single-caller helpers stay beside their caller.

## Stage Responsibilities

### Stage 1 Inference

Stage 1 retains tracking, calibration, CPU/Torch compatibility guards, and the current production entry point. Its shared outputs remain at the sequence root:

- `predictions.json`
- `homography_per_frame.json`
- Stage 1 logs and intermediate data

### Stage 2B Commentary

Stage 2B receives the sequence root and writes commentary artifacts below `comments/`.

`direct` produces event-first video commentary. `hybrid` first produces the same direct baseline, then adds only verified tactical observations. Hybrid writes:

- `comments/events.json`
- `comments/event_spine.json`
- `comments/commentary_direct.json`
- `comments/tactical_candidates.json`
- `comments/commentary.json`
- `comments/relations.json`
- `comments/radar/`

Direct mode omits relations, radar, and tactical candidates, and writes its accepted result to `comments/commentary.json`.

The clip stays at `<sequence>/clip.mp4`.

### Stage 3 TTS

Stage 3 reads only `comments/commentary.json` and uses Fun-CosyVoice3 for `default`, `clone`, or `both`.

- `default` uses `codes/CosyVoice/asset/zero_shot_prompt.wav` and its known transcript.
- `clone` defaults to repository-root `voice_sample.wav`.
- Clone mode uses `inference_zero_shot` when a matching prompt transcript is supplied, otherwise `inference_cross_lingual`.
- `--prompt-wav` overrides the default clone sample.
- `--language zh|en` is supported; one language is generated per invocation.

For each segment, Stage 3 synthesizes normal text and measures the actual duration with FFprobe. On overflow it retries once with the pre-generated `fallback_text_zh` or `fallback_text_en`. A second overflow fails explicitly; audio is never silently truncated.

Stage 3 writes all audio, caches, and videos below `voice/`. It always creates a no-effects baseline and a default deliverable:

- `voice/commentary_<language>_default.mp3` for default mode
- `voice/commentary_<language>.mp3` for clone mode
- `voice/tts_segments/`
- `voice/raw_final_video[_en].mp4`
- `voice/final_video[_en].mp4`

When clone is requested, clone audio owns the final video. Default-only mode uses the default voice for both baseline and final video.

### Stage 4 Effects

Stage 4 reads Stage 1 tracking/homography plus `comments/events.json`, writes root `annotated_video.mp4`, then muxes the existing Stage 3 audio into a temporary final video. Only after FFmpeg succeeds does a same-filesystem rename replace `voice/final_video[_en].mp4`.

The Stage 3 `raw_final_video[_en].mp4` remains unchanged for comparison and rollback.

## Minimal Legacy Extraction

Before deleting former stages:

- Move `FrameData`, frame loading, team/role majority voting, nearest-player possession, and possession segmentation into `stage2b/digest.py`.
- Replace the CSV-backed schema with a closed event catalog in `stage2b/events.py`. It includes `football.corner`, descriptions, and `importance_base`.
- Move the single ARK request path, direct-video prompt, JSON parsing, pacing validation, and commentary writing into `stage2b/generate.py`.
- Move supported relation quantities, aggregations, and predicate evaluation into `relations/query.py`.
- Move the tactical glossary to `stage2b/concepts.yaml`.
- Implement candidate generation, scheduling, composition, auditing, and event-only fallback in `stage2b/hybrid.py`.

Do not migrate detector candidates, classification, evidence-frame creation, event enrichment, former event verification, generic LLM adapters, adapter factories, or full-clip tactical narration.

## Deletion Inventory

Delete these canonical obsolete areas after retained behavior is extracted:

- `pipeline/stage2_events/`
- `pipeline/stage4_commentary/`
- `pipeline/tactics/`
- `pipeline/stage3_effects/render_preview.py`
- former `pipeline/stage5_tts/` after its retained files are moved to `stage3_tts/`
- `reference_football.csv`
- `pitch_distances.py`
- obsolete Stage 1 probes and recompute scripts
- obsolete clip-index utilities
- generated artifacts and `__pycache__` below source directories

Delete obsolete scripts:

- `scripts/run_stage2.sh`
- `scripts/run_stage5.sh`
- `scripts/run_tts.sh`
- `scripts/smoke_doubao_tts.py`
- former Stage 2/4/caption verification and one-off probe launchers

Delete all repository tests except:

- `tests/test_calibration_guard.py`
- new `tests/test_hybrid_smoke.py`

Delete empty test package markers and shared fixtures.

External model repositories, model weights, datasets, `.omc/`, `AccessKey.txt`, and `voice_sample.wav` are user-owned and are not committed. No credential file is read by tests or included in Git staging.

## Output Layout

```text
outputs/<SEQ>/
├── predictions.json
├── homography_per_frame.json
├── clip.mp4
├── annotated_video.mp4
├── comments/
│   ├── events.json
│   ├── event_spine.json
│   ├── commentary_direct.json
│   ├── tactical_candidates.json
│   ├── commentary.json
│   ├── relations.json
│   └── radar/
└── voice/
    ├── tts_segments/
    ├── commentary_zh_default.mp3
    ├── commentary_zh.mp3
    ├── raw_final_video.mp4
    └── final_video.mp4
```

English audio/video names use `_en`.

## Existing Output Migration

Migrate `SNGS-116`, `SNGS-117`, and `SNGS-148`. Merge `SNGS-116-2b` and `SNGS-117-2b` into their base sequence directories.

- Shared videos and Stage 1 artifacts remain at the sequence root.
- Commentary, relations, and radar data move to `comments/`.
- Audio, segment caches, and final videos move to `voice/`.
- Existing direct `-2b` commentary is preserved as the baseline where available.
- Verify every migrated target exists and is non-empty before deleting the two `-2b` directories.
- Migration is a one-time execution step and is not implemented as a permanent compatibility layer.
- Output artifacts remain ignored and uncommitted.

## Launchers

The five retained scripts are thin wrappers. They set defaults, translate environment variables into CLI arguments, and call `python -m ...`. Python CLIs own validation and actionable errors.

- `run_stage1.sh`
- `run_stage2b.sh`
- `run_stage3.sh`
- `run_stage4.sh`
- `setup_cosyvoice.sh`

## Error Handling

- Missing Stage 1 artifacts, video frames, FFmpeg/FFprobe, ARK credentials, CosyVoice model files, clone reference audio, commentary, or base video fail with direct actionable errors.
- ARK JSON receives one schema-guided retry, then fails.
- Hybrid failures fall back to the direct event commentary.
- Missing relations/radar data removes tactical candidates but does not block direct commentary.
- TTS overflow receives one fallback-text retry, then fails without truncation.
- Stage 4 keeps the previous final video unless its complete replacement succeeds.

## Verification

Automated verification is intentionally small:

```bash
python -m compileall -q pipeline
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/run_stage4.sh scripts/setup_cosyvoice.sh
pytest -q tests/test_calibration_guard.py tests/test_hybrid_smoke.py
python -m pipeline.stage2b.run --help
python -m pipeline.stage3_tts.run --help
python -m pipeline.stage4_effects.run --help
```

A real SNGS-116 run is a hard pre-commit gate:

1. Stage 2B direct and hybrid complete.
2. The corner sequence and later key events remain present.
3. CosyVoice default and clone complete without audio truncation.
4. Stage 3 produces baseline and final videos.
5. Stage 4 safely updates the final video.
6. Only then may the single implementation commit be created.

The 50-clip blinded A/B protocol remains a future release gate. No evaluation package is built in this work.

## Acceptance Criteria

- Only the approved numbered stage packages remain.
- No retained code imports former Stage 2, Stage 4 commentary, tactics, Stage 5, Edge TTS, or Doubao TTS modules.
- Stage 2B defaults to hybrid and retains explicit direct mode.
- Sequence outputs follow the root/comments/voice contract with no legacy fallback.
- Stage 3 uses only Fun-CosyVoice3 and supports default, clone, both, Chinese, and English.
- Stage 4 never leaves a partial final video.
- Only the two approved test files remain.
- SNGS-116 passes the real end-to-end gate.
- User-owned credentials, models, assets, and unrelated working-tree changes remain untouched.
