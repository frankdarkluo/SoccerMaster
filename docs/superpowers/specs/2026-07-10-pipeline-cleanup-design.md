# Pipeline Cleanup Design

**Goal:** Reduce the repository to the active Stage 1 → Stage 2B → Stage 5 workflow while preserving Stage 3 visual effects and topology as optional runnable capabilities.

**Decision:** Stage 2B replaces the former Stage 2 and Stage 4 path. The repository has one canonical Python package under `pipeline/`, one canonical launcher directory under `scripts/`, and one output directory per sequence.

## Scope

Retain:

- Stage 1 inference and its live calibration, CPU, and Torch compatibility guards.
- Stage 2B direct video-to-commentary generation.
- Stage 3 visual effects, excluding the unused isolated preview renderer.
- Stage 5 Edge TTS preview and CosyVoice final synthesis.
- Formation topology analysis and rendering support.
- The external model repositories and data under `codes/`.

Remove:

- The former Stage 2 detector/classifier pipeline.
- The former Stage 4 commentary pipeline.
- Root-level mirrors of canonical packages and scripts.
- Manual probes, obsolete launchers, unused adapters, and test-only code.
- Generated run artifacts stored below `pipeline/`.
- The `tests/` directory, per the explicit project decision.

## Canonical Structure

```text
pipeline/
├── config.py
├── run.py
├── stage1_inference/
├── stage2b/
│   ├── __init__.py
│   ├── digest.py
│   ├── generate_direct.py
│   ├── run.py
│   └── video.py
├── stage3_effects/
├── stage5_tts/
│   ├── adapters/
│   │   ├── cosyvoice_adapter.py
│   │   └── edge_tts_adapter.py
│   ├── mux.py
│   ├── pace_filter.py
│   ├── run.py
│   ├── synthesize.py
│   └── tts_adapter.py
├── topology/
└── utils/
    ├── labels.py
    ├── pitch.py
    └── video.py

scripts/
├── run_stage1.sh
├── run_stage2b.sh
├── run_stage3.sh
└── setup_cosyvoice.sh
```

No additional shared-helper package is introduced. Helpers with one retained caller stay beside that caller.

## Helper Extraction

### Stage 2B tracking digest

Move the minimal retained behavior from the former Stage 2 into `pipeline/stage2b/digest.py`:

- `FrameData` and `PossessionSegment` data containers.
- `load_frames` for reading Stage 1 `predictions.json`.
- Team and role majority voting.
- Nearest-player possession resolution and possession segmentation.

Preserve current behavior during the move. Do not retain candidate detection, classification, evidence-frame creation, event enrichment, schema objects, or verification code.

### Stage 2B ARK and JSON handling

Move the minimal retained behavior from the former Stage 4 into `pipeline/stage2b/generate_direct.py`:

- Base64 `video_url` request construction for ARK.
- The closed Stage 2B event menu and concise descriptions.
- Commentary segment normalization.
- Pacing validation.
- `events.json` and `commentary.json` writing.
- One retry after malformed JSON or pacing violations.

Use one direct ARK call function. Do not retain the generic `LLMAdapter`, backend factory, image-sampling path, OpenAI adapter, Qwen adapter, mock adapter, or prompt-builder hierarchy.

### Environment loading

Rename the existing repository `.env` loader to a backend-neutral helper in `pipeline/config.py`. Stage 2B and Stage 5 reuse it. Existing environment variables continue to take priority over `.env` values.

## Stage 5 Consolidation

Replace `make_raw_final_video.py` and `make_final_video.py` with:

```bash
python -m pipeline.stage5_tts.run --output-dir outputs/<SEQ> --voice preview
python -m pipeline.stage5_tts.run --output-dir outputs/<SEQ> --voice clone
python -m pipeline.stage5_tts.run --output-dir outputs/<SEQ> --voice both
```

The runner reuses the existing synthesis, pacing, mux, Edge TTS, and CosyVoice code. It does not add an adapter factory.

Base-video selection is:

1. Explicit `--video`, when supplied.
2. `<output-dir>/annotated_video.mp4`, when present.
3. `<output-dir>/clip.mp4` otherwise.

`--voice both` runs the inexpensive Edge preview first and CosyVoice second. Existing per-segment caches remain in use unless `--force` is supplied.

Remove the Doubao TTS adapter, its smoke script, the HTML/terminal preview helper, and legacy TTS launchers.

## Runtime Data Flow

All stages use one output directory now that there is no A/B arm:

```text
Stage 1
  predictions.json
  homography_per_frame.json

Stage 2B
  clip.mp4
  events.json
  commentary.json

Optional Stage 3
  annotated_video.mp4

Stage 5
  commentary_<language>_default.mp3
  commentary_<language>.mp3
  raw_final_video[_en].mp4
  final_video[_en].mp4
  tts_segments/
```

`scripts/run_stage2b.sh` writes into the Stage 1 output directory instead of a `<SEQ>-2b` sibling. Optional Stage 3 can therefore read Stage 1 predictions/homography and Stage 2B events without copying artifacts.

## Canonical-Copy Reconciliation

The root mirrors are not blindly deleted because several contain newer behavior than their canonical `pipeline/` counterparts. Before removal, reconcile only the retained behavior:

- Preserve Stage 1 timing, batch override, path override, and visualization-control changes from the newer root `stage1_inference/run_gsr.py`.
- Preserve the Stage 3 FFmpeg fallback behavior from the newer root renderer.
- Preserve Stage 2B video input, segment normalization, Stage 5 `--video`, English output naming, and direct preview-link simplifications through the new canonical Stage 2B and Stage 5 implementations.
- Discard differences that belong only to the removed Stage 2 or Stage 4 paths.

All user-owned unrelated working-tree changes remain untouched.

## Deletion Inventory

Delete the following canonical obsolete areas:

- `pipeline/stage2_events/`
- `pipeline/stage4_commentary/`
- `pipeline/stage3_effects/render_preview.py`
- `pipeline/stage5_tts/adapters/doubao_tts.py`
- `pipeline/stage5_tts/make_raw_final_video.py`
- `pipeline/stage5_tts/make_final_video.py`
- `pipeline/stage5_tts/preview.py`
- `pipeline/stage1_inference/probe_concat_tracklets_by_reid.py`
- `pipeline/stage1_inference/probe_heatmap_peaks.py`
- `pipeline/stage1_inference/check_recompute.py`
- `pipeline/stage1_inference/recompute_calibration.sh`
- `pipeline/utils/clip_index_to_events.py`
- `pipeline/utils/validate_clip_index.py`
- `pipeline/SNGS-117-2b/`
- `tests/`

Delete obsolete canonical scripts:

- `scripts/run_caption_verify.sh`
- `scripts/run_stage2.sh`
- `scripts/run_stage4.sh`
- `scripts/run_stage5.sh`
- `scripts/run_tts.sh`
- `scripts/smoke_doubao_tts.py`

Delete root-only or mirrored application code:

- `__init__.py`, `config.py`, `run.py`, and `video.py` at repository root.
- Root `stage1_inference/`, `stage2_events/`, `stage2b/`, `stage3_effects/`, `stage4_commentary/`, `stage5_tts/`, `topology/`, and `utils/` directories.
- Root `run_caption_verify.sh`, `run_stage1.sh`, `run_stage2.sh`, `run_stage2b.sh`, `run_stage3.sh`, `run_stage4.sh`, `run_stage5.sh`, `run_tts.sh`, `setup_cosyvoice.sh`, `smoke_doubao_tts.py`, and `probe_ark_video.py` after the retained Stage 2B launcher is placed under `scripts/`.

Keep the current deletions of obsolete standalone minimap/distance tools and former Stage 2 tests.

## Error Handling

- Missing Stage 1 artifacts, clip frames, FFmpeg/FFprobe, ARK credentials, ARK video support, commentary, or base video fail with a direct actionable error.
- ARK output receives one retry for malformed JSON or pacing violations, then fails.
- There is no frame-sampling fallback when ARK video input fails.
- Stage 5 preserves cached TTS files unless forced.
- Stage 3 falls back to OpenCV MP4 output when FFmpeg is unavailable and logs that the result may not be browser-safe.

## Verification

No tests or test-only self-checks remain, per explicit decision. Verification is limited to production-code loading and static checks:

```bash
python -m compileall -q pipeline
bash -n scripts/run_stage1.sh scripts/run_stage2b.sh scripts/run_stage3.sh scripts/setup_cosyvoice.sh
python -m pipeline.stage2b.run --help
python -m pipeline.stage5_tts.run --help
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary" pipeline scripts
```

The final `rg` command must return no matches. A live Stage 1 GPU run, ARK Stage 2B run, optional Stage 3 render, and CosyVoice run remain manual checks because they require large models, hardware, credentials, or network access.

## Acceptance Criteria

- Only one canonical copy of application code exists.
- Stage 2B has no imports from the removed Stage 2 or Stage 4 packages.
- Stage 1, Stage 2B, optional Stage 3, and Stage 5 command entry points load successfully.
- Stage 3 visual effects and `pipeline/topology/` remain present and runnable.
- Stage 5 supports Edge preview, CosyVoice clone, or both through one runner.
- A sequence uses one output directory from Stage 1 through Stage 5.
- No test directory or test-only code remains.
- No unrelated working-tree edits are included in the cleanup.

## Ponytail Review Summary

The cleanup removes duplicate packages, two replaced pipeline stages, unused diagnostics, one-off probes, redundant wrappers, obsolete adapters, and test-only files. The estimated result is approximately 13,000 fewer Python and shell lines, after accounting for the small helper extraction and consolidated Stage 5 runner.

