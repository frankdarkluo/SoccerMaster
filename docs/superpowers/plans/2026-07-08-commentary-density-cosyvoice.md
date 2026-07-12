# Commentary Density + 1.5x Length + CosyVoice3 TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Denser (≥1 segment per ~5s), livelier (1.5x longer, energy-tagged) commentary that is spoken in full — never text-truncated — by a locally-run CosyVoice3 voice clone with per-segment emotion control.

**Architecture:** Stage 2 already guarantees event density via the uncommitted `fill_buildup_bins` (one `football.buildup` possession event per 5s bin — role-agnostic). This plan (a) teaches the Stage 4 prompt to use those events + formation topology, raises the text budget 1.5x, adds a per-segment `energy` enum emitted by Doubao, and validates pacing with a single retry; (b) rips text truncation out of Stage 5, replacing it with speech-rate control (CosyVoice `speed` per energy tier → ffmpeg `atempo` ≤1.35x → back-to-back spill; audio fade-cut only to protect a following highlight); (c) adds a `CosyVoiceAdapter` (Fun-CosyVoice3-0.5B, instruct mode, `voice_sample.wav` reference, cuda-if-available) as the new default clone backend, demoting Doubao TTS; (d) fixes the `$LANG` env collision and auto-generates `topo.json` when Stage 4 runs standalone.

**Tech Stack:** Python 3.10, pytest, ffmpeg/ffprobe, Doubao ARK API (Stage 4 LLM), CosyVoice3 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` via huggingface_hub; repo cloned, no vLLM), torchaudio, edge-tts (preview step unchanged).

**Decisions locked in the grill-me session (2026-07-08):**
- Density fixed at the source; NO new Stage 2 detectors (VLM audit proved the 9–23s SNGS-116 gap is genuinely quiet play; `fill_buildup_bins` covers it generically for any role).
- Text budget 1.5x: ZH 7 chars/s, EN 4.5 words/s. 2x rejected (unintelligible speech rate).
- Pacing enforcement: prompt rules + ONE validation retry (duration-relative: ≥1 segment/5s, max window 7s); accept after retry with a warning; parse failure after retry = hard error (no more silent 0–30s fallback segment).
- Energy: Doubao emits per-segment `energy` ∈ {calm, engaged, excited, explosive}; keyword rules remain as fallback.
- CosyVoice3 instruct mode (`inference_instruct2`): reference wav only, NO transcript needed. `voice_sample.wav` (repo root, 25s) is the reference.
- Never delete words. Priority: speed → spill (lag acceptable) → fade-cut only before a locked highlight.
- Stage 5 runs on the GPU server; device = cuda if available else cpu (CosyVoice resolves this internally).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/run_stage4.sh` | Modify | `LANG` → `COMMENTARY_LANGS` |
| `pipeline/run.py` | Modify | topo.json autogen in `run_stage4`; `cosyvoice` branch in `_build_tts_adapter` |
| `pipeline/stage4_commentary/prompt_builder.py` | Modify | 1.5x budget, cadence + energy + style rules, buildup guidance, topo sampling |
| `pipeline/stage4_commentary/postprocess.py` | Modify | `energy` normalization, strict parsing (raise), `validate_pacing` |
| `pipeline/stage4_commentary/generate.py` | Modify | validation + single retry, real duration in video_info |
| `pipeline/stage5_tts/pace_filter.py` | Modify | delete all text truncation; keep short-segment merging |
| `pipeline/stage5_tts/tts_adapter.py` | Modify | add `ENERGY_ENGAGED` |
| `pipeline/stage5_tts/synthesize.py` | Modify | energy field priority; placement planner with tempo stretch |
| `pipeline/stage5_tts/adapters/edge_tts_adapter.py` | Modify | engaged prosody tier |
| `pipeline/stage5_tts/adapters/doubao_tts.py` | Modify | engaged prosody tier |
| `pipeline/stage5_tts/adapters/cosyvoice_adapter.py` | Create | CosyVoice3 instruct-mode clone adapter |
| `scripts/setup_cosyvoice.sh` | Create | one-time repo clone + HF weight download |
| `pipeline/config.py` | Modify | `tts_backend` default → `"cosyvoice"` |
| `pipeline/stage5_tts/make_final_video.py` | Modify | Doubao → CosyVoice for the clone step |
| `scripts/run_stage5.sh` | Modify | step-2 wording; drop Doubao TTS env requirement |
| `tests/test_stage4_pacing.py` | Create | prompt rules, postprocess energy, validate_pacing, retry loop |
| `tests/test_stage5_speech.py` | Create | pace_filter no-truncation, energy priority, placement planner, instruct mapping |

Note: `tests/` is currently deleted in the working tree (mid-refactor state). New test files are self-contained — no conftest fixtures needed. Do not resurrect the deleted test files; that's the Stage 2 redesign's business.

---

### Task 1: Fix `$LANG` env collision in run_stage4.sh

**Files:**
- Modify: `scripts/run_stage4.sh`

The script uses `LANG` as its language-list variable, which collides with the system locale (`LANG=en_US.UTF-8`), producing `"language": ["en_US.UTF-8"]` in commentary.json and a garbage prompt line "Generate BOTH EN_US.UTF-8 commentary".

- [ ] **Step 1: Rename the variable**

In `scripts/run_stage4.sh`, make these three edits:

Line 13 (usage comment):
```bash
#   LLM_BACKEND=doubao COMMENTARY_LANGS="en zh" bash scripts/run_stage4.sh outputs/SNGS-148
```

Line 65:
```bash
COMMENTARY_LANGS="${COMMENTARY_LANGS:-en zh}"
```

Line 72:
```bash
echo "  languages:    $COMMENTARY_LANGS"
```

Line 108 (inside the embedded python heredoc):
```python
    languages="${COMMENTARY_LANGS}".split(),
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/run_stage4.sh && grep -n "LANG" scripts/run_stage4.sh`
Expected: syntax OK; every remaining match is `COMMENTARY_LANGS`, zero bare `LANG=` or `$LANG`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_stage4.sh
git commit -m "fix(stage4): rename LANG env var to COMMENTARY_LANGS to avoid locale collision"
```

---

### Task 2: Auto-generate topo.json when Stage 4 runs standalone

**Files:**
- Modify: `pipeline/run.py` (function `run_stage4`, currently lines 210–223)

`topo.json` is computed at the end of Stage 3 only; standalone `run_stage4.sh` runs never get it (SNGS-116 has none), so the prompt's `[Formation Context]` section silently vanishes. Topology is pure computation from predictions.json — no model calls.

- [ ] **Step 1: Add the fallback generation**

Replace `run_stage4` in `pipeline/run.py` with:

```python
def run_stage4(config: PipelineConfig) -> None:
    from pipeline.stage4_commentary.generate import generate_commentary

    if not config.topology_json.exists() and config.predictions_json.exists():
        from pipeline.stage3_effects.topology_analysis import run_topology_analysis
        log.info("topo.json missing — computing from %s", config.predictions_json)
        run_topology_analysis(
            config.predictions_json,
            config.topology_json,
            fps=config.fps,
        )

    visual = config.annotated_video if config.annotated_video.exists() else None
    if visual is None and config.topdown_video.exists():
        visual = config.topdown_video

    generate_commentary(
        config.events_json,
        config.commentary_json,
        config=config,
        visual_input=visual,
        topo_json_path=config.topology_json if config.topology_json.exists() else None,
    )
```

- [ ] **Step 2: Verify on real data**

Run:
```bash
PYTHONPATH=. python -c "
from pathlib import Path
from pipeline.stage3_effects.topology_analysis import run_topology_analysis
p = run_topology_analysis(Path('outputs/SNGS-116/predictions.json'), Path('outputs/SNGS-116/topo.json'), fps=25)
import json; recs = json.loads(p.read_text()); print(len(recs), 'records, first:', recs[0]['t_start'], recs[0]['team'])
"
```
Expected: prints a record count > 0 and a first record. (This also leaves a real topo.json in place for the Task 10 smoke run.)

- [ ] **Step 3: Commit**

```bash
git add pipeline/run.py
git commit -m "feat(stage4): auto-compute topo.json from predictions when missing"
```

---

### Task 3: Prompt overhaul — 1.5x budget, cadence, energy, style, buildup + topo context

**Files:**
- Modify: `pipeline/stage4_commentary/prompt_builder.py`
- Test: `tests/test_stage4_pacing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage4_pacing.py`:

```python
"""Stage 4 pacing/energy tests: prompt rules, postprocess, validation, retry."""
import json
from pathlib import Path

import pytest

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage4_commentary.prompt_builder import build_commentary_prompt


def _write_events(tmp_path: Path, duration_s: float = 30.0) -> Path:
    data = {
        "video_info": {"source": "fixture", "fps": 25, "duration_s": duration_s},
        "events": [
            {"event_id": "evt_001", "timestamp_s": 2.0, "event_code": "football.pass",
             "player_jersey": "10", "player_team": "left", "importance": 0.3, "tags": {}},
            {"event_id": "evt_002", "timestamp_s": 12.0, "event_code": "football.buildup",
             "player_jersey": "1", "player_team": "right", "importance": 0.1,
             "tags": {"filler": "density"}},
        ],
    }
    p = tmp_path / "events.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_topo(tmp_path: Path) -> Path:
    records = []
    for t in range(0, 28):  # stride-1s windows like the real analyzer
        for team in ("left", "right"):
            records.append({
                "t_start": float(t), "t_end": float(t + 3), "team": team,
                "possession_tag": "in" if team == "left" else "out",
                "block_height_m": 30.0 + t, "block_depth_m": 25.0,
                "block_width_m": 40.0,
            })
    p = tmp_path / "topo.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


def test_prompt_has_new_budget_and_cadence(tmp_path):
    prompt = build_commentary_prompt(
        _write_events(tmp_path), EventSchema(), ["en", "zh"],
    )
    assert "7 characters per second" in prompt
    assert "4.5 words per second" in prompt
    assert "at least 6 segments" in prompt          # 30s // 5
    assert "more than 7 seconds" in prompt
    assert '"calm"' in prompt and '"explosive"' in prompt  # energy enum
    assert "filler=density" in prompt               # buildup guidance
    # old budget gone
    assert "5 characters per second" not in prompt
    assert "3.3 words per second" not in prompt


def test_prompt_min_segments_scales_with_duration(tmp_path):
    prompt = build_commentary_prompt(
        _write_events(tmp_path, duration_s=60.0), EventSchema(), ["en", "zh"],
    )
    assert "at least 12 segments" in prompt
    assert "60-second" in prompt


def test_prompt_topo_sampled_across_full_clip(tmp_path):
    prompt = build_commentary_prompt(
        _write_events(tmp_path), EventSchema(), ["en", "zh"],
        topo_json_path=_write_topo(tmp_path),
    )
    assert "[Formation Context]" in prompt
    # old code took records[:6] (t=0..2 only); sampled output must reach late windows
    assert "t~25" in prompt or "t~25.0" in prompt
    assert "possession=in" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v`
Expected: FAIL — assertions on "7 characters per second", "at least 6 segments", "t~25" etc.

- [ ] **Step 3: Implement the prompt changes**

In `pipeline/stage4_commentary/prompt_builder.py`, inside `build_commentary_prompt`, replace everything from `lang_str = ...` through the first `parts.append(f"""...""")` block with:

```python
    duration_s = float(events_data.get("video_info", {}).get("duration_s", 30.0))
    min_segments = max(1, int(duration_s // 5))

    lang_str = " and ".join(languages).upper()
    parts.append(f"""You are a professional football commentator. Generate vivid, flowing commentary for this {duration_s:.0f}-second match clip.

RULES:
1. Use ONLY the timestamps from the event list. Never invent timestamps.
2. Use the EXACT terminology from the tag display names shown in parentheses.
3. Events whose Tags contain filler=density are quiet build-up moments found by
   the tracking system: narrate possession, team shape, off-ball movement, and
   the flow of play there — never invent shots, goals or duels for them.
4. Refer to players by jersey number (or name if roster provided).
5. Generate BOTH {lang_str} commentary for each segment.
6. For events marked HIGHLIGHT, use more excited/vivid language.
7. Output a valid JSON array only. Each object MUST use these keys:
   timestamp_s (number), end_s (number), text_en (string), text_zh (string),
   energy (string), events_referenced (array of event id strings, e.g. ["evt_070"]).
   Do not use alternate key names.
8. energy is the emotional intensity for the voice actor — exactly one of:
   "calm" (routine possession, quiet build-up),
   "engaged" (something is developing: a forward pass, a press, rising interest),
   "excited" (shot, dangerous attack, successful tackle/interception, any
   exclamation-mark moment),
   "explosive" (goal).

STYLE:
9. Be vivid and specific: ball trajectory, player movement, spatial detail
   (flank, box, half-space). Vary sentence structure between segments. Avoid
   bland filler like "both teams contest possession".

PACING (critical for TTS):
10. Each segment is spoken aloud. Budget: Chinese ≤ 7 characters per second,
    English ≤ 4.5 words per second. Example: a 5-second window → max ~35
    Chinese chars or ~22 English words. Fill each window — short is worse.
11. Produce a talking point roughly every 5 seconds: at least {min_segments}
    segments, and NO segment may span more than 7 seconds. Split long quiet
    stretches using the build-up events and Formation Context.
12. Do NOT create segments shorter than 2 seconds. Merge rapid events into one.
13. Lines are spoken back-to-back with no pauses between them.""")
```

- [ ] **Step 4: Implement the topo sampling change**

In the same function, replace the `[Formation Context]` block (`if topo_json_path and topo_json_path.exists(): ...`) with:

```python
    if topo_json_path and topo_json_path.exists():
        parts.append("\n[Formation Context]")
        with open(topo_json_path, encoding="utf-8") as f:
            topo = json.load(f)
        records = topo if isinstance(topo, list) else topo.get("records", [])
        # one record per team per ~5s bin, spanning the whole clip
        sampled: dict = {}
        for record in records:
            t = float(record.get("t_start", record.get("window_start_s", 0.0)))
            sampled.setdefault((record.get("team"), int(t // 5)), record)
        for record in sampled.values():
            t = record.get("t_start", record.get("window_start_s", "?"))
            team = record.get("team", "?")
            height = record.get("block_height_m")
            depth = record.get("block_depth_m")
            poss = record.get("possession_tag", "unknown")
            parts.append(
                f"t~{t}s {team}: height={height}m, depth={depth}m, possession={poss}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/stage4_commentary/prompt_builder.py tests/test_stage4_pacing.py
git commit -m "feat(stage4): 1.5x text budget, 5s cadence, energy enum, vivid style, full-clip topo sampling"
```

---

### Task 4: Postprocess — energy normalization, strict parsing, pacing validator

**Files:**
- Modify: `pipeline/stage4_commentary/postprocess.py`
- Test: `tests/test_stage4_pacing.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage4_pacing.py`:

```python
from pipeline.stage4_commentary.postprocess import (
    parse_commentary_output,
    validate_pacing,
)


def _seg_json(ts, end, energy=None, **extra):
    d = {"timestamp_s": ts, "end_s": end, "text_en": "x", "text_zh": "y",
         "events_referenced": []}
    if energy is not None:
        d["energy"] = energy
    d.update(extra)
    return d


def test_parse_keeps_valid_energy_drops_invalid():
    raw = json.dumps([
        _seg_json(0, 5, energy="EXCITED"),
        _seg_json(5, 10, energy="thrilled"),
        _seg_json(10, 15),
    ])
    segs = parse_commentary_output(raw)
    assert segs[0]["energy"] == "excited"
    assert "energy" not in segs[1]
    assert "energy" not in segs[2]


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_commentary_output("Sorry, I cannot produce commentary.")
    with pytest.raises(ValueError):
        parse_commentary_output("[{broken json]")


def test_validate_pacing_flags_count_and_window():
    segs = [
        {"timestamp_s": 0.0, "end_s": 9.0},
        {"timestamp_s": 9.0, "end_s": 23.5},   # 14.5s window
        {"timestamp_s": 23.5, "end_s": 30.0},
    ]
    problems = validate_pacing(segs, duration_s=30.0)
    assert any("at least 6" in p for p in problems)
    assert any("14.5" in p for p in problems)


def test_validate_pacing_accepts_good_output():
    segs = [{"timestamp_s": i * 5.0, "end_s": (i + 1) * 5.0} for i in range(6)]
    assert validate_pacing(segs, duration_s=30.0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v -k "parse or validate"`
Expected: FAIL — `validate_pacing` doesn't exist; garbage input returns the fallback segment instead of raising; energy not normalized.

- [ ] **Step 3: Implement**

In `pipeline/stage4_commentary/postprocess.py`:

Add near the top (after imports):

```python
ENERGY_LEVELS = ("calm", "engaged", "excited", "explosive")
MAX_WINDOW_S = 7.0
SEGMENT_EVERY_S = 5.0
```

In `_normalize_segment`, after the `if seg.get("event_code"):` block, add:

```python
    energy = str(seg.get("energy", "")).strip().lower()
    if energy in ENERGY_LEVELS:
        out["energy"] = energy
```

Replace `parse_commentary_output` entirely:

```python
def parse_commentary_output(raw_text: str) -> List[dict]:
    """Extract JSON array of commentary segments. Raises ValueError when the
    model output contains no parseable JSON array — callers retry, then fail."""
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("LLM output contains no JSON array")
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output JSON array is malformed: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("LLM output JSON is not an array")
    return [
        _normalize_segment(seg, i)
        for i, seg in enumerate(data)
        if isinstance(seg, dict)
    ]
```

Add after `parse_commentary_output`:

```python
def validate_pacing(segments: List[dict], duration_s: float) -> List[str]:
    """Return human-readable pacing violations (empty list = OK).

    Rules from the 2026-07-08 grill-me session: at least one segment per ~5s
    of video, and no single segment window longer than 7s.
    """
    problems: List[str] = []
    min_segments = max(1, int(duration_s // SEGMENT_EVERY_S))
    if len(segments) < min_segments:
        problems.append(
            f"only {len(segments)} segments; need at least {min_segments} "
            f"for a {duration_s:.0f}s clip (one talking point per ~5s)"
        )
    for seg in segments:
        window = float(seg.get("end_s", 0)) - float(seg.get("timestamp_s", 0))
        if window > MAX_WINDOW_S:
            problems.append(
                f"segment {seg.get('timestamp_s')}-{seg.get('end_s')}s spans "
                f"{window:.1f}s (max {MAX_WINDOW_S:.0f}s) — split it using "
                "build-up events or formation context"
            )
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage4_commentary/postprocess.py tests/test_stage4_pacing.py
git commit -m "feat(stage4): energy normalization, strict output parsing, pacing validator"
```

---

### Task 5: Generate — single validation retry, hard failure, real duration

**Files:**
- Modify: `pipeline/stage4_commentary/generate.py`
- Test: `tests/test_stage4_pacing.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage4_pacing.py`:

```python
from pipeline.stage4_commentary.generate import generate_commentary


class FakeAdapter:
    """Returns queued responses; records call count."""
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def generate(self, prompt, visual_input=None):
        self.prompts.append(prompt)
        self.calls += 1
        return self.responses.pop(0)


def _good_json():
    return json.dumps([
        {"timestamp_s": i * 5.0, "end_s": (i + 1) * 5.0, "text_en": "e",
         "text_zh": "z", "energy": "calm", "events_referenced": []}
        for i in range(6)
    ])


def _sparse_json():
    return json.dumps([
        {"timestamp_s": 0.0, "end_s": 9.0, "text_en": "e", "text_zh": "z",
         "events_referenced": []},
        {"timestamp_s": 9.0, "end_s": 30.0, "text_en": "e", "text_zh": "z",
         "events_referenced": []},
    ])


def test_retry_on_pacing_violation_then_accept(tmp_path):
    adapter = FakeAdapter([_sparse_json(), _good_json()])
    out = generate_commentary(
        _write_events(tmp_path), tmp_path / "commentary.json", adapter=adapter,
    )
    assert adapter.calls == 2
    assert "pacing problems" in adapter.prompts[1]
    data = json.loads(Path(out).read_text())
    assert len(data["commentary"]) == 6
    assert data["video_info"]["duration_s"] == 30.0


def test_second_violation_is_accepted_with_warning(tmp_path):
    adapter = FakeAdapter([_sparse_json(), _sparse_json()])
    out = generate_commentary(
        _write_events(tmp_path), tmp_path / "commentary.json", adapter=adapter,
    )
    assert adapter.calls == 2
    data = json.loads(Path(out).read_text())
    assert len(data["commentary"]) == 2  # accepted as-is after one retry


def test_double_parse_failure_raises(tmp_path):
    adapter = FakeAdapter(["no json here", "still no json"])
    with pytest.raises(RuntimeError):
        generate_commentary(
            _write_events(tmp_path), tmp_path / "commentary.json", adapter=adapter,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v -k retry`
Expected: FAIL — no retry loop exists; single call; garbage produces fallback not RuntimeError.

- [ ] **Step 3: Implement**

In `pipeline/stage4_commentary/generate.py`:

Add `import logging` to the imports and `log = logging.getLogger(__name__)` after them. Extend the postprocess import:

```python
from pipeline.stage4_commentary.postprocess import (
    parse_commentary_output,
    validate_pacing,
    write_commentary_json,
)
```

In `generate_commentary`, replace the block from `raw_output = adapter.generate(prompt, visual_input)` through `segments = parse_commentary_output(raw_output)` and the `video_info = {...}` dict with:

```python
    with open(events_json_path, encoding="utf-8") as f:
        duration_s = float(
            json.load(f).get("video_info", {}).get("duration_s", 30.0)
        )

    def _attempt(p: str):
        raw = adapter.generate(p, visual_input)
        try:
            segs = parse_commentary_output(raw)
        except ValueError as exc:
            return None, [str(exc)]
        return segs, validate_pacing(segs, duration_s)

    segments, problems = _attempt(prompt)
    if problems:
        log.warning("Commentary attempt 1 rejected: %s — retrying once", problems)
        retry_prompt = (
            prompt
            + "\n\nYour previous answer had pacing problems:\n- "
            + "\n- ".join(problems)
            + "\nRegenerate the FULL JSON array fixing every problem."
        )
        segments, problems = _attempt(retry_prompt)
        if segments is None:
            raise RuntimeError(
                f"Commentary generation failed after retry: {problems}"
            )
        if problems:
            log.warning("Pacing problems remain after retry (accepted): %s", problems)

    model_name = getattr(adapter, "model", config.llm_backend)
    video_info = {
        "source": str(config.clip_dir),
        "duration_s": duration_s,
        "fps": config.fps,
    }
```

(The `model_info` dict and `write_commentary_json` call below stay unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage4_pacing.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage4_commentary/generate.py tests/test_stage4_pacing.py
git commit -m "feat(stage4): pacing validation with single retry; fail hard on unparseable output"
```

---

### Task 6: Strip text truncation from pace_filter (keep merging)

**Files:**
- Modify: `pipeline/stage5_tts/pace_filter.py` (full rewrite — the file shrinks from ~250 to ~80 lines)
- Test: `tests/test_stage5_speech.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage5_speech.py`:

```python
"""Stage 5 tests: no text truncation, energy priority, placement planner, CosyVoice mapping."""
import pytest

from pipeline.stage5_tts.pace_filter import filter_segments


def _seg(ts, end, zh="中文" * 50, en="word " * 80):
    return {"timestamp_s": ts, "end_s": end, "text_zh": zh, "text_en": en.strip(),
            "events_referenced": []}


def test_long_text_is_never_truncated():
    zh = "这是一段远远超出旧字数预算的超长中文解说词，" * 10
    seg = _seg(0.0, 3.0, zh=zh)
    out = filter_segments([seg])
    assert out[0]["text_zh"] == zh          # verbatim — no chars deleted
    assert out[0]["text_en"] == seg["text_en"]


def test_short_segments_still_merge():
    segs = [_seg(0.0, 1.0, zh="一", en="a"), _seg(1.0, 4.0, zh="二", en="b")]
    out = filter_segments(segs)
    assert len(out) == 1
    assert out[0]["text_zh"] == "一，二"
    assert out[0]["timestamp_s"] == 0.0 and out[0]["end_s"] == 4.0


def test_truncation_machinery_is_gone():
    import pipeline.stage5_tts.pace_filter as pf
    for name in ("_truncate_zh", "_truncate_en", "_enforce_budget",
                 "ZH_CHARS_PER_S", "EN_WORDS_PER_S"):
        assert not hasattr(pf, name), f"{name} should have been deleted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v`
Expected: FAIL — long text gets truncated; `_truncate_zh` etc. exist.

- [ ] **Step 3: Rewrite the file**

Replace the entire contents of `pipeline/stage5_tts/pace_filter.py` with:

```python
"""Merge short commentary segments for TTS.

Text is NEVER truncated (2026-07-08 decision: fitting is a speech-rate
problem — CosyVoice speed, then atempo, then spill — never a delete-words
problem). This module only merges segments shorter than MIN_SEGMENT_S with
their neighbours until each is at least TARGET_MIN_S long.
"""
from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)

MIN_SEGMENT_S = 2.0
TARGET_MIN_S = 3.0


def filter_segments(segments: List[dict]) -> List[dict]:
    """Return a new list with short segments merged. Text passes through verbatim."""
    merged = _merge_short(segments)
    log.info("Pace filter: %d segments → %d after merge", len(segments), len(merged))
    return merged


def _merge_short(segments: List[dict]) -> List[dict]:
    if not segments:
        return []

    out: list[dict] = []
    buf: dict | None = None

    for seg in segments:
        if buf is None:
            buf = _copy_seg(seg)
            continue

        buf_dur = buf["end_s"] - buf["timestamp_s"]

        if buf_dur < MIN_SEGMENT_S:
            buf = _merge_pair(buf, seg)
            continue

        seg_dur = seg["end_s"] - seg["timestamp_s"]
        if seg_dur < MIN_SEGMENT_S and buf_dur < TARGET_MIN_S:
            buf = _merge_pair(buf, seg)
            continue

        out.append(buf)
        buf = _copy_seg(seg)

    if buf is not None:
        if out and (buf["end_s"] - buf["timestamp_s"]) < MIN_SEGMENT_S:
            out[-1] = _merge_pair(out[-1], buf)
        else:
            out.append(buf)

    return out


def _copy_seg(seg: dict) -> dict:
    out = {
        "timestamp_s": seg["timestamp_s"],
        "end_s": seg["end_s"],
        "text_zh": seg.get("text_zh", ""),
        "text_en": seg.get("text_en", ""),
        "events_referenced": list(seg.get("events_referenced", [])),
    }
    if seg.get("energy"):
        out["energy"] = seg["energy"]
    return out


def _merge_pair(a: dict, b: dict) -> dict:
    joiner_zh = "，" if a.get("text_zh") and b.get("text_zh") else ""
    joiner_en = " " if a.get("text_en") and b.get("text_en") else ""
    merged = {
        "timestamp_s": a["timestamp_s"],
        "end_s": b["end_s"],
        "text_zh": a.get("text_zh", "") + joiner_zh + b.get("text_zh", ""),
        "text_en": a.get("text_en", "") + joiner_en + b.get("text_en", ""),
        "events_referenced": list(
            dict.fromkeys(a.get("events_referenced", []) + b.get("events_referenced", []))
        ),
    }
    # keep the hotter energy of the pair
    order = ("calm", "engaged", "excited", "explosive")
    energies = [e for e in (a.get("energy"), b.get("energy")) if e in order]
    if energies:
        merged["energy"] = max(energies, key=order.index)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage5_tts/pace_filter.py tests/test_stage5_speech.py
git commit -m "feat(stage5): remove all text truncation from pace filter; keep merging, propagate energy"
```

---

### Task 7: Energy plumbing — engaged tier + energy-field priority

**Files:**
- Modify: `pipeline/stage5_tts/tts_adapter.py`
- Modify: `pipeline/stage5_tts/synthesize.py` (function `energy_for_segment`)
- Modify: `pipeline/stage5_tts/adapters/edge_tts_adapter.py`
- Modify: `pipeline/stage5_tts/adapters/doubao_tts.py`
- Test: `tests/test_stage5_speech.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage5_speech.py`:

```python
from pipeline.stage5_tts.synthesize import energy_for_segment
from pipeline.stage5_tts.tts_adapter import (
    ENERGY_ENGAGED, ENERGY_EXCITED, ENERGY_EXPLOSIVE, ENERGY_NORMAL,
)


def test_energy_field_takes_priority_over_keywords():
    seg = {"energy": "calm", "text_zh": "球进了！进球！", "text_en": "GOAL!",
           "events_referenced": []}
    assert energy_for_segment(seg) == ENERGY_NORMAL   # calm maps to normal


def test_energy_field_four_tiers():
    assert energy_for_segment({"energy": "engaged"}) == ENERGY_ENGAGED
    assert energy_for_segment({"energy": "excited"}) == ENERGY_EXCITED
    assert energy_for_segment({"energy": "explosive"}) == ENERGY_EXPLOSIVE


def test_keyword_fallback_when_energy_missing_or_bad():
    assert energy_for_segment({"text_zh": "球进了！", "text_en": ""}) == ENERGY_EXPLOSIVE
    assert energy_for_segment({"energy": "bogus", "text_zh": "射门！",
                               "text_en": ""}) == ENERGY_EXCITED


def test_adapters_have_engaged_prosody():
    from pipeline.stage5_tts.adapters.edge_tts_adapter import _PROSODY as edge_p
    from pipeline.stage5_tts.adapters.doubao_tts import _PROSODY as doubao_p
    assert ENERGY_ENGAGED in edge_p and ENERGY_ENGAGED in doubao_p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v -k energy`
Expected: FAIL — `ENERGY_ENGAGED` doesn't exist.

- [ ] **Step 3: Implement**

`pipeline/stage5_tts/tts_adapter.py` — replace the three constant lines with:

```python
# Energy tiers for football commentary delivery.
ENERGY_NORMAL = "normal"        # calm narration
ENERGY_ENGAGED = "engaged"      # something developing — tone lifts
ENERGY_EXCITED = "excited"      # shots, dangerous attacks, won duels
ENERGY_EXPLOSIVE = "explosive"  # goals
```

`pipeline/stage5_tts/synthesize.py` — extend the tts_adapter import to include `ENERGY_ENGAGED`, then add above `energy_for_segment`:

```python
# Stage 4 emits energy ∈ {calm, engaged, excited, explosive}; map onto tiers.
_STAGE4_ENERGY = {
    "calm": ENERGY_NORMAL,
    "engaged": ENERGY_ENGAGED,
    "excited": ENERGY_EXCITED,
    "explosive": ENERGY_EXPLOSIVE,
}
```

and insert at the top of `energy_for_segment` (before `event_codes = event_codes or {}`):

```python
    tagged = _STAGE4_ENERGY.get(str(seg.get("energy", "")).strip().lower())
    if tagged is not None:
        return tagged
```

`pipeline/stage5_tts/adapters/edge_tts_adapter.py` — import `ENERGY_ENGAGED` and add to `_PROSODY`:

```python
    ENERGY_ENGAGED: {"rate": "+20%", "pitch": "+4Hz", "volume": "+5%"},
```

`pipeline/stage5_tts/adapters/doubao_tts.py` — import `ENERGY_ENGAGED`, add to `_PROSODY`:

```python
    ENERGY_ENGAGED: {"speech_rate": 20, "pitch_rate": 1, "loudness_rate": 5},
```

and to `_STYLE_PREFIX`:

```python
    ENERGY_ENGAGED: "用投入、期待的足球解说语气说：",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage5_tts/tts_adapter.py pipeline/stage5_tts/synthesize.py \
        pipeline/stage5_tts/adapters/edge_tts_adapter.py pipeline/stage5_tts/adapters/doubao_tts.py \
        tests/test_stage5_speech.py
git commit -m "feat(stage5): 4-tier energy with stage4 energy field priority, keyword fallback"
```

---

### Task 8: Placement planner — tempo stretch before trim, spill for normal lines

**Files:**
- Modify: `pipeline/stage5_tts/synthesize.py` (`_plan_placement` and `assemble_timeline`)
- Test: `tests/test_stage5_speech.py` (append)

Current behavior: normal lines spill (start at `max(ts, prev_end)`), highlights lock to their timestamp, and a line colliding with the next planned start gets audio-trimmed. New behavior per the locked decisions: before any trim, speed the line up with `atempo` (≤1.35x); trims can only be forced by a *locked highlight* boundary or the clip end — normal successors just spill later.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage5_speech.py`:

```python
from pipeline.stage5_tts.synthesize import _plan_placement


def test_normal_lines_spill_without_stretch_or_trim():
    # 6s of audio in a 3s slot, next line is normal → full duration, spill
    plan = _plan_placement([0.0, 3.0], [6.0, 2.0], [False, False], 30.0)
    (s0, t0, a0), (s1, t1, a1) = plan
    assert (s0, t0, a0) == (0.0, 1.0, 6.0)
    assert s1 == 6.0          # pushed back-to-back, not trimmed


def test_stretch_to_fit_before_locked_highlight():
    # 5s of audio, highlight locked at t=4 → tempo 1.25 fits exactly, no trim
    plan = _plan_placement([0.0, 4.0], [5.0, 2.0], [False, True], 30.0)
    s0, tempo0, allowed0 = plan[0]
    assert s0 == 0.0
    assert tempo0 == pytest.approx(1.25)
    assert allowed0 == pytest.approx(4.0)
    assert plan[1][0] == 4.0  # highlight stays locked


def test_stretch_caps_at_135_then_trims():
    # 8s of audio before a highlight at t=4 → needs 2.0x; cap 1.35, trim rest
    plan = _plan_placement([0.0, 4.0], [8.0, 2.0], [False, True], 30.0)
    s0, tempo0, allowed0 = plan[0]
    assert tempo0 == pytest.approx(1.35)
    assert allowed0 == pytest.approx(4.0)      # effective 8/1.35=5.93s > 4s → trim


def test_last_segment_bounded_by_clip_end():
    plan = _plan_placement([28.0], [5.0], [False], 30.0)
    s0, tempo0, allowed0 = plan[0]
    assert s0 == 28.0
    assert tempo0 == pytest.approx(min(5.0 / 2.0, 1.35))
    assert allowed0 == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v -k "spill or stretch or bounded"`
Expected: FAIL — `_plan_placement` returns 2-tuples and trims against any next start.

- [ ] **Step 3: Implement the planner**

In `pipeline/stage5_tts/synthesize.py`, replace `_plan_placement` with:

```python
_MAX_STRETCH = 1.35


def _plan_placement(
    starts_wanted: List[float],
    durations: List[float],
    highlights: List[bool],
    total_duration_s: float,
) -> list[tuple[float, float, float]]:
    """Compute (start, tempo, allowed_duration) for each segment.

    Normal lines flow back-to-back: start at ``max(timestamp_s, prev_end)`` and
    spill freely (commentary lag is acceptable — 2026-07-08 decision). Highlight
    lines always start exactly at their ``timestamp_s``. The only hard
    boundaries are a following locked highlight and the end of the clip; a line
    hitting one is first sped up (tempo ≤ _MAX_STRETCH) and only the remainder
    is trimmed via allowed_duration. Words are never deleted from the text.
    """
    n = len(starts_wanted)
    plan: list[tuple[float, float, float]] = []
    prev_end = 0.0
    for i in range(n):
        ts, dur, is_hi = starts_wanted[i], durations[i], highlights[i]
        start = max(ts, 0.0) if is_hi else max(ts, prev_end)

        boundary: Optional[float] = None
        if i + 1 < n and highlights[i + 1]:
            boundary = starts_wanted[i + 1]
        elif i + 1 == n:
            boundary = total_duration_s

        tempo = 1.0
        allowed = dur
        if boundary is not None and start + dur > boundary:
            room = max(boundary - start, 0.0)
            allowed = room
            if room > 0:
                tempo = min(dur / room, _MAX_STRETCH)

        eff_dur = min(dur / tempo, allowed)
        plan.append((start, tempo, allowed))
        prev_end = start + eff_dur
    return plan
```

- [ ] **Step 4: Apply tempo in assemble_timeline**

Still in `synthesize.py`, in `assemble_timeline`, replace the per-segment filter loop (`for idx, ((start, allowed), (_, _, dur, _)) in enumerate(zip(plan, valid)):` and its body) with:

```python
    for idx, ((start, tempo, allowed), (_, _, dur, _)) in enumerate(zip(plan, valid)):
        if allowed <= 0.05:
            continue
        inp = idx + 1
        label = f"d{idx}"
        chain = f"[{inp}]"
        eff_dur = dur
        if tempo > 1.01:
            chain += f"atempo={tempo:.3f},"
            eff_dur = dur / tempo
        # Trim + fade only for what tempo could not absorb (locked highlight
        # or clip end ahead).
        if eff_dur > allowed + 0.01:
            fade = min(_FADE_S, allowed / 2)
            chain += f"atrim=0:{allowed:.3f},afade=t=out:st={max(allowed - fade, 0):.3f}:d={fade:.3f},"
        delay_ms = int(round(start * 1000))
        chain += f"adelay={delay_ms}|{delay_ms}[{label}]"
        filters.append(chain)
        mix_inputs.append(f"[{label}]")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v`
Expected: all PASS.

- [ ] **Step 6: Smoke the ffmpeg chain with real audio**

Run (uses cached edge-tts segments from the previous SNGS-116 run):

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from pipeline.stage5_tts.synthesize import synthesize_commentary
from pipeline.stage5_tts.adapters.edge_tts_adapter import EdgeTTSAdapter
out = synthesize_commentary(
    Path('outputs/SNGS-116/commentary.json'), Path('outputs/SNGS-116'),
    language='zh', adapter=EdgeTTSAdapter(language='zh'),
    events_json=Path('outputs/SNGS-116/events.json'), voice_tag='default',
    audio_path=Path('outputs/SNGS-116/commentary_zh_planner_smoke.mp3'))
print('OK', out)
"
```
Expected: prints `OK .../commentary_zh_planner_smoke.mp3`; file plays. Then delete the smoke file: `rm outputs/SNGS-116/commentary_zh_planner_smoke.mp3`.

- [ ] **Step 7: Commit**

```bash
git add pipeline/stage5_tts/synthesize.py tests/test_stage5_speech.py
git commit -m "feat(stage5): placement planner with atempo stretch (cap 1.35x); trim only at locked highlights/clip end"
```

---

### Task 9: CosyVoiceAdapter + setup script + backend wiring

**Files:**
- Create: `pipeline/stage5_tts/adapters/cosyvoice_adapter.py`
- Create: `scripts/setup_cosyvoice.sh`
- Modify: `pipeline/config.py` (line 66: `tts_backend` default)
- Modify: `pipeline/run.py` (`_build_tts_adapter`)
- Modify: `pipeline/stage5_tts/make_final_video.py`
- Modify: `scripts/run_stage5.sh`
- Test: `tests/test_stage5_speech.py` (append)

- [ ] **Step 1: Write the failing tests** (pure mapping logic — no model load)

Append to `tests/test_stage5_speech.py`:

```python
def test_cosyvoice_instruct_and_speed_mapping():
    from pipeline.stage5_tts.adapters.cosyvoice_adapter import SPEED, instruct_for
    assert SPEED[ENERGY_NORMAL] == 1.0
    assert SPEED[ENERGY_EXPLOSIVE] > SPEED[ENERGY_EXCITED] > SPEED[ENERGY_ENGAGED]
    zh = instruct_for(ENERGY_EXPLOSIVE, "zh")
    assert "亢奋" in zh or "高亢" in zh
    en = instruct_for(ENERGY_NORMAL, "en")
    assert "commentator" in en.lower()
    # unknown energy falls back to normal
    assert instruct_for("bogus", "zh") == instruct_for(ENERGY_NORMAL, "zh")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v -k cosyvoice`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the adapter**

Create `pipeline/stage5_tts/adapters/cosyvoice_adapter.py`:

```python
"""CosyVoice3 local TTS adapter (Fun-CosyVoice3-0.5B, instruct-mode voice clone).

Clones the commentator's voice from a reference wav — instruct mode needs NO
transcript of the reference — and controls emotion per energy tier via
instruct_text plus a speed multiplier. Runs on cuda when available, else cpu
(CosyVoice resolves the device internally).

Env vars (all optional, defaults in parentheses):
    COSYVOICE_REPO        path to the cloned FunAudioLLM/CosyVoice repo
                          (<repo_root>/codes/CosyVoice)
    COSYVOICE_MODEL_DIR   model weights dir
                          (<repo_root>/pretrained_models/Fun-CosyVoice3-0.5B)
    COSYVOICE_PROMPT_WAV  reference voice wav (<repo_root>/voice_sample.wav)

One-time setup: bash scripts/setup_cosyvoice.sh
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from pipeline.config import REPO_ROOT
from pipeline.stage5_tts.tts_adapter import (
    ENERGY_ENGAGED,
    ENERGY_EXCITED,
    ENERGY_EXPLOSIVE,
    ENERGY_NORMAL,
    TTSAdapter,
)

log = logging.getLogger(__name__)

INSTRUCT_ZH = {
    ENERGY_NORMAL: "用平稳、自然的中文足球解说语气播报。",
    ENERGY_ENGAGED: "用投入而期待的中文足球解说语气播报，语调略微上扬。",
    ENERGY_EXCITED: "用激动、急促的中文足球解说语气播报，情绪高涨。",
    ENERGY_EXPLOSIVE: "用极度亢奋的中文足球解说语气呐喊播报，像进球瞬间一样高亢有力！",
}
INSTRUCT_EN = {
    ENERGY_NORMAL: "Speak as a calm, natural English football commentator.",
    ENERGY_ENGAGED: "Speak as an engaged football commentator, tone lifting with anticipation.",
    ENERGY_EXCITED: "Speak as a thrilled football commentator, fast and full of excitement.",
    ENERGY_EXPLOSIVE: "Shout like a football commentator at the moment of a goal, ecstatic and powerful!",
}
SPEED = {
    ENERGY_NORMAL: 1.0,
    ENERGY_ENGAGED: 1.05,
    ENERGY_EXCITED: 1.15,
    ENERGY_EXPLOSIVE: 1.2,
}


def instruct_for(energy: str, language: str) -> str:
    table = INSTRUCT_ZH if language == "zh" else INSTRUCT_EN
    return table.get(energy, table[ENERGY_NORMAL])


class CosyVoiceAdapter(TTSAdapter):
    """Fun-CosyVoice3-0.5B via inference_instruct2 (reference wav + instruction).

    The model and reference audio load lazily on first synthesize() so that
    constructing the adapter (e.g. for CLI --help) costs nothing.
    """

    def __init__(
        self,
        language: str = "zh",
        model_dir: str | Path | None = None,
        prompt_wav: str | Path | None = None,
    ) -> None:
        self.language = language
        self.model_dir = Path(
            model_dir
            or os.environ.get(
                "COSYVOICE_MODEL_DIR",
                REPO_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
            )
        )
        self.prompt_wav = Path(
            prompt_wav
            or os.environ.get("COSYVOICE_PROMPT_WAV", REPO_ROOT / "voice_sample.wav")
        )
        self._model = None
        self._prompt_speech = None

    def _load(self) -> None:
        if self._model is not None:
            return
        repo = Path(os.environ.get("COSYVOICE_REPO", REPO_ROOT / "codes" / "CosyVoice"))
        if not repo.is_dir():
            raise RuntimeError(
                f"CosyVoice repo not found: {repo} — run scripts/setup_cosyvoice.sh"
            )
        if not self.model_dir.is_dir():
            raise RuntimeError(
                f"CosyVoice model dir not found: {self.model_dir} — run scripts/setup_cosyvoice.sh"
            )
        if not self.prompt_wav.is_file():
            raise RuntimeError(f"Reference voice wav not found: {self.prompt_wav}")
        for p in (str(repo), str(repo / "third_party" / "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.file_utils import load_wav

        log.info("Loading CosyVoice3 from %s …", self.model_dir)
        self._model = AutoModel(model_dir=str(self.model_dir))
        self._prompt_speech = load_wav(str(self.prompt_wav), 16000)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        energy: str = ENERGY_NORMAL,
    ) -> Path:
        self._load()
        import torch
        import torchaudio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        instruct = instruct_for(energy, self.language)
        speed = SPEED.get(energy, 1.0)

        chunks = [
            out["tts_speech"]
            for out in self._model.inference_instruct2(
                text, instruct, self._prompt_speech, stream=False, speed=speed
            )
        ]
        wav_path = output_path.with_suffix(".wav")
        torchaudio.save(str(wav_path), torch.cat(chunks, dim=1), self._model.sample_rate)

        # synthesize.py caches/trims mp3 files — convert and clean up.
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path),
             "-c:a", "libmp3lame", "-q:a", "2", str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg wav→mp3 failed: {result.stderr[-400:]}")
        wav_path.unlink()
        return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_stage5_speech.py -v -k cosyvoice`
Expected: PASS (mapping test needs no model).

- [ ] **Step 5: Create the setup script**

Create `scripts/setup_cosyvoice.sh` (then `chmod +x scripts/setup_cosyvoice.sh`):

```bash
#!/usr/bin/env bash
# One-time CosyVoice3 setup: clone repo + download Fun-CosyVoice3-0.5B weights (HuggingFace).
# vLLM is NOT required — plain PyTorch inference is used.
#
# Usage: bash scripts/setup_cosyvoice.sh
# Env:   COSYVOICE_REPO       (default codes/CosyVoice)
#        COSYVOICE_MODEL_DIR  (default pretrained_models/Fun-CosyVoice3-0.5B)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_DIR="${COSYVOICE_REPO:-codes/CosyVoice}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "--- Cloning CosyVoice repo → $REPO_DIR"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO_DIR"
fi

echo "--- Installing CosyVoice requirements"
pip install -r "$REPO_DIR/requirements.txt"

echo "--- Downloading Fun-CosyVoice3-0.5B-2512 weights → $MODEL_DIR"
python - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=sys.argv[1])
PY

echo "CosyVoice ready: repo=$REPO_DIR model=$MODEL_DIR"
```

- [ ] **Step 6: Wire the backend**

`pipeline/config.py` line 66 — change:

```python
    tts_backend: str = "cosyvoice"
```

`pipeline/run.py` `_build_tts_adapter` — add before the `doubao_tts` branch:

```python
    if config.tts_backend == "cosyvoice":
        from pipeline.stage5_tts.adapters.cosyvoice_adapter import CosyVoiceAdapter
        return CosyVoiceAdapter(language=config.tts_language)
```

`pipeline/stage5_tts/make_final_video.py` — in `make_final_video`, replace the import line

```python
    from pipeline.stage5_tts.adapters.doubao_tts import DoubaoTTSAdapter
```
with
```python
    from pipeline.stage5_tts.adapters.cosyvoice_adapter import CosyVoiceAdapter
```
and replace the synth call's adapter + log line:
```python
        log.info("Synthesizing 王楚淇 voice (%s) with CosyVoice3 …", language)
        synthesize_commentary(
            commentary_json,
            output_dir,
            language=language,
            adapter=CosyVoiceAdapter(language=language),
            events_json=events_json if events_json.exists() else None,
            voice_tag="wang",
            audio_path=audio_path,
        )
```
Also update the module docstring lines 4–5 and 11 to:
```
Uses CosyVoice3 (Fun-CosyVoice3-0.5B, instruct-mode clone of voice_sample.wav).
Replaces the default-voice track from step 1 with the cloned voice.
...
    - CosyVoice set up via scripts/setup_cosyvoice.sh (repo + weights + voice_sample.wav)
```
(`load_ark_env()` can stay — it's harmless env loading — but remove the now-unused import if the linter complains.)

`scripts/run_stage5.sh` — change the step-2 lines:
```bash
echo "--- Step 2/2: 王楚淇 clone voice (CosyVoice3) ---"
```
and the header comment `# Step 2: doubao_tts 王楚淇 clone → final_video.mp4` to `# Step 2: CosyVoice3 王楚淇 clone → final_video.mp4`.

- [ ] **Step 7: Verify the API signature against the cloned repo (GPU server)**

CosyVoice3's `AutoModel` / `inference_instruct2` shape was confirmed from the project README + example.py, but pin it against the real code after cloning:

Run (on the GPU server, after `bash scripts/setup_cosyvoice.sh`):
```bash
grep -n "inference_instruct2\|class AutoModel\|def inference" codes/CosyVoice/cosyvoice/cli/cosyvoice.py | head -20
sed -n '1,60p' codes/CosyVoice/example.py
```
Expected: `inference_instruct2(tts_text, instruct_text, prompt_speech..., stream=..., speed=...)` exists. **If the signature differs (e.g. no `speed` kwarg, or prompt is a path not a tensor), adjust `CosyVoiceAdapter.synthesize` accordingly — the atempo stage from Task 8 already covers pace if `speed` is unavailable.**

- [ ] **Step 8: Smoke one real synthesis (GPU server)**

Run:
```bash
PYTHONPATH=. python -c "
from pathlib import Path
from pipeline.stage5_tts.adapters.cosyvoice_adapter import CosyVoiceAdapter
a = CosyVoiceAdapter(language='zh')
out = a.synthesize('三十三号禁区外一脚远射，球直挂死角！', Path('/tmp/cosy_smoke.mp3'), energy='excited')
print('OK', out, out.stat().st_size, 'bytes')
"
```
Expected: `OK /tmp/cosy_smoke.mp3 <nonzero> bytes`; listen — voice resembles voice_sample.wav, delivery is excited.

- [ ] **Step 9: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
chmod +x scripts/setup_cosyvoice.sh
git add pipeline/stage5_tts/adapters/cosyvoice_adapter.py scripts/setup_cosyvoice.sh \
        pipeline/config.py pipeline/run.py pipeline/stage5_tts/make_final_video.py \
        scripts/run_stage5.sh tests/test_stage5_speech.py
git commit -m "feat(stage5): CosyVoice3 instruct-mode clone adapter as default TTS backend"
```

---

### Task 10: End-to-end verification on SNGS-116

No code — evidence gathering. Requires ARK API key (`.env`) for Stages 2/4; Step 3 requires the GPU server for CosyVoice.

- [ ] **Step 1: Regenerate events + commentary**

Run:
```bash
FORCE=1 bash scripts/run_stage2.sh outputs/SNGS-116   # new candidate flow + fill_buildup_bins
FORCE=1 bash scripts/run_stage4.sh outputs/SNGS-116   # new prompt + retry + topo autogen
```
Then inspect:
```bash
PYTHONPATH=. python -c "
import json
d = json.load(open('outputs/SNGS-116/commentary.json'))
segs = d['commentary']
print('segments:', len(segs))
for s in segs:
    w = s['end_s'] - s['timestamp_s']
    print(f\"{s['timestamp_s']:6.2f}-{s['end_s']:6.2f} ({w:4.1f}s) energy={s.get('energy','-'):9s} zh={len(s['text_zh'])}ch {s['text_zh'][:30]}\")
assert len(segs) >= 6, 'density requirement failed'
assert max(s['end_s'] - s['timestamp_s'] for s in segs) <= 7.0 + 1e-6, 'window requirement failed'
assert d['language'] == ['en', 'zh'], 'LANG fix failed'
print('DENSITY/WINDOW/LANG OK')
"
```
Expected: `segments: >= 6`, no window over 7s, energies present, `DENSITY/WINDOW/LANG OK`. The old 9–23.5s void must now contain buildup-grounded segments.

- [ ] **Step 2: Preview voice (local, edge-tts)**

Run: `FORCE=1 TTS_LANGUAGE=zh bash scripts/run_stage5.sh outputs/SNGS-116` — step 1 only needs edge-tts; if running on the Mac without CosyVoice, let step 2 fail with the clear setup error and check `raw_final_video.mp4`.
Expected: full sentences audible (no truncation), continuous commentary across the former dead zone, calm→excited delivery differences.

- [ ] **Step 3: Clone voice (GPU server)**

Run:
```bash
bash scripts/setup_cosyvoice.sh          # one-time
FORCE=1 python -m pipeline.stage5_tts.make_final_video --output-dir outputs/SNGS-116 --force
```
Expected: `final_video.mp4` with 王楚淇-cloned voice; goal/shot lines land on their video moments; energy tiers audible.

- [ ] **Step 4: Commit any output-schema docs touched + wrap up**

```bash
git status   # confirm only intended files changed
```
Then use superpowers:finishing-a-development-branch.

---

## Self-Review Notes

- **Spec coverage:** ①更多事件/无解说空档 → Tasks 2, 3, 4, 5 + existing `fill_buildup_bins` (verified wired at run.py:161, BIN_S=5.0); ①句子1.5x更生动 → Task 3; ②不删字提语速 → Tasks 6, 8, 9 (speed param); ③CosyVoice3+HF、克隆 → Task 9; ④额外改进 → Tasks 1 (LANG), 2 (topo), 4/5 (fail-hard parse). 情绪控制 → Tasks 3 (energy enum), 7 (plumbing), 9 (instruct_text).
- **Known risk, contained:** exact `AutoModel`/`inference_instruct2` signature pinned in Task 9 Step 7 against the cloned repo before first real synthesis; `speed` kwarg fallback is the Task 8 atempo stage.
- **Deliberately untouched:** Stage 2 detector/classifier (uncommitted redesign in progress), deleted `tests/` files, edge-tts preview flow, Doubao TTS adapter (kept as non-default option).
