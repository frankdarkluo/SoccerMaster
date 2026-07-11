# Hybrid Event-First Tactical Commentary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained Stage 2B package whose default hybrid mode preserves direct visible events and adds only code-verified tactical observations.

**Architecture:** One ARK call function supports full-video observation, local event verification, radar-based tactical candidates, and composition. Stage 2B owns event/catalog/composition behavior; `pipeline.relations` owns numeric query resolution. Execute this plan before the cleanup plan so obsolete packages are deleted only after replacements pass.

**Tech Stack:** Python 3.10+, OpenAI-compatible ARK API, JSON, FFmpeg/FFprobe, OpenCV, PyYAML, pytest.

## Global Constraints

- Work in the current `main` checkout; do not create a worktree.
- Preserve existing user changes; do not reset, restore, stash, or rewrite unrelated files.
- Never read or stage `AccessKey.txt`; ARK credentials come from environment variables or local `.env`.
- Package name is exactly `pipeline.stage2b`.
- Modes are exactly `direct` and `hybrid`; hybrid is the default.
- Write Stage 2B artifacts below `outputs/<SEQ>/comments/`; only `clip.mp4` remains at the sequence root.
- Do not introduce an adapter factory, base LLM class, schema library, or new dependency.
- Do not delete old packages in this plan; the cleanup plan removes them after replacement verification.
- Add tests only to `tests/test_hybrid_smoke.py`.
- Do not commit after tasks. The cleanup plan creates the single final commit.

## File Map

**Create:** `pipeline/stage2b/{__init__.py,concepts.yaml,digest.py,events.py,generate.py,hybrid.py,run.py,video.py}`, `pipeline/relations/query.py`, `tests/test_hybrid_smoke.py`.

**Modify:** `pipeline/config.py`, `pipeline/relations/{build.py,kinematics.py,snapshots.py}`, `pipeline/run.py`, `scripts/run_stage2b.sh`.

---

### Task 1: Extract tracking primitives and a closed event catalog

**Files:**
- Create: `pipeline/stage2b/__init__.py`
- Create: `pipeline/stage2b/events.py`
- Create: `pipeline/stage2b/digest.py`
- Create: `pipeline/stage2b/concepts.yaml`
- Modify: `pipeline/relations/build.py`
- Modify: `pipeline/relations/kinematics.py`
- Modify: `pipeline/relations/snapshots.py`
- Test: `tests/test_hybrid_smoke.py`

**Interfaces:**
- Produces `EventDef`, `get_event(code)`, `event_prompt_menu()`.
- Produces `FrameData`, `PossessionSegment`, `load_frames(Path)`, `build_tracking_digest(Path, fps)`.
- Produces team/role majority voting and possession segmentation used by relations.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_hybrid_smoke.py`:

```python
import json
from pathlib import Path

from pipeline.stage2b.digest import build_tracking_digest, load_frames
from pipeline.stage2b.events import event_prompt_menu, get_event


def write_predictions(path: Path) -> Path:
    path.write_text(json.dumps({
        "images": [{"image_id": "000001", "file_name": "000001.jpg"}],
        "annotations": [
            {
                "image_id": "000001", "track_id": 1,
                "bbox_pitch": {"x_bottom_middle": -44.0, "y_bottom_middle": 0.0},
                "attributes": {"role": "goalkeeper", "team": "left", "jersey": "1"},
            },
            {
                "image_id": "000001", "track_id": 99,
                "bbox_pitch": {"x_bottom_middle": -43.8, "y_bottom_middle": 0.0},
                "attributes": {"role": "ball"},
            },
        ],
    }), encoding="utf-8")
    return path


def test_stage2b_catalog_and_digest_smoke(tmp_path):
    predictions = write_predictions(tmp_path / "predictions.json")
    frames = load_frames(predictions)
    assert frames[0].players[0]["jersey"] == "1"
    assert get_event("football.corner").importance_base >= 0.35
    assert "football.corner" in event_prompt_menu()
    digest = build_tracking_digest(predictions, fps=25.0)
    assert "left" in digest
    assert "#1" in digest
```

- [ ] **Step 2: Verify the test fails**

Run: `pytest -q tests/test_hybrid_smoke.py::test_stage2b_catalog_and_digest_smoke`

Expected: `ModuleNotFoundError: No module named 'pipeline.stage2b'`.

- [ ] **Step 3: Create the minimal event catalog**

Create `pipeline/stage2b/events.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EventDef:
    code: str
    description: str
    display_name_zh: str
    importance_base: float


_EVENTS = {
    item.code: item
    for item in [
        EventDef("football.corner", "Corner setup or delivery", "角球", 0.35),
        EventDef("football.pass", "Ball delivered to a teammate", "传球", 0.15),
        EventDef("football.clearance", "Defender removes danger", "解围", 0.40),
        EventDef("football.interception", "Opponent wins the ball", "抢断", 0.55),
        EventDef("football.dribble", "Controlled ball carry", "带球", 0.20),
        EventDef("football.tackle", "Challenge wins the ball", "铲抢", 0.60),
        EventDef("football.shoot", "Attempt toward goal", "射门", 0.75),
        EventDef("football.goal", "Ball enters the goal", "进球", 1.00),
        EventDef("football.save", "Goalkeeper prevents a goal", "扑救", 0.80),
        EventDef("football.goal_kick", "Long goalkeeper/defender restart", "开大脚", 0.25),
        EventDef("football.buildup", "Controlled possession advances", "组织推进", 0.10),
        EventDef("football.pressing", "Coordinated pressure", "逼抢", 0.20),
    ]
}


def get_event(code: str) -> EventDef | None:
    return _EVENTS.get(code)


def event_prompt_menu() -> str:
    return "\n".join(
        f"- {event.code}: {event.description} ({event.display_name_zh})"
        for event in _EVENTS.values()
    )
```

No CSV loader, tag hierarchy, or extension mechanism is permitted.

- [ ] **Step 4: Move only retained tracking code**

Create `digest.py` by moving `FrameData` and `PossessionSegment` from `stage2_events/types.py`, `load_frames` from `stage2_events/detector.py`, and the four possession/team helpers from `stage2_events/possession.py`. Change `load_frames` to accept `Path`.

Add:

```python
def build_tracking_digest(path: Path, fps: float) -> str:
    frames = load_frames(path)
    teams = resolve_team_by_track(frames)
    roles = resolve_role_by_track(frames)
    possessions = possession_segments(frames, teams)
    seen = {}
    for frame in frames:
        for player in frame.players:
            track_id = player.get("track_id")
            if track_id is not None:
                seen[int(track_id)] = (
                    teams.get(int(track_id)),
                    roles.get(int(track_id), "player"),
                    str(player.get("jersey") or ""),
                )
    roster = [
        f"- track {track}: {team} {role} #{jersey or '?'}"
        for track, (team, role, jersey) in sorted(seen.items())
    ]
    windows = [
        f"- {(seg.start_fid - 1) / fps:.2f}-{(seg.end_fid - 1) / fps:.2f}s: "
        f"{seg.team} #{seg.jersey or '?'}"
        for seg in possessions
    ]
    return "\n".join(["[Tracked roster]", *roster, "[Possession windows]", *windows])
```

- [ ] **Step 5: Retarget relations imports**

Use:

```python
# relations/build.py
from pipeline.stage2b.digest import FrameData, load_frames

# relations/kinematics.py
from pipeline.stage2b.digest import FrameData, resolve_role_by_track, resolve_team_by_track

# relations/snapshots.py
from pipeline.stage2b.digest import FrameData
```

Update `relations.build.main()` to pass a `Path` to `load_frames`.

- [ ] **Step 6: Move and trim the glossary**

Create `stage2b/concepts.yaml` from the current glossary, retaining only `counter_attack`, `positional_attack`, `overlap_run`, `depth_run`, `width_run`, `switch_play`, `numerical_superiority`, `high_press`, and `low_block`. Each entry contains only `id`, bilingual names, and a concise description.

- [ ] **Step 7: Verify**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py::test_stage2b_catalog_and_digest_smoke
python -m py_compile pipeline/stage2b/events.py pipeline/stage2b/digest.py
```

Expected: one test passes. Do not commit.

---

### Task 2: Implement one ARK function and the direct observer

**Files:**
- Create: `pipeline/stage2b/video.py`
- Create: `pipeline/stage2b/generate.py`
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:**
- `ark_chat(prompt, *, video_path=None, image_paths=None, temperature=0.7, max_tokens=4096) -> str`.
- `observe_direct(video_path, digest, duration_s, languages, call=ark_chat) -> tuple[list[dict], list[dict]]`.
- `verify_event_window(video_path, event, call=ark_chat) -> dict`.

- [ ] **Step 1: Add a failing corner-observer test**

```python
from pipeline.stage2b.generate import observe_direct


def test_direct_observer_preserves_corner():
    reply = json.dumps({
        "events": [{
            "event_id": "evt_001", "start_s": 0.0, "end_s": 5.0,
            "event_code": "football.corner", "player_team": "left",
            "player_jersey": "", "actors": ["left_team"],
            "outcome": "corner_taken", "confidence": "medium",
            "confidence_reasons": ["directly_visible"],
            "suggested_wording_zh": "左侧球队准备主罚角球。",
            "suggested_wording_en": "The left team prepares the corner.",
            "energy": "engaged",
        }],
        "commentary": [{
            "kind": "event", "timestamp_s": 0.0, "end_s": 5.0,
            "text_zh": "左侧球队准备主罚角球。",
            "text_en": "The left team prepares the corner.",
            "fallback_text_zh": "左侧球队主罚角球。",
            "fallback_text_en": "The left team takes a corner.",
            "energy": "engaged", "events_referenced": ["evt_001"],
        }],
    })

    def fake_call(prompt, **kwargs):
        assert "football.corner" in prompt
        assert kwargs["temperature"] == 0.7
        return reply

    events, commentary = observe_direct(
        Path("clip.mp4"), "digest", 30.0, ["en", "zh"], call=fake_call
    )
    assert events[0]["event_code"] == "football.corner"
    assert commentary[0]["events_referenced"] == ["evt_001"]
```

Run: `pytest -q tests/test_hybrid_smoke.py::test_direct_observer_preserves_corner`

Expected: missing `observe_direct`.

- [ ] **Step 2: Implement video helpers**

`video.py` must provide FFprobe duration, frame-to-MP4 construction, and:

```python
def extract_window(source: Path, start_s: float, end_s: float, target: Path) -> Path:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{max(0.0, start_s):.3f}",
            "-i", str(source), "-t", f"{max(0.1, end_s - start_s):.3f}",
            "-c:v", "libx264", "-an", str(target),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg window extraction failed")
    return target
```

- [ ] **Step 3: Implement the ARK function**

Use `OpenAI(base_url=..., api_key=...)`. Encode MP4 as `data:video/mp4;base64,...` and PNG as `data:image/png;base64,...`. Read `ARK_API_KEY`, optional `ARK_BASE_URL`, and optional `ARK_RESPONSES_MODEL`. Missing credentials raise an actionable error. Do not read `AccessKey.txt`.

- [ ] **Step 4: Implement strict direct parsing**

The direct prompt includes the closed event menu, tracking digest, bilingual/fallback fields, and exact energy enum. Validate known codes, unique ids, finite ordered times, clip bounds, bilingual text, fallback text, and references. Retry invalid JSON once with errors; reject the second failure. Never convert arbitrary prose into one 30-second segment.

- [ ] **Step 5: Implement local event verification**

Use `TemporaryDirectory` and `extract_window(event.start_s - 1, event.end_s + 1)`. Call ARK at `temperature=0.1` and require structured code, midpoint, team, jersey, outcome, direct visibility, and disagreements.

- [ ] **Step 6: Verify**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py::test_direct_observer_preserves_corner
python -m py_compile pipeline/stage2b/generate.py pipeline/stage2b/video.py
```

Expected: pass. Do not commit.

---

### Task 3: Move numeric query resolution into relations

**Files:**
- Create: `pipeline/relations/query.py`
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:** `resolve_query(relations, query) -> dict` and `predicate_passes(result, predicate) -> bool`.

- [ ] **Step 1: Add pass/fail assertions**

```python
from pipeline.relations.query import predicate_passes, resolve_query


def test_relation_predicates_are_computed_by_code():
    relations = {"snapshots": [{
        "t": 20.0, "ball": {"speed": 4.0}, "players": [],
        "teams": {"right": {"opp_line_x": -24.0, "n_within_15m_of_ball": 4}},
    }]}
    query = {
        "t0": 18.0, "t1": 23.0, "team": "right",
        "quantity": "opp_line_x", "agg": "mean",
    }
    result = resolve_query(relations, query)
    assert result == {"value": -24.0, "n_samples": 1}
    assert predicate_passes(result, {"op": "<=", "threshold": -20.0})
    assert not predicate_passes(result, {"op": ">", "threshold": -20.0})
    assert resolve_query(relations, {**query, "agg": "median"})["value"] is None
```

Run the test and expect a missing-module failure.

- [ ] **Step 2: Move and simplify resolver logic**

Move supported quantities, aggregations, operators, `resolve_query`, and predicate evaluation from `tactics/verify.py`. Public predicate code:

```python
def predicate_passes(result: dict, predicate: dict) -> bool:
    op = predicate.get("op") if isinstance(predicate, dict) else None
    threshold = predicate.get("threshold") if isinstance(predicate, dict) else None
    return (
        op in PREDICATES
        and type(threshold) in (int, float)
        and math.isfinite(threshold)
        and result.get("value") is not None
        and PREDICATES[op](result["value"], threshold)
    )
```

Retain auditable `UNSUPPORTED:` and `NO DATA` results. Reject boolean thresholds.

- [ ] **Step 3: Verify**

Run: `pytest -q tests/test_hybrid_smoke.py::test_relation_predicates_are_computed_by_code`

Expected: pass. Do not commit.

---

### Task 4: Implement hybrid candidates, audit, and fallback

**Files:**
- Create: `pipeline/stage2b/hybrid.py`
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:**
- `assign_confidence(event, verification, state_ok) -> str`.
- `verify_candidates(relations, proposed) -> list[dict]`.
- `candidate_windows(events, duration_s, phase_scope) -> list[dict]`.
- `audit_commentary(segments, events, candidates, duration_s) -> list[str]`.
- `compose_hybrid(..., call=ark_chat) -> list[dict]`.

- [ ] **Step 1: Add the SNGS-116 regression**

```python
from pipeline.stage2b.hybrid import audit_commentary, verify_candidates


def test_sngs116_corner_survives_and_failed_tactic_is_removed():
    events = [{
        "event_id": "evt_001", "start_s": 0.0, "end_s": 5.0,
        "event_code": "football.corner", "player_team": "left",
        "outcome": "corner_taken", "confidence": "high",
    }]
    proposed = [{
        "candidate_id": "tac_001",
        "window": {"start_s": 0.0, "end_s": 5.0},
        "concept_id": "high_press", "phase_scope": "local",
        "evidence_queries": [{
            "query": {
                "t0": 0.0, "t1": 5.0, "team": "right",
                "quantity": "n_within_15m_of_ball", "agg": "mean",
            },
            "predicate": {"op": ">=", "threshold": 5.0},
        }],
    }]
    relations = {"snapshots": [{
        "t": 2.0, "ball": {"speed": 0.0}, "players": [],
        "teams": {"right": {"n_within_15m_of_ball": 2, "opp_line_x": -20.0}},
    }]}
    assert verify_candidates(relations, proposed) == []

    segments = [{
        "kind": "event", "timestamp_s": 0.0, "end_s": 5.0,
        "text_zh": "左侧球队准备主罚角球。",
        "text_en": "The left team prepares the corner.",
        "fallback_text_zh": "左侧球队主罚角球。",
        "fallback_text_en": "The left team takes the corner.",
        "energy": "engaged", "events_referenced": ["evt_001"],
        "tactical_candidates_referenced": [],
        "event_claims": [{
            "event_id": "evt_001", "event_code": "football.corner",
            "player_team": "left", "outcome": "corner_taken",
            "assertion_strength": "certain",
        }],
    }]
    assert audit_commentary(segments, events, [], 30.0) == []
```

Run and expect missing hybrid functions.

- [ ] **Step 2: Implement confidence assignment**

High requires matching event code, midpoint delta at most 1.0 second, compatible team/outcome, exact spoken jersey, direct visibility, no disagreements, and `state_ok`. Conflicts return low; remaining cases return medium.

- [ ] **Step 3: Implement candidate verification**

Validate concepts against YAML; require one to three queries; require query windows inside candidate windows; resolve every query; compute and overwrite `result`, `predicate_passed`, and `verified`; retain only candidates whose every predicate passes.

- [ ] **Step 4: Implement sparse scheduling**

Generate causal, event-free gap, and completed-phase windows. Standalone gaps require 4.0 seconds. Apply upper bounds: fragmented 30 seconds ≤1, incomplete attack ≤1, complete attack ≤3. Never overlap required event narration.

- [ ] **Step 5: Implement audit and repair**

Audit finite ordered non-overlapping times, clip bounds, enums, bilingual normal/fallback text, required high/visible-medium events, absent low events, exact event claims, verified tactical references, and tactical count. Compose at temperature 0.7. Retry once with audit errors. A second failure returns direct commentary unchanged. Reworded high events require a low-temperature semantic equivalence result.

- [ ] **Step 6: Verify**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py::test_sngs116_corner_survives_and_failed_tactic_is_removed
python -m py_compile pipeline/stage2b/hybrid.py pipeline/relations/query.py
```

Expected: pass. Do not commit.

---

### Task 5: Add Stage 2B CLI and comments contract

**Files:**
- Create: `pipeline/stage2b/run.py`
- Modify: `pipeline/config.py`
- Modify: `pipeline/run.py`
- Replace: `scripts/run_stage2b.sh`
- Extend: `tests/test_hybrid_smoke.py`

**Interfaces:**
- CLI: `python -m pipeline.stage2b.run OUTPUT_DIR [--clip-dir PATH] [--mode direct|hybrid] [--force]`.
- Function: `run_stage2b(output_dir, clip_dir, mode="hybrid", force=False, call=ark_chat, duration_s=None) -> Path`.

- [ ] **Step 1: Add an offline direct-run test**

```python
from pipeline.stage2b.run import run_stage2b


def test_stage2b_offline_writes_comments_contract(tmp_path):
    output = tmp_path / "SNGS-116"
    output.mkdir()
    write_predictions(output / "predictions.json")
    (output / "clip.mp4").write_bytes(b"fake")
    reply = json.dumps({
        "events": [{
            "event_id": "evt_001", "start_s": 0.0, "end_s": 5.0,
            "event_code": "football.corner", "player_team": "left",
            "player_jersey": "", "actors": ["left_team"],
            "outcome": "corner_taken", "confidence": "medium",
            "confidence_reasons": ["directly_visible"],
            "suggested_wording_zh": "左侧球队准备主罚角球。",
            "suggested_wording_en": "The left team prepares the corner.",
            "energy": "engaged",
        }],
        "commentary": [{
            "kind": "event", "timestamp_s": 0.0, "end_s": 5.0,
            "text_zh": "左侧球队准备主罚角球。",
            "text_en": "The left team prepares the corner.",
            "fallback_text_zh": "左侧球队主罚角球。",
            "fallback_text_en": "The left team takes the corner.",
            "energy": "engaged", "events_referenced": ["evt_001"],
        }],
    })
    result = run_stage2b(
        output, tmp_path, mode="direct", force=True,
        call=lambda prompt, **kwargs: reply, duration_s=30.0,
    )
    assert result == output / "comments" / "commentary.json"
    assert (output / "comments" / "events.json").is_file()
    assert (output / "comments" / "event_spine.json").is_file()
```

Run and expect a missing runner/signature failure.

- [ ] **Step 2: Add exact config paths**

Add `comments_dir`, `voice_dir`, `events_json`, `event_spine_json`, `commentary_direct_json`, `tactical_candidates_json`, `commentary_json`, `relations_json`, and `radar_dir` properties. Remove old detector/commentary/TTS flags. Keep `commentary_mode="hybrid"`, `snapshot_hz=2.0`, `radar_hz=1.0`, and `llm_max_images=32`.

- [ ] **Step 3: Implement orchestration**

Direct writes events, spine, and `commentary.json`. Hybrid also writes direct baseline, relations, radar, verified candidates, and final commentary. If relations/radar fail, log the error and copy direct commentary to final. Reuse existing output unless `--force`.

- [ ] **Step 4: Replace the shell launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m pipeline.stage2b.run "$@"
```

Python supplies defaults and validation.

- [ ] **Step 5: Remove old orchestration branches**

Slim `pipeline/run.py` to Stage 1 plus calls to the new Stage 2B entry point. Do not retain aliases for legacy Stage 2, Stage 4 commentary, tactics, or Stage 5; cleanup plan adds renamed Stage 3/4 calls.

- [ ] **Step 6: Verify all offline Stage 2B behavior**

Run:

```bash
pytest -q tests/test_hybrid_smoke.py
python -m pipeline.stage2b.run --help
bash -n scripts/run_stage2b.sh
rg -n "pipeline\.stage2_events|pipeline\.stage4_commentary|pipeline\.tactics" pipeline/stage2b pipeline/relations
```

Expected: tests and CLI pass; stale-import scan has no output. Do not commit.

---

### Task 6: Verify Stage 2B on SNGS-116

**Files:** Verify only `outputs/SNGS-116/`.

**Interfaces:**
- Consumes: the completed offline Stage 2B implementation and local ARK credentials.
- Produces: real direct/hybrid SNGS-116 evidence for the cleanup plan.

- [ ] **Step 1: Generate direct then hybrid**

```bash
bash scripts/run_stage2b.sh outputs/SNGS-116 --mode direct --force
cp outputs/SNGS-116/comments/commentary.json outputs/SNGS-116/comments/commentary_direct.json
bash scripts/run_stage2b.sh outputs/SNGS-116 --mode hybrid --force
```

Expected: all direct/hybrid JSON and radar artifacts exist and are non-empty.

- [ ] **Step 2: Check hard event facts**

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/SNGS-116/comments")
spine_data = json.loads((root / "event_spine.json").read_text())
events = spine_data.get("events", spine_data)
assert "football.corner" in [event["event_code"] for event in events]

commentary_data = json.loads((root / "commentary.json").read_text())
segments = commentary_data.get("commentary", commentary_data.get("segments", []))
refs = {ref for segment in segments for ref in segment.get("events_referenced", [])}
required = {
    event["event_id"] for event in events
    if event["confidence"] == "high"
    or (
        event["confidence"] == "medium"
        and "directly_visible" in event.get("confidence_reasons", [])
    )
}
assert required <= refs
opening = " ".join(s["text_zh"] for s in segments if s["timestamp_s"] < 6)
assert "角球" in opening
assert "高位逼抢" not in opening
PY
```

Expected: assertions pass. Do not commit; continue with the cleanup plan.

## Acceptance Checklist

- [ ] Stage 2B imports no old Stage 2/4/tactics package.
- [ ] Direct and hybrid share one event spine.
- [ ] Hybrid is the default.
- [ ] Corner is in the closed catalog.
- [ ] Predicates are code-computed.
- [ ] Failed/no-data tactics are omitted.
- [ ] Composition failure returns direct commentary.
- [ ] Every segment contains bilingual fallback text.
- [ ] The only added test file is `test_hybrid_smoke.py`.
- [ ] SNGS-116 passes event-preservation checks.
- [ ] No commit exists yet.
