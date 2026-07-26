# Stage 2 VLM Classification Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stage2's rule-based event *typing* with VLM classification: the rule engine emits **untyped key-moment candidates** (precise timestamp + actor + structural facts), and one Doubao call per candidate picks the action from a closed 10-code menu, judges success/failure, and fills visual tags.

**Architecture:** `detector.py` no longer decides "this is a shot" — it detects 5 structural signals (kick, possession change, carry, pressure, geometry), merges them into ≤20 candidates per 30s clip (bin-aware: every 5s bin gets ≥1), and hands each to `classify.py` with a neutral fact block (actor role incl. `goalkeeper`, zone, ball speed/direction, possession before/after) plus annotated evidence frames (−0.5s…+2.0s). The VLM's answer (`action|none`, outcome, attribution, visual tags) becomes the event; `none` drops the candidate. A rule-side `football.buildup` filler guarantees every 5s bin ends with ≥1 event. Fixes the motivating bug: GK long balls (开大脚) labeled `football.shoot` because `_shots()` only checked ball speed + x-band and ignored the `role: goalkeeper` attribute predictions.json already carries.

**Tech Stack:** Python 3, `cv2`, `numpy`, `pytest`, existing `DoubaoAPIAdapter`/`QwenLocalAdapter` (frame-burst image input).

---

## Design Decisions (locked via 2026-07-07 interview)

| # | Decision |
|---|---|
| Granularity | Per-moment classification. Rules own timestamps (possession segments); one VLM call per candidate. |
| Menu | Closed 10 codes: pass, shoot, goal, clearance, interception, dribble, tackle, pressing, save, **goal_kick (new CSV row)** + `none`→drop. Foul is **out of scope** (phase 2). |
| Triggers | Union of 5 signals: kick (segment end + speed), possession change, carry-past-opponent, sustained pressure, geometric (goal line / box entry). Merge within 0.5s. |
| Cap & density | ≤20 candidates / 30s clip, selected bin-aware (5s bins, each ≥1 candidate first, rest by strength). Every 5s bin must end with ≥1 **event**; empty bins get a rule-emitted `football.buildup` filler (second new CSV row, no VLM call). Never force the VLM to fabricate. |
| Prompt | Neutral fact block (role/jersey/team, zone, ball speed+direction, possession before/after, next stable holder) + annotated frames. **No type prior.** |
| Call topology | Single call returns `{action|none, outcome, actor correction, receiver, confidence, tags, reason}`. |
| VLM dependency | Fail-hard without a backend. Delete `--verify-events` flag; keep `--verify-backend` (doubao/qwen_local). MockVLM in tests. |
| Tag split | VLM fills visual groups (body_part, foot_technique, shot_posture, ball_trajectory, pattern_of_play); enricher keeps geometric groups (pitch_zone, shot_distance, pass_distance, pass_direction). Geometry wins conflicts (enricher runs after classification). |
| Outcome | success/failure for 7 duel codes {pass, goal_kick, clearance, interception, tackle, pressing, dribble}. shoot/goal/save exempt. **Failure no longer halves importance.** |
| Evidence window | Asymmetric −0.5s…+2.0s, ≤12 frames, sparse tail sampling. |
| Model | New `ARK_CLASSIFY_MODEL` env var for classification calls (set it to your strongest vision Doubao endpoint in `.env`); stage4 commentary keeps the lite default. |
| Files | `verify.py`→`classify.py`, `Verdict`→`Classification`, new `Candidate` dataclass. `events_detected.json` becomes untyped candidates (v4). `events_verification.json` keeps its name as the classification audit (`action` replaces `corrected_event_code`). Stage4 gap-filler **deleted** (buildup makes it redundant). `events.json` final format unchanged — stage3/5 untouched. |
| Acceptance | pytest+MockVLM for deterministic logic (incl. GK goal-kick fixture) + real-Doubao golden run on SNGS-116/148 with human review: zero role-impossible labels. |

## File Structure

```
reference_football.csv           MODIFY — +2 rows: football.goal_kick, football.buildup
pipeline/stage2_events/
├── types.py         MODIFY — add Candidate + Classification; delete Verdict (Task 11)
├── schema.py        MODIFY — drop pattern_of_play from COMPUTABLE_TAGS
├── possession.py    MODIFY — add resolve_role_by_track (rest unchanged)
├── detector.py      REPLACE — signal detectors → candidates; merge/select; buildup filler;
│                              keeps load_frames, dedup_events, write_events_json, compose_assists
├── evidence.py      MODIFY — build_candidate_frames (generic overlays, asymmetric window);
│                              keeps _homography_valid / image_space_goal_point (tests exist)
├── classify.py      CREATE — fact block, menu prompt, parse, candidate→event, orchestration
├── verify.py        DELETE (Task 11)
└── enricher.py      KEEP — unchanged (its pattern_of_play `.get(...)` already preserves VLM value)

pipeline/config.py               MODIFY — drop verify_events, verify_window_s, ball_speed_shot_threshold_mps
pipeline/run.py                  MODIFY — new run_stage2, _build_classify_adapter, CLI flag removal
pipeline/stage4_commentary/
├── prompt_builder.py            MODIFY — delete gap-filler (_find_gaps, _load_gap_filler_events, block)
└── generate.py                  MODIFY — drop verification_audit_path parameter

tests/
├── conftest.py                        CREATE — GK-boot + pass/interception fixtures, MockVLMAdapter
├── test_schema_codes.py               CREATE
├── test_candidate_types.py            CREATE
├── test_detector_candidates.py        CREATE
├── test_candidate_select.py           CREATE
├── test_evidence_candidates.py        CREATE
├── test_classify.py                   CREATE
├── test_buildup_fill.py               CREATE
├── test_stage2_integration.py         CREATE
├── test_verify_apply_verdict.py       DELETE (Task 11)
└── test_commentary_gap_fill.py        DELETE (Task 10)
```

**Constants** (module-level in `detector.py` unless noted):

```python
KICK_SPEED_MPS = 8.0             # ball acceleration that marks a kick moment
PRESSURE_RADIUS_M = 4.0          # opponent inside this = pressing
PRESSURE_MIN_FRAMES = 10         # ≥0.4s @25fps of sustained pressure
CARRY_MIN_DISPLACEMENT_M = 6.0   # carry length for a dribble-ish moment
GOAL_LINE_TOL_M = 1.5
MERGE_WINDOW_S = 0.5             # raw signals within this collapse into one candidate
CANDIDATE_CAP = 20               # per clip
BIN_S = 5.0                      # density bin width
CLASSIFY_PRE_S = 0.5             # evidence window before the moment   (evidence.py)
CLASSIFY_POST_S = 2.0            # evidence window after the moment    (evidence.py)
SIGNAL_STRENGTH = {"goal_line": 1.0, "kick": 0.7, "possession_win": 0.6,
                   "carry": 0.5, "box_entry": 0.5, "pressure": 0.4}
```

---

## Task 0: Ontology — two new CSV rows + visual tag regrouping

**Files:**
- Modify: `reference_football.csv`
- Modify: `pipeline/stage2_events/schema.py:46`
- Test: `tests/test_schema_codes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_codes.py`:

```python
from pipeline.stage2_events.schema import EventSchema


def test_goal_kick_and_buildup_exist():
    schema = EventSchema()
    gk = schema.get_event("football.goal_kick")
    assert gk is not None
    assert gk.display_name_cn == "开大脚"
    assert 0 < gk.importance_base < 0.5
    bu = schema.get_event("football.buildup")
    assert bu is not None
    assert bu.display_name_cn == "控球推进"
    assert 0 < bu.importance_base < 0.3


def test_pattern_of_play_is_visual_not_computable():
    schema = EventSchema()
    assert "pattern_of_play" in schema.visual_tag_groups()
    assert "pattern_of_play" not in schema.computable_tag_groups()
    # geometric groups stay computable
    assert "pitch_zone" in schema.computable_tag_groups()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gluo/Desktop/SoccerMaster && python3 -m pytest tests/test_schema_codes.py -v`
Expected: FAIL — `get_event("football.goal_kick")` returns None.

- [ ] **Step 3: Add the two CSV rows**

The CSV header is:
`event_id,sport,family,code,display_name_cn,display_name_en,description,level_hint,enabled,source_type,importance_base,tags,trigger_notes,negative_flag,version,notes,Parent items,父记录`

Insert these two lines right after the `football.clearance` row (line 27):

```csv
football.goal_kick,football,possession,goal_kick,开大脚,Goal Kick / Long Boot,Goalkeeper or defender boots the ball long upfield to relieve pressure or restart play,core,,model_direct,0.25,"possession, signal",long high-speed ball launched from own defensive zone,,v4-20260707,,,
football.buildup,football,possession,buildup,控球推进,Build-up Play,Team retains possession and advances the ball without a discrete key action,signal,,model_direct,0.2,"possession, filler",rule-side density filler for silent 5s bins — never VLM-classified,,v4-20260707,,,
```

- [ ] **Step 4: Move pattern_of_play out of COMPUTABLE_TAGS**

In `pipeline/stage2_events/schema.py`, change line 46:

```python
COMPUTABLE_TAGS = {"pitch_zone", "shot_distance", "pass_distance", "pass_direction"}
```

(Only `pattern_of_play` is removed. `enricher.py` needs no change: its line 87 `event.tags.get("pattern_of_play", "open_play")` already preserves a VLM-provided value and defaults to `open_play` otherwise.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schema_codes.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add reference_football.csv pipeline/stage2_events/schema.py tests/test_schema_codes.py
git commit -m "feat(stage2): add goal_kick + buildup codes; pattern_of_play becomes a visual tag"
```

---

## Task 1: Test infrastructure — fixtures + MockVLMAdapter

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py` (empty, if missing)

- [ ] **Step 1: Create tests/__init__.py if missing**

Empty file.

- [ ] **Step 2: Write conftest.py**

Two scripted clips at 25 fps. **GK-boot clip** (the motivating bug): keeper #1 (role `goalkeeper`, left team, defends x=−52.5) holds the ball inside his own penalty area frames 1–10, then boots it — ball travels −44→+8 over frames 11–20 (≈13 m/s, spike ≥ KICK_SPEED), teammate #9 receives at midfield frames 21–30. Correct label is goal_kick, and the old rules would have called the receive-side speed a shot if it happened near a goal. **Pass/interception clip** reuses the proven script: #7 left → #10 left (pass), then #4 right wins it (interception).

```python
import json
from pathlib import Path

import numpy as np
import pytest


def _ann(ann_id, image_id, track_id, role, team, jersey, px, py,
         bx=100, by=200, bw=40, bh=100):
    return {
        "id": str(ann_id), "image_id": str(image_id), "track_id": track_id,
        "supercategory": "object", "category_id": 1,
        "bbox_image": {"x": bx, "y": by, "w": bw, "h": bh,
                       "x_center": bx + bw / 2, "y_center": by + bh / 2},
        "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                       "x_bottom_left": px - 0.3, "y_bottom_left": py,
                       "x_bottom_right": px + 0.3, "y_bottom_right": py},
        "attributes": {"role": role, "team": team, "jersey": jersey},
    }


def _ball_ann(ann_id, image_id, px, py):
    return {
        "id": str(ann_id), "image_id": str(image_id), "track_id": 99,
        "supercategory": "object", "category_id": 4,
        "bbox_image": {"x": 500, "y": 400, "w": 12, "h": 12,
                       "x_center": 506, "y_center": 406},
        "bbox_pitch": {"x_bottom_middle": px, "y_bottom_middle": py,
                       "x_bottom_left": px, "y_bottom_left": py,
                       "x_bottom_right": px, "y_bottom_right": py},
        "attributes": {"role": "ball"},
    }


def _write_predictions(tmp_path, name, n_frames, ball_for, players_for):
    images, annotations = [], []
    aid = 1
    for f in range(1, n_frames + 1):
        image_id = f"3148{f:06d}"
        images.append({"image_id": image_id, "file_name": f"{f:06d}.jpg",
                       "width": 1920, "height": 1080})
        annotations.append(_ball_ann(aid, image_id, *ball_for(f))); aid += 1
        for (tid, role, team, jersey, px, py) in players_for(f):
            annotations.append(_ann(aid, image_id, tid, role, team, jersey, px, py)); aid += 1
    data = {"info": {"name": "FIXT", "fps": 25}, "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "player"}, {"id": 4, "name": "ball"}]}
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def gk_boot_predictions(tmp_path):
    """Keeper #1 (left, own box at x=-44) boots long to #9 at x=+8. 30 frames."""
    def ball_for(f):
        if f <= 10:
            return (-44.0, 0.0)                    # at the keeper's feet
        if f <= 20:
            return (-44.0 + (f - 10) * 5.2, 0.0)   # 5.2 m/frame = 130 m/s spike... scaled below
        return (8.0, 0.0)                          # settled at #9

    # NOTE: 5.2 m/frame is unrealistically fast but safely above KICK_SPEED_MPS;
    # what matters is monotonic travel from own box to midfield.
    def players_for(f):
        return [
            (1, "goalkeeper", "left", "1", -44.0, 0.0),
            (9, "player", "left", "9", 8.0, 0.0),
            (4, "player", "right", "4", 20.0, 5.0),
        ]
    return _write_predictions(tmp_path, "gk_boot.json", 30, ball_for, players_for)


@pytest.fixture
def pass_intercept_predictions(tmp_path):
    """#7 left → #10 left (pass), then #4 right wins it (interception). 25 frames."""
    def holder_for(f):
        if 1 <= f <= 6:
            return 7
        if 9 <= f <= 14:
            return 10
        if 17 <= f <= 22:
            return 4
        return None

    def ball_for(f):
        if f <= 8:
            return (min(f * 1.3, 8.0), 0.0)
        if f <= 16:
            return (8.0 + (f - 8) * 1.0, 0.0)
        return (16.0, 0.0)

    def players_for(f):
        h = holder_for(f)
        out = []
        for (tid, team, jersey, hx) in [(7, "left", "7", 0.0),
                                        (10, "left", "10", 8.0),
                                        (4, "right", "4", 16.0)]:
            px = ball_for(f)[0] if h == tid else hx
            out.append((tid, "player", team, jersey, px, 0.0))
        out.append((50, "referee", None, "", ball_for(f)[0], 1.0))
        return out
    return _write_predictions(tmp_path, "pass_intercept.json", 25, ball_for, players_for)


@pytest.fixture
def homography_file(tmp_path):
    frames = {}
    H = [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]]
    H_inv = np.linalg.inv(np.array(H)).tolist()
    for f in range(1, 31):
        frames[f"3148{f:06d}"] = {"H": H, "H_inv": H_inv, "valid": True}
    p = tmp_path / "homography_per_frame.json"
    p.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return p


@pytest.fixture
def frames_dir(tmp_path):
    import cv2
    d = tmp_path / "img1"
    d.mkdir()
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for f in range(1, 31):
        cv2.imwrite(str(d / f"{f:06d}.jpg"), img)
    return d


class MockVLMAdapter:
    """Injectable stand-in for DoubaoAPIAdapter/QwenLocalAdapter.

    `script` maps a substring of the prompt -> raw JSON string the model returns.
    Records every (prompt, visual_input) call for assertions.
    """
    def __init__(self, script=None, default='{"action": "none", "reason": "quiet"}'):
        self.script = script or {}
        self.default = default
        self.calls = []

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        self.calls.append((prompt, visual_input))
        for key, resp in self.script.items():
            if key in prompt:
                return resp
        return self.default


@pytest.fixture
def mock_adapter():
    return MockVLMAdapter
```

- [ ] **Step 3: Verify pytest still collects cleanly**

Run: `python3 -m pytest tests/ -q --collect-only | tail -3`
Expected: existing tests collected, no conftest errors.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: fixtures for GK boot + pass/interception clips, MockVLMAdapter"
```

---

## Task 2: types.py — Candidate + Classification dataclasses

**Files:**
- Modify: `pipeline/stage2_events/types.py`
- Test: `tests/test_candidate_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_candidate_types.py`:

```python
from pipeline.stage2_events.types import Candidate, Classification


def test_candidate_to_dict_roundtrip():
    c = Candidate(
        candidate_id="cand_001", frame_id=250, timestamp_s=10.0,
        signals=["kick", "possession_change"], strength=0.75,
        track_id=1, jersey="1", team="left", role="goalkeeper",
        ball_speed_mps=13.2, ball_xy=(-44.0, 0.0), ball_direction="toward_opponent_goal",
        prev_holder=None,
        next_holder={"track_id": 9, "jersey": "9", "team": "left",
                     "role": "player", "start_fid": 271},
    )
    d = c.to_dict()
    assert d["candidate_id"] == "cand_001"
    assert d["signals"] == ["kick", "possession_change"]
    assert d["role"] == "goalkeeper"
    assert d["next_holder"]["jersey"] == "9"
    assert d["ball_speed_mps"] == 13.2


def test_classification_defaults():
    cls = Classification()
    assert cls.action == "none"
    assert cls.outcome is None
    assert cls.tags == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_candidate_types.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Append the dataclasses to types.py**

Append to `pipeline/stage2_events/types.py` (leave `Verdict` in place for now — it is deleted with verify.py in Task 11):

```python
@dataclass
class Candidate:
    """An untyped key moment proposed by the rule engine. The VLM names the action."""
    candidate_id: str
    frame_id: int
    timestamp_s: float
    signals: List[str]
    strength: float
    track_id: Optional[int] = None
    jersey: Optional[str] = None
    team: Optional[str] = None
    role: str = "player"
    ball_speed_mps: Optional[float] = None
    ball_xy: Optional[Tuple[float, float]] = None
    ball_direction: Optional[str] = None  # toward_opponent_goal | toward_own_goal | lateral
    prev_holder: Optional[dict] = None    # {track_id, jersey, team, role, end_fid}
    next_holder: Optional[dict] = None    # {track_id, jersey, team, role, start_fid}

    def to_dict(self) -> dict:
        d = {
            "candidate_id": self.candidate_id,
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_id": self.frame_id,
            "signals": self.signals,
            "strength": round(self.strength, 2),
            "track_id": int(self.track_id) if self.track_id is not None else None,
            "jersey": self.jersey,
            "team": self.team,
            "role": self.role,
        }
        if self.ball_speed_mps is not None:
            d["ball_speed_mps"] = round(self.ball_speed_mps, 1)
        if self.ball_xy is not None:
            d["ball_xy"] = [round(self.ball_xy[0], 1), round(self.ball_xy[1], 1)]
        if self.ball_direction:
            d["ball_direction"] = self.ball_direction
        if self.prev_holder:
            d["prev_holder"] = self.prev_holder
        if self.next_holder:
            d["next_holder"] = self.next_holder
        return d


@dataclass
class Classification:
    """Parsed VLM answer for one candidate."""
    action: str = "none"                 # full code (football.pass) or "none"
    outcome: Optional[str] = None        # success | failure | None
    actor_jersey: Optional[str] = None
    actor_team: Optional[str] = None
    receiver_jersey: Optional[str] = None
    confidence: float = 0.5
    tags: Dict[str, str] = field(default_factory=dict)
    reason: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_candidate_types.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/types.py tests/test_candidate_types.py
git commit -m "feat(stage2): Candidate + Classification dataclasses"
```

---

## Task 3: possession.py — role_by_track; detector.py — signal detectors

**Files:**
- Modify: `pipeline/stage2_events/possession.py` (append one function)
- Replace: `pipeline/stage2_events/detector.py`
- Test: `tests/test_detector_candidates.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detector_candidates.py`:

```python
from pipeline.stage2_events.detector import detect_candidates, load_frames
from pipeline.stage2_events.possession import (
    possession_segments,
    resolve_role_by_track,
    resolve_team_by_track,
)
from pipeline.stage2_events.types import FrameData


def _candidates(path):
    frames = load_frames(str(path))
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    return detect_candidates(frames, segs, team, role, fps=25)


def test_role_by_track_majority_vote(gk_boot_predictions):
    frames = load_frames(str(gk_boot_predictions))
    role = resolve_role_by_track(frames)
    assert role[1] == "goalkeeper"
    assert role[9] == "player"


def test_gk_boot_yields_kick_candidate_with_gk_facts(gk_boot_predictions):
    cands = _candidates(gk_boot_predictions)
    kicks = [c for c in cands if "kick" in c.signals]
    assert kicks, "keeper's boot must produce a kick candidate"
    k = kicks[0]
    assert k.role == "goalkeeper" and k.jersey == "1" and k.team == "left"
    assert k.ball_speed_mps and k.ball_speed_mps >= 8.0
    assert k.next_holder and k.next_holder["jersey"] == "9"
    # untyped: a Candidate has no event_code attribute at all
    assert not hasattr(k, "event_code")


def test_cross_team_switch_yields_possession_win(pass_intercept_predictions):
    cands = _candidates(pass_intercept_predictions)
    wins = [c for c in cands if "possession_win" in c.signals]
    assert len(wins) == 1
    w = wins[0]
    assert w.jersey == "4" and w.team == "right"
    assert w.prev_holder and w.prev_holder["team"] == "left"


def test_same_team_switch_yields_kick_not_win(pass_intercept_predictions):
    cands = _candidates(pass_intercept_predictions)
    # the #7 -> #10 same-team hand-off appears as a kick/transition moment
    kicks = [c for c in cands if "kick" in c.signals and c.jersey == "7"]
    assert kicks
    assert kicks[0].next_holder["jersey"] == "10"


def _pressure_frames():
    """#3 right sits 2m from holder #7 left for 15 frames."""
    frames = []
    for f in range(1, 21):
        fd = FrameData(frame_id=f, ball_xy=(0.0, 0.0))
        fd.players = [
            {"track_id": 7, "x": 0.0, "y": 0.0, "role": "player", "team": "left", "jersey": "7"},
            {"track_id": 3, "x": 2.0, "y": 0.0, "role": "player", "team": "right", "jersey": "3"},
        ]
        frames.append(fd)
    return frames


def test_sustained_pressure_yields_pressure_candidate():
    frames = _pressure_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    pressures = [c for c in cands if "pressure" in c.signals]
    assert pressures
    p = pressures[0]
    assert p.jersey == "3" and p.team == "right"       # actor = the presser
    assert p.prev_holder and p.prev_holder["jersey"] == "7"  # the player under pressure


def _carry_frames():
    """#7 left carries 9m with opponent #3 nearby."""
    frames = []
    for f in range(1, 11):
        bx = (f - 1) * 1.0
        fd = FrameData(frame_id=f, ball_xy=(bx, 0.0))
        fd.players = [
            {"track_id": 7, "x": bx, "y": 0.0, "role": "player", "team": "left", "jersey": "7"},
            {"track_id": 3, "x": 3.0, "y": 0.5, "role": "player", "team": "right", "jersey": "3"},
        ]
        frames.append(fd)
    return frames


def test_long_carry_past_opponent_yields_carry_candidate():
    frames = _carry_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    assert [c for c in cands if "carry" in c.signals]


def _goal_line_frames():
    """Ball crosses the +x goal line inside the posts."""
    frames = []
    seq = {1: 47.0, 2: 47.0, 3: 47.0, 4: 50.0, 5: 52.4}
    for f in range(1, 6):
        fd = FrameData(frame_id=f, ball_xy=(seq[f], 0.0))
        fd.players = [{"track_id": 9, "x": 47.0, "y": 0.0, "role": "player",
                       "team": "left", "jersey": "9"}]
        frames.append(fd)
    return frames


def test_goal_line_crossing_yields_geometry_candidate():
    frames = _goal_line_frames()
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = detect_candidates(frames, segs, team, role, fps=25)
    assert [c for c in cands if "goal_line" in c.signals]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_detector_candidates.py -v`
Expected: FAIL — `resolve_role_by_track` / `detect_candidates` not defined.

- [ ] **Step 3: Append resolve_role_by_track to possession.py**

```python
def resolve_role_by_track(frames: List[FrameData]) -> Dict[int, str]:
    """Majority-vote role per track (player / goalkeeper / referee / other)."""
    votes: Dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        for player in frame.players:
            track_id = player.get("track_id")
            role = player.get("role")
            if track_id is None or not role:
                continue
            votes[int(track_id)][role] += 1
    return {tid: counts.most_common(1)[0][0] for tid, counts in votes.items()}
```

- [ ] **Step 4: Replace detector.py**

Replace the whole file. `load_frames` is byte-identical to the current one; `dedup_events`/`write_events_json`/`compose_assists` are carried over with `EVENT_GAP_S` extended; everything typed (`EventDetector`, `_passes`, `_shots`, …) is gone.

```python
"""Untyped key-moment candidate detection from predictions.json.

The rule engine no longer names actions. It detects 5 structural signals —
kick (segment end + ball speed), possession change, carry-past-opponent,
sustained pressure, geometry (goal line / box entry) — merges them into
candidates (timestamp + actor + facts), and classify.py asks a VLM to name
the action. Typed helpers that run AFTER classification (dedup, assists,
buildup density filler, events.json writer) also live here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Candidate, Event, FrameData, PossessionSegment
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.pitch import GOAL_X, GOAL_Y_HALF, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH

KICK_SPEED_MPS = 8.0
PRESSURE_RADIUS_M = 4.0
PRESSURE_MIN_FRAMES = 10
CARRY_MIN_DISPLACEMENT_M = 6.0
CARRY_OPPONENT_RADIUS_M = 4.0
GOAL_LINE_TOL_M = 1.5
MERGE_WINDOW_S = 0.5
CANDIDATE_CAP = 20
BIN_S = 5.0
ASSIST_WINDOW_S = 5.0

SIGNAL_STRENGTH = {
    "goal_line": 1.0,
    "kick": 0.7,
    "possession_win": 0.6,
    "carry": 0.5,
    "box_entry": 0.5,
    "pressure": 0.4,
}

EVENT_GAP_S = {
    "football.goal": 2.0,
    "football.shoot": 1.0,
    "football.pass": 0.4,
    "football.clearance": 1.0,
    "football.interception": 0.8,
    "football.dribble": 1.0,
    "football.assist": 1.0,
    "football.tackle": 0.8,
    "football.pressing": 1.0,
    "football.save": 1.0,
    "football.goal_kick": 1.0,
    "football.buildup": 5.0,
}


def load_frames(path: str) -> List[FrameData]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_frame, _ = frame_index_from_labels(data)
    frames_dict: Dict[int, FrameData] = {
        frame_num: FrameData(frame_id=frame_num)
        for frame_num in sorted(set(image_id_to_frame.values()))
    }

    for ann in data.get("annotations", []):
        frame_num = image_id_to_frame.get(str(ann.get("image_id", "")))
        if frame_num is None:
            continue

        bp = ann.get("bbox_pitch")
        if not isinstance(bp, dict):
            continue

        x = bp.get("x_bottom_middle")
        y = bp.get("y_bottom_middle")
        if x is None or y is None:
            continue

        attrs = ann.get("attributes", {}) or {}
        if attrs.get("role") == "ball":
            frames_dict[frame_num].ball_xy = (float(x), float(y))
        else:
            frames_dict[frame_num].players.append({
                "track_id": ann.get("track_id"),
                "x": float(x),
                "y": float(y),
                "role": attrs.get("role", "other"),
                "team": attrs.get("team"),
                "jersey": attrs.get("jersey", ""),
            })

    return [frames_dict[frame_num] for frame_num in sorted(frames_dict.keys())]


def ball_positions(frames: List[FrameData]) -> Dict[int, Tuple[float, float]]:
    return {f.frame_id: f.ball_xy for f in frames if f.ball_xy is not None}


def ball_velocities(ball_pos: Dict[int, Tuple[float, float]], fps: int) -> Dict[int, float]:
    velocities: Dict[int, float] = {}
    fids = sorted(ball_pos)
    for i in range(1, len(fids)):
        f1, f2 = fids[i - 1], fids[i]
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        dx = ball_pos[f2][0] - ball_pos[f1][0]
        dy = ball_pos[f2][1] - ball_pos[f1][1]
        velocities[f2] = math.hypot(dx, dy) / dt
    return velocities


def _holder_info(segment: PossessionSegment, role_by_track: Dict[int, str],
                 fid_key: str) -> dict:
    return {
        "track_id": segment.track_id,
        "jersey": segment.jersey,
        "team": segment.team,
        "role": role_by_track.get(segment.track_id, "player"),
        fid_key: segment.start_fid if fid_key == "start_fid" else segment.end_fid,
    }


def _max_speed_after(velocities: Dict[int, float], fid: int, horizon: int = 5) -> Optional[float]:
    speeds = [velocities[f] for f in range(fid, fid + horizon + 1) if f in velocities]
    return max(speeds) if speeds else None


def _ball_direction(ball_pos: Dict[int, Tuple[float, float]], fid: int,
                    team: Optional[str], horizon: int = 5) -> Optional[str]:
    pts = [ball_pos[f] for f in range(fid, fid + horizon + 1) if f in ball_pos]
    if len(pts) < 2 or team not in ("left", "right"):
        return None
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    attack_sign = 1 if team != "right" else -1  # left attacks +x (evidence.py convention)
    toward = dx * attack_sign
    if abs(toward) < abs(dy) * 0.5:
        return "lateral"
    return "toward_opponent_goal" if toward > 0 else "toward_own_goal"


def _raw(signal: str, frame_id: int, fps: int, **kw) -> Candidate:
    return Candidate(
        candidate_id="",
        frame_id=frame_id,
        timestamp_s=frame_id / fps,
        signals=[signal],
        strength=SIGNAL_STRENGTH[signal],
        **kw,
    )


def _kick_candidates(segments, velocities, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    for i, seg in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        speed = _max_speed_after(velocities, seg.end_fid)
        if nxt is None and (speed is None or speed < KICK_SPEED_MPS):
            continue  # possession simply faded; not a kick moment
        out.append(_raw(
            "kick", seg.end_fid, fps,
            track_id=seg.track_id, jersey=seg.jersey, team=seg.team,
            role=role_by_track.get(seg.track_id, "player"),
            ball_speed_mps=speed,
            ball_xy=ball_pos.get(seg.end_fid),
            ball_direction=_ball_direction(ball_pos, seg.end_fid, seg.team),
            next_holder=_holder_info(nxt, role_by_track, "start_fid") if nxt else None,
        ))
    return out


def _possession_win_candidates(segments, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    for a, b in zip(segments, segments[1:]):
        if a.team is None or b.team is None or a.team == b.team:
            continue
        out.append(_raw(
            "possession_win", b.start_fid, fps,
            track_id=b.track_id, jersey=b.jersey, team=b.team,
            role=role_by_track.get(b.track_id, "player"),
            ball_xy=ball_pos.get(b.start_fid),
            prev_holder=_holder_info(a, role_by_track, "end_fid"),
        ))
    return out


def _carry_candidates(segments, frames, team_by_track, role_by_track,
                      ball_pos, fps) -> List[Candidate]:
    out = []
    frame_by_id = {f.frame_id: f for f in frames}
    for seg in segments:
        displacement = math.hypot(seg.end_xy[0] - seg.start_xy[0],
                                  seg.end_xy[1] - seg.start_xy[1])
        if displacement < CARRY_MIN_DISPLACEMENT_M:
            continue
        engaged_fid = None
        min_d = None
        for fid in range(seg.start_fid, seg.end_fid + 1):
            frame = frame_by_id.get(fid)
            if frame is None or frame.ball_xy is None:
                continue
            bx, by = frame.ball_xy
            for player in frame.players:
                tid = player.get("track_id")
                if tid is not None and team_by_track.get(int(tid)) == seg.team:
                    continue
                if player.get("role") == "referee":
                    continue
                d = math.hypot(player["x"] - bx, player["y"] - by)
                if d <= CARRY_OPPONENT_RADIUS_M and (min_d is None or d < min_d):
                    min_d, engaged_fid = d, fid
        if engaged_fid is None:
            continue
        out.append(_raw(
            "carry", engaged_fid, fps,
            track_id=seg.track_id, jersey=seg.jersey, team=seg.team,
            role=role_by_track.get(seg.track_id, "player"),
            ball_xy=ball_pos.get(engaged_fid),
        ))
    return out


def _pressure_candidates(segments, frames, team_by_track, role_by_track,
                         ball_pos, fps) -> List[Candidate]:
    out = []
    frame_by_id = {f.frame_id: f for f in frames}
    for seg in segments:
        run_start = None
        run_len = 0
        presser = None
        for fid in range(seg.start_fid, seg.end_fid + 1):
            frame = frame_by_id.get(fid)
            opp = None
            if frame is not None and frame.ball_xy is not None:
                bx, by = frame.ball_xy
                best = None
                for player in frame.players:
                    tid = player.get("track_id")
                    if tid is None or player.get("role") == "referee":
                        continue
                    if team_by_track.get(int(tid)) == seg.team:
                        continue
                    d = math.hypot(player["x"] - bx, player["y"] - by)
                    if d <= PRESSURE_RADIUS_M and (best is None or d < best[0]):
                        best = (d, player)
                opp = best[1] if best else None
            if opp is not None:
                if run_start is None:
                    run_start, presser = fid, opp
                run_len += 1
                if run_len == PRESSURE_MIN_FRAMES:
                    out.append(_raw(
                        "pressure", run_start, fps,
                        track_id=int(presser["track_id"]),
                        jersey=presser.get("jersey"),
                        team=team_by_track.get(int(presser["track_id"])),
                        role=role_by_track.get(int(presser["track_id"]), "player"),
                        ball_xy=ball_pos.get(run_start),
                        prev_holder=_holder_info(seg, role_by_track, "end_fid"),
                    ))
            else:
                run_start, run_len, presser = None, 0, None
    return out


def _in_penalty_area(bx: float, by: float) -> bool:
    return abs(bx) >= (GOAL_X - PENALTY_AREA_LENGTH) and abs(by) <= PENALTY_AREA_WIDTH / 2


def _last_holder_before(segments, frame_id) -> Optional[PossessionSegment]:
    best = None
    for seg in segments:
        if seg.end_fid <= frame_id and (best is None or seg.end_fid > best.end_fid):
            best = seg
    return best


def _geometry_candidates(segments, ball_pos, role_by_track, fps) -> List[Candidate]:
    out = []
    fids = sorted(ball_pos)
    seen_goal_line = False
    for i, fid in enumerate(fids):
        bx, by = ball_pos[fid]
        crossed = abs(abs(bx) - GOAL_X) <= GOAL_LINE_TOL_M and abs(by) <= GOAL_Y_HALF
        entered = (
            i > 0
            and _in_penalty_area(bx, by)
            and not _in_penalty_area(*ball_pos[fids[i - 1]])
        )
        signal = "goal_line" if crossed and not seen_goal_line else (
            "box_entry" if entered else None)
        if crossed:
            seen_goal_line = True  # one goal-line candidate per contiguous crossing
        elif not entered:
            seen_goal_line = False
        if signal is None:
            continue
        holder = _last_holder_before(segments, fid)
        out.append(_raw(
            signal, fid, fps,
            track_id=holder.track_id if holder else None,
            jersey=holder.jersey if holder else None,
            team=holder.team if holder else None,
            role=role_by_track.get(holder.track_id, "player") if holder else "player",
            ball_xy=ball_pos.get(fid),
        ))
    return out


def detect_candidates(
    frames: List[FrameData],
    segments: List[PossessionSegment],
    team_by_track: Dict[int, Optional[str]],
    role_by_track: Dict[int, str],
    fps: int = 25,
) -> List[Candidate]:
    """All raw signal candidates, time-sorted, unmerged and without ids."""
    ball_pos = ball_positions(frames)
    velocities = ball_velocities(ball_pos, fps)
    raw: List[Candidate] = []
    raw += _kick_candidates(segments, velocities, ball_pos, role_by_track, fps)
    raw += _possession_win_candidates(segments, ball_pos, role_by_track, fps)
    raw += _carry_candidates(segments, frames, team_by_track, role_by_track, ball_pos, fps)
    raw += _pressure_candidates(segments, frames, team_by_track, role_by_track, ball_pos, fps)
    raw += _geometry_candidates(segments, ball_pos, role_by_track, fps)
    return sorted(raw, key=lambda c: c.timestamp_s)


# ---------- typed helpers that run AFTER classification ----------

def dedup_events(events: List[Event]) -> List[Event]:
    by_code: Dict[str, List[Event]] = {}
    for ev in events:
        by_code.setdefault(ev.event_code, []).append(ev)
    kept: List[Event] = []
    for code, group in by_code.items():
        gap = EVENT_GAP_S.get(code, 1.0)
        for ev in sorted(group, key=lambda e: e.timestamp_s):
            if kept and kept[-1].event_code == code and ev.timestamp_s - kept[-1].timestamp_s < gap:
                prev = kept[-1]
                better = ev.importance > prev.importance or (
                    ev.importance == prev.importance and ev.confidence > prev.confidence
                )
                if better:
                    kept[-1] = ev
            else:
                kept.append(ev)
    return sorted(kept, key=lambda e: e.timestamp_s)


def write_events_json(events: List[Event], output_path, video_info: Optional[dict] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info or {},
        "schema_version": "v3-20260319",
        "events": [e.to_dict() for e in events],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_candidates_json(candidates: List[Candidate], output_path,
                          video_info: Optional[dict] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_info": video_info or {},
        "schema_version": "v4-candidates-20260707",
        "candidates": [c.to_dict() for c in candidates],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compose_assists(events: List[Event]) -> List[Event]:
    """Append assist events for a pass whose receiver scores within ASSIST_WINDOW_S.

    Runs POST-classification so only surviving passes and goals qualify.
    """
    schema = EventSchema()
    goals = [e for e in events if e.event_code == "football.goal"]
    passes = [e for e in events if e.event_code == "football.pass"]
    ev_def = schema.get_event("football.assist")
    out = list(events)
    counter = len(events)
    for goal in goals:
        cands = [
            p for p in passes
            if 0 < (goal.timestamp_s - p.timestamp_s) < ASSIST_WINDOW_S
            and p.target_team == goal.player_team
            and (p.target_jersey == goal.player_jersey or goal.player_jersey is None)
        ]
        if not cands:
            continue
        last = max(cands, key=lambda p: p.timestamp_s)
        counter += 1
        out.append(Event(
            event_id=f"evt_{counter:03d}",
            timestamp_s=last.timestamp_s,
            frame_id=last.frame_id,
            event_code="football.assist",
            display_name_en=ev_def.display_name_en,
            display_name_cn=ev_def.display_name_cn,
            importance=ev_def.importance_base,
            player_jersey=last.player_jersey,
            player_team=last.player_team,
            target_jersey=last.target_jersey,
            target_team=last.target_team,
            confidence=0.8,
            description_hint=f"assist: #{last.player_jersey} before goal",
        ))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_detector_candidates.py -v`
Expected: 7 PASS. (Other suites are broken at this point — verify.py imports still resolve, run.py doesn't; that's expected mid-rewrite and is fixed by Tasks 9–11.)

- [ ] **Step 6: Commit**

```bash
git add pipeline/stage2_events/possession.py pipeline/stage2_events/detector.py tests/test_detector_candidates.py
git commit -m "feat(stage2): untyped 5-signal candidate detection replaces typed detectors"
```

---

## Task 4: Candidate merge + bin-aware selection (cap 20, every 5s bin covered)

**Files:**
- Modify: `pipeline/stage2_events/detector.py`
- Test: `tests/test_candidate_select.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidate_select.py`:

```python
from pipeline.stage2_events.detector import merge_candidates, select_candidates
from pipeline.stage2_events.types import Candidate


def _cand(t, signals, strength, jersey="7", fps=25):
    return Candidate(
        candidate_id="", frame_id=int(t * fps), timestamp_s=t,
        signals=list(signals), strength=strength,
        track_id=int(jersey), jersey=jersey, team="left",
    )


def test_merge_within_half_second_unions_signals():
    a = _cand(4.00, ["kick"], 0.7)
    b = _cand(4.30, ["possession_win"], 0.6, jersey="4")
    c = _cand(6.00, ["carry"], 0.5)
    merged = merge_candidates([a, b, c], window_s=0.5)
    assert len(merged) == 2
    first = merged[0]
    assert set(first.signals) == {"kick", "possession_win"}
    assert first.jersey == "7"                    # actor of the strongest signal wins
    assert first.strength == 0.7 + 0.05           # max + 0.05 per extra signal
    assert merged[0].candidate_id == "cand_001"   # ids assigned after merge
    assert merged[1].candidate_id == "cand_002"


def test_selection_guarantees_every_bin_then_fills_by_strength():
    cands = []
    # bin 0 (0-5s): one weak candidate — must survive despite low strength
    cands.append(_cand(2.0, ["pressure"], 0.4))
    # bin 1 (5-10s): 30 strong candidates — capped
    for i in range(30):
        cands.append(_cand(5.1 + i * 0.15, ["kick"], 0.9))
    # bins 2-5 empty
    selected = select_candidates(cands, duration_s=30.0, cap=20, bin_s=5.0)
    assert len(selected) == 20
    assert any(c.timestamp_s < 5.0 for c in selected), "weak bin-0 candidate must be kept"
    assert selected == sorted(selected, key=lambda c: c.timestamp_s)


def test_selection_under_cap_keeps_everything():
    cands = [_cand(1.0, ["kick"], 0.7), _cand(9.0, ["carry"], 0.5)]
    assert len(select_candidates(cands, duration_s=30.0)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_candidate_select.py -v`
Expected: FAIL — `merge_candidates` not defined.

- [ ] **Step 3: Implement merge + select in detector.py**

Append after `detect_candidates`:

```python
def merge_candidates(raw: List[Candidate], window_s: float = MERGE_WINDOW_S) -> List[Candidate]:
    """Greedy time-clustering: signals within window_s collapse into one candidate.

    The strongest member donates actor/facts; signals are unioned; strength is
    max + 0.05 per extra signal. Ids (cand_001..) are assigned here.
    """
    merged: List[Candidate] = []
    cluster: List[Candidate] = []

    def _flush():
        if not cluster:
            return
        best = max(cluster, key=lambda c: c.strength)
        signals: List[str] = []
        for c in sorted(cluster, key=lambda c: -c.strength):
            for s in c.signals:
                if s not in signals:
                    signals.append(s)
        best.signals = signals
        best.strength = round(max(c.strength for c in cluster) + 0.05 * (len(cluster) - 1), 2)
        for c in cluster:
            if best.prev_holder is None and c.prev_holder is not None:
                best.prev_holder = c.prev_holder
            if best.next_holder is None and c.next_holder is not None:
                best.next_holder = c.next_holder
            if best.ball_speed_mps is None and c.ball_speed_mps is not None:
                best.ball_speed_mps = c.ball_speed_mps
        merged.append(best)

    for cand in sorted(raw, key=lambda c: c.timestamp_s):
        if cluster and cand.timestamp_s - cluster[0].timestamp_s < window_s:
            cluster.append(cand)
        else:
            _flush()
            cluster = [cand]
    _flush()

    for i, cand in enumerate(merged, start=1):
        cand.candidate_id = f"cand_{i:03d}"
    return merged


def select_candidates(
    candidates: List[Candidate],
    duration_s: float,
    cap: int = CANDIDATE_CAP,
    bin_s: float = BIN_S,
) -> List[Candidate]:
    """Bin-aware top-k: each bin_s bin keeps its strongest candidate first,
    remaining slots go to the globally strongest of the rest."""
    if len(candidates) <= cap:
        return sorted(candidates, key=lambda c: c.timestamp_s)

    n_bins = max(1, math.ceil(duration_s / bin_s))
    by_bin: Dict[int, List[Candidate]] = {}
    for cand in candidates:
        b = min(int(cand.timestamp_s // bin_s), n_bins - 1)
        by_bin.setdefault(b, []).append(cand)

    guaranteed = [max(group, key=lambda c: c.strength) for group in by_bin.values()]
    chosen = set(id(c) for c in guaranteed)
    rest = sorted(
        (c for c in candidates if id(c) not in chosen),
        key=lambda c: -c.strength,
    )
    for cand in rest:
        if len(guaranteed) >= cap:
            break
        guaranteed.append(cand)
    return sorted(guaranteed, key=lambda c: c.timestamp_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_candidate_select.py tests/test_detector_candidates.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/detector.py tests/test_candidate_select.py
git commit -m "feat(stage2): candidate merge (0.5s) + bin-aware cap of 20 with per-bin guarantee"
```

---

## Task 5: evidence.py — generic candidate overlays, asymmetric window

**Files:**
- Modify: `pipeline/stage2_events/evidence.py`
- Test: `tests/test_evidence_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evidence_candidates.py`:

```python
from pipeline.stage2_events.detector import (
    detect_candidates, load_frames, merge_candidates,
)
from pipeline.stage2_events.evidence import (
    build_bbox_index, build_candidate_frames, load_homography,
)
from pipeline.stage2_events.possession import (
    possession_segments, resolve_role_by_track, resolve_team_by_track,
)


def test_candidate_burst_is_asymmetric_and_annotated(
        tmp_path, gk_boot_predictions, homography_file, frames_dir):
    frames = load_frames(str(gk_boot_predictions))
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = merge_candidates(detect_candidates(frames, segs, team, role, fps=25))
    kick = next(c for c in cands if "kick" in c.signals)

    idx = build_bbox_index(str(gk_boot_predictions))
    homo = load_homography(str(homography_file))
    paths = build_candidate_frames(
        kick, frames_dir, idx, homo, str(gk_boot_predictions),
        tmp_path / "burst", fps=25,
    )
    assert paths and all(p.exists() for p in paths)
    assert len(paths) <= 12
    fnums = [int(p.stem.split("_")[-1]) for p in paths]
    # asymmetric: more context after the moment than before
    assert max(fnums) - kick.frame_id > kick.frame_id - min(fnums)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evidence_candidates.py -v`
Expected: FAIL — `build_candidate_frames` not defined.

- [ ] **Step 3: Add build_candidate_frames to evidence.py**

Keep `load_homography`, `project_pitch_to_image`, `build_bbox_index`, `_box`, `_center`, `_homography_valid`, `image_space_goal_point`, `_frame_to_image_id` exactly as they are (`tests/test_evidence_fallback.py` depends on two of them). **Do NOT delete `build_evidence_frames` / `_receiver_bbox_for_frame` yet** — `verify.py` still imports the former until Task 11 removes both files' dead code together. Add:

```python
CLASSIFY_PRE_S = 0.5
CLASSIFY_POST_S = 2.0


def build_candidate_frames(
    candidate,
    frames_dir: Path,
    bbox_index: Dict[Tuple[int, int], dict],
    homo: Dict[str, dict],
    predictions_json_path: str,
    out_dir: Path,
    fps: int = 25,
    pre_s: float = CLASSIFY_PRE_S,
    post_s: float = CLASSIFY_POST_S,
    max_frames: int = 12,
) -> List[Path]:
    """Annotated JPG burst for an untyped candidate: actor red box, ball green
    circle, next-holder amber box, goal arrow only when the ball is inside a
    penalty-area band (structural fact, not a type hint)."""
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_to_image = _frame_to_image_id(predictions_json_path)

    lo = max(1, candidate.frame_id - int(round(pre_s * fps)))
    hi = candidate.frame_id + int(round(post_s * fps))
    candidate_fnums = [f for f in range(lo, hi + 1)
                       if (frames_dir / f"{f:06d}.jpg").exists()]
    if not candidate_fnums:
        return []
    if len(candidate_fnums) > max_frames:
        step = len(candidate_fnums) / max_frames
        candidate_fnums = [candidate_fnums[int(i * step)] for i in range(max_frames)]

    near_goal = (
        candidate.ball_xy is not None
        and abs(candidate.ball_xy[0]) >= (GOAL_X - PENALTY_AREA_LENGTH)
    )
    goal_x = GOAL_X if candidate.team != "right" else -GOAL_X
    next_track = (candidate.next_holder or {}).get("track_id")

    out_paths: List[Path] = []
    for fnum in candidate_fnums:
        img = cv2.imread(str(frames_dir / f"{fnum:06d}.jpg"))
        if img is None:
            continue

        actor_bbox = None
        if candidate.track_id is not None:
            actor_bbox = bbox_index.get((fnum, int(candidate.track_id)))
            if actor_bbox:
                _box(img, actor_bbox, ACTOR_COLOR)

        if next_track is not None:
            receiver_bbox = bbox_index.get((fnum, int(next_track)))
            if receiver_bbox:
                _box(img, receiver_bbox, RECEIVER_COLOR, thickness=2, expand=4)

        ball_bbox = bbox_index.get((fnum, 99))
        if ball_bbox:
            cx, cy = _center(ball_bbox)
            cv2.circle(img, (cx, cy), 8, BALL_COLOR, 2)

        if near_goal and actor_bbox:
            image_id = frame_to_image.get(fnum)
            goal_point = None
            if image_id is not None and _homography_valid(homo, image_id):
                goal_point = project_pitch_to_image(homo, image_id, goal_x, 0.0)
            if goal_point is None and image_id is not None and not _homography_valid(homo, image_id):
                gx, gy = image_space_goal_point(actor_bbox, candidate.team, img.shape[1])
                goal_point = (float(gx), float(gy))
            if goal_point is not None:
                ax, ay = _center(actor_bbox)
                cv2.arrowedLine(img, (ax, ay),
                                (int(goal_point[0]), int(goal_point[1])),
                                GOAL_ARROW_COLOR, 2, tipLength=0.05)

        out = out_dir / f"{candidate.candidate_id}_{fnum:06d}.jpg"
        cv2.imwrite(str(out), img)
        out_paths.append(out)

    return out_paths
```

Also add `PENALTY_AREA_LENGTH` to the `pipeline.utils.pitch` import at the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_evidence_candidates.py tests/test_evidence_fallback.py -v`
Expected: all PASS (fallback tests untouched).

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/evidence.py tests/test_evidence_candidates.py
git commit -m "feat(stage2): generic candidate evidence burst with -0.5s/+2.0s window"
```

---

## Task 6: classify.py — fact block, menu prompt, parser, candidate→event

**Files:**
- Create: `pipeline/stage2_events/classify.py`
- Test: `tests/test_classify.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classify.py`:

```python
from pipeline.stage2_events.classify import (
    DUEL_OUTCOME_CODES, MENU, build_classify_prompt, candidate_to_event,
    parse_classification,
)
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Candidate, Classification


def _gk_candidate():
    return Candidate(
        candidate_id="cand_001", frame_id=250, timestamp_s=10.0,
        signals=["kick", "possession_win"], strength=0.75,
        track_id=1, jersey="1", team="left", role="goalkeeper",
        ball_speed_mps=13.2, ball_xy=(-44.0, 0.0),
        ball_direction="toward_opponent_goal",
        next_holder={"track_id": 9, "jersey": "9", "team": "left",
                     "role": "player", "start_fid": 271},
    )


def test_prompt_states_facts_without_type_prior():
    prompt = build_classify_prompt(_gk_candidate(), EventSchema())
    assert "goalkeeper" in prompt
    assert "own penalty area" in prompt
    assert "13.2" in prompt
    assert "#9" in prompt                       # next stable holder fact
    assert "football.goal_kick" in prompt       # menu offered
    assert '"action"' in prompt                 # JSON contract
    # no type prior: the words below must only appear inside the menu listing,
    # never as "candidate shot/pass" phrasing
    assert "candidate \"" not in prompt.lower()
    assert "flagged a candidate" not in prompt.lower()


def test_parse_valid_answer():
    raw = ('Sure. {"action": "goal_kick", "outcome": "success", '
           '"actor_jersey": "1", "actor_team": "left", "receiver_jersey": "9", '
           '"confidence": 0.9, "tags": {"body_part": "right_foot"}, '
           '"reason": "keeper boots long"}')
    cls = parse_classification(raw)
    assert cls.action == "football.goal_kick"   # prefix normalized
    assert cls.outcome == "success"
    assert cls.confidence == 0.9
    assert cls.tags == {"body_part": "right_foot"}


def test_parse_malformed_and_invalid_action_become_none():
    assert parse_classification("no json here").action == "none"
    assert parse_classification('{"action": "football.rocket"}').action == "none"
    assert parse_classification('{"action": "none"}').action == "none"


def test_candidate_to_event_maps_schema_fields():
    schema = EventSchema()
    cls = Classification(action="football.goal_kick", outcome="success",
                         confidence=0.9, tags={"body_part": "right_foot"})
    ev = candidate_to_event(_gk_candidate(), cls, schema, counter=1)
    assert ev.event_code == "football.goal_kick"
    assert ev.event_id == "evt_001"
    assert ev.timestamp_s == 10.0               # timestamp authority = rules
    assert ev.player_jersey == "1" and ev.player_team == "left"
    assert ev.tags["outcome"] == "success"
    assert ev.tags["body_part"] == "right_foot"
    assert ev.importance == schema.get_event("football.goal_kick").importance_base


def test_failure_outcome_does_not_halve_importance():
    schema = EventSchema()
    base = schema.get_event("football.pass").importance_base
    cls = Classification(action="football.pass", outcome="failure", confidence=0.8)
    ev = candidate_to_event(_gk_candidate(), cls, schema, counter=2)
    assert ev.tags["outcome"] == "failure"
    assert ev.importance == base                # NOT halved


def test_outcome_only_kept_for_duel_codes():
    schema = EventSchema()
    cls = Classification(action="football.shoot", outcome="success", confidence=0.8)
    ev = candidate_to_event(_gk_candidate(), cls, schema, counter=3)
    assert "outcome" not in ev.tags
    assert "football.shoot" not in DUEL_OUTCOME_CODES


def test_none_returns_no_event_and_vlm_corrections_apply():
    schema = EventSchema()
    assert candidate_to_event(_gk_candidate(), Classification(action="none"),
                              schema, counter=4) is None
    cls = Classification(action="football.pass", actor_jersey="10",
                         actor_team="right", receiver_jersey="4", confidence=0.7)
    ev = candidate_to_event(_gk_candidate(), cls, schema, counter=5)
    assert ev.player_jersey == "10" and ev.player_team == "right"
    assert ev.target_jersey == "4" and ev.target_team == "right"


def test_invalid_tags_are_dropped():
    schema = EventSchema()
    cls = Classification(action="football.shoot", confidence=0.8,
                         tags={"body_part": "tentacle", "made_up_group": "x",
                               "shot_posture": "volley"})
    ev = candidate_to_event(_gk_candidate(), cls, schema, counter=6)
    assert ev.tags.get("shot_posture") == "volley"
    assert "body_part" not in ev.tags
    assert "made_up_group" not in ev.tags


def test_menu_is_exactly_ten_codes():
    assert len(MENU) == 10
    assert "football.foul" not in MENU
    assert "football.buildup" not in MENU       # filler is rule-side only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_classify.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Create classify.py**

```python
"""VLM classification of untyped rule-engine candidates.

The rule engine proposes key moments (timestamp + actor + structural facts);
a VLM (Doubao API default, local Qwen switchable) watches an annotated frame
burst and names the action from a closed menu, judges success/failure for duel
actions, corrects attribution, and optionally fills visual tags. "none" drops
the candidate. Timestamps are never changed by the VLM.

Fail-hard: no try/except around infra or logic. The ONLY guard is model-text
-> JSON parsing, which maps malformed output to a deterministic action="none".
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.stage2_events.evidence import (
    build_bbox_index,
    build_candidate_frames,
    load_homography,
)
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Candidate, Classification, Event
from pipeline.utils.pitch import GOAL_X, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH

MENU = [
    "football.pass",
    "football.shoot",
    "football.goal",
    "football.clearance",
    "football.interception",
    "football.dribble",
    "football.tackle",
    "football.pressing",
    "football.save",
    "football.goal_kick",
]

DUEL_OUTCOME_CODES = {
    "football.pass",
    "football.goal_kick",
    "football.clearance",
    "football.interception",
    "football.tackle",
    "football.pressing",
    "football.dribble",
}

RECEIVER_CODES = {"football.pass", "football.goal_kick"}

CLASSIFY_CLIPS_DIR = "classify_clips"

_SUCCESS = {"success", "successful", "succeeded", "true", "yes", "1",
            "complete", "completed", "won"}
_FAILURE = {"failure", "failed", "fail", "unsuccessful", "false", "no", "0",
            "incomplete", "lost"}

SIGNAL_FACTS = {
    "kick": "the ball suddenly accelerated away from the actor",
    "possession_win": "possession changed to the actor's team at this moment",
    "carry": "the actor carried the ball several meters with an opponent close by",
    "pressure": "the actor closed down the opposing ball holder",
    "goal_line": "the ball crossed the goal-line plane between the posts",
    "box_entry": "the ball entered the penalty area",
}


def normalize_outcome(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _SUCCESS:
        return "success"
    if s in _FAILURE:
        return "failure"
    return None


def zone_text(ball_xy: Optional[Tuple[float, float]], team: Optional[str]) -> str:
    if ball_xy is None:
        return "unknown zone"
    bx, by = ball_xy
    attack_sign = 1 if team != "right" else -1  # left attacks +x
    toward_opp = bx * attack_sign
    in_box_band = abs(bx) >= (GOAL_X - PENALTY_AREA_LENGTH) and abs(by) <= PENALTY_AREA_WIDTH / 2
    if toward_opp < 0:
        return "own penalty area" if in_box_band else "own half"
    return "opponent penalty area" if in_box_band else "opponent half"


def fact_block(candidate: Candidate) -> str:
    lines = [
        f"- Time t={candidate.timestamp_s:.2f}s. Actor (red box): "
        f"#{candidate.jersey or '?'}, {candidate.team or 'unknown'} team, "
        f"role: {candidate.role}.",
        f"- Ball location: {zone_text(candidate.ball_xy, candidate.team)} (actor's perspective).",
    ]
    if candidate.ball_speed_mps is not None:
        direction = (candidate.ball_direction or "unknown direction").replace("_", " ")
        lines.append(f"- Ball leaves at ~{candidate.ball_speed_mps:.1f} m/s, {direction}.")
    if candidate.prev_holder:
        p = candidate.prev_holder
        lines.append(
            f"- Possession before: #{p.get('jersey') or '?'} ({p.get('team')}, {p.get('role')})."
        )
    if candidate.next_holder:
        n = candidate.next_holder
        lines.append(
            f"- Next stable possession: #{n.get('jersey') or '?'} "
            f"({n.get('team')}, {n.get('role')}) — amber box in the frames."
        )
    else:
        lines.append("- No stable possession within ~2s after this moment "
                     "(ball out of tracking, loose, or play stopped).")
    observed = [SIGNAL_FACTS[s] for s in candidate.signals if s in SIGNAL_FACTS]
    if observed:
        lines.append("- Observed by tracking: " + "; ".join(observed) + ".")
    return "\n".join(lines)


def menu_text(schema: EventSchema) -> str:
    lines = []
    for code in MENU:
        ev = schema.get_event(code)
        lines.append(f"- {code}: {ev.description} ({ev.display_name_cn} / {ev.display_name_en})")
    lines.append("- none: nothing notable happened at this moment")
    return "\n".join(lines)


def visual_tag_menu(schema: EventSchema) -> str:
    lines = []
    for group_code in schema.visual_tag_groups():
        tg = schema.get_tag_group(group_code)
        if tg is None or not tg.values:
            continue
        applies = ", ".join(tg.applies_to) if tg.applies_to else "any"
        values = ", ".join(v.code for v in tg.values)
        lines.append(f"- {group_code} (applies to: {applies}): {values}")
    return "\n".join(lines)


def build_classify_prompt(candidate: Candidate, schema: EventSchema) -> str:
    return (
        "You are a professional football analyst. A tracking system marked a key "
        "moment in this clip, WITHOUT deciding what happened. The frames show the "
        "moment (red box = actor, green circle = ball, amber box = next player to "
        "hold the ball, yellow arrow = direction of the nearest goal when relevant).\n\n"
        "Facts from the tracking system (reliable, use them):\n"
        f"{fact_block(candidate)}\n\n"
        "Decide what the actor did at this moment. Choose EXACTLY ONE action:\n"
        f"{menu_text(schema)}\n\n"
        "Rules:\n"
        "1) Use football sense: a goalkeeper or last defender booting a long ball "
        "out of the defensive zone is a goal_kick or clearance, never a shoot. A "
        "shoot must be a deliberate attempt at the OPPONENT goal.\n"
        "2) outcome — only for pass, goal_kick, clearance, interception, tackle, "
        "pressing, dribble: did the action achieve its purpose? (pass/goal_kick "
        "reaches a teammate; clearance relieves danger; interception/tackle wins "
        "the ball; pressing forces a turnover or rushed clearance; dribble beats "
        "the opponent). Otherwise null.\n"
        "3) actor_jersey / actor_team (left|right) — correct the actor if the red "
        "box is on the wrong player. receiver_jersey — for pass/goal_kick only.\n"
        "4) tags — optional visual detail, only when clearly visible; pick values "
        "from these groups (leave out anything uncertain):\n"
        f"{visual_tag_menu(schema)}\n"
        "5) confidence — 0.0-1.0, your certainty in the chosen action.\n\n"
        "Output ONLY JSON:\n"
        '{"action": "<one menu code or none>", "outcome": "success|failure|null", '
        '"actor_jersey": "<number or empty>", "actor_team": "left|right|", '
        '"receiver_jersey": "<number or empty>", "confidence": 0.0, '
        '"tags": {"<group>": "<value>"}, "reason": "<short>"}'
    )


def _normalize_action(raw) -> str:
    s = str(raw or "none").strip().lower()
    if s in ("", "none", "null", "nothing"):
        return "none"
    if not s.startswith("football."):
        s = f"football.{s}"
    return s if s in MENU else "none"


def parse_classification(raw: str) -> Classification:
    """Extract the first JSON object from model text. Narrow guard: malformed -> none."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return Classification(action="none", reason="no-json")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Classification(action="none", reason="bad-json")

    actor_jersey = data.get("actor_jersey")
    receiver_jersey = data.get("receiver_jersey")
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    tags = data.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    return Classification(
        action=_normalize_action(data.get("action")),
        outcome=normalize_outcome(data.get("outcome")),
        actor_jersey=(str(actor_jersey).strip() or None) if actor_jersey else None,
        actor_team=data.get("actor_team") if data.get("actor_team") in ("left", "right") else None,
        receiver_jersey=(str(receiver_jersey).strip() or None) if receiver_jersey else None,
        confidence=confidence,
        tags={str(k): str(v) for k, v in tags.items()},
        reason=str(data.get("reason") or "")[:400],
    )


def _valid_tags(tags: Dict[str, str], action: str, schema: EventSchema) -> Dict[str, str]:
    base_code = action.split(".", 1)[1]
    out = {}
    for group_code, value in tags.items():
        tg = schema.get_tag_group(group_code)
        if tg is None:
            continue
        if tg.applies_to and not any(
            base_code == a or base_code.startswith(a) for a in tg.applies_to
        ):
            continue
        if not any(v.code == value for v in tg.values):
            continue
        out[group_code] = value
    return out


def candidate_to_event(
    candidate: Candidate,
    cls: Classification,
    schema: EventSchema,
    counter: int,
) -> Optional[Event]:
    """Build a typed Event from a candidate + VLM answer. None when action is none."""
    if cls.action == "none":
        return None

    ev_def = schema.get_event(cls.action)
    jersey = cls.actor_jersey or candidate.jersey
    team = cls.actor_team or candidate.team

    event = Event(
        event_id=f"evt_{counter:03d}",
        timestamp_s=candidate.timestamp_s,
        frame_id=candidate.frame_id,
        event_code=cls.action,
        display_name_en=ev_def.display_name_en,
        display_name_cn=ev_def.display_name_cn,
        importance=ev_def.importance_base,
        player_jersey=jersey,
        player_team=team,
        track_id=candidate.track_id,
        ball_speed_mps=candidate.ball_speed_mps,
        confidence=cls.confidence,
        description_hint=f"vlm: {cls.reason}" if cls.reason else "vlm-classified",
    )
    event.tags["classified"] = "vlm"

    if cls.action in DUEL_OUTCOME_CODES and cls.outcome is not None:
        event.tags["outcome"] = cls.outcome  # failure does NOT reduce importance

    if cls.action in RECEIVER_CODES:
        receiver_jersey = cls.receiver_jersey or (candidate.next_holder or {}).get("jersey")
        if receiver_jersey:
            event.target_jersey = str(receiver_jersey)
            event.target_team = team

    for group, value in _valid_tags(cls.tags, cls.action, schema).items():
        event.tags[group] = value

    return event


def cleanup_classify_artifacts(output_dir: Path) -> None:
    p = Path(output_dir) / CLASSIFY_CLIPS_DIR
    if p.is_dir():
        shutil.rmtree(p)


def classify_candidates(
    candidates: List[Candidate],
    predictions_json_path: str,
    frames_dir: Path,
    output_dir: Path,
    adapter,
    homography_path: str,
    schema: Optional[EventSchema] = None,
    fps: int = 25,
) -> Tuple[List[Event], List[dict]]:
    """One VLM call per candidate. Returns (events, audit). Writes the audit to
    events_verification.json (same filename as before; fields updated)."""
    output_dir = Path(output_dir)
    clip_dir = output_dir / CLASSIFY_CLIPS_DIR
    clip_dir.mkdir(parents=True, exist_ok=True)

    schema = schema or EventSchema()
    bbox_index = build_bbox_index(predictions_json_path)
    homo = load_homography(homography_path)

    events: List[Event] = []
    audit: List[dict] = []

    for candidate in candidates:
        frames = build_candidate_frames(
            candidate, frames_dir, bbox_index, homo,
            predictions_json_path, clip_dir / candidate.candidate_id, fps=fps,
        )
        if not frames:
            cls = Classification(action="none", reason="no-frames")
        else:
            raw = adapter.generate(build_classify_prompt(candidate, schema), frames)
            cls = parse_classification(raw)

        event = candidate_to_event(candidate, cls, schema, counter=len(events) + 1)
        audit.append({
            "candidate_id": candidate.candidate_id,
            "timestamp_s": candidate.timestamp_s,
            "signals": candidate.signals,
            "rule_actor_jersey": candidate.jersey,
            "rule_actor_team": candidate.team,
            "rule_actor_role": candidate.role,
            "action": cls.action,
            "outcome": cls.outcome,
            "actor_jersey": cls.actor_jersey,
            "actor_team": cls.actor_team,
            "receiver_jersey": cls.receiver_jersey,
            "confidence": cls.confidence,
            "reason": cls.reason,
            "kept": event is not None,
        })
        if event is not None:
            events.append(event)

    (output_dir / "events_verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return events, audit
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_classify.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/classify.py tests/test_classify.py
git commit -m "feat(stage2): VLM classification — fact block, 10-code menu, parse, candidate→event"
```

---

## Task 7: Orchestration test — classify_candidates with MockVLM

**Files:**
- Test: `tests/test_classify.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classify.py`:

```python
import json as _json

from pipeline.stage2_events.classify import classify_candidates
from pipeline.stage2_events.detector import (
    detect_candidates, load_frames, merge_candidates,
)
from pipeline.stage2_events.possession import (
    possession_segments, resolve_role_by_track, resolve_team_by_track,
)


def test_classify_candidates_end_to_end_with_mock(
        tmp_path, gk_boot_predictions, homography_file, frames_dir, mock_adapter):
    frames = load_frames(str(gk_boot_predictions))
    team = resolve_team_by_track(frames)
    role = resolve_role_by_track(frames)
    segs = possession_segments(frames, team)
    cands = merge_candidates(detect_candidates(frames, segs, team, role, fps=25))
    assert cands

    adapter = mock_adapter(script={
        # any prompt mentioning the goalkeeper actor -> goal_kick
        "role: goalkeeper": ('{"action": "goal_kick", "outcome": "success", '
                             '"confidence": 0.9, "reason": "long boot"}'),
    })
    events, audit = classify_candidates(
        cands, str(gk_boot_predictions), frames_dir, tmp_path,
        adapter, str(homography_file), fps=25,
    )
    assert any(e.event_code == "football.goal_kick" for e in events)
    assert not any(e.event_code == "football.shoot" for e in events)
    assert len(audit) == len(cands)
    assert len(adapter.calls) == len(cands)          # one call per candidate

    saved = _json.loads((tmp_path / "events_verification.json").read_text())
    assert saved[0]["candidate_id"].startswith("cand_")
    assert "action" in saved[0] and "corrected_event_code" not in saved[0]
```

- [ ] **Step 2: Run test to verify it passes** (implementation landed in Task 6)

Run: `python3 -m pytest tests/test_classify.py -v`
Expected: all PASS. If FAIL, fix classify.py — do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_classify.py
git commit -m "test(stage2): classify orchestration — one call per candidate, GK boot becomes goal_kick"
```

---

## Task 8: Density guarantee — buildup filler for empty 5s bins

**Files:**
- Modify: `pipeline/stage2_events/detector.py`
- Test: `tests/test_buildup_fill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_buildup_fill.py`:

```python
from pipeline.stage2_events.detector import fill_buildup_bins
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.types import Event, PossessionSegment


def _event(t, code="football.pass"):
    return Event(
        event_id=f"evt_{int(t*10):03d}", timestamp_s=t, frame_id=int(t * 25),
        event_code=code, display_name_en="Pass", display_name_cn="传球",
        importance=0.15, player_jersey="7", player_team="left",
    )


def _seg(track, team, jersey, start_fid, end_fid):
    return PossessionSegment(track_id=track, team=team, jersey=jersey,
                             start_fid=start_fid, end_fid=end_fid,
                             start_xy=(0.0, 0.0), end_xy=(5.0, 0.0))


def test_empty_bins_get_buildup_from_dominant_segment():
    events = [_event(2.0), _event(27.0)]               # bins 1,2,3,4 (5-25s) empty
    segments = [
        _seg(7, "left", "7", 50, 200),                 # 2-8s
        _seg(4, "right", "4", 210, 700),               # 8.4-28s — dominates middle bins
    ]
    out = fill_buildup_bins(events, segments, duration_s=30.0, fps=25,
                            schema=EventSchema())
    fillers = [e for e in out if e.event_code == "football.buildup"]
    assert len(fillers) == 4
    for f in fillers:
        assert f.player_team in ("left", "right")
        assert f.tags.get("filler") == "density"
    # every 5s bin now has at least one event
    for b in range(6):
        assert any(b * 5.0 <= e.timestamp_s < (b + 1) * 5.0 for e in out), f"bin {b} empty"


def test_no_filler_when_bins_already_covered():
    events = [_event(2.0), _event(7.0), _event(12.0), _event(17.0),
              _event(22.0), _event(27.0)]
    out = fill_buildup_bins(events, [], duration_s=30.0, fps=25, schema=EventSchema())
    assert not [e for e in out if e.event_code == "football.buildup"]


def test_empty_bins_fall_back_to_last_known_possession():
    events = [_event(2.0)]
    segments = [_seg(7, "left", "7", 25, 100)]         # data only in 1-4s
    out = fill_buildup_bins(events, segments, duration_s=30.0, fps=25,
                            schema=EventSchema())
    fillers = [e for e in out if e.event_code == "football.buildup"]
    # bins 1-5 have no overlapping segment; fallback = last segment before the bin
    assert fillers, "fallback to last known possession still fills bins"
    assert all(f.player_jersey == "7" for f in fillers)


def test_bin_with_no_tracking_data_at_all_is_skipped_honestly():
    events = [_event(2.0)]
    out = fill_buildup_bins(events, [], duration_s=30.0, fps=25,
                            schema=EventSchema())
    # no possession segments anywhere -> nothing truthful to fabricate
    assert out == sorted(events, key=lambda e: e.timestamp_s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_buildup_fill.py -v`
Expected: FAIL — `fill_buildup_bins` not defined.

- [ ] **Step 3: Implement fill_buildup_bins in detector.py**

Append:

```python
def fill_buildup_bins(
    events: List[Event],
    segments: List[PossessionSegment],
    duration_s: float,
    fps: int,
    schema: EventSchema,
    bin_s: float = BIN_S,
) -> List[Event]:
    """Density guarantee: every bin_s bin must hold >=1 event. Empty bins get a
    rule-emitted football.buildup for the bin's dominant possession segment
    (fallback: last segment ending before the bin). A bin with zero tracking
    data anywhere before it is skipped — we never fabricate."""
    ev_def = schema.get_event("football.buildup")
    out = list(events)
    counter = len(events)
    n_bins = max(1, math.ceil(duration_s / bin_s))

    for b in range(n_bins):
        lo, hi = b * bin_s, (b + 1) * bin_s
        if any(lo <= e.timestamp_s < hi for e in out):
            continue

        lo_fid, hi_fid = int(lo * fps), int(hi * fps)
        best = None
        best_overlap = 0
        for seg in segments:
            overlap = min(seg.end_fid, hi_fid) - max(seg.start_fid, lo_fid)
            if overlap > best_overlap:
                best, best_overlap = seg, overlap
        if best is None:
            prior = [s for s in segments if s.end_fid < lo_fid]
            best = max(prior, key=lambda s: s.end_fid) if prior else None
        if best is None:
            continue  # nothing truthful to say about this bin

        if best_overlap > 0:
            frame_id = (max(best.start_fid, lo_fid) + min(best.end_fid, hi_fid)) // 2
        else:
            frame_id = int((lo + hi) / 2 * fps)
        counter += 1
        filler = Event(
            event_id=f"evt_{counter:03d}",
            timestamp_s=frame_id / fps,
            frame_id=frame_id,
            event_code="football.buildup",
            display_name_en=ev_def.display_name_en,
            display_name_cn=ev_def.display_name_cn,
            importance=ev_def.importance_base,
            player_jersey=best.jersey,
            player_team=best.team,
            track_id=best.track_id,
            confidence=0.9,
            description_hint="buildup: density filler from possession geometry",
        )
        filler.tags["filler"] = "density"
        out.append(filler)

    return sorted(out, key=lambda e: e.timestamp_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_buildup_fill.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage2_events/detector.py tests/test_buildup_fill.py
git commit -m "feat(stage2): buildup density filler — every 5s bin ends with an event"
```

---

## Task 9: Wiring — run.py + config.py

**Files:**
- Modify: `pipeline/run.py` (run_stage2, adapter builder, CLI)
- Modify: `pipeline/config.py:50-58`
- Test: `tests/test_stage2_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_stage2_integration.py`:

```python
import json

from pipeline.config import PipelineConfig
from pipeline.run import run_stage2


def test_run_stage2_full_flow_with_mock(
        tmp_path, gk_boot_predictions, homography_file, frames_dir, mock_adapter):
    config = PipelineConfig(
        clip_dir=frames_dir.parent,
        output_dir=tmp_path,
        existing_predictions_json=gk_boot_predictions,
        existing_homography_json=homography_file,
        fps=25,
    )
    adapter = mock_adapter(script={
        "role: goalkeeper": ('{"action": "goal_kick", "outcome": "success", '
                             '"confidence": 0.9, "reason": "long boot"}'),
    })
    n = run_stage2(config, classify_adapter=adapter)
    assert n > 0

    detected = json.loads((tmp_path / "events_detected.json").read_text())
    assert detected["schema_version"] == "v4-candidates-20260707"
    assert detected["candidates"] and "signals" in detected["candidates"][0]
    assert "event_code" not in detected["candidates"][0]      # untyped

    final = json.loads((tmp_path / "events.json").read_text())
    assert final["schema_version"] == "v3-20260319"           # downstream contract
    codes = [e["event_code"] for e in final["events"]]
    assert "football.goal_kick" in codes
    assert "football.shoot" not in codes
    # density: fixture is 30 frames = 1.2s -> single bin, covered
    assert final["events"]

    audit = json.loads((tmp_path / "events_verification.json").read_text())
    assert all("action" in a for a in audit)


def test_run_stage2_fails_hard_without_backend(
        tmp_path, gk_boot_predictions, homography_file, frames_dir, monkeypatch):
    import pytest
    for var in ("ARK_API_KEY", "DOUBAO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # keep load_ark_env from re-reading the real .env on dev machines
    import pipeline.stage4_commentary.generate as gen
    monkeypatch.setattr(gen, "load_ark_env", lambda: None)
    config = PipelineConfig(
        clip_dir=frames_dir.parent,
        output_dir=tmp_path,
        existing_predictions_json=gk_boot_predictions,
        existing_homography_json=homography_file,
        fps=25,
    )
    with pytest.raises(RuntimeError):
        run_stage2(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage2_integration.py -v`
Expected: FAIL — `run_stage2` has the old signature/flow.

- [ ] **Step 3: Rewrite run_stage2 and the adapter builder in run.py**

Replace `run_stage2` and `_build_verify_adapter` with:

```python
def run_stage2(config: PipelineConfig, classify_adapter=None) -> int:
    from pipeline.stage2_events.classify import (
        classify_candidates,
        cleanup_classify_artifacts,
    )
    from pipeline.stage2_events.detector import (
        compose_assists,
        dedup_events,
        detect_candidates,
        fill_buildup_bins,
        load_frames,
        merge_candidates,
        select_candidates,
        write_candidates_json,
        write_events_json,
    )
    from pipeline.stage2_events.enricher import enrich_events
    from pipeline.stage2_events.possession import (
        possession_segments,
        resolve_role_by_track,
        resolve_team_by_track,
    )
    from pipeline.stage2_events.schema import EventSchema

    schema = EventSchema()
    frames = load_frames(str(config.predictions_json))
    team_by_track = resolve_team_by_track(frames)
    role_by_track = resolve_role_by_track(frames)
    segments = possession_segments(frames, team_by_track)
    duration_s = (max(f.frame_id for f in frames) / config.fps) if frames else 0.0

    raw = detect_candidates(frames, segments, team_by_track, role_by_track, fps=config.fps)
    candidates = select_candidates(merge_candidates(raw), duration_s)
    log.info("Stage 2 candidates: %d raw -> %d selected", len(raw), len(candidates))

    video_info = {
        "source": str(config.frames_dir), "fps": config.fps,
        "duration_s": round(duration_s, 2),
        "total_frames": max((f.frame_id for f in frames), default=0),
    }
    write_candidates_json(candidates, config.output_dir / "events_detected.json", video_info)

    adapter = classify_adapter or _build_classify_adapter(config)
    log.info("Classifying %d candidates with %s ...", len(candidates), config.verify_backend)
    events, audit = classify_candidates(
        candidates, str(config.predictions_json), config.frames_dir,
        config.output_dir, adapter, str(config.homography_json),
        schema=schema, fps=config.fps,
    )
    log.info("Classification: %d candidates, %d events kept", len(audit), len(events))
    cleanup_classify_artifacts(config.output_dir)

    events = fill_buildup_bins(events, segments, duration_s, config.fps, schema)
    events = enrich_events(dedup_events(compose_assists(events)), frames)
    write_events_json(events, config.output_dir / "events.json", video_info)
    return len(events)


def _build_classify_adapter(config: PipelineConfig):
    import os

    if config.verify_backend == "qwen_local":
        from pipeline.stage4_commentary.adapters.qwen_local import QwenLocalAdapter

        return QwenLocalAdapter(model_path=config.verify_model_path)
    if config.verify_backend == "doubao":
        from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter
        from pipeline.stage4_commentary.generate import load_ark_env

        load_ark_env()
        if not (os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")):
            raise RuntimeError(
                "Stage 2 classification requires a VLM backend. Set ARK_API_KEY "
                "(and optionally ARK_CLASSIFY_MODEL for a vision-strong endpoint) "
                "or use --verify-backend qwen_local."
            )
        return DoubaoAPIAdapter(model=os.environ.get("ARK_CLASSIFY_MODEL"))
    raise ValueError(f"Unknown classify backend: {config.verify_backend}")
```

(`DoubaoAPIAdapter(model=None)` falls through to its existing env/default chain, so `ARK_CLASSIFY_MODEL` cleanly overrides just the classification calls. Document in `.env`: `ARK_CLASSIFY_MODEL=<your strongest vision Doubao endpoint>`.)

In `build_arg_parser()` delete the `--verify-events` argument block; in `config_from_args()` delete `verify_events=args.verify_events,`. Keep `--verify-backend` as is.

- [ ] **Step 4: Trim config.py**

In `pipeline/config.py` Stage 2 block, delete these three fields (keep `event_importance_threshold`, `min_event_gap_s`, `verify_backend`, `verify_model_path`):

```python
    ball_speed_shot_threshold_mps: float = 10.0   # DELETE
    verify_events: bool = False                    # DELETE
    verify_window_s: float = 0.5                   # DELETE
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_stage2_integration.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/run.py pipeline/config.py tests/test_stage2_integration.py
git commit -m "feat(stage2): wire candidate→classify→buildup pipeline; VLM is a hard dependency"
```

---

## Task 10: Stage 4 — delete the gap-filler (buildup makes it redundant)

**Files:**
- Modify: `pipeline/stage4_commentary/prompt_builder.py`
- Modify: `pipeline/stage4_commentary/generate.py:57-84`
- Modify: `pipeline/run.py` (run_stage4)
- Delete: `tests/test_commentary_gap_fill.py`

- [ ] **Step 1: Delete gap-filler code**

In `prompt_builder.py`:
- Delete `_find_gaps` (lines ~10-18) and `_load_gap_filler_events` (lines ~21-50).
- In `build_commentary_prompt`: remove the `verification_audit_path` and `gap_threshold_s` parameters, and delete the block from `confirmed_timestamps = ...` through the `[Unconfirmed Activity]` loop (keep the `duration_s = ...` line only if it is referenced later in the function — check before deleting).

In `generate.py`: remove the `verification_audit_path` parameter from `generate_commentary` and from the `build_commentary_prompt(...)` call.

In `run.py` `run_stage4`: delete the `verification_audit_path` variable and argument (events_verification.json is still written by stage2 as an audit log — nothing reads it anymore, which is fine).

- [ ] **Step 2: Delete the test**

```bash
git rm tests/test_commentary_gap_fill.py
```

- [ ] **Step 3: Verify nothing references the removed names**

Run: `grep -rn "gap_filler\|_find_gaps\|verification_audit" pipeline/ tests/ --include="*.py"`
Expected: no output.

Run: `python3 -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add -A pipeline/stage4_commentary pipeline/run.py
git commit -m "refactor(stage4): drop unconfirmed-activity gap filler — buildup events guarantee density"
```

---

## Task 11: Cleanup — delete verify.py, Verdict, stale test

**Files:**
- Delete: `pipeline/stage2_events/verify.py`
- Delete: `tests/test_verify_apply_verdict.py`
- Modify: `pipeline/stage2_events/types.py` (remove Verdict)

- [ ] **Step 1: Delete files and the Verdict dataclass**

```bash
git rm pipeline/stage2_events/verify.py tests/test_verify_apply_verdict.py
```

Remove the `Verdict` dataclass (the last class) from `pipeline/stage2_events/types.py`.

Now that verify.py is gone, also delete its two dead dependencies from `pipeline/stage2_events/evidence.py`: the `build_evidence_frames` function and `_receiver_bbox_for_frame` (deferred from Task 5).

- [ ] **Step 2: Verify no dangling references**

Run: `grep -rn "verify_events\|from pipeline.stage2_events.verify\|Verdict\|build_evidence_frames\|_receiver_bbox_for_frame" pipeline/ tests/ --include="*.py"`
Expected: no output.

- [ ] **Step 3: Full suite**

Run: `python3 -m pytest tests/ -q`
Expected: all PASS, none skipped unexpectedly.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(stage2): remove verify.py and Verdict — classification replaced verification"
```

---

## Task 12: Golden run — real Doubao on SNGS-116 / SNGS-148 (manual acceptance)

**Files:** none (manual verification; archive old outputs first)

- [ ] **Step 1: Archive old outputs for diff**

```bash
cp outputs/SNGS-116/events.json outputs/SNGS-116/events.v3-rules.json
cp outputs/SNGS-148/events.json outputs/SNGS-148/events.v3-rules.json
```

- [ ] **Step 2: Set the classification model and run**

Ensure `.env` has `ARK_API_KEY` and (recommended) `ARK_CLASSIFY_MODEL=<strongest vision Doubao endpoint>`.

```bash
python3 -m pipeline.run --clip-dir <path-to-SNGS-116-clip> \
  --output-dir outputs/SNGS-116 \
  --existing-predictions-json outputs/SNGS-116/predictions.json \
  --existing-homography-json outputs/SNGS-116/homography_per_frame.json \
  --force
```

Repeat for SNGS-148. If you still have the clip where the original GK-boot-labeled-as-shot misjudgment appeared, run it too.

- [ ] **Step 3: Acceptance checklist (human review against the video)**

- [ ] Zero role-impossible labels (no `shoot` by a goalkeeper from his own box; GK long balls come out as `goal_kick`/`clearance`).
- [ ] Every 5s bin of the final `events.json` has ≥1 event; `buildup` fillers appear only in genuinely quiet stretches.
- [ ] ≤20 classified candidates per clip (check `events_detected.json` length + `events_verification.json`).
- [ ] Spot-check 5 outcomes (pass/interception/pressing): agree with what the video shows.
- [ ] Visual tags, when present, are plausible (no `bicycle_kick` on a tap-in); absent tags are fine.
- [ ] Diff against `events.v3-rules.json`: the phantom near-duplicate shots (e.g. SNGS-116 t=6.3/7.3) are gone or correctly retyped.

- [ ] **Step 4: Commit the archived baselines**

```bash
git add outputs/SNGS-116/events.v3-rules.json outputs/SNGS-148/events.v3-rules.json
git commit -m "chore: archive rule-typed events.json baselines before VLM classification"
```

---

## Self-Review Notes

- **Spec coverage:** all 16 locked decisions map to tasks (menu/T0+T6, triggers/T3, cap+density/T4+T8, fact block/T6, single call/T6-7, fail-hard/T9, tag split/T0+T6+enricher-as-is, outcome/T6, window/T5, foul-exclusion/T6 menu test, model/T9, file plan/T9-11, acceptance/T12).
- **Deliberate scope cuts:** no per-candidate cache (rerun cost ≤20 calls), no low-confidence retry branch, no set-piece triggers — all phase 2.
- **Mid-plan breakage window:** the pytest suite stays green at every task boundary (verify.py and its imports stay intact until Task 11; no old test imports the deleted `EventDetector`). What IS broken between Tasks 3 and 9 is `run_stage2` itself — its function-body imports reference the new detector API before Task 9 rewires it — so don't run the actual pipeline in that window; tests only.
- **Type consistency check:** `Candidate`/`Classification` fields used in detector/classify/evidence/run match Task 2 definitions; `build_candidate_frames` signature in Task 5 matches its call in Task 6's `classify_candidates`; `run_stage2(config, classify_adapter=None)` matches Task 9's tests.
