# Auto Evidence Adapter (GSR Trajectory → ClipEvidence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive `ClipEvidence` automatically from GSR trajectories (`outputs/<clip>/predictions.json`) and measure per-field agreement against the hand-encoded `recognition_evidence.jsonl`, including how much checker replay precision drops when auto evidence replaces hand evidence.

**Architecture:** Pure deterministic trajectory inference — no model calls, no use of `comments/events.json` labels (those are event-model output, not facts). Pipeline: ball track → kinematics (kicks, dead-ball periods) → possession timeline (reusing `pipeline/stage2b/digest.py`) → restart classification by ball position → delivery/landing inference → `ClipEvidence`. Every inference that falls below its threshold yields `None`/no-event, so the downstream checker returns `insufficient` — never guess to raise coverage.

**Tech Stack:** Python 3, numpy (already a dependency of `pipeline/topology`), pytest. No new dependencies.

**Prerequisites (verify before Task 1):**
- The triage plan (`2026-07-17-recognition-error-triage-and-checkers.md`) is fully executed: `pipeline/tactics_qa/` exists with `evidence.py`, `checkers.py`, `replay.py`; `benchmark/tactical_prototypes/recognition_review.jsonl` and `recognition_evidence.jsonl` exist.
- `outputs/<clip>/predictions.json` exists for at least SNGS-116. Run `ls outputs/ | head` to see coverage; clips without outputs are handled as `no_data`, not errors.

**Coordinate conventions (from real SNGS-116 data — do not re-derive):**
- `bbox_pitch` is in meters, origin at pitch center; nominal bounds x ∈ [−52.5, 52.5], y ∈ [−34, 34] (105×68 pitch).
- **Homography noise pushes points out of bounds** (real ball sample: x=53.8, y=−36.6). All location classifiers must tolerate up to 3 m beyond nominal bounds. Never filter out-of-bounds points.
- `predictions.json` top level: `{info: {name, n_frames, fps}, images: [...], annotations: [...], categories: [...]}`. Ball annotations have `category_id: 4`; players 1, goalkeepers 2, referees 3. Player team is `attributes.team` ∈ {"left","right"}. Frame linkage is `image_id`; frame index is the trailing number of `file_name` (e.g. `000155.jpg` → 155) — `image_id` values like `"3116000001"` embed the clip id, use `file_name`.
- fps is 25 (read it from `info.fps`, don't hardcode).
- `team` in evidence is relative to the claim's attacking team; the review rows don't record left/right, so the adapter maps sides via possession at the claim window (Task 6 Step 2 defines the rule).

**Branch:** `feat/auto-evidence-adapter` off the branch where the triage plan landed.

## Implementation status (completed 2026-07-17)

The deterministic adapter and report pipeline are implemented without new dependencies or model-derived event inputs. Final scoped verification: 48/48 `tactics_qa` tests and 217/217 top-level repository tests passed; the SNGS-116 real-data anchor detected the 6.2s corner, `corner_cross`, and `near_post` landing directly from `predictions.json`.

- Local coverage is only 3/72 reviewed claims; 69 are explicitly `no_data`.
- Checker verdicts match hand evidence for 1/3 covered claims (33.3%), with zero automatic false vetoes.
- SNGS-117 and SNGS-148 abstain because the open-play key-pass kick is not detected; their delivery checker result is `insufficient`.
- SNGS-116 exposes a hand/auto timing mismatch: hand evidence places the corner at the 3.0s window origin, while trajectory kinematics detects the actual kick at 6.2s. The hand evidence was not changed.
- Running the generator twice produced byte-identical auto-evidence and report hashes.

This completes the adapter experiment but does **not** pass the ≥80% gate for live-pipeline integration. The next dependency is broader Stage 1 output coverage, followed by improving open-play key-pass detection from trajectories; agent topology and checker thresholds should remain unchanged.

The unchecked boxes below are retained as the original TDD execution recipe, not as current completion state. Canonical generated assets use the repository ID `corner-near-far-post` instead of the obsolete underscore spelling present in some examples below.

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `pipeline/tactics_qa/evidence.py` | Modify | Add `contested_touch` to `CHAOS_KINDS` |
| `pipeline/tactics_qa/gsr_io.py` | Create | Load predictions.json → ball track + per-team player tracks |
| `pipeline/tactics_qa/kinematics.py` | Create | Ball speed, kick events, dead-ball periods |
| `pipeline/tactics_qa/possession_timeline.py` | Create | Team possession runs → regain / contested-touch events |
| `pipeline/tactics_qa/restart_infer.py` | Create | Dead-ball + kick position → restart kind |
| `pipeline/tactics_qa/delivery_infer.py` | Create | Key-pass delivery_kind + corner landing zone |
| `pipeline/tactics_qa/auto_evidence.py` | Create | Compose everything into `ClipEvidence` |
| `pipeline/tactics_qa/agreement.py` | Create | Field-level agreement + auto-vs-hand replay comparison |
| `scripts/build_auto_evidence.py` | Create | CLI: generate auto evidence + agreement report |
| `tests/tactics_qa/test_kinematics.py` etc. | Create | One test file per module (paths in each task) |
| `benchmark/tactical_prototypes/recognition_evidence_auto.jsonl` | Generate | Auto evidence for covered claims |
| `benchmark/tactical_prototypes/evidence_agreement_report.json` + `.md` | Generate | The acceptance deliverable |

All thresholds live as module constants named `*_M`, `*_MPS`, `*_S` so the agreement report can dump them; do not scatter magic numbers.

---

### Task 1: Extend the evidence schema for trajectory-derived chaos events

**Files:**
- Modify: `pipeline/tactics_qa/evidence.py`
- Modify: `tests/tactics_qa/test_evidence.py`

- [ ] **Step 1: Write the failing test** (append to `tests/tactics_qa/test_evidence.py`)

```python
def test_contested_touch_is_a_chaos_kind():
    from pipeline.tactics_qa.evidence import CHAOS_KINDS, EVENT_KINDS
    assert "contested_touch" in CHAOS_KINDS
    assert "contested_touch" in EVENT_KINDS
    # existing checker semantics unchanged
    E = __import__("pipeline.tactics_qa.evidence", fromlist=["PossessionEvent"])
    E.PossessionEvent(t=1.0, team="attacking", kind="contested_touch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tactics_qa/test_evidence.py -v`
Expected: new test FAILS (`contested_touch` unknown), old tests PASS.

- [ ] **Step 3: Implement**

In `pipeline/tactics_qa/evidence.py` change:

```python
CHAOS_KINDS = {"clearance", "header_flick", "aerial_duel", "contested_touch"}
```

Rationale comment to add above it: GSR has no ball height, so aerial chaos cannot be distinguished from ground scrambles; `contested_touch` is the trajectory-derived generic for rapid alternating possession.

- [ ] **Step 4: Run the whole existing suite** (checkers count chaos via `CHAOS_KINDS`, so nothing else changes)

Run: `python3 -m pytest tests/tactics_qa/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/tactics_qa/evidence.py tests/tactics_qa/test_evidence.py
git commit -m "feat(tactics_qa): contested_touch chaos kind for trajectory evidence"
```

---

### Task 2: GSR loading

**Files:**
- Create: `pipeline/tactics_qa/gsr_io.py`
- Test: `tests/tactics_qa/test_gsr_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_gsr_io.py
from pipeline.tactics_qa.gsr_io import load_gsr, GsrClip


def make_predictions(ball_xy_by_frame, players=()):
    """Minimal synthetic predictions.json dict. ball_xy_by_frame: {fid: (x, y)}."""
    images, annotations = [], []
    fids = sorted(set(ball_xy_by_frame) | {f for f, *_ in players})
    for fid in fids:
        images.append({"image_id": f"900{fid:07d}", "file_name": f"{fid:06d}.jpg"})
    for fid, (x, y) in ball_xy_by_frame.items():
        annotations.append({
            "image_id": f"900{fid:07d}", "category_id": 4, "track_id": 99,
            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
            "attributes": {"role": "ball", "team": None},
        })
    for fid, track_id, team, x, y in players:
        annotations.append({
            "image_id": f"900{fid:07d}", "category_id": 1, "track_id": track_id,
            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
            "attributes": {"role": "player", "team": team, "jersey": ""},
        })
    return {"info": {"name": "test", "n_frames": max(fids), "fps": 25},
            "images": images, "annotations": annotations,
            "categories": [{"id": 4, "name": "ball"}, {"id": 1, "name": "player"}]}


def test_load_gsr_ball_and_players(tmp_path):
    import json
    pred = make_predictions({1: (0.0, 0.0), 2: (1.0, 0.5)},
                            players=[(1, 7, "left", 10.0, 5.0)])
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(pred))
    clip = load_gsr(p)
    assert isinstance(clip, GsrClip)
    assert clip.fps == 25
    assert clip.ball[1] == (0.0, 0.0) and clip.ball[2] == (1.0, 0.5)
    assert clip.players[1][0] == (7, "left", 10.0, 5.0)


def test_missing_ball_frames_absent_not_interpolated(tmp_path):
    import json
    pred = make_predictions({1: (0.0, 0.0), 5: (2.0, 0.0)})
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(pred))
    clip = load_gsr(p)
    assert 3 not in clip.ball  # gaps stay gaps; downstream must handle them
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/tactics_qa/test_gsr_io.py -v` — Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/gsr_io.py
"""Load GSR predictions.json into frame-indexed tracks.

Frame ids come from file_name (e.g. '000155.jpg' -> 155); image_id embeds the
clip id and is only used for joining annotations to images. Missing frames are
left missing — no interpolation here.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

BALL_CATEGORY = 4
PERSON_CATEGORIES = {1, 2}  # player, goalkeeper (referees excluded)


@dataclass
class GsrClip:
    fps: float
    n_frames: int
    ball: dict = field(default_factory=dict)     # {fid: (x, y)}
    players: dict = field(default_factory=dict)  # {fid: [(track_id, team, x, y), ...]}


def _fid(file_name: str) -> int:
    return int(Path(file_name).stem)


def load_gsr(path) -> GsrClip:
    with open(path, encoding="utf-8") as fp:
        pred = json.load(fp)
    fid_by_image = {im["image_id"]: _fid(im["file_name"]) for im in pred["images"]}
    clip = GsrClip(fps=float(pred["info"]["fps"]), n_frames=int(pred["info"]["n_frames"]))
    for ann in pred["annotations"]:
        pitch = ann.get("bbox_pitch") or {}
        x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
        if x is None or y is None:
            continue
        fid = fid_by_image.get(ann["image_id"])
        if fid is None:
            continue
        if ann.get("category_id") == BALL_CATEGORY:
            clip.ball[fid] = (float(x), float(y))
        elif ann.get("category_id") in PERSON_CATEGORIES:
            team = (ann.get("attributes") or {}).get("team")
            clip.players.setdefault(fid, []).append(
                (int(ann["track_id"]), team, float(x), float(y)))
    return clip
```

- [ ] **Step 4: Run to verify pass**, then **Step 5: Commit**

```bash
python3 -m pytest tests/tactics_qa/test_gsr_io.py -v
git add pipeline/tactics_qa/gsr_io.py tests/tactics_qa/test_gsr_io.py
git commit -m "feat(tactics_qa): GSR predictions loader"
```

---

### Task 3: Ball kinematics — kicks and dead-ball periods

**Files:**
- Create: `pipeline/tactics_qa/kinematics.py`
- Test: `tests/tactics_qa/test_kinematics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_kinematics.py
from pipeline.tactics_qa.kinematics import ball_speeds, detect_kicks, dead_ball_periods


def _stationary_then_kick(fps=25):
    """Ball still at (50, -33) for 2s, then moves 1 m/frame (25 m/s) for 1s."""
    ball = {}
    for f in range(0, 50):
        ball[f] = (50.0, -33.0)
    for i, f in enumerate(range(50, 75)):
        ball[f] = (50.0 - i * 1.0, -33.0 + i * 0.4)
    return ball


def test_ball_speeds_smoothed():
    speeds = ball_speeds(_stationary_then_kick(), fps=25)
    assert speeds[10] < 0.5
    assert speeds[60] > 15.0


def test_detect_kick_at_transition():
    kicks = detect_kicks(_stationary_then_kick(), fps=25)
    assert len(kicks) == 1
    k = kicks[0]
    assert 1.8 <= k.t <= 2.4                      # around frame 50
    # smoothing delays detection by ~2 frames, so the ball has moved ~2m and
    # the smoothed speed is still ramping up — tolerances account for that
    assert abs(k.xy[0] - 50.0) < 3.5
    assert k.speed_mps > 8.0


def test_dead_ball_period_found():
    periods = dead_ball_periods(_stationary_then_kick(), fps=25)
    assert len(periods) == 1
    start, end = periods[0]
    assert start <= 0.2 and 1.8 <= end <= 2.2


def test_gap_in_track_does_not_fabricate_kick():
    ball = {f: (0.0, 0.0) for f in range(0, 25)}
    ball.update({f: (30.0, 0.0) for f in range(50, 75)})  # 1s gap then far away
    kicks = detect_kicks(ball, fps=25)
    assert kicks == []  # displacement across a gap is not a kick
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/tactics_qa/test_kinematics.py -v` → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/kinematics.py
"""Ball kinematics from a frame-indexed 2D track. Deterministic; gaps in the
track produce no measurements rather than spikes."""
import math
from dataclasses import dataclass

SMOOTH_WINDOW_FRAMES = 5
MAX_STEP_GAP_FRAMES = 3        # speed only measured across <=3 missing frames
KICK_MIN_SPEED_MPS = 8.0
KICK_PRE_MAX_SPEED_MPS = 3.0   # ball must be slow shortly before the kick
KICK_LOOKBACK_S = 0.4
DEAD_MAX_SPEED_MPS = 0.8
DEAD_MIN_DURATION_S = 1.0


@dataclass(frozen=True)
class Kick:
    t: float
    xy: tuple
    speed_mps: float


def ball_speeds(ball: dict, fps: float) -> dict:
    """{fid: smoothed speed m/s}. Only frames with a near-neighbour measurement."""
    fids = sorted(ball)
    raw = {}
    for a, b in zip(fids, fids[1:]):
        if b - a > MAX_STEP_GAP_FRAMES:
            continue
        (x1, y1), (x2, y2) = ball[a], ball[b]
        raw[b] = math.hypot(x2 - x1, y2 - y1) * fps / (b - a)
    out = {}
    for f in raw:
        neigh = [raw[g] for g in range(f - SMOOTH_WINDOW_FRAMES, f + 1) if g in raw]
        out[f] = sum(neigh) / len(neigh)
    return out


def detect_kicks(ball: dict, fps: float) -> list:
    speeds = ball_speeds(ball, fps)
    kicks, last_kick_f = [], -10 ** 9
    lookback = int(KICK_LOOKBACK_S * fps)
    for f in sorted(speeds):
        if speeds[f] < KICK_MIN_SPEED_MPS or f - last_kick_f < lookback:
            continue
        prior = [speeds[g] for g in range(f - lookback, f) if g in speeds]
        if not prior or min(prior) > KICK_PRE_MAX_SPEED_MPS:
            continue
        xy = ball.get(f) or ball.get(f - 1)
        if xy is None:
            continue
        kicks.append(Kick(t=f / fps, xy=xy, speed_mps=speeds[f]))
        last_kick_f = f
    return kicks


def dead_ball_periods(ball: dict, fps: float) -> list:
    """[(start_s, end_s)] where the ball is continuously near-stationary."""
    speeds = ball_speeds(ball, fps)
    periods, start = [], None
    fids = sorted(speeds)
    for f in fids:
        slow = speeds[f] < DEAD_MAX_SPEED_MPS
        if slow and start is None:
            start = f
        elif not slow and start is not None:
            if (f - start) / fps >= DEAD_MIN_DURATION_S:
                periods.append((start / fps, f / fps))
            start = None
    if start is not None and (fids[-1] - start) / fps >= DEAD_MIN_DURATION_S:
        periods.append((start / fps, fids[-1] / fps))
    # a track that begins stationary counts from t=0
    if periods and periods[0][0] <= (fids[0] + SMOOTH_WINDOW_FRAMES + 1) / fps:
        periods[0] = (0.0, periods[0][1])
    return periods
```

- [ ] **Step 4: Run to verify pass**; adjust only test tolerances if the smoothing shifts edges by a frame or two — never widen `KICK_PRE_MAX_SPEED_MPS`/`KICK_MIN_SPEED_MPS` just to pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/tactics_qa/kinematics.py tests/tactics_qa/test_kinematics.py
git commit -m "feat(tactics_qa): ball kinematics (kicks, dead-ball periods)"
```

---

### Task 4: Possession timeline → regain and contested-touch events

**Files:**
- Create: `pipeline/tactics_qa/possession_timeline.py`
- Test: `tests/tactics_qa/test_possession_timeline.py`

Reuse `pipeline/stage2b/digest.py` (`FrameData`, `possession_segments`, `resolve_team_by_track`) rather than re-deriving holder logic. `possession_segments` yields per-player runs; this module merges them into per-team runs and classifies transitions.

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_possession_timeline.py
from pipeline.tactics_qa.possession_timeline import team_runs, transition_events

FPS = 25


def _run(team, start_s, end_s):
    return {"team": team, "start_s": start_s, "end_s": end_s}


def test_team_runs_merge_same_team_players():
    segs = [  # (track_id, team, start_fid, end_fid)
        (1, "left", 0, 49), (2, "left", 55, 99), (3, "right", 110, 200),
    ]
    runs = team_runs(segs, fps=FPS)
    assert [r["team"] for r in runs] == ["left", "right"]
    assert runs[0]["start_s"] == 0.0 and runs[0]["end_s"] == 99 / FPS


def test_controlled_regain_emitted():
    runs = [_run("right", 0.0, 5.0), _run("left", 5.5, 9.0)]
    evs = transition_events(runs, attacking_team="left")
    assert [(e.kind, e.team) for e in evs] == [("controlled_recovery", "attacking")]
    assert 5.0 <= evs[0].t <= 5.6


def test_short_alternation_becomes_contested_touches():
    # note: consecutive same-team runs never occur here — team_runs merges them
    # before transition_events sees the list, and this fixture mirrors that
    runs = [_run("right", 0.0, 3.5), _run("left", 3.7, 4.3), _run("right", 4.5, 5.1),
            _run("left", 5.3, 5.8), _run("right", 6.0, 6.5), _run("left", 6.7, 12.0)]
    evs = transition_events(runs, attacking_team="left")
    kinds = [e.kind for e in evs]
    assert kinds.count("contested_touch") >= 3
    assert kinds[-1] == "controlled_recovery"  # the final stable run still counts


def test_no_transition_no_events():
    runs = [_run("left", 0.0, 12.0)]
    assert transition_events(runs, attacking_team="left") == []
```

- [ ] **Step 2: Run to verify failure** → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/possession_timeline.py
"""Team possession runs and transition classification.

A run is 'stable' if it lasts >= STABLE_RUN_MIN_S; a change of possession into
a stable attacking run is a controlled_recovery; runs shorter than
CONTESTED_RUN_MAX_S alternating between teams are contested touches. Runs in
between (0.9s-2.0s) emit nothing — deliberately: ambiguous evidence must not
become a confident event.
"""
from .evidence import PossessionEvent

STABLE_RUN_MIN_S = 2.0
CONTESTED_RUN_MAX_S = 0.9
MERGE_SAME_TEAM_GAP_S = 1.0


def team_runs(segments, fps: float) -> list:
    """segments: iterable of (track_id, team, start_fid, end_fid) or objects
    with those attributes (digest.PossessionSegment works). Returns merged
    [{'team', 'start_s', 'end_s'}] ordered by time; None-team segments dropped."""
    norm = []
    for s in segments:
        if isinstance(s, tuple):
            _, team, sf, ef = s
        else:
            team, sf, ef = s.team, s.start_fid, s.end_fid
        if team is None:
            continue
        norm.append({"team": team, "start_s": sf / fps, "end_s": ef / fps})
    norm.sort(key=lambda r: r["start_s"])
    merged = []
    for r in norm:
        if merged and merged[-1]["team"] == r["team"] \
                and r["start_s"] - merged[-1]["end_s"] <= MERGE_SAME_TEAM_GAP_S:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], r["end_s"])
        else:
            merged.append(dict(r))
    return merged


def transition_events(runs: list, attacking_team: str) -> list:
    """PossessionEvents at each change of team possession."""
    events = []
    for prev, cur in zip(runs, runs[1:]):
        if prev["team"] == cur["team"]:
            continue
        rel = "attacking" if cur["team"] == attacking_team else "defending"
        dur = cur["end_s"] - cur["start_s"]
        if dur >= STABLE_RUN_MIN_S:
            if rel == "attacking":
                events.append(PossessionEvent(
                    t=cur["start_s"], team="attacking", kind="controlled_recovery"))
        elif dur <= CONTESTED_RUN_MAX_S:
            events.append(PossessionEvent(
                t=cur["start_s"], team=rel, kind="contested_touch"))
        # ambiguous durations: no event
    return events
```

- [ ] **Step 4: Run to verify pass**, **Step 5: Full suite**, **Step 6: Commit**

```bash
python3 -m pytest tests/tactics_qa/ -v
git add pipeline/tactics_qa/possession_timeline.py tests/tactics_qa/test_possession_timeline.py
git commit -m "feat(tactics_qa): team possession timeline and transition events"
```

---

### Task 5: Restart inference from dead-ball location

**Files:**
- Create: `pipeline/tactics_qa/restart_infer.py`
- Test: `tests/tactics_qa/test_restart_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_restart_infer.py
from pipeline.tactics_qa.restart_infer import classify_restart_location, infer_restarts
from pipeline.tactics_qa.kinematics import Kick


def test_location_classification():
    assert classify_restart_location((0.5, -0.3)) == "kickoff"
    assert classify_restart_location((49.0, 3.0)) == "goal_kick"
    assert classify_restart_location((-49.5, -5.0)) == "goal_kick"
    # real SNGS-116 corner kick was at (53.8, -36.6): out of nominal bounds
    assert classify_restart_location((53.8, -36.6)) == "corner"
    assert classify_restart_location((10.0, -34.5)) == "throw_in"
    assert classify_restart_location((20.0, 10.0)) == "free_kick"
    assert classify_restart_location((60.0, 0.0)) is None  # too far out: noise


def test_infer_restart_requires_dead_ball_before_kick():
    kicks = [Kick(t=6.2, xy=(53.8, -36.6), speed_mps=20.0)]
    dead = [(3.0, 6.1)]
    rs = infer_restarts(kicks, dead)
    assert len(rs) == 1 and rs[0].kind == "corner" and rs[0].t == 6.2

    assert infer_restarts(kicks, dead_periods=[]) == []      # moving ball: open play
    assert infer_restarts(kicks, dead_periods=[(0.0, 2.0)]) == []  # stale dead ball
```

- [ ] **Step 2: Run to verify failure** → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/restart_infer.py
"""Restart classification: a kick following a dead-ball period, classified by
where the ball sat. Free kicks are the fallback for in-field dead-ball
restarts; anything further than OUT_OF_PLAY_TOLERANCE_M beyond the pitch is
projection noise and yields None (no restart claimed)."""
from dataclasses import dataclass

PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0
OUT_OF_PLAY_TOLERANCE_M = 3.0
CENTER_SPOT_RADIUS_M = 3.0
GOAL_AREA_DEPTH_M = 7.0        # generous: goal kicks are taken inside ~5.5m box
GOAL_AREA_HALF_WIDTH_M = 11.0
CORNER_RADIUS_M = 4.0
THROW_IN_BAND_M = 1.5          # |y| within this of the touchline
DEAD_TO_KICK_MAX_GAP_S = 0.6


@dataclass(frozen=True)
class Restart:
    t: float
    xy: tuple
    kind: str


def classify_restart_location(xy) -> str | None:
    x, y = xy
    ax, ay = abs(x), abs(y)
    if ax > PITCH_HALF_LENGTH_M + OUT_OF_PLAY_TOLERANCE_M \
            or ay > PITCH_HALF_WIDTH_M + OUT_OF_PLAY_TOLERANCE_M:
        return None
    if ax <= CENTER_SPOT_RADIUS_M and ay <= CENTER_SPOT_RADIUS_M:
        return "kickoff"
    near_goal_line = ax >= PITCH_HALF_LENGTH_M - CORNER_RADIUS_M
    near_touch_line = ay >= PITCH_HALF_WIDTH_M - CORNER_RADIUS_M
    if near_goal_line and near_touch_line:
        return "corner"
    if ax >= PITCH_HALF_LENGTH_M - GOAL_AREA_DEPTH_M and ay <= GOAL_AREA_HALF_WIDTH_M:
        return "goal_kick"
    if ay >= PITCH_HALF_WIDTH_M - THROW_IN_BAND_M:
        return "throw_in"
    return "free_kick"


def infer_restarts(kicks, dead_periods) -> list:
    out = []
    for k in kicks:
        preceded = any(end <= k.t <= end + DEAD_TO_KICK_MAX_GAP_S or start <= k.t <= end
                       for start, end in dead_periods)
        if not preceded:
            continue
        kind = classify_restart_location(k.xy)
        if kind is not None:
            out.append(Restart(t=k.t, xy=k.xy, kind=kind))
    return out
```

Note: throw-ins are hand-thrown so kick detection may miss them (no foot strike, lower speed); that shows up as a `missing` in the agreement report, which is the honest outcome — do not lower `KICK_MIN_SPEED_MPS` globally to catch them.

- [ ] **Step 4: Run to verify pass**, **Step 5: Commit**

```bash
python3 -m pytest tests/tactics_qa/test_restart_infer.py -v
git add pipeline/tactics_qa/restart_infer.py tests/tactics_qa/test_restart_infer.py
git commit -m "feat(tactics_qa): restart inference from dead-ball location"
```

---

### Task 6: Delivery kind and corner landing zone

**Files:**
- Create: `pipeline/tactics_qa/delivery_infer.py`
- Test: `tests/tactics_qa/test_delivery_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_delivery_infer.py
from pipeline.tactics_qa.delivery_infer import classify_delivery, corner_landing_zone
from pipeline.tactics_qa.kinematics import Kick
from pipeline.tactics_qa.restart_infer import Restart


def test_restart_delivery_wins_over_geometry():
    k = Kick(t=6.2, xy=(53.8, -36.6), speed_mps=20.0)
    d = classify_delivery(k, end_xy=(48.0, -5.0), attack_sign=1,
                          restarts=[Restart(6.2, (53.8, -36.6), "corner")])
    assert d == "corner_cross"


def test_cross_from_wide_channel_into_box():
    k = Kick(t=10.0, xy=(40.0, -28.0), speed_mps=18.0)
    assert classify_delivery(k, end_xy=(47.0, -2.0), attack_sign=1, restarts=[]) == "cross"


def test_long_clearance_from_defensive_third():
    k = Kick(t=3.0, xy=(-40.0, 5.0), speed_mps=28.0)
    assert classify_delivery(k, end_xy=(10.0, 0.0), attack_sign=1, restarts=[]) == "long_clearance"


def test_forward_ground_pass_is_open_play_pass():
    k = Kick(t=8.0, xy=(0.0, 5.0), speed_mps=12.0)
    assert classify_delivery(k, end_xy=(20.0, 3.0), attack_sign=1, restarts=[]) == "open_play_pass"


def test_ambiguous_returns_none():
    k = Kick(t=8.0, xy=(0.0, 5.0), speed_mps=12.0)
    assert classify_delivery(k, end_xy=None, attack_sign=1, restarts=[]) is None


def test_corner_landing_zone_near_far_center():
    # corner from y=-36.6 (negative side); landing y=-5.2 → same side → near post
    # (matches SNGS-116 human verdict 前点)
    assert corner_landing_zone(corner_y=-36.6, landing_xy=(50.0, -5.2)) == "near_post"
    assert corner_landing_zone(corner_y=-36.6, landing_xy=(50.0, 6.0)) == "far_post"
    assert corner_landing_zone(corner_y=-36.6, landing_xy=(50.0, 0.5)) == "center"
    assert corner_landing_zone(corner_y=-36.6, landing_xy=None) is None
```

- [ ] **Step 2: Run to verify failure** → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/delivery_infer.py
"""Delivery classification for the claim's key pass, and corner landing zones.

attack_sign: +1 if the attacking team attacks toward x=+52.5, else -1.
end_xy: where the ball is next controlled (next stable holder position), or
None when tracking loses the ball — in which case geometry-based kinds are
unavailable and the function returns None rather than guessing.
GSR has no ball height, so 'cross' is inferred purely from wide-origin +
box-destination geometry.
"""
WIDE_CHANNEL_MIN_Y_M = 20.0
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
CLEARANCE_MIN_SPEED_MPS = 22.0
CLEARANCE_MIN_DISTANCE_M = 30.0
DEFENSIVE_THIRD_MAX_X_M = -17.5   # in attack-normalized coords
PASS_MIN_FORWARD_M = 3.0

_RESTART_DELIVERY = {"corner": "corner_cross", "goal_kick": "goal_kick",
                     "throw_in": "throw_in", "free_kick": "free_kick"}


def _in_box(x, y, attack_sign):
    return (x * attack_sign >= 52.5 - BOX_DEPTH_M) and abs(y) <= BOX_HALF_WIDTH_M


def classify_delivery(kick, end_xy, attack_sign, restarts) -> str | None:
    for r in restarts:
        if abs(r.t - kick.t) < 0.2 and r.kind in _RESTART_DELIVERY:
            return _RESTART_DELIVERY[r.kind]
    if end_xy is None:
        return None
    x0, y0 = kick.xy
    x1, y1 = end_xy
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    forward = (x1 - x0) * attack_sign
    if (kick.speed_mps >= CLEARANCE_MIN_SPEED_MPS and dist >= CLEARANCE_MIN_DISTANCE_M
            and x0 * attack_sign <= DEFENSIVE_THIRD_MAX_X_M):
        return "long_clearance"
    if abs(y0) >= WIDE_CHANNEL_MIN_Y_M and _in_box(x1, y1, attack_sign) \
            and abs(y1) < abs(y0):
        return "cross"
    if forward >= PASS_MIN_FORWARD_M:
        return "open_play_pass"
    return None


def corner_landing_zone(corner_y, landing_xy) -> str | None:
    """near_post = landing on the corner's side of goal centre, far_post =
    opposite side, center = within CENTER_BAND of y=0."""
    CENTER_BAND_M = 2.0
    if landing_xy is None:
        return None
    _, y = landing_xy
    side = -1.0 if corner_y < 0 else 1.0
    y_toward_corner = y * side
    if y_toward_corner >= CENTER_BAND_M:
        return "near_post"
    if y_toward_corner <= -CENTER_BAND_M:
        return "far_post"
    return "center"
```

- [ ] **Step 4: Run to verify pass**, **Step 5: Commit**

```bash
python3 -m pytest tests/tactics_qa/test_delivery_infer.py -v
git add pipeline/tactics_qa/delivery_infer.py tests/tactics_qa/test_delivery_infer.py
git commit -m "feat(tactics_qa): delivery kind and corner landing zone inference"
```

---

### Task 7: Compose auto ClipEvidence per claim

**Files:**
- Create: `pipeline/tactics_qa/auto_evidence.py`
- Test: `tests/tactics_qa/test_auto_evidence.py`

- [ ] **Step 1: Write the failing test** (unit level with synthetic GsrClip; the real-data smoke test comes in Task 8)

```python
# tests/tactics_qa/test_auto_evidence.py
from pipeline.tactics_qa.auto_evidence import attacking_side, build_claim_evidence
from pipeline.tactics_qa.gsr_io import GsrClip


def _clip_with_ball(ball):
    return GsrClip(fps=25, n_frames=max(ball) + 1, ball=ball, players={})


def test_attacking_side_from_possession_runs():
    runs = [{"team": "left", "start_s": 0.0, "end_s": 3.0},
            {"team": "right", "start_s": 3.5, "end_s": 12.0}]
    # claim window 4-10s: right holds possession for most of it
    assert attacking_side(runs, {"start_s": 4.0, "end_s": 10.0}) == "right"


def test_attacking_side_ambiguous_returns_none():
    runs = [{"team": "left", "start_s": 4.0, "end_s": 7.0},
            {"team": "right", "start_s": 7.2, "end_s": 10.0}]
    assert attacking_side(runs, {"start_s": 4.0, "end_s": 10.0}) is None


def test_build_claim_evidence_marks_unknowns():
    ball = {f: (0.0, 0.0) for f in range(0, 10)}  # nearly no data
    ev = build_claim_evidence(
        clip=_clip_with_ball(ball), clip_uid="t:x", tactic_id="fast_break_pattern",
        window={"start_s": 5.0, "end_s": 15.0})
    assert ev.delivery_kind is None
    assert ev.corner_landing_zone is None
    assert ev.source == "gsr_trajectory"
```

- [ ] **Step 2: Run to verify failure** → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/auto_evidence.py
"""Compose ClipEvidence from GSR trajectories for one reviewed claim.

Side mapping: the review rows never say left/right, so the attacking side is
inferred as the team holding possession for the majority (>=60%) of the claim
window. If no side reaches 60%, the claim gets no possession-relative events
(checkers then return insufficient) — ambiguity must surface, not be hidden.
"""
from .evidence import ClipEvidence
from .kinematics import detect_kicks, dead_ball_periods
from .possession_timeline import team_runs, transition_events
from .restart_infer import infer_restarts
from .delivery_infer import classify_delivery, corner_landing_zone
from .gsr_io import GsrClip

ATTACK_SIDE_MIN_SHARE = 0.6
EVENT_BAND_BEFORE_S = 5.0      # events kept from window_start - 5s ...
EVENT_BAND_AFTER_S = 5.0       # ... to window_start + 5s (checker bands are ±3s)
LANDING_SEARCH_S = 4.0
HOLDER_NEAR_BALL_M = 1.5


def attacking_side(runs, window) -> str | None:
    ws, we = window["start_s"], window["end_s"]
    share = {}
    for r in runs:
        overlap = max(0.0, min(we, r["end_s"]) - max(ws, r["start_s"]))
        share[r["team"]] = share.get(r["team"], 0.0) + overlap
    total = sum(share.values())
    if not total:
        return None
    team, best = max(share.items(), key=lambda kv: kv[1])
    return team if best / total >= ATTACK_SIDE_MIN_SHARE else None


def _possession_segments(clip: GsrClip):
    """Adapt GsrClip.players to digest.possession_segments input."""
    from pipeline.stage2b.digest import FrameData, possession_segments
    frames = []
    for fid in sorted(set(clip.players) | set(clip.ball)):
        players = [{"track_id": tid, "role": "player", "team": team,
                    "jersey": "", "x": x, "y": y}
                   for tid, team, x, y in clip.players.get(fid, [])]
        # FrameData.ball_xy is an Optional[Tuple[float, float]] — see digest.py
        frames.append(FrameData(frame_id=fid, players=players,
                                ball_xy=clip.ball.get(fid)))
    team_by_track = {}
    for fid_players in clip.players.values():
        for tid, team, _, _ in fid_players:
            team_by_track.setdefault(tid, team)
    return possession_segments(frames, team_by_track)


def _attack_sign(runs, side, clip: GsrClip, window) -> int | None:
    """+1 if `side` attacks toward +x. Estimated from ball drift while that
    side holds possession inside the window; None when displacement < 5 m."""
    xs = []
    for r in runs:
        if r["team"] != side:
            continue
        lo = max(r["start_s"], window["start_s"])
        hi = min(r["end_s"], window["end_s"])
        pts = [clip.ball[f] for f in sorted(clip.ball)
               if lo <= f / clip.fps <= hi]
        if len(pts) >= 2:
            xs.append(pts[-1][0] - pts[0][0])
    drift = sum(xs)
    if abs(drift) < 5.0:
        return None
    return 1 if drift > 0 else -1


def build_claim_evidence(clip: GsrClip, clip_uid: str, tactic_id: str,
                         window: dict) -> ClipEvidence:
    kicks = detect_kicks(clip.ball, clip.fps)
    dead = dead_ball_periods(clip.ball, clip.fps)
    restarts = infer_restarts(kicks, dead)
    runs = team_runs(_possession_segments(clip), clip.fps)
    side = attacking_side(runs, window)

    ws = window["start_s"]
    events, notes = [], []
    for r in restarts:
        if ws - EVENT_BAND_BEFORE_S <= r.t <= ws + EVENT_BAND_AFTER_S:
            team = "attacking"  # restart taker side is not needed by restart_gate
            from .evidence import PossessionEvent
            events.append(PossessionEvent(t=r.t, team=team, kind=r.kind))
            notes.append(f"restart {r.kind}@{r.t:.1f}s at {r.xy}")
    if side is not None:
        for e in transition_events(runs, attacking_team=side):
            if ws - EVENT_BAND_BEFORE_S <= e.t <= ws + EVENT_BAND_AFTER_S:
                events.append(e)
    else:
        notes.append("attacking side ambiguous: no possession events emitted")

    delivery = None
    sign = _attack_sign(runs, side, clip, window) if side else None
    window_kicks = [k for k in kicks if ws - 2.0 <= k.t <= window["end_s"]]
    if window_kicks and sign is not None:
        key = window_kicks[0]
        end_xy = _next_control_point(clip, key, runs)
        delivery = classify_delivery(key, end_xy, sign, restarts)
    landing = None
    if tactic_id == "corner_near_far_post":
        corner = next((r for r in restarts if r.kind == "corner"), None)
        if corner is not None:
            landing = corner_landing_zone(
                corner.xy[1], _first_box_touch(clip, corner))
    return ClipEvidence(clip_uid=clip_uid, tactic_id=tactic_id, events=events,
                        delivery_kind=delivery, corner_landing_zone=landing,
                        source="gsr_trajectory", notes="; ".join(notes))


def _next_control_point(clip: GsrClip, kick, runs):
    """Ball position when a player next controls it after the kick, else None."""
    for f in sorted(clip.ball):
        t = f / clip.fps
        if t <= kick.t + 0.2:
            continue
        if t > kick.t + LANDING_SEARCH_S:
            break
        bx, by = clip.ball[f]
        for tid, team, px, py in clip.players.get(f, []):
            if ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 <= HOLDER_NEAR_BALL_M:
                return (bx, by)
    return None


def _first_box_touch(clip: GsrClip, corner):
    """First ball position after the corner kick within reach of any player
    (the contested first contact)."""
    for f in sorted(clip.ball):
        t = f / clip.fps
        if t <= corner.t + 0.2 or t > corner.t + LANDING_SEARCH_S:
            continue
        bx, by = clip.ball[f]
        for tid, team, px, py in clip.players.get(f, []):
            if ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 <= HOLDER_NEAR_BALL_M:
                return (bx, by)
    return None
```

- [ ] **Step 4: Run to verify pass**, then full suite; **Step 5: Commit**

```bash
python3 -m pytest tests/tactics_qa/ -v
git add pipeline/tactics_qa/auto_evidence.py tests/tactics_qa/test_auto_evidence.py
git commit -m "feat(tactics_qa): compose auto ClipEvidence from GSR trajectories"
```

---

### Task 8: Real-data smoke test on SNGS-116

**Files:**
- Test: `tests/tactics_qa/test_smoke_sngs116.py`

- [ ] **Step 1: Write the test** (skips when outputs are absent, e.g. on CI)

```python
# tests/tactics_qa/test_smoke_sngs116.py
"""Anchored to known facts about outputs/SNGS-116: a left-team corner is taken
around t=6.2s from (53.8, -36.6); the human review row for SNGS-116
(corner_near_far_post) is verdict=correct with correction 前点 (near post)."""
import os
import pytest

PRED = "outputs/SNGS-116/predictions.json"
pytestmark = pytest.mark.skipif(not os.path.exists(PRED),
                                reason="SNGS-116 GSR outputs not present")


def test_corner_restart_detected():
    from pipeline.tactics_qa.gsr_io import load_gsr
    from pipeline.tactics_qa.kinematics import detect_kicks, dead_ball_periods
    from pipeline.tactics_qa.restart_infer import infer_restarts
    clip = load_gsr(PRED)
    restarts = infer_restarts(detect_kicks(clip.ball, clip.fps),
                              dead_ball_periods(clip.ball, clip.fps))
    corners = [r for r in restarts if r.kind == "corner"]
    assert corners, f"no corner found; restarts={restarts}"
    assert any(4.0 <= r.t <= 8.5 for r in corners)


def test_corner_landing_zone_is_near_post():
    from pipeline.tactics_qa.gsr_io import load_gsr
    from pipeline.tactics_qa.auto_evidence import build_claim_evidence
    clip = load_gsr(PRED)
    ev = build_claim_evidence(clip, "soccernetgs:SNGS-116", "corner_near_far_post",
                              {"start_s": 3.0, "end_s": 9.0})
    assert ev.corner_landing_zone == "near_post", ev.notes
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/tactics_qa/test_smoke_sngs116.py -v`
Expected: PASS on the machine with outputs. If the corner isn't detected, debug in this order: (1) dump `dead_ball_periods` — is the pre-corner stillness ≥ 1.0s? (2) dump `detect_kicks` around t=6.2s; (3) only then consider a threshold change, and record any change plus its reason in the commit message. If `near_post` fails, print `ev.notes` and check `_first_box_touch` picked the header contest, not a later scramble.

- [ ] **Step 3: Commit**

```bash
git add tests/tactics_qa/test_smoke_sngs116.py
git commit -m "test(tactics_qa): SNGS-116 real-data smoke test for corner inference"
```

---

### Task 9: Agreement report and auto-evidence replay

**Files:**
- Create: `pipeline/tactics_qa/agreement.py`
- Create: `scripts/build_auto_evidence.py`
- Test: `tests/tactics_qa/test_agreement.py`
- Generate: `benchmark/tactical_prototypes/recognition_evidence_auto.jsonl`, `evidence_agreement_report.json`, `evidence_agreement_report.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_agreement.py
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent as E
from pipeline.tactics_qa.agreement import claim_agreement

W = {"start_s": 10.0, "end_s": 20.0}


def test_claim_agreement_fields():
    hand = ClipEvidence("t:a", "fast_break_pattern",
                        [E(10.0, "attacking", "kickoff")], delivery_kind="long_clearance")
    auto = ClipEvidence("t:a", "fast_break_pattern",
                        [E(10.2, "attacking", "kickoff")], delivery_kind=None,
                        source="gsr_trajectory")
    a = claim_agreement(hand, auto, W)
    assert a["restart_at_origin"] == "match"        # both find a kickoff in the R4 band
    assert a["delivery_kind"] == "auto_missing"     # hand has it, auto doesn't
    assert a["has_controlled_regain"] == "match"    # both False
    assert a["chaotic"] == "match"                  # both False


def test_verdict_level_agreement():
    hand = ClipEvidence("t:b", "corner_near_far_post", corner_landing_zone="center")
    auto = ClipEvidence("t:b", "corner_near_far_post", corner_landing_zone="near_post",
                        source="gsr_trajectory")
    a = claim_agreement(hand, auto, W)
    assert a["corner_landing_zone"] == "mismatch"
    assert a["checker_verdicts_equal"] is False
```

- [ ] **Step 2: Run to verify failure** → `ImportError`

- [ ] **Step 3: Implement**

```python
# pipeline/tactics_qa/agreement.py
"""Per-claim field agreement between hand-encoded and auto evidence.

Raw event lists are not directly comparable (timing granularity differs), so
agreement is measured on the checker-relevant predicates derived from them:
restart at window origin (R4 band), controlled regain presence and chaos flag
(R7 band), delivery_kind, corner_landing_zone, and finally whether the full
checker verdict set comes out identical. Values per field:
match | mismatch | auto_missing | hand_missing | both_missing.
"""
from .checkers import run_checkers, RESTART_LOOKBACK_S, REGAIN_WINDOW_S
from .evidence import RESTART_KINDS, CONTROLLED_REGAIN_KINDS, CHAOS_KINDS


def _restart_at_origin(ev, window):
    ws = window["start_s"]
    kinds = [e.kind for e in ev.events
             if e.kind in RESTART_KINDS and ws - RESTART_LOOKBACK_S <= e.t <= ws + 1.0]
    return kinds[0] if kinds else None


def _regain_predicates(ev, window):
    ws = window["start_s"]
    near = [e for e in ev.events if ws - REGAIN_WINDOW_S <= e.t <= ws + REGAIN_WINDOW_S]
    has_regain = any(e.team == "attacking" and e.kind in CONTROLLED_REGAIN_KINDS
                     for e in near)
    chaotic = sum(1 for e in near if e.kind in CHAOS_KINDS) > 2
    return has_regain, chaotic


def _cmp(hand_val, auto_val):
    if hand_val is None and auto_val is None:
        return "both_missing"
    if auto_val is None:
        return "auto_missing"
    if hand_val is None:
        return "hand_missing"
    return "match" if hand_val == auto_val else "mismatch"


def claim_agreement(hand, auto, window) -> dict:
    hr, ar = _restart_at_origin(hand, window), _restart_at_origin(auto, window)
    h_regain, h_chaos = _regain_predicates(hand, window)
    a_regain, a_chaos = _regain_predicates(auto, window)
    hand_verdicts = sorted((c.checker, c.verdict) for c in run_checkers(hand, window))
    auto_verdicts = sorted((c.checker, c.verdict) for c in run_checkers(auto, window))
    return {
        "clip_uid": hand.clip_uid,
        "tactic_id": hand.tactic_id,
        "restart_at_origin": _cmp(hr, ar),
        "has_controlled_regain": "match" if h_regain == a_regain else "mismatch",
        "chaotic": "match" if h_chaos == a_chaos else "mismatch",
        "delivery_kind": _cmp(hand.delivery_kind, auto.delivery_kind),
        "corner_landing_zone": _cmp(hand.corner_landing_zone, auto.corner_landing_zone),
        "checker_verdicts_equal": hand_verdicts == auto_verdicts,
        "hand_verdicts": hand_verdicts,
        "auto_verdicts": auto_verdicts,
    }
```

- [ ] **Step 4: Run to verify pass** → `python3 -m pytest tests/tactics_qa/test_agreement.py -v`

- [ ] **Step 5: Write the CLI**

```python
# scripts/build_auto_evidence.py
"""Generate auto evidence for every reviewed claim with GSR outputs, then
compare against hand evidence and re-run the replay under auto evidence.

Usage: python3 scripts/build_auto_evidence.py [--outputs-dir outputs]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tactics_qa.gsr_io import load_gsr                      # noqa: E402
from pipeline.tactics_qa.auto_evidence import build_claim_evidence   # noqa: E402
from pipeline.tactics_qa.evidence import load_evidence_jsonl         # noqa: E402
from pipeline.tactics_qa.agreement import claim_agreement            # noqa: E402
from pipeline.tactics_qa.replay import replay, to_markdown           # noqa: E402

BASE = Path("benchmark/tactical_prototypes")


def main() -> None:
    outputs_dir = Path(sys.argv[sys.argv.index("--outputs-dir") + 1]) \
        if "--outputs-dir" in sys.argv else Path("outputs")
    review = [json.loads(l) for l in open(BASE / "recognition_review.jsonl")]
    hand = load_evidence_jsonl(str(BASE / "recognition_evidence.jsonl"))

    auto, coverage = {}, Counter()
    with open(BASE / "recognition_evidence_auto.jsonl", "w", encoding="utf-8") as out:
        for r in review:
            pred = outputs_dir / r["clip_id"] / "predictions.json"
            if not pred.exists():
                coverage["no_data"] += 1
                continue
            clip = load_gsr(pred)
            ev = build_claim_evidence(clip, r["clip_uid"], r["tactic_id"], r["window"])
            auto[(ev.clip_uid, ev.tactic_id)] = ev
            out.write(json.dumps(ev.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            coverage["covered"] += 1

    agreements = [claim_agreement(hand[k], auto[k], r["window"])
                  for r in review
                  for k in [(r["clip_uid"], r["tactic_id"])]
                  if k in auto and k in hand]

    covered_review = [r for r in review
                      if (r["clip_uid"], r["tactic_id"]) in auto]
    rep_auto = replay(covered_review, auto)
    rep_hand = replay(covered_review, hand)  # same subset, hand evidence

    fields = ["restart_at_origin", "has_controlled_regain", "chaotic",
              "delivery_kind", "corner_landing_zone"]
    summary = {f: Counter(a[f] for a in agreements) for f in fields}
    summary["checker_verdicts_equal"] = Counter(
        a["checker_verdicts_equal"] for a in agreements)

    report = {"schema_version": "evidence-agreement-v1",
              "coverage": dict(coverage),
              "field_agreement": {k: dict(v) for k, v in summary.items()},
              "claims": agreements,
              "replay_auto": rep_auto["per_tactic"],
              "replay_hand_same_subset": rep_hand["per_tactic"]}
    with open(BASE / "evidence_agreement_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, sort_keys=True)

    lines = ["# 自动证据一致率报告", "",
             f"覆盖：{coverage['covered']} 条有 GSR，{coverage['no_data']} 条无数据", "",
             "| 字段 | match | mismatch | auto_missing | hand_missing | both_missing |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for f in fields:
        c = summary[f]
        lines.append(f"| {f} | {c.get('match',0)} | {c.get('mismatch',0)} "
                     f"| {c.get('auto_missing',0)} | {c.get('hand_missing',0)} "
                     f"| {c.get('both_missing',0)} |")
    eq = summary["checker_verdicts_equal"]
    lines += ["", f"checker 判定完全一致：{eq.get(True,0)}/{len(agreements)}", "",
              "## 自动证据回放（同一覆盖子集） ", "", to_markdown(rep_auto), "",
              "## 手工证据回放（同一覆盖子集，对照） ", "", to_markdown(rep_hand)]
    with open(BASE / "evidence_agreement_report.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"covered={coverage['covered']} no_data={coverage['no_data']} "
          f"verdict_equal={eq.get(True,0)}/{len(agreements)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it and sanity-check the report**

Run: `python3 scripts/build_auto_evidence.py`
Expected console: coverage counts (non-SNGS rows 6/7/13/14 are `no_data` by design; SNGS rows depend on which outputs exist on this machine).

Sanity checks on `evidence_agreement_report.md`:
- `corner_landing_zone` mismatch rows: inspect each — zone thresholds (`CENTER_BAND_M`) may need one adjustment; if changed, re-run and note it.
- `auto_missing` should dominate over `mismatch` everywhere: the adapter is designed to abstain, and abstention shows up as `insufficient`, not wrong vetoes. **A high `mismatch` count is a bug signal; a high `auto_missing` count is a perception-coverage finding — report it, don't paper over it.**
- Compare `replay_auto` vs `replay_hand_same_subset` per-tactic `precision_after` and `false_vetoes`. False vetoes under auto evidence are the headline risk number.

- [ ] **Step 7: Full suite + commit**

```bash
python3 -m pytest tests/tactics_qa/ -v
git add pipeline/tactics_qa/agreement.py tests/tactics_qa/test_agreement.py \
        scripts/build_auto_evidence.py \
        benchmark/tactical_prototypes/recognition_evidence_auto.jsonl \
        benchmark/tactical_prototypes/evidence_agreement_report.json \
        benchmark/tactical_prototypes/evidence_agreement_report.md
git commit -m "feat(tactics_qa): auto-vs-hand evidence agreement and replay comparison"
```

---

### Task 10: Wrap-up

- [ ] **Step 1: Prepend a summary to `evidence_agreement_report.md`** (manually, ~8 lines):
per-field agreement percentages, checker-verdict agreement rate, `precision_after` auto vs hand per P0 tactic, false-veto count under auto evidence, and the top-3 causes of `auto_missing` (e.g. throw-in kicks undetected, attacking-side ambiguity, ball-track gaps). These three causes are the prioritized perception backlog for the next iteration.

- [ ] **Step 2: Update `benchmark/README.md`** with the two new files (one row each, same style as existing table).

- [ ] **Step 3: Final commit**

```bash
git add benchmark/tactical_prototypes/evidence_agreement_report.md benchmark/README.md
git commit -m "docs(benchmark): auto-evidence agreement summary and perception backlog"
```

**Done criteria for the whole plan:**
1. `python3 -m pytest tests/tactics_qa/` green (smoke test may skip off-server).
2. `recognition_evidence_auto.jsonl` generated for every reviewed claim with GSR outputs; the rest counted as `no_data`.
3. `evidence_agreement_report.md` shows per-field agreement, checker-verdict agreement, and auto-vs-hand replay precision on the same subset, with false vetoes under auto evidence explicitly listed.
4. SNGS-116 smoke test passes on the server (corner detected, `near_post` landing).
5. No threshold was tuned to convert `auto_missing` into guesses; every threshold change is recorded in a commit message with its reason.

**Explicit non-goals:** no use of `comments/events.json` or any VLM output as evidence; no changes to `checkers.py` semantics; no wiring into the live stage2 pipeline (that's the next plan, after agreement numbers are known); no work on deferred tactics; no re-encoding of the hand evidence to make agreement look better — if hand encoding turns out wrong on inspection, fix it in a dedicated commit that names the row and reason.
