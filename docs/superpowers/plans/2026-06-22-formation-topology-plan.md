# Formation Topology V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure, 0-GPU post-processor that turns per-frame player pitch landing points (`bbox_pitch`) into a per-window, per-team structural signal (block height/depth/width, 5×3 lane-band occupancy, and D–M / M–F inter-line gaps) for downstream LLM tactical commentary.

**Architecture:** Read `Labels-GameState.json` → normalize each team into a canonical attacking frame (attack → +x) → slide a time window taking each track's median position → compute clustering-free metrics always, and split players into depth lines by their largest x-gaps (coverage-gated) to report inter-line distances. Possession is a metadata tag from the ball's nearest player; geometry never depends on it.

**Tech Stack:** Python 3.10, numpy, scipy (ConvexHull), pytest. No GPU, no ML. Same code path on GT `bbox_pitch` (validation) and calibrated SoccerMaster/PnLCalib output (production).

---

## File Structure

```
formation_topology/
  __init__.py        # empty package marker
  pitch.py           # pitch constants + canonical-frame transform
  io_gamestate.py    # load Labels-GameState.json -> List[Detection]
  metrics.py         # clustering-free metrics + lane/band bucketing
  lines.py           # simplified depth-line split -> inter-line gaps
  windows.py         # sliding windows + per-track median position
  possession.py      # per-window possession tag (nearest player to ball)
  pipeline.py        # infer attack dirs + analyze() -> records
  cli.py             # argparse runner -> writes records JSON
tests/
  test_pitch.py  test_io_gamestate.py  test_metrics.py  test_lines.py
  test_windows.py  test_possession.py  test_pipeline.py  test_cli.py
```

Each module has one responsibility and a small interface. `metrics.py` / `lines.py` are pure (numpy in, numbers out). `pipeline.py` is the only orchestrator.

---

## Setup

- [ ] **Create the package skeleton and install deps**

Run (from repo root, alongside your existing `pitch_distances.py`):
```bash
mkdir -p formation_topology tests
touch formation_topology/__init__.py
python -m pip install --break-system-packages numpy scipy pytest
```

Integration note: place `formation_topology/` next to `pitch_distances.py`; it reads the same `Labels-GameState.json`. If your export's top-level key isn't `annotations` or attribute names differ, adjust them in `load_detections` (Task 2) — everything downstream consumes the `Detection` dataclass, not the raw JSON.

---

## Task 1: Pitch constants + canonical-frame transform

**Files:**
- Create: `formation_topology/pitch.py`
- Test: `tests/test_pitch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pitch.py
import numpy as np
from formation_topology.pitch import canonicalize, HALF_LENGTH, HALF_WIDTH


def test_constants():
    assert HALF_LENGTH == 52.5
    assert HALF_WIDTH == 34.0


def test_canonicalize_flips_negative_direction_team():
    # team attacking -x (own goal at +52.5 side) -> rotate 180deg about origin
    pts = np.array([[40.0, 10.0], [30.0, -5.0]])
    out = canonicalize(pts, attack_dir=-1)
    np.testing.assert_allclose(out, [[-40.0, -10.0], [-30.0, 5.0]])


def test_canonicalize_identity_for_positive_direction_team():
    pts = np.array([[-40.0, 10.0]])
    out = canonicalize(pts, attack_dir=1)
    np.testing.assert_allclose(out, [[-40.0, 10.0]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pitch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.pitch'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/pitch.py
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0   # 52.5  (x: own goal -52.5 -> opponent goal +52.5)
HALF_WIDTH = PITCH_WIDTH / 2.0     # 34.0  (y: sideline to sideline)

# Canonical-frame buckets (attack -> +x).
BAND_EDGES_X = (-HALF_LENGTH / 3.0, HALF_LENGTH / 3.0)   # (-17.5, 17.5): defensive/middle/attacking thirds
LANE_EDGES_Y = (-20.16, -7.0, 7.0, 20.16)                # 5 lanes: wing/half-space/center/half-space/wing


def canonicalize(pts, attack_dir):
    """Rotate a team's points into the canonical attacking frame (attack -> +x).

    attack_dir: +1 if the team already attacks +x, -1 otherwise.
    A -1 team is rotated 180 deg about the origin: (x, y) -> (-x, -y). This puts the
    own goal at -52.5 and preserves left/right handedness (a rotation, not a reflection),
    so lane indices mean the same flank for both teams.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if attack_dir < 0:
        return -pts
    return pts.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pitch.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/__init__.py formation_topology/pitch.py tests/test_pitch.py
git commit -m "feat(topo): pitch constants and canonical-frame transform"
```

---

## Task 2: Load Labels-GameState.json into Detection rows

**Files:**
- Create: `formation_topology/io_gamestate.py`
- Test: `tests/test_io_gamestate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io_gamestate.py
import json
from formation_topology.io_gamestate import load_detections


def test_load_skips_null_pitch_and_parses_fields(tmp_path):
    sample = {"annotations": [
        {"id": 1, "image_id": 1, "track_id": 7,
         "attributes": {"role": "player", "team": "left", "jersey": 7},
         "bbox_pitch": {"x_bottom_middle": -40.0, "y_bottom_middle": 5.0}},
        {"id": 2, "image_id": 1, "track_id": 99,
         "attributes": {"role": "ball"},
         "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}},
        {"id": 3, "image_id": 1, "track_id": 8,
         "attributes": {"role": "player", "team": "right", "jersey": 9},
         "bbox_pitch": None},  # calibration failed -> must be skipped
    ]}
    p = tmp_path / "Labels-GameState.json"
    p.write_text(json.dumps(sample))

    dets = load_detections(str(p))
    assert len(dets) == 2  # the null-pitch detection is dropped
    d0 = dets[0]
    assert (d0.frame, d0.track_id, d0.team, d0.role, d0.jersey_number) == (1, 7, "left", "player", 7)
    assert d0.x == -40.0 and d0.y == 5.0
    assert dets[1].role == "ball" and dets[1].team is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_io_gamestate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.io_gamestate'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/io_gamestate.py
import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    frame: int                  # image_id
    track_id: int
    team: Optional[str]         # 'left' | 'right' | None
    role: str                   # 'player' | 'goalkeeper' | 'referee' | 'ball' | 'other'
    jersey_number: Optional[int]
    x: float                    # bbox_pitch.x_bottom_middle (meters)
    y: float                    # bbox_pitch.y_bottom_middle (meters)


def load_detections(json_path: str) -> List[Detection]:
    """Parse a SoccerNet GameState Labels-GameState.json into Detection rows.

    Drops any detection whose bbox_pitch is missing/None (calibration failed that frame).
    Uses x_bottom_middle / y_bottom_middle (the player's feet/ground point).
    """
    with open(json_path) as f:
        data = json.load(f)

    dets: List[Detection] = []
    for a in data["annotations"]:
        bp = a.get("bbox_pitch")
        if not isinstance(bp, dict):
            continue
        x = bp.get("x_bottom_middle")
        y = bp.get("y_bottom_middle")
        if x is None or y is None:
            continue
        attrs = a.get("attributes", {}) or {}
        jn = attrs.get("jersey")
        dets.append(Detection(
            frame=int(a["image_id"]),
            track_id=int(a.get("track_id", a["id"])),
            team=attrs.get("team"),
            role=attrs.get("role", "other"),
            jersey_number=int(jn) if jn not in (None, "") else None,
            x=float(x),
            y=float(y),
        ))
    return dets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_io_gamestate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/io_gamestate.py tests/test_io_gamestate.py
git commit -m "feat(topo): load Labels-GameState.json into Detection rows"
```

---

## Task 3: Clustering-free metrics + lane/band bucketing

**Files:**
- Create: `formation_topology/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
from formation_topology import metrics


def test_height_depth_width():
    pts = np.array([[-40, 0], [-38, 0], [10, 0], [10, 20], [10, -20]], dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    assert abs(metrics.block_height(pts) - np.mean(x)) < 1e-6
    assert abs(metrics.block_depth(pts) - (np.percentile(x, 90) - np.percentile(x, 10))) < 1e-6
    assert abs(metrics.block_width(pts) - (np.percentile(y, 90) - np.percentile(y, 10))) < 1e-6
    assert abs(metrics.centroid_y(pts) - np.mean(y)) < 1e-6


def test_band_and_lane_counts_and_overload():
    # 2 in defensive third (x < -17.5), 1 middle, 2 attacking (x > 17.5)
    pts = np.array([[-40, -25], [-30, -10], [0, 0], [30, 10], [40, 25]], dtype=float)
    assert metrics.band_counts(pts) == [2, 1, 2]
    # lanes by y edges (-20.16, -7, 7, 20.16): -25->0, -10->1, 0->2, 10->3, 25->4
    assert metrics.lane_counts(pts) == [1, 1, 1, 1, 1]
    assert abs(metrics.side_overload(pts) - 0.0) < 1e-6


def test_side_overload_signed():
    # all on lanes 0/1 (one flank) -> overload +1.0
    pts = np.array([[0, -25], [0, -10]], dtype=float)
    assert abs(metrics.side_overload(pts) - 1.0) < 1e-6


def test_hull_area_triangle():
    pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    assert abs(metrics.hull_area(pts) - 50.0) < 1e-6


def test_hull_area_degenerate_returns_zero():
    assert metrics.hull_area(np.array([[0, 0], [5, 0]], dtype=float)) == 0.0
    assert metrics.hull_area(np.array([[0, 0], [5, 0], [10, 0]], dtype=float)) == 0.0  # collinear
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/metrics.py
import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:  # older scipy
    from scipy.spatial.qhull import QhullError

from .pitch import BAND_EDGES_X, LANE_EDGES_Y


def block_height(pts):
    """Mean x: how high up the pitch the team sits (canonical frame, attack +x)."""
    return float(np.mean(pts[:, 0]))


def block_depth(pts):
    """Front-to-back spread along x (p90 - p10, robust to outliers)."""
    x = pts[:, 0]
    return float(np.percentile(x, 90) - np.percentile(x, 10))


def block_width(pts):
    """Side-to-side spread along y (p90 - p10)."""
    y = pts[:, 1]
    return float(np.percentile(y, 90) - np.percentile(y, 10))


def centroid_y(pts):
    """Mean y: ball-side shift."""
    return float(np.mean(pts[:, 1]))


def hull_area(pts):
    """Convex-hull area (compactness). 0.0 for <3 points or collinear points."""
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)  # 2D 'volume' is area
    except QhullError:
        return 0.0


def band_counts(pts):
    """Players per horizontal third [defensive, middle, attacking] by canonical x."""
    lo, hi = BAND_EDGES_X
    counts = [0, 0, 0]
    for xi in pts[:, 0]:
        if xi < lo:
            counts[0] += 1
        elif xi < hi:
            counts[1] += 1
        else:
            counts[2] += 1
    return counts


def lane_counts(pts):
    """Players per vertical lane [wing, half-space, center, half-space, wing] by y."""
    counts = [0, 0, 0, 0, 0]
    for yi in pts[:, 1]:
        idx = int(np.searchsorted(LANE_EDGES_Y, yi, side="right"))
        counts[idx] += 1
    return counts


def side_overload(pts):
    """Signed flank asymmetry in [-1, 1]: (lanes 0+1 - lanes 3+4) / total."""
    lc = lane_counts(pts)
    total = sum(lc)
    if total == 0:
        return 0.0
    return float((lc[0] + lc[1] - lc[3] - lc[4]) / total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/metrics.py tests/test_metrics.py
git commit -m "feat(topo): clustering-free metrics and lane/band bucketing"
```

---

## Task 4: Simplified depth-line split → inter-line gaps

**Files:**
- Create: `formation_topology/lines.py`
- Test: `tests/test_lines.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lines.py
from formation_topology.lines import detect_lines


def test_three_clear_lines():
    # back ~ -40, mid ~ -10, front ~ +20 (canonical, attack +x); gaps ~30m each
    xs = [-41, -40, -39, -40, -11, -10, -9, 19, 20, 21]
    count, gaps = detect_lines(xs, gap_delta=7.0)
    assert count == 3
    assert len(gaps) == 2
    assert 28 < gaps[0] < 32   # D-M
    assert 28 < gaps[1] < 32   # M-F


def test_two_lines():
    xs = [-40, -41, -39, 5, 6, 4]
    count, gaps = detect_lines(xs, gap_delta=7.0)
    assert count == 2
    assert len(gaps) == 1
    assert 43 < gaps[0] < 47   # ~45m


def test_no_significant_gap_falls_back():
    xs = [-5, -4, -3, -2, -1, 0, 1, 2]
    count, gaps = detect_lines(xs, gap_delta=7.0)
    assert count == 1
    assert gaps == []


def test_too_few_points():
    assert detect_lines([3.0], gap_delta=7.0) == (1, [])
    assert detect_lines([], gap_delta=7.0) == (0, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.lines'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/lines.py
import numpy as np


def detect_lines(xs, gap_delta=7.0, max_lines=3):
    """Split players into depth lines by their largest x-gaps.

    Returns (line_count, inter_line_gaps_m):
      - sort xs, look at gaps between consecutive players
      - take up to (max_lines - 1) largest gaps that exceed gap_delta as split points
      - inter_line_gaps = differences between adjacent line x-centroids, ordered by depth
    If no gap exceeds gap_delta (lines indistinguishable) or <2 players, no split is made
    and the caller should fall back to band occupancy.
    """
    xs = np.sort(np.asarray(xs, dtype=float))
    n = len(xs)
    if n == 0:
        return 0, []
    if n == 1:
        return 1, []

    gaps = np.diff(xs)
    candidates = [(g, i) for i, g in enumerate(gaps) if g > gap_delta]
    if not candidates:
        return 1, []

    candidates.sort(reverse=True)                       # largest gaps first
    n_splits = min(len(candidates), max_lines - 1)
    split_idx = sorted(i for _, i in candidates[:n_splits])

    groups, start = [], 0
    for si in split_idx:
        groups.append(xs[start:si + 1])
        start = si + 1
    groups.append(xs[start:])

    centroids = [float(np.mean(g)) for g in groups]
    inter = [round(centroids[k + 1] - centroids[k], 2) for k in range(len(centroids) - 1)]
    return len(groups), inter
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lines.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/lines.py tests/test_lines.py
git commit -m "feat(topo): largest-gap depth-line split with inter-line gaps"
```

---

## Task 5: Sliding windows + per-track median position

**Files:**
- Create: `formation_topology/windows.py`
- Test: `tests/test_windows.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_windows.py
from formation_topology.io_gamestate import Detection
from formation_topology.windows import sliding_windows


def _d(frame, tid, team, role, x, y):
    return Detection(frame=frame, track_id=tid, team=team, role=role,
                     jersey_number=None, x=x, y=y)


def test_window_groups_by_team_and_medians_tracks():
    dets = [
        _d(1, 7, "left", "player", -40.0, 0.0),
        _d(2, 7, "left", "player", -42.0, 0.0),    # track 7 median x = -41
        _d(1, 8, "left", "player", -30.0, 10.0),
        _d(1, 1, "right", "player", 40.0, 0.0),
        _d(1, 50, "left", "goalkeeper", -50.0, 0.0),  # GK excluded from team point set
        _d(1, 99, None, "ball", 0.0, 0.0),
    ]
    wins = sliding_windows(dets, fps=10.0, win_s=2.0, stride_s=2.0)
    assert len(wins) == 1
    w0 = wins[0]
    assert set(w0["teams"].keys()) == {"left", "right"}
    left = w0["teams"]["left"]
    assert len(left["players"]) == 2  # GK not counted
    idx = left["track_ids"].index(7)
    assert abs(left["players"][idx][0] + 41.0) < 1e-6
    assert w0["ball"] is not None and abs(w0["ball"][0]) < 1e-6


def test_in_play_filter():
    dets = [
        _d(1, 7, "left", "player", -40.0, 0.0),
        _d(2, 7, "left", "player", -40.0, 0.0),
    ]
    wins = sliding_windows(dets, fps=10.0, win_s=2.0, stride_s=2.0, in_play_frames={1})
    # frame 2 dropped; track 7 still present from frame 1
    assert len(wins[0]["teams"]["left"]["players"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_windows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.windows'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/windows.py
import numpy as np
from collections import defaultdict
from typing import List, Optional, Set

from .io_gamestate import Detection


def sliding_windows(detections: List[Detection], fps: float,
                    win_s: float = 3.0, stride_s: float = 1.0,
                    in_play_frames: Optional[Set[int]] = None):
    """Build sliding windows of per-track median positions.

    Returns a list of dicts:
      {"t_start", "t_end",
       "teams": {team: {"players": ndarray(N, 2), "track_ids": [int, ...]}},
       "ball": ndarray(2,) | None}

    Only role == 'player' goes into team point sets (GK excluded). role == 'ball'
    -> ball position. Each track's representative point is its component-wise median
    over the window's in-play frames. Always emits >= 1 window; no trailing partial.
    """
    if not detections:
        return []
    if in_play_frames is not None:
        detections = [d for d in detections if d.frame in in_play_frames]
    if not detections:
        return []

    max_t = max(d.frame for d in detections) / fps

    windows = []
    t = 0.0
    while True:
        t_end = t + win_s
        lo, hi = t * fps, t_end * fps
        in_win = [d for d in detections if lo <= d.frame <= hi]

        by_team_track = defaultdict(lambda: defaultdict(list))
        ball_pts = []
        for d in in_win:
            if d.role == "ball":
                ball_pts.append((d.x, d.y))
            elif d.role == "player" and d.team in ("left", "right"):
                by_team_track[d.team][d.track_id].append((d.x, d.y))

        teams = {}
        for team, tracks in by_team_track.items():
            pts, tids = [], []
            for tid, xy in tracks.items():
                pts.append(np.median(np.asarray(xy, dtype=float), axis=0))
                tids.append(tid)
            teams[team] = {"players": np.asarray(pts), "track_ids": tids}

        ball = np.median(np.asarray(ball_pts), axis=0) if ball_pts else None
        windows.append({"t_start": round(t, 3), "t_end": round(t_end, 3),
                        "teams": teams, "ball": ball})

        t += stride_s
        if t >= max_t:
            break
    return windows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_windows.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/windows.py tests/test_windows.py
git commit -m "feat(topo): sliding windows with per-track median positions"
```

---

## Task 6: Per-window possession tag

**Files:**
- Create: `formation_topology/possession.py`
- Test: `tests/test_possession.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_possession.py
import numpy as np
from formation_topology.possession import possession_team


def test_possession_nearest_player_to_ball():
    w = {"ball": np.array([39.0, 1.0]),
         "teams": {
             "left": {"players": np.array([[-40.0, 0.0], [-30.0, 5.0]]), "track_ids": [7, 8]},
             "right": {"players": np.array([[40.0, 0.0], [20.0, 2.0]]), "track_ids": [1, 2]},
         }}
    assert possession_team(w) == "right"


def test_possession_unknown_without_ball():
    w = {"ball": None,
         "teams": {"left": {"players": np.array([[0.0, 0.0]]), "track_ids": [7]}}}
    assert possession_team(w) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_possession.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.possession'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/possession.py
import numpy as np
from typing import Optional


def possession_team(window) -> Optional[str]:
    """Team of the player whose window-median position is nearest the ball.

    Returns 'left' | 'right', or None if there is no ball / no players in the window.
    Distance is computed in raw coordinates (rotation-invariant), so no canonicalization
    is needed here.
    """
    ball = window.get("ball")
    if ball is None:
        return None
    best_team, best_d = None, float("inf")
    for team, info in window["teams"].items():
        pts = info["players"]
        if len(pts) == 0:
            continue
        d = float(np.min(np.linalg.norm(pts - ball, axis=1)))
        if d < best_d:
            best_d, best_team = d, team
    return best_team
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_possession.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/possession.py tests/test_possession.py
git commit -m "feat(topo): per-window possession tag from nearest player to ball"
```

---

## Task 7: Pipeline — infer attack directions + analyze()

**Files:**
- Create: `formation_topology/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from formation_topology.io_gamestate import Detection
from formation_topology.pipeline import analyze, infer_attack_dirs


def _d(frame, tid, team, role, x, y):
    return Detection(frame=frame, track_id=tid, team=team, role=role,
                     jersey_number=None, x=x, y=y)


def _clip():
    """30 frames @ 10fps. Left = deep compact low block (3 lines), right pressing high.
    Ball sits next to a right player deep in left's half -> right in possession."""
    dets = []
    backs = [(-40, y) for y in (-20, -7, 7, 20)]
    mids = [(-30, y) for y in (-12, 0, 12)]
    fronts = [(-22, y) for y in (-6, 0, 6)]
    left_outfield = backs + mids + fronts  # 10 players
    for f in range(1, 31):
        dets.append(_d(f, 100, "left", "goalkeeper", -50.0, 0.0))   # left attacks +x
        for i, (x, y) in enumerate(left_outfield):
            dets.append(_d(f, i + 1, "left", "player", x, y))
        dets.append(_d(f, 200, "right", "goalkeeper", 50.0, 0.0))   # right attacks -x
        for i, (x, y) in enumerate([(-15, 0), (-5, 10), (-5, -10)]):
            dets.append(_d(f, 300 + i, "right", "player", x, y))
        dets.append(_d(f, 99, None, "ball", -16.0, 0.0))
    return dets


def test_attack_dirs_from_gk():
    dirs = infer_attack_dirs(_clip())
    assert dirs["left"] == 1 and dirs["right"] == -1


def test_low_block_record():
    recs = analyze(_clip(), fps=10.0, win_s=3.0, stride_s=3.0, min_coverage=8, gap_delta=6.0)
    left = [r for r in recs if r["team"] == "left"]
    assert left, "expected a left-team record"
    r = left[0]
    assert r["coverage_n"] == 10
    assert r["block_height_m"] < -25       # deep low block in canonical frame
    assert r["line_count"] == 3            # back/mid/front separated by largest gaps
    assert len(r["inter_line_gaps_m"]) == 2
    assert r["possession_tag"] == "out"    # ball is nearest a right player
    assert r["low_confidence"] is False


def test_below_coverage_skips_lines():
    dets = [
        _d(1, 100, "left", "goalkeeper", -50.0, 0.0),
        _d(1, 1, "left", "player", -30.0, 0.0),
        _d(2, 1, "left", "player", -30.0, 0.0),
    ]
    recs = analyze(dets, fps=10.0, win_s=3.0, stride_s=3.0, min_coverage=8)
    left = [r for r in recs if r["team"] == "left"]
    assert left[0]["line_count"] is None
    assert left[0]["low_confidence"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/pipeline.py
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .io_gamestate import Detection
from .pitch import canonicalize
from . import metrics
from .lines import detect_lines
from .windows import sliding_windows
from .possession import possession_team

MIN_COVERAGE = 8


def infer_attack_dirs(detections: List[Detection]) -> Dict[str, int]:
    """Per team, attack_dir = -sign(mean GK x).

    A GK at x > 0 defends the right goal, so that team attacks -x (dir -1). When a team
    has no GK detections, fall back to the side convention (left: +1, right: -1).
    """
    gk_x = defaultdict(list)
    for d in detections:
        if d.role == "goalkeeper" and d.team in ("left", "right"):
            gk_x[d.team].append(d.x)
    dirs = {}
    for team in ("left", "right"):
        if gk_x[team]:
            dirs[team] = -1 if float(np.mean(gk_x[team])) > 0 else 1
        else:
            dirs[team] = 1 if team == "left" else -1
    return dirs


def analyze(detections: List[Detection], fps: float,
            win_s: float = 3.0, stride_s: float = 1.0,
            min_coverage: int = MIN_COVERAGE, gap_delta: float = 7.0,
            in_play_frames: Optional[Set[int]] = None) -> List[dict]:
    """Produce one record per (window, team) on the visible outfield players.

    Clustering-free metrics are always emitted. Depth lines (and inter-line gaps) are
    attempted only when coverage_n >= min_coverage; otherwise line fields are null and
    low_confidence is True.
    """
    dirs = infer_attack_dirs(detections)
    windows = sliding_windows(detections, fps, win_s, stride_s, in_play_frames)

    records: List[dict] = []
    for w in windows:
        poss = possession_team(w)
        for team, info in w["teams"].items():
            raw = info["players"]
            if len(raw) == 0:
                continue
            pts = canonicalize(raw, dirs[team])
            coverage_n = len(pts)

            tag = "in" if poss == team else ("unknown" if poss is None else "out")
            rec = {
                "t_start": w["t_start"], "t_end": w["t_end"], "team": team,
                "possession_tag": tag,
                "block_height_m": round(metrics.block_height(pts), 2),
                "block_depth_m": round(metrics.block_depth(pts), 2),
                "block_width_m": round(metrics.block_width(pts), 2),
                "hull_area_m2": round(metrics.hull_area(pts), 1),
                "centroid_y_m": round(metrics.centroid_y(pts), 2),
                "lane_counts": metrics.lane_counts(pts),
                "band_counts": metrics.band_counts(pts),
                "side_overload": round(metrics.side_overload(pts), 3),
                "coverage_n": coverage_n,
            }
            if coverage_n >= min_coverage:
                lc, gaps = detect_lines(pts[:, 0], gap_delta=gap_delta)
                rec["line_count"] = lc
                rec["inter_line_gaps_m"] = gaps
                rec["low_confidence"] = lc < 2        # couldn't separate distinct lines
            else:
                rec["line_count"] = None
                rec["inter_line_gaps_m"] = []
                rec["low_confidence"] = True
            records.append(rec)
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add formation_topology/pipeline.py tests/test_pipeline.py
git commit -m "feat(topo): pipeline analyze() + GK-based attack-direction inference"
```

---

## Task 8: CLI runner

**Files:**
- Create: `formation_topology/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import json
from formation_topology.cli import main


def test_cli_writes_records(tmp_path):
    ann = []
    for f in range(1, 11):
        ann.append({"id": f, "image_id": f, "track_id": 7,
                    "attributes": {"role": "player", "team": "left"},
                    "bbox_pitch": {"x_bottom_middle": -30.0, "y_bottom_middle": 0.0}})
    inp = tmp_path / "Labels-GameState.json"
    out = tmp_path / "topo.json"
    inp.write_text(json.dumps({"annotations": ann}))

    main(["--input", str(inp), "--output", str(out),
          "--fps", "10", "--win", "3", "--stride", "3"])

    recs = json.loads(out.read_text())
    assert isinstance(recs, list) and len(recs) >= 1
    assert recs[0]["team"] == "left"
    assert recs[0]["coverage_n"] == 1
    assert recs[0]["line_count"] is None  # below default min-coverage -> no lines
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'formation_topology.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# formation_topology/cli.py
import argparse
import json

from .io_gamestate import load_detections
from .pipeline import analyze


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Formation topology V0: per-window, per-team team-shape signal.")
    p.add_argument("--input", required=True, help="Labels-GameState.json path")
    p.add_argument("--output", required=True, help="output records JSON path")
    p.add_argument("--fps", type=float, required=True, help="video frame rate")
    p.add_argument("--win", type=float, default=3.0, help="window length (s)")
    p.add_argument("--stride", type=float, default=1.0, help="window stride (s)")
    p.add_argument("--min-coverage", type=int, default=8,
                   help="min visible outfielders to attempt line split")
    p.add_argument("--gap-delta", type=float, default=7.0,
                   help="min x-gap (m) to count as a line boundary")
    args = p.parse_args(argv)

    dets = load_detections(args.input)
    recs = analyze(dets, fps=args.fps, win_s=args.win, stride_s=args.stride,
                   min_coverage=args.min_coverage, gap_delta=args.gap_delta)
    with open(args.output, "w") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(recs)} records to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite + commit**

```bash
python -m pytest tests/ -v
git add formation_topology/cli.py tests/test_cli.py
git commit -m "feat(topo): CLI runner writing per-window records JSON"
```
Expected: all tests pass (21 passed).

---

## Validation (after the suite is green)

Run on a real in-play segment, not the kickoff clip:

```bash
python -m formation_topology.cli \
  --input /path/to/Labels-GameState.json --output topo.json \
  --fps 25 --win 3 --stride 1
```

Contrastive check (no formation ground truth exists): cut one "team sitting in a settled low block" segment and one "team pressing high / building up" segment from the NAS World Cup footage. A correct module shows clearly divergent `block_height_m`, `hull_area_m2`, and `block_depth_m` between them (low/small/compact vs high/large/stretched). Eyeball the per-window records against the existing minimap. Do not validate on the 30s kickoff clip — no settled block has formed there.

---

## Self-Review (completed during planning)

- **Spec coverage:** anchor A / visible-team-only / canonical frame + possession-decoupled (i) → Tasks 1, 5, 7; input contract (bbox_pitch, role filter, team left/right) → Task 2; the football "four reads" + 5×3 occupancy → Task 3; simplified largest-gap line split + inter-line gaps + fallback → Task 4; coverage gate + low-confidence → Task 7; possession tag → Task 6; output schema → Task 7; CLI/integration → Task 8; validation → Validation section. No formation-label ("4-3-3") output is produced, per the locked decision.
- **Placeholder scan:** none — every code step is complete and runnable.
- **Sandbox verification:** the full package + all 8 test files were built from this plan verbatim and run — **21 passed** (numpy 2.4, scipy 1.17, pytest 9.1, Python 3.12). An end-to-end synthetic low-block clip produced a sensible record (`band_counts [10,0,0]`, `block_height -32m`, `inter_line_gaps [8.0, 7.0]`, `possession out`). Codex should reproduce green before touching real data.
- **Type consistency:** `Detection(frame, track_id, team, role, jersey_number, x, y)`, window dict `{t_start, t_end, teams{team:{players, track_ids}}, ball}`, `canonicalize(pts, attack_dir)`, `detect_lines(xs, gap_delta, max_lines) -> (count, gaps)`, and `analyze(...)` record keys are used identically across Tasks 2–8.
- **Open integration flag (not a code placeholder):** Task 2 assumes the GSR top-level key `annotations` and attribute names `role`/`team`/`jersey`. Confirm against your actual export (the same file `pitch_distances.py` reads); adjust only `load_detections` if they differ.

---

## Execution Handoff

Plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
2. **Inline Execution** — execute tasks in this session via superpowers:executing-plans, batch execution with checkpoints.

Which approach?
