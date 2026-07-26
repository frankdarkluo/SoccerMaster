# Tactical Commentary v2 (State-to-Reasoning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the typed-event commentary path with a "code measures, LLM reasons" pipeline: relations.json + radar frames + tactical glossary → two-pass doubao narration (narrate → self-ask verify) → commentary.json consumed by the unchanged Stage-5 TTS.

**Architecture:** New `pipeline/relations/` package computes per-snapshot relational measurements from `predictions.json` (reusing `stage2_events.detector.load_frames`, `stage2_events.possession`, `pipeline/topology`). New `pipeline/tactics/` package holds the concept glossary (YAML), the Pass-1 narrate prompt, and the Pass-2 GameSight-style verifier. A new `run_stage2b` orchestrates: relations → radar → narrate → verify → `commentary.json` (existing schema, so Stage 5 needs zero changes). `classify.py`/old stage-4 prompt path stay in the repo but are not on this path.

**Tech Stack:** Python 3.10, numpy, PIL (Pillow), PyYAML, pytest, existing `DoubaoAPIAdapter` (OpenAI-compatible ARK client), existing `postprocess.parse_commentary_output` / `write_commentary_json`.

**Spec:** `docs/superpowers/specs/2026-07-09-tactical-commentary-v2-design.md`

---

## File Structure

```
pipeline/relations/
    __init__.py
    kinematics.py     # per-track smoothed position/velocity series
    snapshots.py      # per-snapshot relational measurements (the core)
    build.py          # orchestrate → relations.json (+ CLI)
    radar.py          # render 1Hz top-view PNG frames
pipeline/tactics/
    __init__.py
    concepts.yaml     # tactical concept glossary (v1: 15 entries)
    kb.py             # load/validate glossary, render prompt block
    narrate.py        # Pass 1: prompt build + LLM call → draft segments
    verify.py         # Pass 2: self-ask queries → mechanical resolve → rewrite
pipeline/run.py       # add run_stage2b()
pipeline/config.py    # add commentary_mode + paths + knobs
pipeline/stage4_commentary/adapters/doubao_api.py  # image cap configurable
scripts/run_stage2b.sh
tests/
    conftest.py
    relations/test_kinematics.py
    relations/test_snapshots.py
    relations/test_build.py
    relations/test_radar.py
    tactics/test_kb.py
    tactics/test_narrate.py
    tactics/test_verify.py
    test_stage2b_e2e.py
```

Conventions used throughout: pitch coords are meters, origin at pitch center, x ∈ [-52.5, 52.5], y ∈ [-34, 34] (matches `bbox_pitch`). `attack_dir[team] ∈ {+1, -1}` = sign of x-direction that team attacks (from `pipeline.topology.analysis.infer_attack_dirs`). All "relational" quantities are signed by the *player's own team's* attack direction, so "+" always means "toward the goal they attack".

**Test environment note:** run tests in the conda env (`tracklab_release`) on the machine with the repo. If pytest is missing: `pip install pytest`. All non-LLM tests run offline; LLM tests use a fake adapter injected in the test.

---

### Task 1: Scaffold packages and test bed

**Files:**
- Create: `pipeline/relations/__init__.py`, `pipeline/tactics/__init__.py`
- Create: `tests/conftest.py`, `tests/relations/__init__.py`, `tests/tactics/__init__.py`

- [ ] **Step 1: Create empty package inits**

`pipeline/relations/__init__.py` and `pipeline/tactics/__init__.py`, each:

```python
```

(empty file)

- [ ] **Step 2: Create tests/conftest.py with a synthetic-frames factory**

```python
"""Shared fixtures: build synthetic FrameData sequences for unit tests."""
from __future__ import annotations

import pytest

from pipeline.stage2_events.types import FrameData


def make_frames(specs, n_frames, fps=25.0, ball=None):
    """Build FrameData list from per-player motion specs.

    specs: list of dicts:
        {"track_id": 7, "team": "left", "jersey": "7", "role": "player",
         "start": (x0, y0), "vel": (vx, vy)}   # vel in m/s
    ball: optional {"start": (x, y), "vel": (vx, vy)}
    """
    frames = []
    for fid in range(1, n_frames + 1):
        t = (fid - 1) / fps
        players = []
        for s in specs:
            players.append({
                "track_id": s["track_id"],
                "x": s["start"][0] + s["vel"][0] * t,
                "y": s["start"][1] + s["vel"][1] * t,
                "role": s.get("role", "player"),
                "team": s.get("team"),
                "jersey": s.get("jersey", ""),
            })
        frame = FrameData(frame_id=fid, players=players)
        if ball is not None:
            frame.ball_xy = (
                ball["start"][0] + ball["vel"][0] * t,
                ball["start"][1] + ball["vel"][1] * t,
            )
        frames.append(frame)
    return frames


@pytest.fixture
def frames_factory():
    return make_frames
```

- [ ] **Step 3: Verify pytest collects (no tests yet, expect "no tests ran")**

Run: `cd <repo-root> && python -m pytest tests/ -q`
Expected: `no tests ran` (exit code 5 is fine at this point)

- [ ] **Step 4: Commit**

```bash
git add pipeline/relations/__init__.py pipeline/tactics/__init__.py tests/
git commit -m "chore: scaffold relations/tactics packages and test bed"
```

---

### Task 2: kinematics.py — smoothed per-track series

**Files:**
- Create: `pipeline/relations/kinematics.py`
- Test: `tests/relations/test_kinematics.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tests.conftest import make_frames
from pipeline.relations.kinematics import build_tracks


def test_constant_velocity_track_recovers_speed():
    # player moving +x at 6 m/s for 2 s @25fps
    frames = make_frames(
        [{"track_id": 7, "team": "left", "jersey": "7",
          "start": (0.0, 0.0), "vel": (6.0, 0.0)}],
        n_frames=50,
    )
    tracks = build_tracks(frames, fps=25.0, smooth_s=0.5)
    tr = tracks[7]
    # after warmup, speed ~6 m/s, vy ~0
    assert tr.speed[-1] == pytest.approx(6.0, abs=0.5)
    assert tr.vy[-1] == pytest.approx(0.0, abs=0.3)
    assert tr.team == "left"
    assert tr.jersey == "7"


def test_track_with_gaps_reports_coverage():
    frames = make_frames(
        [{"track_id": 3, "team": "right", "jersey": "3",
          "start": (10.0, 5.0), "vel": (0.0, 0.0)}],
        n_frames=40,
    )
    # remove player from 30 of 40 frames -> coverage 0.25
    for f in frames[10:]:
        f.players = []
    tracks = build_tracks(frames, fps=25.0)
    assert tracks[3].coverage == pytest.approx(0.25, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/relations/test_kinematics.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.relations.kinematics`

- [ ] **Step 3: Implement pipeline/relations/kinematics.py**

```python
"""Per-track smoothed position/velocity series from FrameData.

Measurements only. Velocities are EMA-smoothed central differences; the EMA
horizon (smooth_s) suppresses GSR jitter without inventing anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pipeline.stage2_events.possession import resolve_role_by_track, resolve_team_by_track
from pipeline.stage2_events.types import FrameData


@dataclass
class TrackSeries:
    track_id: int
    team: Optional[str]
    role: str
    jersey: str
    fids: List[int] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    coverage: float = 0.0  # fraction of frames this track appears in

    def at(self, fid: int) -> Optional[int]:
        """Index of the sample at frame id `fid`, or None."""
        try:
            return self.fids.index(fid)
        except ValueError:
            return None


def _majority_jersey(frames: List[FrameData], track_id: int) -> str:
    from collections import Counter
    votes = Counter()
    for fr in frames:
        for p in fr.players:
            if p.get("track_id") == track_id and p.get("jersey"):
                votes[p["jersey"]] += 1
    return votes.most_common(1)[0][0] if votes else ""


def build_tracks(frames: List[FrameData], fps: float, smooth_s: float = 0.5) -> Dict[int, TrackSeries]:
    teams = resolve_team_by_track(frames)
    roles = resolve_role_by_track(frames)
    n_frames = len(frames)

    raw: Dict[int, List[tuple]] = {}
    for fr in frames:
        for p in fr.players:
            tid = p.get("track_id")
            if tid is None:
                continue
            raw.setdefault(int(tid), []).append((fr.frame_id, p["x"], p["y"]))

    alpha = 1.0 - pow(0.5, 1.0 / max(1.0, smooth_s * fps))  # EMA half-life = smooth_s
    tracks: Dict[int, TrackSeries] = {}
    for tid, samples in raw.items():
        samples.sort()
        tr = TrackSeries(
            track_id=tid,
            team=teams.get(tid),
            role=roles.get(tid, "player"),
            jersey=_majority_jersey(frames, tid),
        )
        ema_vx = ema_vy = 0.0
        for i, (fid, x, y) in enumerate(samples):
            tr.fids.append(fid)
            tr.x.append(x)
            tr.y.append(y)
            if i == 0:
                inst_vx = inst_vy = 0.0
            else:
                pf, px, py = samples[i - 1]
                dt = max(1, fid - pf) / fps
                inst_vx = (x - px) / dt
                inst_vy = (y - py) / dt
            ema_vx = alpha * inst_vx + (1 - alpha) * ema_vx
            ema_vy = alpha * inst_vy + (1 - alpha) * ema_vy
            tr.vx.append(ema_vx)
            tr.vy.append(ema_vy)
            tr.speed.append((ema_vx ** 2 + ema_vy ** 2) ** 0.5)
        tr.coverage = len(samples) / max(1, n_frames)
        tracks[tid] = tr
    return tracks


def ball_series(frames: List[FrameData], fps: float, smooth_s: float = 0.2):
    """(fids, x, y, speed) for the ball. Shorter smoothing: kicks are fast."""
    fids, xs, ys, speeds = [], [], [], []
    alpha = 1.0 - pow(0.5, 1.0 / max(1.0, smooth_s * fps))
    ema = 0.0
    prev = None
    for fr in frames:
        if fr.ball_xy is None:
            continue
        x, y = fr.ball_xy
        if prev is not None:
            pf, px, py = prev
            dt = max(1, fr.frame_id - pf) / fps
            inst = (((x - px) ** 2 + (y - py) ** 2) ** 0.5) / dt
        else:
            inst = 0.0
        ema = alpha * inst + (1 - alpha) * ema
        fids.append(fr.frame_id)
        xs.append(x)
        ys.append(y)
        speeds.append(ema)
        prev = (fr.frame_id, x, y)
    return fids, xs, ys, speeds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/relations/test_kinematics.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/relations/kinematics.py tests/relations/test_kinematics.py
git commit -m "feat(relations): per-track smoothed kinematics from FrameData"
```

---

### Task 3: snapshots.py — relational measurements (the core)

**Files:**
- Create: `pipeline/relations/snapshots.py`
- Test: `tests/relations/test_snapshots.py`

Quantities per snapshot (all measurements, no labels):
ball x/y/speed; carrier (nearest same-team player within 3m, hysteresis left to
possession segments in build.py — here carrier = nearest player within radius);
per qualifying player (coverage ≥ 0.8, role player/goalkeeper): x, y, speed,
`rel_x`/`rel_y` vs carrier signed by own attack_dir, `dist_ball`,
`dist_nearest_opp`, `depth_vs_line` (x-distance past opponents' second-last
defender, signed by own attack_dir); per team: defensive `line_x`, and
`n_within_15m_of_ball`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tests.conftest import make_frames
from pipeline.relations.kinematics import build_tracks
from pipeline.relations.snapshots import build_snapshots


def _two_team_frames():
    # left attacks +x (all left players moving/positioned toward +x)
    specs = [
        # carrier: left #10 at (0,0) holding ball
        {"track_id": 1, "team": "left", "jersey": "10", "start": (0.0, 0.0), "vel": (2.0, 0.0)},
        # left #7 wide right, ahead of carrier (candidate overlap geometry)
        {"track_id": 2, "team": "left", "jersey": "7", "start": (5.0, 20.0), "vel": (7.0, 0.0)},
        # right defenders
        {"track_id": 3, "team": "right", "jersey": "4", "start": (15.0, 0.0), "vel": (0.0, 0.0)},
        {"track_id": 4, "team": "right", "jersey": "5", "start": (20.0, 5.0), "vel": (0.0, 0.0)},
        {"track_id": 5, "team": "right", "jersey": "1", "start": (50.0, 0.0), "vel": (0.0, 0.0),
         "role": "goalkeeper"},
    ]
    ball = {"start": (0.3, 0.0), "vel": (2.0, 0.0)}
    return make_frames(specs, n_frames=100, ball=ball)


def test_snapshot_relational_quantities():
    frames = _two_team_frames()
    tracks = build_tracks(frames, fps=25.0)
    snaps = build_snapshots(frames, tracks, fps=25.0, hz=2.0)
    assert len(snaps) == pytest.approx(8, abs=1)  # 4 s clip @2Hz

    s = snaps[-1]  # late snapshot, EMA warmed up
    assert s["ball"]["speed"] == pytest.approx(2.0, abs=0.5)
    assert s["carrier"]["jersey"] == "10"

    p7 = next(p for p in s["players"] if p["jersey"] == "7")
    # #7 ahead of carrier toward +x goal -> rel_x > 0, wide -> |rel_y| > 10
    assert p7["rel_x"] > 0
    assert abs(p7["rel_y"]) > 10
    # nearest opponent to #7 is right #5 at (20,5)
    assert p7["dist_nearest_opp"] == pytest.approx(
        ((p7["x"] - 20.0) ** 2 + (p7["y"] - 5.0) ** 2) ** 0.5, abs=0.5)

    team = s["teams"]["left"]
    # right's second-last defender (excluding GK at x=50): x=20 -> line_x=20
    assert team["opp_line_x"] == pytest.approx(20.0, abs=0.5)


def test_low_coverage_player_excluded():
    frames = _two_team_frames()
    # cripple track 2's visibility
    for fr in frames[10:]:
        fr.players = [p for p in fr.players if p["track_id"] != 2]
    tracks = build_tracks(frames, fps=25.0)
    snaps = build_snapshots(frames, tracks, fps=25.0, hz=2.0)
    jerseys = {p["jersey"] for p in snaps[-1]["players"]}
    assert "7" not in jerseys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/relations/test_snapshots.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.relations.snapshots`

- [ ] **Step 3: Implement pipeline/relations/snapshots.py**

```python
"""Per-snapshot relational measurements. Measurements only — no tactic labels."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from pipeline.relations.kinematics import TrackSeries, ball_series
from pipeline.stage2_events.types import FrameData
from pipeline.topology.analysis import infer_attack_dirs
from pipeline.topology.io_gamestate import Detection

CARRIER_RADIUS_M = 3.0
LOCAL_RADIUS_M = 15.0
MIN_COVERAGE = 0.8


def _detections_for_attack_dir(frames: List[FrameData]) -> List[Detection]:
    dets = []
    for fr in frames:
        for p in fr.players:
            if p.get("team") in ("left", "right"):
                dets.append(Detection(
                    frame=fr.frame_id, track_id=p.get("track_id") or -1,
                    x=p["x"], y=p["y"], team=p["team"],
                    role=p.get("role", "player"), jersey=p.get("jersey", ""),
                ))
    return dets


def _sample_track(tr: TrackSeries, fid: int) -> Optional[dict]:
    """Nearest stored sample within 3 frames of fid."""
    best, best_d = None, 4
    for i, f in enumerate(tr.fids):
        d = abs(f - fid)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    return {"x": tr.x[best], "y": tr.y[best], "speed": tr.speed[best]}


def build_snapshots(frames: List[FrameData], tracks: Dict[int, TrackSeries],
                    fps: float, hz: float = 2.0) -> List[dict]:
    attack_dirs = infer_attack_dirs(_detections_for_attack_dir(frames))
    b_fids, b_x, b_y, b_speed = ball_series(frames, fps)
    ball_by_fid = {f: (x, y, s) for f, x, y, s in zip(b_fids, b_x, b_y, b_speed)}

    qualifying = {
        tid: tr for tid, tr in tracks.items()
        if tr.coverage >= MIN_COVERAGE and tr.role in ("player", "goalkeeper")
        and tr.team in ("left", "right")
    }

    step = max(1, int(round(fps / hz)))
    snapshots = []
    for fid in range(1, frames[-1].frame_id + 1, step):
        ball = ball_by_fid.get(fid)
        if ball is None:  # tolerate short ball dropouts
            near = [f for f in ball_by_fid if abs(f - fid) <= 3]
            if not near:
                continue
            ball = ball_by_fid[min(near, key=lambda f: abs(f - fid))]
        bx, by, bs = ball

        samples = {}
        for tid, tr in qualifying.items():
            s = _sample_track(tr, fid)
            if s is not None:
                s.update(track_id=tid, team=tr.team, jersey=tr.jersey, role=tr.role)
                samples[tid] = s

        # carrier: nearest player to ball within radius
        carrier = None
        best = CARRIER_RADIUS_M
        for s in samples.values():
            d = math.hypot(s["x"] - bx, s["y"] - by)
            if d < best:
                best, carrier = d, s

        players_out = []
        teams_out = {}
        for team in ("left", "right"):
            adir = attack_dirs.get(team, 1)
            own = [s for s in samples.values() if s["team"] == team]
            opp = [s for s in samples.values() if s["team"] != team]
            opp_outfield = [s for s in opp if s["role"] != "goalkeeper"]
            xs = sorted((s["x"] * adir for s in opp_outfield), reverse=True)
            opp_line_x_signed = xs[1] if len(xs) >= 2 else (xs[0] if xs else None)
            teams_out[team] = {
                "attack_dir": adir,
                "opp_line_x": (opp_line_x_signed * adir) if opp_line_x_signed is not None else None,
                "n_within_15m_of_ball": sum(
                    1 for s in own if math.hypot(s["x"] - bx, s["y"] - by) <= LOCAL_RADIUS_M),
            }
            for s in own:
                row = {
                    "track_id": s["track_id"], "team": team, "jersey": s["jersey"],
                    "role": s["role"],
                    "x": round(s["x"], 1), "y": round(s["y"], 1),
                    "speed": round(s["speed"], 1),
                    "dist_ball": round(math.hypot(s["x"] - bx, s["y"] - by), 1),
                }
                if carrier is not None and s["track_id"] != carrier["track_id"]:
                    row["rel_x"] = round((s["x"] - carrier["x"]) * adir, 1)
                    row["rel_y"] = round(s["y"] - carrier["y"], 1)
                if opp_outfield:
                    row["dist_nearest_opp"] = round(min(
                        math.hypot(s["x"] - o["x"], s["y"] - o["y"]) for o in opp_outfield), 1)
                if opp_line_x_signed is not None:
                    row["depth_vs_line"] = round(s["x"] * adir - opp_line_x_signed, 1)
                players_out.append(row)

        snapshots.append({
            "t": round((fid - 1) / fps, 2),
            "frame_id": fid,
            "ball": {"x": round(bx, 1), "y": round(by, 1), "speed": round(bs, 1)},
            "carrier": ({"track_id": carrier["track_id"], "jersey": carrier["jersey"],
                         "team": carrier["team"]} if carrier else None),
            "players": players_out,
            "teams": teams_out,
        })
    return snapshots
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/relations/test_snapshots.py -q`
Expected: 2 passed. If `infer_attack_dirs` picks the wrong direction for the tiny
synthetic scene, pin it by moving all left players' start x to negative half
(the function infers from average positions) — adjust the fixture, not the code.

- [ ] **Step 5: Commit**

```bash
git add pipeline/relations/snapshots.py tests/relations/test_snapshots.py
git commit -m "feat(relations): per-snapshot relational measurements"
```

---

### Task 4: build.py — relations.json writer + CLI

**Files:**
- Create: `pipeline/relations/build.py`
- Test: `tests/relations/test_build.py`

- [ ] **Step 1: Write the failing test**

```python
import json

from tests.conftest import make_frames
from pipeline.relations.build import build_relations, write_relations_json


def test_build_and_write(tmp_path, monkeypatch):
    frames = make_frames(
        [{"track_id": 1, "team": "left", "jersey": "10", "start": (-10.0, 0.0), "vel": (2.0, 0.0)},
         {"track_id": 3, "team": "right", "jersey": "4", "start": (15.0, 0.0), "vel": (0.0, 0.0)},
         {"track_id": 4, "team": "right", "jersey": "5", "start": (20.0, 5.0), "vel": (0.0, 0.0)}],
        n_frames=100,
        ball={"start": (-9.7, 0.0), "vel": (2.0, 0.0)},
    )
    rel = build_relations(frames, fps=25.0, snapshot_hz=2.0)
    assert rel["video_info"]["duration_s"] == 4.0
    assert len(rel["snapshots"]) >= 6
    assert "kick_anchors" in rel

    out = tmp_path / "relations.json"
    write_relations_json(rel, out)
    loaded = json.loads(out.read_text())
    assert loaded["snapshots"][0]["ball"]["x"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/relations/test_build.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.relations.build`

- [ ] **Step 3: Implement pipeline/relations/build.py**

```python
"""Assemble relations.json: snapshots + kick anchors + meta. CLI included."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pipeline.relations.kinematics import ball_series, build_tracks
from pipeline.relations.snapshots import build_snapshots
from pipeline.stage2_events.types import FrameData

KICK_SPEED_MPS = 8.0  # same threshold detector.py uses


def _kick_anchors(frames: List[FrameData], fps: float) -> List[dict]:
    """Ball-speed spikes: t where smoothed speed crosses KICK_SPEED_MPS upward."""
    fids, _, _, speeds = ball_series(frames, fps)
    anchors = []
    above = False
    for fid, s in zip(fids, speeds):
        if s >= KICK_SPEED_MPS and not above:
            anchors.append({"t": round((fid - 1) / fps, 2), "ball_speed": round(s, 1)})
            above = True
        elif s < KICK_SPEED_MPS * 0.6:
            above = False
    return anchors


def build_relations(frames: List[FrameData], fps: float, snapshot_hz: float = 2.0) -> dict:
    tracks = build_tracks(frames, fps)
    snapshots = build_snapshots(frames, tracks, fps, hz=snapshot_hz)
    return {
        "schema_version": "relations-v1-20260709",
        "video_info": {
            "duration_s": round(frames[-1].frame_id / fps, 2),
            "fps": fps,
            "n_frames": frames[-1].frame_id,
        },
        "conventions": (
            "Pitch meters, origin center. rel_x/rel_y are player position minus "
            "carrier position, rel_x signed so + means toward the goal this "
            "player's team attacks. depth_vs_line + means beyond the opponents' "
            "second-last outfield defender. Only players tracked >=80% of the "
            "clip appear; absence of a player means NOT OBSERVED, never absence "
            "from the pitch."
        ),
        "snapshots": snapshots,
        "kick_anchors": _kick_anchors(frames, fps),
    }


def write_relations_json(relations: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(relations, ensure_ascii=False, indent=1), encoding="utf-8")
    return output_path


def main() -> None:
    import argparse
    from pipeline.stage2_events.detector import load_frames

    ap = argparse.ArgumentParser(description="predictions.json -> relations.json")
    ap.add_argument("--predictions-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--snapshot-hz", type=float, default=2.0)
    args = ap.parse_args()

    frames = load_frames(str(args.predictions_json))
    rel = build_relations(frames, fps=args.fps, snapshot_hz=args.snapshot_hz)
    write_relations_json(rel, args.out)
    print(f"wrote {args.out}: {len(rel['snapshots'])} snapshots, "
          f"{len(rel['kick_anchors'])} kick anchors")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/relations/test_build.py -q`
Expected: 1 passed

- [ ] **Step 5: Smoke on real data (manual, on the machine with outputs/)**

Run: `python -m pipeline.relations.build --predictions-json outputs/SNGS-116/predictions.json --out outputs/SNGS-116/relations.json`
Expected: ~60 snapshots, a handful of kick anchors. Open the file and eyeball
one snapshot: jersey numbers and positions should match the annotated video.

- [ ] **Step 6: Commit**

```bash
git add pipeline/relations/build.py tests/relations/test_build.py
git commit -m "feat(relations): relations.json builder with kick anchors and CLI"
```

---

### Task 5: radar.py — 1Hz top-view frames

**Files:**
- Create: `pipeline/relations/radar.py`
- Test: `tests/relations/test_radar.py`

- [ ] **Step 1: Write the failing test**

```python
from PIL import Image

from tests.conftest import make_frames
from pipeline.relations.build import build_relations
from pipeline.relations.radar import render_radar_frames


def test_renders_one_png_per_second(tmp_path):
    frames = make_frames(
        [{"track_id": 1, "team": "left", "jersey": "10", "start": (-10.0, 0.0), "vel": (2.0, 0.0)},
         {"track_id": 3, "team": "right", "jersey": "4", "start": (15.0, 0.0), "vel": (0.0, 0.0)}],
        n_frames=100,  # 4 s
        ball={"start": (-9.7, 0.0), "vel": (2.0, 0.0)},
    )
    rel = build_relations(frames, fps=25.0, snapshot_hz=2.0)
    paths = render_radar_frames(rel, tmp_path, hz=1.0)
    assert len(paths) == 4
    img = Image.open(paths[0])
    assert img.size == (630, 408)
    # pitch background green must dominate
    colors = img.convert("RGB").getcolors(maxcolors=100000)
    top = max(colors, key=lambda c: c[0])[1]
    assert top[1] > top[0] and top[1] > top[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/relations/test_radar.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.relations.radar`

- [ ] **Step 3: Implement pipeline/relations/radar.py**

```python
"""Render relations.json snapshots as top-view radar PNGs for the LLM."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from pipeline.config import PITCH_LENGTH, PITCH_WIDTH

SCALE = 6  # px per meter -> 630x408
W, H = int(PITCH_LENGTH * SCALE), int(PITCH_WIDTH * SCALE)
GREEN = (34, 120, 52)
LINE = (240, 240, 240)
TEAM_COLOR = {"left": (40, 90, 220), "right": (220, 60, 50)}
BALL = (250, 220, 40)


def _to_px(x: float, y: float):
    return (int((x + PITCH_LENGTH / 2) * SCALE), int((y + PITCH_WIDTH / 2) * SCALE))


def _draw_pitch(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, W - 1, H - 1], outline=LINE, width=2)
    d.line([W // 2, 0, W // 2, H], fill=LINE, width=2)
    r = int(9.15 * SCALE)
    d.ellipse([W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r], outline=LINE, width=2)
    for side in (-1, 1):
        x0, _ = _to_px(side * PITCH_LENGTH / 2, 0)
        depth = int(16.5 * SCALE) * (-side)
        half_w = int(40.32 / 2 * SCALE)
        d.rectangle(sorted([x0, x0 + depth]) + [H // 2 - half_w, H // 2 + half_w]
                    if False else [min(x0, x0 + depth), H // 2 - half_w,
                                   max(x0, x0 + depth), H // 2 + half_w],
                    outline=LINE, width=2)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_radar_frames(relations: dict, out_dir: Path, hz: float = 1.0) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(14)
    small = _font(11)

    snaps = relations["snapshots"]
    stride = max(1, int(round((1.0 / hz) / max(0.01, snaps[1]["t"] - snaps[0]["t"])))) \
        if len(snaps) > 1 else 1

    paths = []
    for snap in snaps[::stride]:
        img = Image.new("RGB", (W, H), GREEN)
        d = ImageDraw.Draw(img)
        _draw_pitch(d)
        for p in snap["players"]:
            cx, cy = _to_px(p["x"], p["y"])
            color = TEAM_COLOR.get(p["team"], (128, 128, 128))
            r = 10
            if p["role"] == "goalkeeper":
                d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=LINE)
            else:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=LINE)
            if p["jersey"]:
                d.text((cx, cy), p["jersey"], fill=(255, 255, 255),
                       font=small, anchor="mm")
        bx, by = _to_px(snap["ball"]["x"], snap["ball"]["y"])
        d.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=BALL, outline=(0, 0, 0))
        d.text((8, 6), f"t={snap['t']:.1f}s", fill=(255, 255, 60), font=font)
        path = out_dir / f"radar_{snap['t']:06.2f}.png"
        img.save(path)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/relations/test_radar.py -q`
Expected: 1 passed

- [ ] **Step 5: Manual visual check on real data**

```bash
python - <<'EOF'
import json
from pathlib import Path
from pipeline.relations.radar import render_radar_frames
rel = json.loads(Path("outputs/SNGS-116/relations.json").read_text())
print(render_radar_frames(rel, Path("outputs/SNGS-116/radar"), hz=1.0)[:3])
EOF
```
Open 2-3 PNGs: team colors separated, jersey numbers legible, ball visible.
Fix the penalty-box drawing if it looks wrong — the geometry helper is fiddly;
simplify to two rectangles if needed. Legibility > fidelity.

- [ ] **Step 6: Commit**

```bash
git add pipeline/relations/radar.py tests/relations/test_radar.py
git commit -m "feat(relations): radar frame renderer for LLM visual input"
```

---

### Task 6: concepts.yaml + kb.py — tactical glossary

**Files:**
- Create: `pipeline/tactics/concepts.yaml`
- Create: `pipeline/tactics/kb.py`
- Test: `tests/tactics/test_kb.py`

- [ ] **Step 1: Write concepts.yaml (v1 = 17 entries; Frank expands to 30+ later)**

```yaml
# Tactical concept glossary. The LLM may only name concepts from this file and
# must cite relations.json evidence for each. Definitions paraphrased from
# Wikipedia (zh/en) tactics articles; exemplars original.
version: 1
concepts:
  # ---- phases ----
  - id: positional_attack
    name_zh: 阵地战
    name_en: positional attack
    definition_zh: 进攻方长时间控球、对手落位防守后的有组织进攻，通过传导与跑位寻找空当。
    evidence_requirements: 持续同队控球（carrier同队且连续多个快照），对方防线line_x稳定，球速多数时间低于8m/s。
    exemplar_zh: 蓝队耐心传导，阵地战中寻找红队防线的缝隙。
    exemplar_en: The blues probe patiently in a settled positional attack.
  - id: counter_attack
    name_zh: 快速反击
    name_en: counter attack
    definition_zh: 由守转攻后立即向前推进，趁对方防线未落位时打击身后空间。
    evidence_requirements: carrier球权在两队间切换后数秒内，球和进攻方球员整体向前速度高，depth_vs_line快速减小或转正。
    exemplar_zh: 断球之后红队立刻提速，三人快速反击直扑腹地！
    exemplar_en: Turnover — and the reds break at pace!
  - id: transition
    name_zh: 攻防转换
    name_en: transition
    definition_zh: 球权刚易主、双方阵型尚未稳定的短暂阶段。
    evidence_requirements: carrier队伍切换后约3秒内，两队line_x与block形状明显移动。
    exemplar_zh: 球权转换瞬间，场上阵型还没有稳定下来。
    exemplar_en: A moment of transition — neither side is set.
  - id: high_press
    name_zh: 高位逼抢
    name_en: high press
    definition_zh: 无球方在对方半场就地围抢，压缩持球人出球空间。
    evidence_requirements: 无球方多名球员在对方半场（相对无球方attack_dir的前场）且靠近carrier（dist_ball小），carrier的dist_nearest_opp持续小于5m。
    exemplar_zh: 红队高位逼抢，蓝队后场出球非常艰难。
    exemplar_en: The reds press high, squeezing the buildup.
  - id: low_block
    name_zh: 低位防守
    name_en: low block
    definition_zh: 防守方整体退守己方半场，压缩身后空间。
    evidence_requirements: 防守方line_x深且稳定，多数防守球员x位于己方半场，块状紧凑。
    exemplar_zh: 蓝队摆出低位防线，不给身后留任何空间。
    exemplar_en: The blues sit in a deep, compact block.
  # ---- runs (off-ball) ----
  - id: overlap_run
    name_zh: 套边（套上）
    name_en: overlapping run
    definition_zh: 无球队员从持球队友身后沿边路外侧超越持球人，制造边路人数优势。
    evidence_requirements: 同队球员rel_y绝对值大（更靠边）、rel_x由负转正（从身后跑到身前）、speed高。
    exemplar_zh: 7号从外线套上，边路瞬间形成二打一！
    exemplar_en: Number 7 overlaps down the flank — two on one!
  - id: underlap_run
    name_zh: 内切前插（内线套上）
    name_en: underlapping run
    definition_zh: 无球队员从持球人内侧向前插上，攻击肋部空间。
    evidence_requirements: 同队球员|rel_y|小于持球人到边线距离（更靠内）、rel_x由负转正、speed高。
    exemplar_zh: 8号从肋部内线前插，直接冲击防线结合部。
    exemplar_en: Number 8 darts through the half-space.
  - id: depth_run
    name_zh: 前插身后（打身后）
    name_en: run in behind
    definition_zh: 进攻队员冲击对方防线身后的纵深空间。
    evidence_requirements: depth_vs_line由负转正或接近0且speed高、方向向前。
    exemplar_zh: 9号突然启动打身后，防线被彻底拉开！
    exemplar_en: Number 9 bursts in behind the last line!
  - id: pull_wide
    name_zh: 拉边
    name_en: pulling wide
    definition_zh: 进攻队员移动到边路拉开场地宽度，牵扯防守。
    evidence_requirements: |rel_y|或|y|增大到边路区域（|y|>20），横向速度分量明显。
    exemplar_zh: 11号主动拉边，把防线的宽度拉扯开来。
    exemplar_en: Number 11 stretches the pitch wide.
  - id: drop_deep
    name_zh: 回撤接应
    name_en: dropping deep
    definition_zh: 前场队员回撤到持球人附近接应，形成出球点。
    evidence_requirements: rel_x由正变小或为负、向carrier移动、dist_ball减小。
    exemplar_zh: 10号回撤到中场接应，帮助球队稳住节奏。
    exemplar_en: Number 10 drops in to link the play.
  # ---- space / numbers ----
  - id: numerical_superiority
    name_zh: 局部人数优势
    name_en: local numerical superiority
    definition_zh: 球周围区域一方人数多于另一方，形成以多打少。
    evidence_requirements: teams.*.n_within_15m_of_ball差值≥1。
    exemplar_zh: 球周围蓝队形成4打3的人数优势。
    exemplar_en: The blues have a four-v-three around the ball.
  - id: overload_side
    name_zh: 边路人数堆积（强侧堆积）
    name_en: overloading one side
    definition_zh: 进攻方在一侧集中多名球员制造优势，为弱侧转移或强侧渗透做准备。
    evidence_requirements: 多名同队球员y同号且|y|大，或topo.json的side_overload绝对值大。
    exemplar_zh: 蓝队在右路堆积兵力，弱侧留下大片空当。
    exemplar_en: The blues overload the right, leaving the far side open.
  - id: line_break
    name_zh: 穿透防线的传球（穿线球）
    name_en: line-breaking pass
    definition_zh: 传球越过对方一条防守线，使接球人在防线身后拿球。
    evidence_requirements: kick_anchor附近，接球人depth_vs_line大于传球时刻持球人的depth_vs_line且跨越0。
    exemplar_zh: 一脚穿透中场线的直塞，接球人瞬间面对防线！
    exemplar_en: A pass slices through the midfield line!
  - id: press_resistance
    name_zh: 摆脱逼抢
    name_en: press resistance
    definition_zh: 持球人在贴身压迫下通过盘带或出球保住球权并推进。
    evidence_requirements: carrier的dist_nearest_opp持续小于3m期间控球未易主且x向前推进。
    exemplar_zh: 6号背身扛住逼抢，从容分边。
    exemplar_en: Number 6 shrugs off the press and plays out.
  - id: switch_play
    name_zh: 弱侧转移（大范围转移）
    name_en: switch of play
    definition_zh: 用长传把球从一侧快速转移到防守薄弱的另一侧。
    evidence_requirements: kick_anchor伴随ball y坐标短时间内大幅变号/变化（>25m）。
    exemplar_zh: 一脚大范围转移，直接找到弱侧无人盯防的队友！
    exemplar_en: A raking switch finds the free man on the far side!
  - id: decoy_run
    name_zh: 无球牵制（牵制跑位）
    name_en: decoy run
    definition_zh: 无球队员用跑动带走防守者，为队友或传球线路腾出空间，自己并不接球。
    evidence_requirements: 跑动者speed高且未接球（非carrier），其dist_nearest_opp保持小（防守者跟随移动），同时另一同队球员的dist_nearest_opp或空当变大。
    exemplar_zh: 9号斜插带走中卫，为10号在弧顶腾出了拿球空间。
    exemplar_en: Number 9's decoy run drags the centre-back away, freeing space at the top of the box.
  - id: second_ball_protection
    name_zh: 第二落点保护
    name_en: second-ball protection
    definition_zh: 长传/传中/争顶前后，队员提前站位于可能的第二落点区域，保护球权或发动二次进攻。
    evidence_requirements: kick_anchor（高球速长传）后，非争顶队员位于落点附近8-15m环带内且在球落地前已就位（速度低、位置稳定）。
    exemplar_zh: 6号提前落位第二落点，头球摆渡刚被顶出他就把球控制下来。
    exemplar_en: Number 6 is stationed for the second ball and mops it up the moment the header drops.
```

- [ ] **Step 2: Write the failing test**

```python
from pipeline.tactics.kb import load_concepts, render_glossary_block


def test_load_and_validate():
    concepts = load_concepts()
    assert len(concepts) >= 15
    ids = {c["id"] for c in concepts}
    assert "overlap_run" in ids and "counter_attack" in ids
    for c in concepts:
        for key in ("id", "name_zh", "name_en", "definition_zh",
                    "evidence_requirements", "exemplar_zh", "exemplar_en"):
            assert c.get(key), f"{c.get('id')} missing {key}"


def test_glossary_block_contains_all_names():
    concepts = load_concepts()
    block = render_glossary_block(concepts)
    for c in concepts:
        assert c["name_zh"] in block and c["name_en"] in block
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/tactics/test_kb.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.tactics.kb`

- [ ] **Step 4: Implement pipeline/tactics/kb.py**

```python
"""Load and render the tactical concept glossary."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

REQUIRED_KEYS = ("id", "name_zh", "name_en", "definition_zh",
                 "evidence_requirements", "exemplar_zh", "exemplar_en")
DEFAULT_PATH = Path(__file__).parent / "concepts.yaml"


def load_concepts(path: Path = DEFAULT_PATH) -> List[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    concepts = data["concepts"]
    for c in concepts:
        missing = [k for k in REQUIRED_KEYS if not c.get(k)]
        if missing:
            raise ValueError(f"concept {c.get('id')!r} missing keys: {missing}")
    ids = [c["id"] for c in concepts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate concept ids in glossary")
    return concepts


def render_glossary_block(concepts: List[dict]) -> str:
    lines = ["TACTICAL CONCEPT GLOSSARY (you may ONLY name concepts from this list, "
             "and every use MUST cite evidence from the state data):"]
    for c in concepts:
        lines.append(
            f"- [{c['id']}] {c['name_zh']} / {c['name_en']}: {c['definition_zh']}\n"
            f"  evidence needed: {c['evidence_requirements']}\n"
            f"  style: zh「{c['exemplar_zh']}」 en \"{c['exemplar_en']}\""
        )
    return "\n".join(lines)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/tactics/test_kb.py -q`
Expected: 2 passed (install pyyaml if missing: `pip install pyyaml`)

- [ ] **Step 6: Commit**

```bash
git add pipeline/tactics/concepts.yaml pipeline/tactics/kb.py tests/tactics/test_kb.py
git commit -m "feat(tactics): tactical concept glossary v1 (15 entries) + loader"
```

---

### Task 7: Doubao adapter — configurable image cap

**Files:**
- Modify: `pipeline/stage4_commentary/adapters/doubao_api.py` (the `_sample_paths(..., limit=12)` call inside `DoubaoAPIAdapter.generate`)
- Test: `tests/tactics/test_narrate.py` (covered by Task 8's fake adapter; this task needs only a unit test of `_sample_paths`)

- [ ] **Step 1: Write the failing test** (`tests/tactics/test_adapter_images.py`)

```python
from pathlib import Path

from pipeline.stage4_commentary.adapters.doubao_api import DoubaoAPIAdapter, _sample_paths


def test_sample_paths_respects_limit():
    paths = [Path(f"{i}.png") for i in range(40)]
    assert len(_sample_paths(paths, limit=32)) == 32
    assert _sample_paths(paths, limit=32)[0] == paths[0]
    assert _sample_paths(paths, limit=32)[-1] == paths[-1]


def test_adapter_reads_max_images_env(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_IMAGES", "32")
    adapter = DoubaoAPIAdapter()
    assert adapter.max_images == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tactics/test_adapter_images.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'max_images'`

- [ ] **Step 3: Modify doubao_api.py**

In `DoubaoAPIAdapter.__init__`, add (after existing env reads):

```python
        self.max_images = int(os.environ.get("LLM_MAX_IMAGES", "12"))
```

In `generate(...)`, replace the hardcoded call `_sample_paths(paths, limit=12)`
(find it with `grep -n "_sample_paths" doubao_api.py`) with:

```python
        sampled = _sample_paths(paths, limit=self.max_images)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tactics/test_adapter_images.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage4_commentary/adapters/doubao_api.py tests/tactics/test_adapter_images.py
git commit -m "feat(adapter): configurable image cap via LLM_MAX_IMAGES"
```

---

### Task 8: narrate.py — Pass 1

**Files:**
- Create: `pipeline/tactics/narrate.py`
- Test: `tests/tactics/test_narrate.py`

Output segments reuse the existing commentary schema (`timestamp_s`, `end_s`,
`text_en`, `text_zh`, `energy`, `events_referenced`) so Stage 5 works untouched;
one extra key `evidence` (list of strings) is added and passes through
`postprocess._normalize_segment` untouched only if we keep it — check: the
normalizer whitelists keys, so `evidence` must be re-attached after parsing
(handled below by parsing raw JSON ourselves with the same tolerance).

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from pipeline.tactics.narrate import build_narrate_prompt, narrate


class FakeAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None
        self.last_visual = None

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        self.last_prompt = prompt
        self.last_visual = visual_input
        return self.reply


REPLY = json.dumps([{
    "timestamp_s": 0.0, "end_s": 6.0,
    "text_zh": "蓝队后场从容传导，阵地战缓缓展开。",
    "text_en": "The blues build patiently.",
    "energy": "calm",
    "concepts": ["positional_attack"],
    "evidence": ["t=0.5-5.5 carrier team=left continuous, ball speed<8"],
}])


def _mini_relations():
    return {
        "video_info": {"duration_s": 30.0, "fps": 25.0, "n_frames": 750},
        "conventions": "test",
        "snapshots": [
            {"t": 0.5, "frame_id": 13,
             "ball": {"x": -20.0, "y": 0.0, "speed": 3.0},
             "carrier": {"track_id": 1, "jersey": "10", "team": "left"},
             "players": [{"track_id": 1, "team": "left", "jersey": "10",
                          "role": "player", "x": -20.0, "y": 0.0, "speed": 2.0,
                          "dist_ball": 0.5}],
             "teams": {"left": {"attack_dir": 1, "opp_line_x": 10.0,
                                "n_within_15m_of_ball": 3},
                       "right": {"attack_dir": -1, "opp_line_x": -30.0,
                                 "n_within_15m_of_ball": 2}}},
        ],
        "kick_anchors": [{"t": 6.2, "ball_speed": 12.0}],
    }


def test_prompt_contains_state_glossary_rules(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_mini_relations()), encoding="utf-8")
    prompt = build_narrate_prompt(rel_path, languages=["en", "zh"])
    assert "GLOSSARY" in prompt
    assert "套边" in prompt                  # glossary injected
    assert "t=0.5" in prompt                 # snapshot serialized
    assert "kick" in prompt.lower()          # anchors present
    assert "NOT OBSERVED" in prompt or "not observed" in prompt.lower()


def test_narrate_writes_draft(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_mini_relations()), encoding="utf-8")
    adapter = FakeAdapter(REPLY)
    out = narrate(rel_path, tmp_path / "draft.json", adapter,
                  languages=["en", "zh"], radar_paths=[Path("r1.png")])
    segs = json.loads(out.read_text())["segments"]
    assert segs[0]["concepts"] == ["positional_attack"]
    assert adapter.last_visual == [Path("r1.png")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tactics/test_narrate.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.tactics.narrate`

- [ ] **Step 3: Implement pipeline/tactics/narrate.py**

```python
"""Pass 1: narrate from relations.json + radar frames + glossary."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from pipeline.tactics.kb import load_concepts, render_glossary_block


def _serialize_snapshot(s: dict) -> str:
    carrier = s.get("carrier")
    carrier_txt = (f"carrier=#{carrier['jersey']}({carrier['team']})"
                   if carrier else "carrier=none")
    rows = []
    for p in s["players"]:
        bits = [f"#{p['jersey'] or p['track_id']}({p['team'][0]})",
                f"xy=({p['x']},{p['y']})", f"v={p['speed']}"]
        for k in ("rel_x", "rel_y", "dist_ball", "dist_nearest_opp", "depth_vs_line"):
            if k in p:
                bits.append(f"{k}={p[k]}")
        rows.append(" ".join(bits))
    teams = "; ".join(
        f"{t}: opp_line_x={v['opp_line_x']} near_ball={v['n_within_15m_of_ball']}"
        for t, v in s["teams"].items())
    return (f"t={s['t']} ball=({s['ball']['x']},{s['ball']['y']}) "
            f"ball_speed={s['ball']['speed']} {carrier_txt}\n  "
            + "\n  ".join(rows) + f"\n  teams: {teams}")


def build_narrate_prompt(relations_json_path: Path, languages: List[str]) -> str:
    rel = json.loads(Path(relations_json_path).read_text(encoding="utf-8"))
    duration_s = rel["video_info"]["duration_s"]
    min_segments = max(1, int(duration_s // 5))
    lang_str = " and ".join(languages).upper()
    glossary = render_glossary_block(load_concepts())
    snapshots = "\n".join(_serialize_snapshot(s) for s in rel["snapshots"])
    anchors = ", ".join(f"t={a['t']}(v={a['ball_speed']}m/s)"
                        for a in rel["kick_anchors"]) or "none"

    return f"""You are a professional football commentator AND tactical analyst.
You receive (1) top-view radar images of the pitch, (2) a numeric state table,
(3) a tactical concept GLOSSARY. Generate vivid, tactically insightful
commentary for this {duration_s:.0f}-second clip in {lang_str}.

STATE CONVENTIONS: {rel['conventions']}
Players absent from the table are NOT OBSERVED — never describe them.

BALL KICK ANCHORS (ball-speed spikes, likely passes/shots/clearances): {anchors}

STATE TABLE ({len(rel['snapshots'])} snapshots):
{snapshots}

GLOSSARY:
{glossary}

RULES:
1. YOU decide what the key moments are, directly from the radar images and the
   state table. Name actions (pass, shot, tackle...) yourself from evidence.
2. Weave tactical explanation into the flow: use phase concepts to frame
   ("阵地战中...", "反击中...") and run/space concepts to explain WHY players
   move where they move.
3. Every tactical concept you name MUST appear in the GLOSSARY, and every
   segment using one MUST include an "evidence" array quoting the supporting
   numbers (e.g. "t=6.5 #7 rel_x -3.1->+2.4, |rel_y|=20").
4. Never invent players, jersey numbers, or off-screen events.
5. Cover the full clip; at least {min_segments} segments; segments ordered,
   non-overlapping, each 3-10 s.
6. Output ONLY a valid JSON array. Each object keys: timestamp_s (number),
   end_s (number), text_en (string), text_zh (string), energy (one of
   "calm"|"engaged"|"excited"|"explosive"), concepts (array of glossary ids,
   may be empty), evidence (array of strings, required when concepts non-empty).
"""


def _parse_segments(raw: str) -> List[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise ValueError(f"no JSON array in LLM reply: {raw[:200]}")
    segments = json.loads(m.group(0))
    for i, seg in enumerate(segments):
        for key in ("timestamp_s", "end_s", "text_en", "text_zh", "energy"):
            if key not in seg:
                raise ValueError(f"segment {i} missing {key}")
        seg.setdefault("concepts", [])
        seg.setdefault("evidence", [])
    return segments


def narrate(relations_json_path: Path, output_path: Path, adapter,
            languages: List[str],
            radar_paths: Optional[List[Path]] = None) -> Path:
    prompt = build_narrate_prompt(relations_json_path, languages)
    raw = adapter.generate(prompt, radar_paths)
    segments = _parse_segments(raw)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {"schema_version": "draft-v1-20260709", "segments": segments},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tactics/test_narrate.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/tactics/narrate.py tests/tactics/test_narrate.py
git commit -m "feat(tactics): pass-1 narration prompt + draft writer"
```

---

### Task 9: verify.py — Pass 2 (self-ask → resolve → rewrite)

**Files:**
- Create: `pipeline/tactics/verify.py`
- Test: `tests/tactics/test_verify.py`

Design: one LLM call asks structured questions about every segment that names
concepts; the resolver answers them mechanically from relations.json; a second
LLM call rewrites, keeping the same output schema. Query grammar (JSON):

```json
{"segment_index": 0, "question": "was #7 wide and ahead of carrier 6-8s?",
 "query": {"t0": 6.0, "t1": 8.0, "jersey": "7", "team": "left",
           "quantity": "rel_x", "agg": "max"}}
```

`quantity` ∈ {x, y, speed, rel_x, rel_y, dist_ball, dist_nearest_opp,
depth_vs_line, ball_speed, n_within_15m_of_ball}; `agg` ∈ {min, max, mean, last}.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from pipeline.tactics.verify import resolve_query, verify_and_refine


REL = {
    "video_info": {"duration_s": 30.0, "fps": 25.0, "n_frames": 750},
    "conventions": "test",
    "snapshots": [
        {"t": 6.0, "frame_id": 151, "ball": {"x": 0.0, "y": 0.0, "speed": 10.0},
         "carrier": {"track_id": 1, "jersey": "10", "team": "left"},
         "players": [
            {"track_id": 2, "team": "left", "jersey": "7", "role": "player",
             "x": 5.0, "y": 21.0, "speed": 7.5, "dist_ball": 21.0,
             "rel_x": 5.0, "rel_y": 21.0, "dist_nearest_opp": 8.0,
             "depth_vs_line": -3.0}],
         "teams": {"left": {"attack_dir": 1, "opp_line_x": 8.0,
                            "n_within_15m_of_ball": 2},
                   "right": {"attack_dir": -1, "opp_line_x": -30.0,
                             "n_within_15m_of_ball": 3}}},
        {"t": 8.0, "frame_id": 201, "ball": {"x": 4.0, "y": 6.0, "speed": 4.0},
         "carrier": {"track_id": 1, "jersey": "10", "team": "left"},
         "players": [
            {"track_id": 2, "team": "left", "jersey": "7", "role": "player",
             "x": 12.0, "y": 22.0, "speed": 7.8, "dist_ball": 17.9,
             "rel_x": 9.0, "rel_y": 22.0, "dist_nearest_opp": 6.0,
             "depth_vs_line": 1.5}],
         "teams": {"left": {"attack_dir": 1, "opp_line_x": 8.0,
                            "n_within_15m_of_ball": 2},
                   "right": {"attack_dir": -1, "opp_line_x": -30.0,
                             "n_within_15m_of_ball": 3}}},
    ],
    "kick_anchors": [],
}


def test_resolve_query_player_quantity():
    q = {"t0": 6.0, "t1": 8.0, "jersey": "7", "team": "left",
         "quantity": "rel_x", "agg": "max"}
    r = resolve_query(REL, q)
    assert r["value"] == 9.0
    assert r["n_samples"] == 2


def test_resolve_query_team_quantity():
    q = {"t0": 6.0, "t1": 8.0, "team": "left",
         "quantity": "n_within_15m_of_ball", "agg": "mean"}
    assert resolve_query(REL, q)["value"] == 2.0


class ScriptedAdapter:
    """Returns queued replies in order."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_verify_and_refine_roundtrip(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(REL), encoding="utf-8")
    draft = {"schema_version": "draft-v1-20260709", "segments": [{
        "timestamp_s": 6.0, "end_s": 9.0,
        "text_zh": "7号从外线套上！", "text_en": "Seven overlaps!",
        "energy": "excited", "concepts": ["overlap_run"],
        "evidence": ["t=6-8 #7 rel_x 5->9, rel_y 21"]}]}
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")

    ask_reply = json.dumps([{"segment_index": 0,
                             "question": "was 7 ahead and wide of carrier",
                             "query": {"t0": 6.0, "t1": 8.0, "jersey": "7",
                                       "team": "left", "quantity": "rel_x",
                                       "agg": "max"}}])
    final_reply = json.dumps([{
        "timestamp_s": 6.0, "end_s": 9.0,
        "text_zh": "7号沿边路套上，rel_x从5米冲到9米！",
        "text_en": "Seven overlaps down the flank!",
        "energy": "excited", "concepts": ["overlap_run"],
        "evidence": ["resolved: max rel_x=9.0 in 6-8s"]}])
    adapter = ScriptedAdapter([ask_reply, final_reply])

    out = verify_and_refine(draft_path, rel_path, tmp_path / "commentary.json",
                            adapter, languages=["en", "zh"])
    data = json.loads(out.read_text())
    assert data["segments"][0]["text_zh"].startswith("7号")
    # resolver results were fed into the second prompt
    assert "9.0" in adapter.prompts[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tactics/test_verify.py -q`
Expected: FAIL — `ModuleNotFoundError: pipeline.tactics.verify`

- [ ] **Step 3: Implement pipeline/tactics/verify.py**

```python
"""Pass 2: GameSight/FLARE-style self-ask verification against relations.json."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import List

from pipeline.tactics.kb import load_concepts

PLAYER_QUANTITIES = {"x", "y", "speed", "rel_x", "rel_y", "dist_ball",
                     "dist_nearest_opp", "depth_vs_line"}
TEAM_QUANTITIES = {"n_within_15m_of_ball", "opp_line_x"}
AGGS = {"min": min, "max": max, "mean": lambda v: statistics.fmean(v),
        "last": lambda v: v[-1]}


def resolve_query(relations: dict, query: dict) -> dict:
    t0, t1 = float(query["t0"]), float(query["t1"])
    quantity = query["quantity"]
    agg = AGGS[query.get("agg", "mean")]
    values = []
    for snap in relations["snapshots"]:
        if not (t0 - 1e-6 <= snap["t"] <= t1 + 1e-6):
            continue
        if quantity == "ball_speed":
            values.append(snap["ball"]["speed"])
        elif quantity in TEAM_QUANTITIES:
            team = query["team"]
            v = snap["teams"].get(team, {}).get(quantity)
            if v is not None:
                values.append(v)
        elif quantity in PLAYER_QUANTITIES:
            for p in snap["players"]:
                if (str(p.get("jersey")) == str(query.get("jersey"))
                        and (not query.get("team") or p["team"] == query["team"])
                        and quantity in p):
                    values.append(p[quantity])
        else:
            return {"error": f"unknown quantity {quantity}", "n_samples": 0}
    if not values:
        return {"value": None, "n_samples": 0,
                "note": "NO DATA in range — claim unsupported"}
    return {"value": round(agg(values), 2), "n_samples": len(values)}


def _extract_json_array(raw: str) -> list:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise ValueError(f"no JSON array in reply: {raw[:200]}")
    return json.loads(m.group(0))


def _ask_prompt(segments: List[dict], concept_ids: List[str]) -> str:
    return f"""You are auditing draft football commentary against tracked state data.
For EVERY segment that names tactical concepts, produce verification questions
as machine-readable queries. Query grammar:
{{"segment_index": int, "question": str,
  "query": {{"t0": float, "t1": float, "jersey": str?, "team": "left"|"right"?,
             "quantity": one of ["x","y","speed","rel_x","rel_y","dist_ball",
             "dist_nearest_opp","depth_vs_line","ball_speed",
             "n_within_15m_of_ball","opp_line_x"],
             "agg": "min"|"max"|"mean"|"last"}}}}
Ask 1-3 queries per concept claim, time-constrained to the segment's window.
Valid concept ids: {concept_ids}
DRAFT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=1)}
Output ONLY a JSON array of query objects."""


def _refine_prompt(segments: List[dict], resolutions: List[dict],
                   languages: List[str]) -> str:
    lang_str = " and ".join(languages).upper()
    return f"""Rewrite the draft football commentary using the verification results.
RULES:
1. If a query returned "NO DATA" or a value contradicting the claim, REMOVE or
   rewrite that tactical claim (keep the play-by-play if it stands on its own).
2. If verified, keep the claim and you may quote the number naturally.
3. Keep {lang_str} texts, same JSON schema as the draft (timestamp_s, end_s,
   text_en, text_zh, energy, concepts, evidence). Keep segments ordered and
   non-overlapping. Output ONLY the JSON array.
DRAFT:
{json.dumps(segments, ensure_ascii=False, indent=1)}
VERIFICATION RESULTS:
{json.dumps(resolutions, ensure_ascii=False, indent=1)}"""


def verify_and_refine(draft_path: Path, relations_json_path: Path,
                      output_path: Path, adapter,
                      languages: List[str]) -> Path:
    draft = json.loads(Path(draft_path).read_text(encoding="utf-8"))
    relations = json.loads(Path(relations_json_path).read_text(encoding="utf-8"))
    segments = draft["segments"]
    concept_ids = [c["id"] for c in load_concepts()]

    flagged = [s for s in segments if s.get("concepts")]
    resolutions = []
    if flagged:
        queries = _extract_json_array(
            adapter.generate(_ask_prompt(segments, concept_ids)))
        for q in queries:
            res = resolve_query(relations, q.get("query", {}))
            resolutions.append({**q, "result": res})

    final_segments = segments
    if resolutions:
        final_segments = _extract_json_array(
            adapter.generate(_refine_prompt(segments, resolutions, languages)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "schema_version": "commentary-2b-v1-20260709",
        "video_info": relations["video_info"],
        "languages": languages,
        "segments": final_segments,
        "verification": {"n_queries": len(resolutions),
                         "resolutions": resolutions},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tactics/test_verify.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/tactics/verify.py tests/tactics/test_verify.py
git commit -m "feat(tactics): pass-2 self-ask verification with mechanical resolver"
```

---

### Task 10: Config + run_stage2b orchestration + script

**Files:**
- Modify: `pipeline/config.py` (add fields + properties to `PipelineConfig`)
- Modify: `pipeline/run.py` (add `run_stage2b`; branch in `run_pipeline`)
- Create: `scripts/run_stage2b.sh`
- Test: `tests/test_stage2b_e2e.py`

**Stage-5 compatibility check (do this first):** `run_stage5` reads
`config.commentary_json` and passes it to `synthesize_commentary`. Confirm with
`grep -n "commentary_json" pipeline/config.py pipeline/stage5_tts/synthesize.py`
that Stage 5 consumes `segments[*].timestamp_s/end_s/text_zh/text_en/energy` —
our Pass-2 output keeps exactly those keys, plus extra keys (`concepts`,
`evidence`) which Stage 5 must ignore. If `synthesize.py` iterates keys
strictly, adjust it to `.get()` access — note the change in the commit.

- [ ] **Step 1: Add config fields**

In `pipeline/config.py`, inside `PipelineConfig` after the Stage-4 block:

```python
    # --- Stage 2b (state-to-reasoning commentary) ---
    commentary_mode: str = "events"  # "events" (legacy) | "state2b"
    snapshot_hz: float = 2.0
    radar_hz: float = 1.0
    llm_max_images: int = 32
```

And properties (next to the other path properties):

```python
    @property
    def relations_json(self) -> Path:
        return self.output_dir / "relations.json"

    @property
    def radar_dir(self) -> Path:
        return self.output_dir / "radar"

    @property
    def draft_commentary_json(self) -> Path:
        return self.output_dir / "commentary_draft.json"
```

- [ ] **Step 2: Write the failing e2e test** (`tests/test_stage2b_e2e.py`)

```python
import json
from pathlib import Path

from tests.conftest import make_frames
from pipeline.config import PipelineConfig
from pipeline.run import run_stage2b


class ScriptedAdapter:
    def __init__(self, replies):
        self.replies = list(replies)

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        return self.replies.pop(0)


DRAFT = json.dumps([{"timestamp_s": 0.0, "end_s": 5.0,
                     "text_zh": "比赛开始。", "text_en": "Kickoff.",
                     "energy": "calm", "concepts": [], "evidence": []}])


def test_run_stage2b_end_to_end(tmp_path, monkeypatch):
    frames = make_frames(
        [{"track_id": 1, "team": "left", "jersey": "10", "start": (-10.0, 0.0), "vel": (2.0, 0.0)},
         {"track_id": 3, "team": "right", "jersey": "4", "start": (15.0, 0.0), "vel": (0.0, 0.0)},
         {"track_id": 4, "team": "right", "jersey": "5", "start": (20.0, 5.0), "vel": (0.0, 0.0)}],
        n_frames=100,
        ball={"start": (-9.7, 0.0), "vel": (2.0, 0.0)},
    )
    monkeypatch.setattr("pipeline.run._load_frames_for_2b", lambda cfg: frames)

    cfg = PipelineConfig(output_dir=tmp_path, commentary_mode="state2b")
    adapter = ScriptedAdapter([DRAFT])  # no concepts -> no verify calls
    run_stage2b(cfg, adapter=adapter)

    assert cfg.relations_json.exists()
    assert len(list(cfg.radar_dir.glob("*.png"))) >= 3
    data = json.loads(cfg.commentary_json.read_text())
    assert data["segments"][0]["text_zh"] == "比赛开始。"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_stage2b_e2e.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_stage2b'`

- [ ] **Step 4: Implement run_stage2b in pipeline/run.py**

Add near the other stage functions:

```python
def _load_frames_for_2b(config: PipelineConfig):
    from pipeline.stage2_events.detector import load_frames
    return load_frames(str(config.predictions_json))


def run_stage2b(config: PipelineConfig, adapter=None) -> None:
    """State-to-reasoning commentary: relations -> radar -> narrate -> verify."""
    import os

    from pipeline.relations.build import build_relations, write_relations_json
    from pipeline.relations.radar import render_radar_frames
    from pipeline.stage4_commentary.generate import build_adapter
    from pipeline.tactics.narrate import narrate
    from pipeline.tactics.verify import verify_and_refine

    os.environ.setdefault("LLM_MAX_IMAGES", str(config.llm_max_images))
    adapter = adapter or build_adapter(config.llm_backend)

    frames = _load_frames_for_2b(config)
    relations = build_relations(frames, fps=float(config.fps),
                                snapshot_hz=config.snapshot_hz)
    write_relations_json(relations, config.relations_json)

    radar_paths = render_radar_frames(relations, config.radar_dir,
                                      hz=config.radar_hz)

    narrate(config.relations_json, config.draft_commentary_json, adapter,
            languages=config.languages, radar_paths=radar_paths)
    verify_and_refine(config.draft_commentary_json, config.relations_json,
                      config.commentary_json, adapter,
                      languages=config.languages)
```

In `run_pipeline`, where stage 2/3/4 currently run, branch:

```python
    if config.commentary_mode == "state2b":
        run_stage2b(config)
    else:
        ...existing stage2/3/4 calls unchanged...
```

(Keep stage 5 call shared after the branch.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_stage2b_e2e.py -q`
Expected: 1 passed. Then full suite: `python -m pytest tests/ -q` — all green.

- [ ] **Step 6: Create scripts/run_stage2b.sh**

```bash
#!/usr/bin/env bash
# State-to-reasoning commentary (2b): relations -> radar -> narrate -> verify -> TTS.
# Usage: run_stage2b.sh <run_output_dir_with_predictions.json>
set -euo pipefail
RUN_DIR="${1:?usage: run_stage2b.sh <run_output_dir>}"
cd "$(dirname "$0")/.."
python - <<EOF
from pathlib import Path
from pipeline.config import PipelineConfig
from pipeline.run import run_stage2b, run_stage5

cfg = PipelineConfig(
    output_dir=Path("$RUN_DIR"),
    existing_predictions_json=Path("$RUN_DIR") / "predictions.json",
    commentary_mode="state2b",
)
run_stage2b(cfg)
run_stage5(cfg)
EOF
```

Run: `chmod +x scripts/run_stage2b.sh && bash -n scripts/run_stage2b.sh`
Expected: no output (syntax OK)

- [ ] **Step 7: Commit**

```bash
git add pipeline/config.py pipeline/run.py scripts/run_stage2b.sh tests/test_stage2b_e2e.py
git commit -m "feat(pipeline): stage-2b state-to-reasoning commentary path"
```

---

### Task 11: Real-clip golden run + A/B guardrail

**Files:**
- Create: `tests/golden/SNGS-116.relations.md5` (checksum note, see below)

- [ ] **Step 1: Full real run on the GPU machine (doubao credentials in .env)**

```bash
bash scripts/run_stage2b.sh outputs/SNGS-116
```
Expected artifacts in `outputs/SNGS-116/`: `relations.json`, `radar/*.png`,
`commentary_draft.json`, `commentary.json` (with `verification` block),
`commentary_zh.mp3`, `final_video.mp4`.

- [ ] **Step 2: Manual quality gates (record answers in the PR/commit message)**

1. Every concept named in `commentary.json` exists in `concepts.yaml`?
   `python - <<'EOF'` ... check with a 5-line script comparing sets ... `EOF`
```python
import json, yaml
from pathlib import Path
kb = {c["id"] for c in yaml.safe_load(Path("pipeline/tactics/concepts.yaml").read_text())["concepts"]}
data = json.loads(Path("outputs/SNGS-116/commentary.json").read_text())
used = {c for s in data["segments"] for c in s.get("concepts", [])}
assert used <= kb, used - kb
print("concepts OK:", used)
print("verify queries:", data["verification"]["n_queries"],
      "no-data:", sum(1 for r in data["verification"]["resolutions"]
                      if r["result"].get("n_samples") == 0))
```
2. Watch `final_video.mp4` against the old variant-B video for the same clip:
   is the new one at least as good on event correctness, and better on
   tactical "why"? If not, the failure is almost always in the prompt's
   snapshot serialization density — tune `snapshot_hz` (try 1.0 and 3.0)
   before touching code.

- [ ] **Step 3: Freeze the relations golden**

```bash
python -c "import hashlib,pathlib;print(hashlib.md5(pathlib.Path('outputs/SNGS-116/relations.json').read_bytes()).hexdigest())" > tests/golden/SNGS-116.relations.md5
git add tests/golden/SNGS-116.relations.md5
git commit -m "test: freeze SNGS-116 relations.json golden checksum"
```
(Regenerate the checksum deliberately whenever relations code changes; a CI
diff on this file forces a human to re-review real-data output.)

---

## Self-Review

**Spec coverage:** relations.py measurements (Tasks 2–4) ✓; radar frames (Task 5) ✓; glossary + wholesale injection (Task 6, 8) ✓; Pass-1 narrate with evidence citations (Task 8) ✓; Pass-2 self-ask + time-constrained mechanical resolver (Task 9) ✓; retirement of typed-event path = `commentary_mode` branch, legacy left intact (Task 10) ✓; Stage-5 unchanged, compatibility check step (Task 10) ✓; testing pyramid incl. golden + prompt-level concept check (Tasks 2–11) ✓. Visible-region mask from homography: **deliberately cut from v1** (YAGNI — the ≥80% coverage gate plus the "absence = NOT OBSERVED" convention line covers the same risk at zero code); revisit if the LLM hallucinates far-side players in Task 11's manual gate.

**Known judgment calls:** carrier has no hysteresis in snapshots (raw nearest-within-3m); acceptable because the LLM sees the carrier across consecutive snapshots and can smooth mentally — add hysteresis only if Task 11 shows carrier flicker. `energy`/pacing validation from the legacy path (`validate_pacing`) is not reused in 2b v1; if TTS pacing regresses, port the `validate_pacing` retry loop from `stage4_commentary/generate.py` into `narrate()`.

**Type consistency:** `TrackSeries` (Task 2) consumed by `build_snapshots` (Task 3); snapshot dict keys (`rel_x`, `dist_nearest_opp`, `depth_vs_line`, `n_within_15m_of_ball`, `opp_line_x`) match `PLAYER_QUANTITIES`/`TEAM_QUANTITIES` in `verify.py` (Task 9) and the query grammar in `_ask_prompt`; segment schema identical across narrate → verify → stage-5 keys.
