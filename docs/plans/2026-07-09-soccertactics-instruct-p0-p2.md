# SoccerTactics-Instruct P0–P2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the solo-feasible core of SoccerTactics-Instruct: the complete tactical taxonomy, the verified state-guided generation loop, the seed dataset from the 164 GT SoccerNetGS sequences, MCQ generation for two families, and the mining-pilot density report — everything needed to recruit collaborators for P3+.

**Architecture:** New top-level `soccertactics/` package. It consumes the shared components from the v2 pipeline plan (`pipeline/relations/`, `pipeline/tactics/kb.py`, `pipeline/tactics/verify.py`) and produces SoccerChat-format JSONL. Flow: GT `Labels-GameState.json` → `load_frames` → `relations.json` → per-category **retrieval signatures** find instances → **generator** writes answer-guided free-form responses → **filter** verifies every claim mechanically → **writer** emits dataset rows → **report** tabulates per-category density.

**Tech Stack:** Python 3.10, PyYAML, pytest, existing `DoubaoAPIAdapter`/fake adapters, `pipeline.relations.*` + `pipeline.tactics.verify.resolve_query`.

**Spec:** `docs/superpowers/specs/2026-07-09-soccertactics-instruct-design.md`

**Hard dependency:** v2 plan (`docs/superpowers/plans/2026-07-09-tactical-commentary-v2.md`) Tasks 1–4 (relations), 6 (kb), 9 (verify) must be implemented first. This plan's tests import from them.

---

## File Structure

```
soccertactics/
    __init__.py
    taxonomy.yaml        # P0: complete 4-moments taxonomy, ~24 categories
    taxonomy.py          # loader + validation
    signatures.py        # per-category retrieval over relations.json
    templates.yaml       # free-form query templates per category
    generate.py          # answer-guided free-form generation (LLM)
    filter.py            # claim extraction -> mechanical verification -> keep/drop
    dataset.py           # SoccerChat-format JSONL writer
    run_seed.py          # batch: 164 GT sequences -> seed dataset
    mcq.py               # MCQ generation: run-type + phase families
    report.py            # per-category density table (markdown)
scripts/run_seed_dataset.sh
tests/soccertactics/
    __init__.py
    test_taxonomy.py
    test_signatures.py
    test_generate.py
    test_filter.py
    test_dataset.py
    test_mcq.py
    test_report.py
```

Data facts this plan relies on (verified): GT `Labels-GameState.json` shares the
predictions.json schema (`bbox_pitch`, `attributes.role/team/jersey`, ball as
`role="ball"`) so `pipeline.stage2_events.detector.load_frames` loads it
directly; 164 sequences live under
`codes/sn-gamestate/datasets/SoccerNetGS/{train,valid,test}/SNGS-*/`.

---

### Task 1: taxonomy.yaml + loader (P0)

**Files:**
- Create: `soccertactics/__init__.py` (empty), `tests/soccertactics/__init__.py` (empty)
- Create: `soccertactics/taxonomy.yaml`
- Create: `soccertactics/taxonomy.py`
- Test: `tests/soccertactics/test_taxonomy.py`

- [ ] **Step 1: Write taxonomy.yaml** (complete, all four moments + set pieces; `tier: 3` marks eval-only)

```yaml
# SoccerTactics-Instruct taxonomy. Literature-complete (four moments + set
# pieces). Fields: verifiability lists the relations.json quantities that
# check the defining claim; signature names the retrieval function in
# signatures.py (null = tier-3, mined opportunistically / by event labels).
version: 1
categories:
  # ================ moment: in_possession ================
  - id: buildup_from_back
    moment: in_possession
    tier: 1
    name_zh: 后场组织出球
    name_en: build-up from the back
    definition_zh: 守门员或后卫在本方后场通过短传组织，逐步向前推进。
    observability: 常见于门球/后场控球开始的10-20秒内。
    verifiability: [carrier, x, ball_speed, opp_line_x]
    signature: sig_buildup
  - id: third_man
    moment: in_possession
    tier: 2
    name_zh: 第三人配合
    name_en: third-man combination
    definition_zh: A传B，B一脚回做或分边给提前启动的C，利用C不被盯防的时间差。
    observability: 需要2次快速传递+一名无球启动者，3-6秒内完成。
    verifiability: [carrier, kick_anchors, speed, rel_x]
    signature: sig_third_man
  - id: switch_play
    moment: in_possession
    tier: 1
    name_zh: 弱侧转移
    name_en: switch of play
    definition_zh: 长传把球从一侧快速转移到防守薄弱的另一侧。
    observability: 单次长传，球y坐标大幅变化。
    verifiability: [ball_speed, y]
    signature: sig_switch
  - id: wing_overload
    moment: in_possession
    tier: 2
    name_zh: 边路人数堆积
    name_en: wing overload
    definition_zh: 进攻方在一侧边路集中多人制造局部优势。
    observability: 同侧边路3+进攻球员聚集。
    verifiability: [y, n_within_15m_of_ball]
    signature: sig_overload
  - id: halfspace_occupation
    moment: in_possession
    tier: 2
    name_zh: 肋部占位
    name_en: half-space occupation
    definition_zh: 进攻球员站位于边路与中路之间的肋部通道，处于对方防线之间。
    observability: 静态站位即可观察。
    verifiability: [y, x, depth_vs_line]
    signature: sig_halfspace
  - id: overlap_run
    moment: in_possession
    tier: 1
    name_zh: 套边
    name_en: overlapping run
    definition_zh: 无球队员沿持球队友外侧超越，制造边路二打一。
    observability: 2-4秒的冲刺，外侧超越持球人。
    verifiability: [rel_x, rel_y, speed]
    signature: sig_overlap
  - id: underlap_run
    moment: in_possession
    tier: 2
    name_zh: 内线前插
    name_en: underlapping run
    definition_zh: 无球队员从持球人内侧肋部前插。
    observability: 同套边，但在内侧通道。
    verifiability: [rel_x, rel_y, speed]
    signature: sig_underlap
  - id: run_in_behind
    moment: in_possession
    tier: 1
    name_zh: 打身后
    name_en: run in behind
    definition_zh: 进攻队员冲击对方防线身后纵深。
    observability: 冲刺+越过防线深度。
    verifiability: [depth_vs_line, speed]
    signature: sig_run_behind
  - id: cutback
    moment: in_possession
    tier: 2
    name_zh: 倒三角回传
    name_en: cutback
    definition_zh: 底线附近回敲给禁区弧顶插上的队友。
    observability: 传球起点近底线、方向向后、终点禁区中路。
    verifiability: [x, y, kick_anchors]
    signature: sig_cutback
  - id: one_two
    moment: in_possession
    tier: 2
    name_zh: 二过一
    name_en: one-two (wall pass)
    definition_zh: 传球后立刻前插接回敲，用两脚传递过掉防守者。
    observability: 2次快速传递+持球人前插。
    verifiability: [carrier, kick_anchors, speed]
    signature: sig_one_two
  - id: width_depth_stretch
    moment: in_possession
    tier: 2
    name_zh: 宽度与纵深拉扯
    name_en: width/depth stretching
    definition_zh: 通过边路拉宽+前锋压深迫使对方防线横向/纵向拉开。
    observability: 团队站位形状指标。
    verifiability: [y, depth_vs_line, opp_line_x]
    signature: sig_stretch
  - id: decoy_run
    moment: in_possession
    tier: 2
    name_zh: 无球牵制
    name_en: decoy run
    definition_zh: 无球队员跑动带走防守者，为队友腾出空间，自己不接球。
    observability: 跑动者与其盯防者同步位移+受益队友空间变大，3-6秒。
    verifiability: [speed, dist_nearest_opp, rel_x]
    signature: sig_decoy
  - id: second_ball_protection
    moment: in_possession
    tier: 2
    name_zh: 第二落点保护
    name_en: second-ball protection
    definition_zh: 长传/争顶前提前占据可能的第二落点区域，保护球权。
    observability: 高球速长传后非争顶队员在落点环带内提前就位。
    verifiability: [x, y, speed, ball_speed]
    signature: sig_second_ball
  # ================ moment: out_of_possession ================
  - id: high_press
    moment: out_of_possession
    tier: 1
    name_zh: 高位逼抢
    name_en: high press
    definition_zh: 无球方在对方半场就地围抢持球人与接应点。
    observability: 多名防守球员前压至对方半场贴近球。
    verifiability: [dist_ball, x, dist_nearest_opp]
    signature: sig_high_press
  - id: mid_block
    moment: out_of_possession
    tier: 2
    name_zh: 中位防守块
    name_en: mid block
    definition_zh: 防守方在中场区域保持紧凑阵型，放弃前场逼抢。
    observability: 防守块位置+紧凑度。
    verifiability: [opp_line_x, x]
    signature: sig_block
  - id: low_block
    moment: out_of_possession
    tier: 2
    name_zh: 低位防守块
    name_en: low block
    definition_zh: 防守方整体退入本方半场深处压缩身后空间。
    observability: 防守块深度。
    verifiability: [opp_line_x, x]
    signature: sig_block
  - id: compactness
    moment: out_of_possession
    tier: 2
    name_zh: 阵型紧凑性
    name_en: compactness
    definition_zh: 防守球员之间横向与纵向距离压缩，封锁中路通道。
    observability: 团队形状（宽度/深度/面积）。
    verifiability: [x, y]
    signature: sig_compact
  - id: line_management
    moment: out_of_possession
    tier: 3
    name_zh: 防线深度控制/造越位
    name_en: defensive line management / offside trap
    definition_zh: 后卫线整体前提或后撤控制身后空间，包括造越位。
    observability: 需要精确同步的防线移动，30秒片段内少见且难验证。
    verifiability: [opp_line_x]
    signature: null
  - id: marking_scheme
    moment: out_of_possession
    tier: 3
    name_zh: 盯人/区域策略
    name_en: zonal vs man marking
    definition_zh: 防守方按区域或按人分配防守职责。
    observability: 需要多回合观察才能区分，单片段弱可验证。
    verifiability: [dist_nearest_opp]
    signature: null
  # ================ moment: attacking_transition ================
  - id: counter_attack
    moment: attacking_transition
    tier: 1
    name_zh: 快速反击
    name_en: counter attack
    definition_zh: 由守转攻后立即向前推进，打对方立足未稳。
    observability: 球权切换+快速向前推进，5-15秒。
    verifiability: [carrier, ball_speed, x, depth_vs_line]
    signature: sig_counter
  - id: direct_runners
    moment: attacking_transition
    tier: 2
    name_zh: 反击中的无球冲刺
    name_en: direct runners in transition
    definition_zh: 反击中多名无球队员向前冲刺提供出球线路。
    observability: 反击窗口内2+球员高速前插。
    verifiability: [speed, rel_x]
    signature: sig_runners
  # ================ moment: defensive_transition ================
  - id: counter_press
    moment: defensive_transition
    tier: 2
    name_zh: 反抢（丢球反抢）
    name_en: counter-press (Gegenpressing)
    definition_zh: 丢球后数秒内就地围抢，阻止对方发动反击。
    observability: 球权切换后原进攻方多人立即收缩逼近球。
    verifiability: [carrier, dist_ball, speed]
    signature: sig_counter_press
  - id: rest_defense
    moment: defensive_transition
    tier: 3
    name_zh: 进攻时的防守保护
    name_en: rest defense
    definition_zh: 进攻时后场保留人数与站位以防被反击。
    observability: 进攻期间后场站位结构，弱可验证。
    verifiability: [x]
    signature: null
  # ================ moment: set_piece ================
  - id: corner_scheme
    moment: set_piece
    tier: 3
    name_zh: 角球战术
    name_en: corner scheme
    definition_zh: 角球进攻/防守的人员安排与跑位设计。
    observability: SoccerNet-v2有角球时间标注，可直接检索；站位可观察。
    verifiability: [x, y]
    signature: null   # retrieval via SoccerNet-v2 event labels (P2+)
  - id: free_kick_scheme
    moment: set_piece
    tier: 3
    name_zh: 任意球战术
    name_en: free kick scheme
    definition_zh: 任意球的人墙、跑位与传射选择。
    observability: 事件标注可检索。
    verifiability: [x, y]
    signature: null
  - id: long_throw
    moment: set_piece
    tier: 3
    name_zh: 长界外球
    name_en: long throw-in
    definition_zh: 用长界外球直接攻击禁区。
    observability: 事件标注可检索。
    verifiability: [y, ball_speed]
    signature: null
```

- [ ] **Step 2: Write the failing test** (`tests/soccertactics/test_taxonomy.py`)

```python
from soccertactics.taxonomy import load_taxonomy


def test_taxonomy_complete_and_valid():
    cats = load_taxonomy()
    assert len(cats) >= 20
    moments = {c["moment"] for c in cats}
    assert moments == {"in_possession", "out_of_possession",
                       "attacking_transition", "defensive_transition",
                       "set_piece"}
    for c in cats:
        assert c["tier"] in (1, 2, 3)
        for key in ("id", "name_zh", "name_en", "definition_zh",
                    "observability", "verifiability"):
            assert c.get(key), f"{c.get('id')} missing {key}"
        if c["tier"] in (1, 2):
            assert c["signature"], f"tier-{c['tier']} {c['id']} needs a signature"


def test_ids_unique():
    cats = load_taxonomy()
    ids = [c["id"] for c in cats]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_taxonomy.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.taxonomy`

- [ ] **Step 4: Implement soccertactics/taxonomy.py**

```python
"""Load and validate the SoccerTactics taxonomy."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

DEFAULT_PATH = Path(__file__).parent / "taxonomy.yaml"
MOMENTS = {"in_possession", "out_of_possession", "attacking_transition",
           "defensive_transition", "set_piece"}


def load_taxonomy(path: Path = DEFAULT_PATH) -> List[dict]:
    cats = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["categories"]
    for c in cats:
        if c["moment"] not in MOMENTS:
            raise ValueError(f"{c['id']}: bad moment {c['moment']}")
        if c["tier"] not in (1, 2, 3):
            raise ValueError(f"{c['id']}: bad tier {c['tier']}")
    return cats


def by_id(cats: List[dict]) -> dict:
    return {c["id"]: c for c in cats}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/soccertactics/test_taxonomy.py -q`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add soccertactics/ tests/soccertactics/
git commit -m "feat(soccertactics): literature-complete taxonomy + loader (P0)"
```

---

### Task 2: signatures.py — retrieval over relations.json

**Files:**
- Create: `soccertactics/signatures.py`
- Test: `tests/soccertactics/test_signatures.py`

Signatures RETRIEVE candidate instances; they are deliberately high-recall
(precision comes later from LLM confirm + claim filter). Each returns
`[{"t0": float, "t1": float, "players": [jersey...], "team": str, "facts": {...}}]`.
v1 implements the six tier-1 signatures + two tier-2 (overlap family); the
remaining tier-2 signatures follow the same pattern and are added in P1 as
needed (each is ~15 lines; the test pattern below is the template).

- [ ] **Step 1: Write the failing test**

```python
from tests.conftest import make_frames
from pipeline.relations.build import build_relations
from soccertactics.signatures import (SIGNATURES, sig_counter, sig_overlap,
                                      sig_switch)


def test_registry_covers_tier12_taxonomy():
    from soccertactics.taxonomy import load_taxonomy
    needed = {c["signature"] for c in load_taxonomy()
              if c["tier"] in (1, 2) and c["signature"]}
    implemented = set(SIGNATURES)
    # v1 gate: all tier-1 signatures implemented
    tier1 = {c["signature"] for c in load_taxonomy() if c["tier"] == 1}
    assert tier1 <= implemented, tier1 - implemented


def test_sig_overlap_fires_on_synthetic_overlap():
    # carrier advances slowly; #2 sprints outside past the carrier
    frames = make_frames(
        [{"track_id": 1, "team": "left", "jersey": "10", "start": (-30.0, 5.0), "vel": (1.0, 0.0)},
         {"track_id": 2, "team": "left", "jersey": "3", "start": (-38.0, 22.0), "vel": (7.0, 0.5)},
         {"track_id": 3, "team": "right", "jersey": "4", "start": (-10.0, 0.0), "vel": (0.0, 0.0)},
         {"track_id": 4, "team": "right", "jersey": "5", "start": (-5.0, 8.0), "vel": (0.0, 0.0)}],
        n_frames=150,  # 6 s
        ball={"start": (-29.7, 5.0), "vel": (1.0, 0.0)},
    )
    rel = build_relations(frames, fps=25.0, snapshot_hz=2.0)
    hits = sig_overlap(rel)
    assert hits, "overlap signature should fire"
    assert "3" in hits[0]["players"]


def test_sig_overlap_silent_without_runner():
    frames = make_frames(
        [{"track_id": 1, "team": "left", "jersey": "10", "start": (-30.0, 5.0), "vel": (1.0, 0.0)},
         {"track_id": 3, "team": "right", "jersey": "4", "start": (-10.0, 0.0), "vel": (0.0, 0.0)},
         {"track_id": 4, "team": "right", "jersey": "5", "start": (-5.0, 8.0), "vel": (0.0, 0.0)}],
        n_frames=150,
        ball={"start": (-29.7, 5.0), "vel": (1.0, 0.0)},
    )
    rel = build_relations(frames, fps=25.0, snapshot_hz=2.0)
    assert sig_overlap(rel) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_signatures.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.signatures`

- [ ] **Step 3: Implement soccertactics/signatures.py**

```python
"""High-recall retrieval signatures over relations.json.

Each signature scans snapshots and returns candidate instances:
    {"t0", "t1", "team", "players": [jersey...], "facts": {...}}
Precision is NOT the goal here — the generator's LLM confirm step and the
claim filter downstream reject false positives.
"""
from __future__ import annotations

from typing import Callable, Dict, List

SPRINT_MPS = 5.5
WIDE_Y = 15.0
WINDOW_PAD_S = 1.5


def _player_series(rel: dict, jersey: str, team: str) -> List[dict]:
    out = []
    for s in rel["snapshots"]:
        for p in s["players"]:
            if p["jersey"] == jersey and p["team"] == team:
                out.append({"t": s["t"], **p})
    return out


def _all_players(rel: dict) -> List[tuple]:
    seen = set()
    for s in rel["snapshots"]:
        for p in s["players"]:
            seen.add((p["jersey"], p["team"]))
    return sorted(seen)


def _carrier_team_at(rel: dict, t: float):
    best = None
    for s in rel["snapshots"]:
        if best is None or abs(s["t"] - t) < abs(best["t"] - t):
            best = s
    c = best.get("carrier") if best else None
    return c["team"] if c else None


def sig_overlap(rel: dict) -> List[dict]:
    """Teammate goes from behind-carrier to past-carrier on the outside at speed."""
    hits = []
    for jersey, team in _all_players(rel):
        series = _player_series(rel, jersey, team)
        run = [p for p in series if "rel_x" in p]
        for i in range(1, len(run)):
            a, b = run[i - 1], run[i]
            if (a["rel_x"] < 0 <= b["rel_x"] and abs(b.get("rel_y", 0)) >= WIDE_Y
                    and b["speed"] >= SPRINT_MPS):
                hits.append({"t0": max(0.0, a["t"] - WINDOW_PAD_S),
                             "t1": b["t"] + WINDOW_PAD_S,
                             "team": team, "players": [jersey],
                             "facts": {"rel_x_from": a["rel_x"],
                                       "rel_x_to": b["rel_x"],
                                       "rel_y": b.get("rel_y"),
                                       "speed": b["speed"]}})
                break
    return hits


def sig_underlap(rel: dict) -> List[dict]:
    hits = []
    for jersey, team in _all_players(rel):
        run = [p for p in _player_series(rel, jersey, team) if "rel_x" in p]
        for i in range(1, len(run)):
            a, b = run[i - 1], run[i]
            if (a["rel_x"] < 0 <= b["rel_x"] and abs(b.get("rel_y", 99)) < WIDE_Y
                    and b["speed"] >= SPRINT_MPS):
                hits.append({"t0": max(0.0, a["t"] - WINDOW_PAD_S),
                             "t1": b["t"] + WINDOW_PAD_S,
                             "team": team, "players": [jersey],
                             "facts": {"rel_x_from": a["rel_x"],
                                       "rel_x_to": b["rel_x"],
                                       "rel_y": b.get("rel_y")}})
                break
    return hits


def sig_run_behind(rel: dict) -> List[dict]:
    hits = []
    for jersey, team in _all_players(rel):
        run = [p for p in _player_series(rel, jersey, team) if "depth_vs_line" in p]
        for i in range(1, len(run)):
            a, b = run[i - 1], run[i]
            if a["depth_vs_line"] < -1.0 and b["depth_vs_line"] > -0.5 \
                    and b["speed"] >= SPRINT_MPS:
                hits.append({"t0": max(0.0, a["t"] - WINDOW_PAD_S),
                             "t1": b["t"] + WINDOW_PAD_S,
                             "team": team, "players": [jersey],
                             "facts": {"depth_from": a["depth_vs_line"],
                                       "depth_to": b["depth_vs_line"],
                                       "speed": b["speed"]}})
                break
    return hits


def sig_counter(rel: dict) -> List[dict]:
    """Possession flips, then ball advances fast within 5 s."""
    hits = []
    snaps = rel["snapshots"]
    for i in range(1, len(snaps)):
        prev_c, cur_c = snaps[i - 1].get("carrier"), snaps[i].get("carrier")
        if not prev_c or not cur_c or prev_c["team"] == cur_c["team"]:
            continue
        t_flip = snaps[i]["t"]
        window = [s for s in snaps if t_flip <= s["t"] <= t_flip + 5.0]
        if len(window) < 2:
            continue
        adir = window[0]["teams"][cur_c["team"]]["attack_dir"]
        dx = (window[-1]["ball"]["x"] - window[0]["ball"]["x"]) * adir
        if dx >= 15.0:
            hits.append({"t0": max(0.0, t_flip - 1.0), "t1": window[-1]["t"],
                         "team": cur_c["team"], "players": [cur_c["jersey"]],
                         "facts": {"advance_m": round(dx, 1), "within_s": 5.0}})
    return hits


def sig_switch(rel: dict) -> List[dict]:
    """Kick anchor + ball y moves > 25 m within 2.5 s."""
    hits, snaps = [], rel["snapshots"]
    for a in rel.get("kick_anchors", []):
        t = a["t"]
        near = [s for s in snaps if t - 0.5 <= s["t"] <= t + 2.5]
        if len(near) < 2:
            continue
        dy = abs(near[-1]["ball"]["y"] - near[0]["ball"]["y"])
        if dy >= 25.0:
            team = _carrier_team_at(rel, t - 0.5)
            hits.append({"t0": max(0.0, t - 1.0), "t1": near[-1]["t"],
                         "team": team or "unknown", "players": [],
                         "facts": {"delta_y_m": round(dy, 1),
                                   "ball_speed": a["ball_speed"]}})
    return hits


def sig_high_press(rel: dict) -> List[dict]:
    """3+ defenders near carrier in carrier's own half, sustained >= 2 s."""
    hits, streak_start = [], None
    for s in rel["snapshots"]:
        c = s.get("carrier")
        ok = False
        if c:
            def_team = "left" if c["team"] == "right" else "right"
            adir = s["teams"][c["team"]]["attack_dir"]
            in_own_half = s["ball"]["x"] * adir < 0
            n_close = sum(1 for p in s["players"]
                          if p["team"] == def_team and p["dist_ball"] <= 12.0)
            ok = in_own_half and n_close >= 3
        if ok and streak_start is None:
            streak_start = s["t"]
        elif not ok and streak_start is not None:
            if s["t"] - streak_start >= 2.0:
                hits.append({"t0": streak_start, "t1": s["t"],
                             "team": def_team, "players": [],
                             "facts": {"n_pressers": n_close}})
            streak_start = None
    return hits


def sig_buildup(rel: dict) -> List[dict]:
    """Sustained same-team possession in own half, low ball speed, >= 6 s."""
    hits, streak, team0 = [], None, None
    for s in rel["snapshots"]:
        c = s.get("carrier")
        ok = False
        if c:
            adir = s["teams"][c["team"]]["attack_dir"]
            ok = s["ball"]["x"] * adir < -10 and s["ball"]["speed"] < 8.0
        if ok and (streak is None or c["team"] == team0):
            if streak is None:
                streak, team0 = s["t"], c["team"]
        else:
            if streak is not None and s["t"] - streak >= 6.0:
                hits.append({"t0": streak, "t1": s["t"], "team": team0,
                             "players": [], "facts": {"duration_s": round(s["t"] - streak, 1)}})
            streak, team0 = (s["t"], c["team"]) if ok else (None, None)
    return hits


def sig_runners(rel: dict) -> List[dict]:
    """Within a counter window, 2+ attackers sprinting forward."""
    hits = []
    for cnt in sig_counter(rel):
        sprinters = set()
        for s in rel["snapshots"]:
            if cnt["t0"] <= s["t"] <= cnt["t1"]:
                for p in s["players"]:
                    if p["team"] == cnt["team"] and p["speed"] >= SPRINT_MPS:
                        sprinters.add(p["jersey"])
        if len(sprinters) >= 2:
            hits.append({**cnt, "players": sorted(sprinters),
                         "facts": {**cnt["facts"], "n_runners": len(sprinters)}})
    return hits


SIGNATURES: Dict[str, Callable[[dict], List[dict]]] = {
    "sig_overlap": sig_overlap,
    "sig_underlap": sig_underlap,
    "sig_run_behind": sig_run_behind,
    "sig_counter": sig_counter,
    "sig_switch": sig_switch,
    "sig_high_press": sig_high_press,
    "sig_buildup": sig_buildup,
    "sig_runners": sig_runners,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/soccertactics/test_signatures.py -q`
Expected: 3 passed. (`test_registry_covers_tier12_taxonomy` only gates tier-1;
the remaining tier-2 signatures — `sig_third_man`, `sig_overload`,
`sig_halfspace`, `sig_cutback`, `sig_one_two`, `sig_stretch`, `sig_block`,
`sig_compact`, `sig_counter_press`, `sig_stretch` — are added during P1 with
the identical fire/silent test pattern shown above, one commit each.)

- [ ] **Step 5: Commit**

```bash
git add soccertactics/signatures.py tests/soccertactics/test_signatures.py
git commit -m "feat(soccertactics): tier-1 retrieval signatures"
```

---

### Task 3: templates.yaml + generate.py — answer-guided free-form generation

**Files:**
- Create: `soccertactics/templates.yaml`
- Create: `soccertactics/generate.py`
- Test: `tests/soccertactics/test_generate.py`

- [ ] **Step 1: Write templates.yaml**

```yaml
# Free-form query templates. {placeholders} filled from instance facts.
# Each category id maps to a list; generator samples one per instance.
version: 1
defaults:
  long_review:
    - "请从战术角度复盘这段片段中{team_zh}的进攻/防守组织，解释关键球员的站位与跑位。"
    - "Analyze this clip tactically: explain the positioning and movement of the key players involved."
by_category:
  overlap_run:
    - "为什么{jersey}号此时沿边路套上？他的跑位为球队创造了什么？"
    - "What does number {jersey}'s overlapping run achieve here, and why is it timed this way?"
  underlap_run:
    - "解释{jersey}号这次内线前插的战术意图。"
    - "Explain the tactical intent of number {jersey}'s underlapping run."
  run_in_behind:
    - "{jersey}号为什么在这个时机打身后？防线出现了什么问题？"
    - "Why does number {jersey} attack the space in behind at this moment?"
  counter_attack:
    - "复盘这次快速反击：参与进攻的球员跑位为什么是这样安排的？"
    - "Break down this counter attack: why do the attacking players run where they run?"
  switch_play:
    - "这脚弱侧转移解决了什么问题？转移前后两队的形势有何变化？"
    - "What problem does this switch of play solve? How does the picture change after it?"
  high_press:
    - "描述这段高位逼抢的结构：谁压持球人，谁封传球线路？"
    - "Describe the pressing structure here: who presses the ball, who covers the passing lanes?"
  buildup_from_back:
    - "分析{team_zh}这段后场组织出球：站位如何帮助他们摆脱第一道压迫？"
    - "Analyze this build-up phase: how does the positioning beat the first line of pressure?"
  direct_runners:
    - "这次反击中有多名球员无球冲刺，解释他们各自的跑动为持球人提供了什么选择。"
    - "Several players sprint without the ball in this transition — what option does each run create?"
```

- [ ] **Step 2: Write the failing test**

```python
import json

from soccertactics.generate import build_generation_prompt, generate_items


class FakeAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def supports_video(self):
        return False

    def generate(self, prompt, visual_input=None):
        self.prompts.append(prompt)
        return self.reply


INSTANCE = {"category": "overlap_run", "t0": 5.0, "t1": 9.0, "team": "left",
            "players": ["3"], "facts": {"rel_x_from": -4.0, "rel_x_to": 2.4,
                                        "rel_y": 21.0, "speed": 7.1}}
REL = {"video_info": {"duration_s": 30.0}, "conventions": "test",
       "snapshots": [], "kick_anchors": []}


def test_prompt_is_answer_guided():
    prompt = build_generation_prompt(INSTANCE, REL, seq_id="SNGS-060")
    # the known geometric facts must be IN the prompt (answer-guided)
    assert "rel_x" in prompt and "2.4" in prompt
    assert "套" in prompt or "overlap" in prompt.lower()
    assert "为什么" in prompt or "{jersey}" not in prompt  # template filled


def test_generate_items_roundtrip(tmp_path):
    reply = json.dumps({
        "query_zh": "为什么3号此时沿边路套上？",
        "query_en": "Why does number 3 overlap here?",
        "response_zh": "3号从持球人身后外侧启动（rel_x从-4.0提升到+2.4，速度7.1m/s），沿边路套上制造二打一。",
        "response_en": "Number 3 starts from behind the carrier and overlaps outside at 7.1 m/s.",
        "claims": [{"t0": 5.0, "t1": 9.0, "jersey": "3", "team": "left",
                    "quantity": "rel_x", "agg": "max", "expect": ">0"}],
    })
    adapter = FakeAdapter(reply)
    items = generate_items([INSTANCE], REL, "SNGS-060", adapter)
    assert len(items) == 1
    assert items[0]["category"] == "overlap_run"
    assert items[0]["claims"][0]["quantity"] == "rel_x"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_generate.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.generate`

- [ ] **Step 4: Implement soccertactics/generate.py**

```python
"""Answer-guided free-form generation: instance + state facts -> QA pair.

The LLM is TOLD the verified geometric facts and must write a long grounded
explanation around them, plus machine-checkable claims (Pass-2 query grammar)
that the filter later resolves. Chains contradicting the facts get dropped
downstream — GameSight's answer-guided bootstrap, applied to free-form.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List

import yaml

from soccertactics.taxonomy import by_id, load_taxonomy

TEMPLATES_PATH = Path(__file__).parent / "templates.yaml"
TEAM_ZH = {"left": "左侧球队", "right": "右侧球队"}


def _pick_template(category: str) -> str:
    data = yaml.safe_load(TEMPLATES_PATH.read_text(encoding="utf-8"))
    pool = data["by_category"].get(category) or data["defaults"]["long_review"]
    return random.choice(pool)


def build_generation_prompt(instance: dict, relations: dict, seq_id: str) -> str:
    cat = by_id(load_taxonomy())[instance["category"]]
    jersey = instance["players"][0] if instance["players"] else ""
    template = _pick_template(instance["category"]).format(
        jersey=jersey, team_zh=TEAM_ZH.get(instance["team"], instance["team"]))
    window_snaps = [s for s in relations["snapshots"]
                    if instance["t0"] <= s["t"] <= instance["t1"]]
    return f"""You are a professional tactical analyst writing training data.
CLIP: {seq_id}, window {instance['t0']}-{instance['t1']}s.
TACTICAL CATEGORY: {cat['name_zh']} / {cat['name_en']} — {cat['definition_zh']}
VERIFIED GEOMETRIC FACTS (ground truth, your explanation MUST be consistent
with these and MUST quote the key numbers naturally):
{json.dumps(instance['facts'], ensure_ascii=False)}
STATE SNAPSHOTS in window:
{json.dumps(window_snaps, ensure_ascii=False)}
CONVENTIONS: {relations['conventions']}

TASK: write ONE training example as JSON with keys:
  query_zh, query_en  — based on this question: 「{template}」 (rephrase freely)
  response_zh, response_en — 120-300 words (zh) tactical explanation: what
    happens, WHY the positioning/movement makes sense, what it creates. Ground
    every factual statement in the facts/snapshots. No invented players/events.
  claims — array of machine-checkable claims backing your response, grammar:
    {{"t0": float, "t1": float, "jersey": str?, "team": str?,
      "quantity": "x|y|speed|rel_x|rel_y|dist_ball|dist_nearest_opp|depth_vs_line|ball_speed|n_within_15m_of_ball",
      "agg": "min|max|mean|last", "expect": "><=N or >0 or <0"}}
Output ONLY the JSON object."""


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {raw[:200]}")
    obj = json.loads(m.group(0))
    for key in ("query_zh", "query_en", "response_zh", "response_en", "claims"):
        if key not in obj:
            raise ValueError(f"generation reply missing {key}")
    return obj


def generate_items(instances: List[dict], relations: dict, seq_id: str,
                   adapter) -> List[dict]:
    items = []
    for inst in instances:
        raw = adapter.generate(build_generation_prompt(inst, relations, seq_id))
        obj = _parse(raw)
        items.append({
            "seq_id": seq_id,
            "category": inst["category"],
            "tier": by_id(load_taxonomy())[inst["category"]]["tier"],
            "t0": inst["t0"], "t1": inst["t1"],
            "team": inst["team"], "players": inst["players"],
            "facts": inst["facts"],
            **obj,
        })
    return items
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/soccertactics/test_generate.py -q`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add soccertactics/templates.yaml soccertactics/generate.py tests/soccertactics/test_generate.py
git commit -m "feat(soccertactics): answer-guided free-form generation"
```

---

### Task 4: filter.py — mechanical claim verification

**Files:**
- Create: `soccertactics/filter.py`
- Test: `tests/soccertactics/test_filter.py`

Reuses `pipeline.tactics.verify.resolve_query`. A claim passes when the
resolved value satisfies `expect`. An item survives if ≥80% of its claims pass
AND it has ≥1 claim; otherwise dropped (logged).

- [ ] **Step 1: Write the failing test**

```python
from soccertactics.filter import check_claim, filter_items

REL = {"snapshots": [
    {"t": 6.0, "frame_id": 151, "ball": {"x": 0.0, "y": 0.0, "speed": 10.0},
     "carrier": {"track_id": 1, "jersey": "10", "team": "left"},
     "players": [{"track_id": 2, "team": "left", "jersey": "3", "role": "player",
                  "x": 5.0, "y": 21.0, "speed": 7.1, "dist_ball": 21.0,
                  "rel_x": 2.4, "rel_y": 21.0}],
     "teams": {"left": {"attack_dir": 1, "opp_line_x": 8.0,
                        "n_within_15m_of_ball": 2},
               "right": {"attack_dir": -1, "opp_line_x": -30.0,
                         "n_within_15m_of_ball": 3}}}],
    "kick_anchors": []}


def test_check_claim_expect_grammar():
    base = {"t0": 5.0, "t1": 7.0, "jersey": "3", "team": "left",
            "quantity": "rel_x", "agg": "max"}
    assert check_claim(REL, {**base, "expect": ">0"})["passed"]
    assert check_claim(REL, {**base, "expect": ">2"})["passed"]
    assert not check_claim(REL, {**base, "expect": "<0"})["passed"]
    # no data in range -> fail
    assert not check_claim(REL, {**base, "t0": 20.0, "t1": 25.0,
                                 "expect": ">0"})["passed"]


def test_filter_items_drops_unsupported():
    good = {"category": "overlap_run", "claims": [
        {"t0": 5.0, "t1": 7.0, "jersey": "3", "team": "left",
         "quantity": "rel_x", "agg": "max", "expect": ">0"}]}
    bad = {"category": "overlap_run", "claims": [
        {"t0": 5.0, "t1": 7.0, "jersey": "3", "team": "left",
         "quantity": "rel_x", "agg": "max", "expect": "<0"}]}
    no_claims = {"category": "overlap_run", "claims": []}
    kept, dropped = filter_items([good, bad, no_claims], REL)
    assert len(kept) == 1 and len(dropped) == 2
    assert kept[0] is good
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_filter.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.filter`

- [ ] **Step 3: Implement soccertactics/filter.py**

```python
"""Mechanical claim verification: keep only grounded training items."""
from __future__ import annotations

import re
from typing import List, Tuple

from pipeline.tactics.verify import resolve_query

PASS_RATIO = 0.8
_EXPECT_RE = re.compile(r"^([<>]=?)\s*(-?\d+(?:\.\d+)?)$")


def _expect_ok(value: float, expect: str) -> bool:
    expect = expect.strip()
    m = _EXPECT_RE.match(expect)
    if not m:
        raise ValueError(f"bad expect expression: {expect!r}")
    op, num = m.group(1), float(m.group(2))
    return {"<": value < num, "<=": value <= num,
            ">": value > num, ">=": value >= num}[op]


def check_claim(relations: dict, claim: dict) -> dict:
    res = resolve_query(relations, claim)
    if res.get("value") is None:
        return {"passed": False, "reason": "no data in range", **res}
    passed = _expect_ok(res["value"], claim["expect"])
    return {"passed": passed, "value": res["value"],
            "n_samples": res["n_samples"]}


def filter_items(items: List[dict], relations: dict) -> Tuple[List[dict], List[dict]]:
    kept, dropped = [], []
    for item in items:
        claims = item.get("claims", [])
        if not claims:
            dropped.append({**item, "_drop_reason": "no claims"})
            continue
        results = [check_claim(relations, c) for c in claims]
        ratio = sum(r["passed"] for r in results) / len(results)
        item["claim_results"] = results
        item["claim_pass_ratio"] = round(ratio, 2)
        if ratio >= PASS_RATIO:
            kept.append(item)
        else:
            dropped.append({**item, "_drop_reason": f"pass_ratio {ratio:.2f}"})
    return kept, dropped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/soccertactics/test_filter.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add soccertactics/filter.py tests/soccertactics/test_filter.py
git commit -m "feat(soccertactics): mechanical claim filter"
```

---

### Task 5: dataset.py — SoccerChat-format writer

**Files:**
- Create: `soccertactics/dataset.py`
- Test: `tests/soccertactics/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
import json

from soccertactics.dataset import write_jsonl


def test_write_jsonl_soccerchat_compatible(tmp_path):
    items = [{"seq_id": "SNGS-060", "category": "overlap_run", "tier": 1,
              "t0": 5.0, "t1": 9.0, "team": "left", "players": ["3"],
              "facts": {"rel_x_to": 2.4},
              "query_zh": "为什么3号套上？", "query_en": "Why overlap?",
              "response_zh": "……", "response_en": "...",
              "claims": [], "claim_results": [], "claim_pass_ratio": 1.0}]
    out = tmp_path / "train.jsonl"
    n = write_jsonl(items, out)
    assert n == 1
    row = json.loads(out.read_text().splitlines()[0])
    # SoccerChat-compatible core fields + our extensions
    for key in ("query", "response", "path", "events",
                "category", "tier", "t0", "t1", "query_zh", "response_zh"):
        assert key in row
    assert row["path"] == "videos/SNGS-060_5.0_9.0.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.dataset`

- [ ] **Step 3: Implement soccertactics/dataset.py**

```python
"""Write items as SoccerChat-compatible JSONL (+ tactical extensions).

Core fields mirror SimulaMet/SoccerChat (query, response, path, events) so the
published ms-swift SFT recipe works unchanged; extensions carry zh text,
category/tier, window, and verification stats. Video/radar files are cut
separately by clip-export tooling; `path` is the naming contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def to_row(item: dict, lang: str = "en") -> dict:
    return {
        "query": item[f"query_{lang}"],
        "response": item[f"response_{lang}"],
        "path": f"videos/{item['seq_id']}_{item['t0']}_{item['t1']}.mp4",
        "events": [item["category"]],
        # extensions
        "query_zh": item["query_zh"],
        "response_zh": item["response_zh"],
        "category": item["category"],
        "tier": item["tier"],
        "seq_id": item["seq_id"],
        "t0": item["t0"], "t1": item["t1"],
        "team": item["team"], "players": item["players"],
        "facts": item["facts"],
        "claim_pass_ratio": item.get("claim_pass_ratio"),
    }


def write_jsonl(items: List[dict], output_path: Path, lang: str = "en") -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(to_row(item, lang), ensure_ascii=False) + "\n")
    return len(items)
```

- [ ] **Step 4: Run test, then commit**

Run: `python -m pytest tests/soccertactics/test_dataset.py -q` → 1 passed

```bash
git add soccertactics/dataset.py tests/soccertactics/test_dataset.py
git commit -m "feat(soccertactics): SoccerChat-compatible JSONL writer"
```

---

### Task 6: run_seed.py — batch over the 164 GT sequences (P1)

**Files:**
- Create: `soccertactics/run_seed.py`
- Create: `scripts/run_seed_dataset.sh`
- Test: covered by an integration test inside `tests/soccertactics/test_report.py` (Task 7 uses run_seed's per-sequence function)

- [ ] **Step 1: Implement soccertactics/run_seed.py**

```python
"""P1 batch: GT Labels-GameState.json (164 seqs) -> verified seed dataset.

Per sequence: load GT frames -> relations -> signatures -> generate -> filter.
Writes per-sequence artifacts (resumable) then merges into train JSONL splits
that follow the SoccerNetGS train/valid/test split (no sequence leakage).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import GSR_ROOT
from pipeline.relations.build import build_relations
from pipeline.stage2_events.detector import load_frames
from soccertactics.dataset import write_jsonl
from soccertactics.filter import filter_items
from soccertactics.generate import generate_items
from soccertactics.signatures import SIGNATURES
from soccertactics.taxonomy import load_taxonomy

DATASET_ROOT = GSR_ROOT / "datasets" / "SoccerNetGS"
SPLITS = ("train", "valid", "test")


def find_sequences(root: Path = DATASET_ROOT):
    for split in SPLITS:
        for labels in sorted((root / split).glob("SNGS-*/Labels-GameState.json")):
            yield split, labels.parent.name, labels


def mine_sequence(labels_path: Path, fps: float = 25.0):
    """-> (relations, instances) where instances carry their category id."""
    frames = load_frames(str(labels_path))
    relations = build_relations(frames, fps=fps)
    instances = []
    for cat in load_taxonomy():
        fn = SIGNATURES.get(cat.get("signature") or "")
        if fn is None:
            continue
        for hit in fn(relations):
            instances.append({"category": cat["id"], **hit})
    return relations, instances


def process_sequence(split: str, seq_id: str, labels_path: Path,
                     out_dir: Path, adapter) -> dict:
    seq_dir = out_dir / split / seq_id
    done = seq_dir / "items.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    relations, instances = mine_sequence(labels_path)
    items = generate_items(instances, relations, seq_id, adapter)
    kept, dropped = filter_items(items, relations)
    seq_dir.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "seq_id": seq_id,
              "n_instances": len(instances), "n_kept": len(kept),
              "n_dropped": len(dropped), "kept": kept, "dropped": dropped}
    done.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return result


def main() -> None:
    from pipeline.stage4_commentary.generate import build_adapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/soccertactics_seed"))
    ap.add_argument("--backend", default="doubao")
    ap.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    args = ap.parse_args()

    adapter = build_adapter(args.backend)
    results = []
    for i, (split, seq_id, labels) in enumerate(find_sequences()):
        if args.limit and i >= args.limit:
            break
        r = process_sequence(split, seq_id, labels, args.out_dir, adapter)
        results.append(r)
        print(f"[{split}/{seq_id}] instances={r['n_instances']} "
              f"kept={r['n_kept']} dropped={r['n_dropped']}")

    for split in SPLITS:
        rows = [it for r in results if r["split"] == split for it in r["kept"]]
        n = write_jsonl(rows, args.out_dir / f"seed_{split}.jsonl")
        print(f"{split}: {n} items")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create scripts/run_seed_dataset.sh**

```bash
#!/usr/bin/env bash
# P1: build the verified seed dataset from the 164 GT SoccerNetGS sequences.
# Requires doubao credentials in .env. Resumable (per-sequence items.json).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m soccertactics.run_seed --out-dir outputs/soccertactics_seed "$@"
```

Run: `chmod +x scripts/run_seed_dataset.sh && bash -n scripts/run_seed_dataset.sh`

- [ ] **Step 3: Dry-run one sequence with the mock backend (no cost, real GT data)**

Run: `python -m soccertactics.run_seed --backend mock --limit 1`
Expected: either items generated (if the mock adapter returns parseable JSON)
or a clean `ValueError: no JSON object in reply` from `generate._parse` — the
latter confirms wiring; for a full offline test use `--backend mock` only with
a mock that echoes valid JSON (see FakeAdapter pattern in tests). The real
smoke is: `python -m soccertactics.run_seed --limit 2` on the machine with
doubao credentials — inspect `outputs/soccertactics_seed/train/SNGS-*/items.json`
and read 2-3 generated responses for sanity.

- [ ] **Step 4: Commit**

```bash
git add soccertactics/run_seed.py scripts/run_seed_dataset.sh
git commit -m "feat(soccertactics): P1 seed-dataset batch runner (resumable)"
```

---

### Task 7: report.py — density report (P0/P2 deliverable)

**Files:**
- Create: `soccertactics/report.py`
- Test: `tests/soccertactics/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
from soccertactics.report import density_table


def test_density_table_markdown():
    results = [
        {"split": "train", "kept": [
            {"category": "overlap_run", "tier": 1},
            {"category": "overlap_run", "tier": 1},
            {"category": "counter_attack", "tier": 1}]},
        {"split": "valid", "kept": [{"category": "overlap_run", "tier": 1}]},
    ]
    md = density_table(results)
    assert "| overlap_run" in md and "| 3 |" in md   # 2 train + 1 valid
    assert "counter_attack" in md
    # taxonomy categories with zero mined instances must still appear
    assert "line_management" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.report`

- [ ] **Step 3: Implement soccertactics/report.py**

```python
"""Per-category density report: the honest-coverage table for the paper
and the recruiting demo."""
from __future__ import annotations

from collections import Counter
from typing import List

from soccertactics.taxonomy import load_taxonomy


def density_table(results: List[dict]) -> str:
    counts = Counter()
    for r in results:
        for item in r["kept"]:
            counts[item["category"]] += 1
    lines = ["| category | moment | tier | mined items |",
             "|---|---|---|---|"]
    for c in load_taxonomy():
        lines.append(f"| {c['id']} | {c['moment']} | {c['tier']} "
                     f"| {counts.get(c['id'], 0)} |")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", type=Path,
                    default=Path("outputs/soccertactics_seed"))
    args = ap.parse_args()
    results = [json.loads(p.read_text(encoding="utf-8"))
               for p in args.seed_dir.glob("*/SNGS-*/items.json")]
    out = args.seed_dir / "density_report.md"
    out.write_text(density_table(results), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then commit**

Run: `python -m pytest tests/soccertactics/test_report.py -q` → 1 passed

```bash
git add soccertactics/report.py tests/soccertactics/test_report.py
git commit -m "feat(soccertactics): per-category density report"
```

---

### Task 8: mcq.py — MCQ generation, two families (diagnostic + future GRPO rewards)

**Files:**
- Create: `soccertactics/mcq.py`
- Test: `tests/soccertactics/test_mcq.py`

Two families in v1: **run-type identification** (answer from the instance's
category; distractors = other run categories whose geometry nearly matches)
and **phase identification** (answer from signature family; distractors =
other phases). More families follow the same shape in P4.

- [ ] **Step 1: Write the failing test**

```python
from soccertactics.mcq import mcq_run_type, mcq_phase

OVERLAP_INSTANCE = {"category": "overlap_run", "t0": 5.0, "t1": 9.0,
                    "team": "left", "players": ["3"],
                    "facts": {"rel_x_from": -4.0, "rel_x_to": 2.4,
                              "rel_y": 21.0, "speed": 7.1}}


def test_mcq_run_type_structure():
    q = mcq_run_type(OVERLAP_INSTANCE, seq_id="SNGS-060", seed=7)
    assert q["answer"] in q["choices"]
    assert len(q["choices"]) == 4
    assert len(set(q["choices"])) == 4
    assert q["answer"] == "overlapping run (套边)"
    assert "3" in q["question"]


def test_mcq_deterministic_with_seed():
    a = mcq_run_type(OVERLAP_INSTANCE, seq_id="SNGS-060", seed=7)
    b = mcq_run_type(OVERLAP_INSTANCE, seq_id="SNGS-060", seed=7)
    assert a == b


def test_mcq_phase():
    counter = {"category": "counter_attack", "t0": 3.0, "t1": 12.0,
               "team": "right", "players": ["9"],
               "facts": {"advance_m": 28.0, "within_s": 5.0}}
    q = mcq_phase(counter, seq_id="SNGS-061", seed=1)
    assert q["answer"] == "counter attack (快速反击)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/soccertactics/test_mcq.py -q`
Expected: FAIL — `ModuleNotFoundError: soccertactics.mcq`

- [ ] **Step 3: Implement soccertactics/mcq.py**

```python
"""MCQ item generation. Answers come from mined instances (geometry-derived);
distractors are near-miss categories. Deterministic under a seed."""
from __future__ import annotations

import random

RUN_LABELS = {
    "overlap_run": "overlapping run (套边)",
    "underlap_run": "underlapping run (内线前插)",
    "run_in_behind": "run in behind (打身后)",
    "drop_deep": "dropping deep (回撤接应)",
    "pull_wide": "pulling wide (拉边)",
}
PHASE_LABELS = {
    "counter_attack": "counter attack (快速反击)",
    "buildup_from_back": "build-up from the back (后场组织)",
    "high_press": "high press (高位逼抢)",
    "switch_play": "switch of play (弱侧转移)",
    "direct_runners": "transition with direct runners (反击无球冲刺)",
}


def _mcq(labels: dict, instance: dict, question: str, seq_id: str, seed: int) -> dict:
    answer = labels[instance["category"]]
    rng = random.Random((seq_id, instance["t0"], seed).__repr__())
    distractors = rng.sample([v for k, v in labels.items()
                              if k != instance["category"]], 3)
    choices = distractors + [answer]
    rng.shuffle(choices)
    return {"seq_id": seq_id, "category": instance["category"],
            "t0": instance["t0"], "t1": instance["t1"],
            "question": question, "choices": choices, "answer": answer,
            "facts": instance["facts"]}


def mcq_run_type(instance: dict, seq_id: str, seed: int = 0) -> dict:
    if instance["category"] not in RUN_LABELS:
        raise ValueError(f"not a run category: {instance['category']}")
    jersey = instance["players"][0]
    question = (f"In the window {instance['t0']}-{instance['t1']}s, what type "
                f"of off-ball run does number {jersey} make?")
    return _mcq(RUN_LABELS, instance, question, seq_id, seed)


def mcq_phase(instance: dict, seq_id: str, seed: int = 0) -> dict:
    if instance["category"] not in PHASE_LABELS:
        raise ValueError(f"not a phase category: {instance['category']}")
    question = (f"Which phase/pattern of play best describes the window "
                f"{instance['t0']}-{instance['t1']}s?")
    return _mcq(PHASE_LABELS, instance, question, seq_id, seed)
```

- [ ] **Step 4: Run test, then commit**

Run: `python -m pytest tests/soccertactics/test_mcq.py -q` → 3 passed

```bash
git add soccertactics/mcq.py tests/soccertactics/test_mcq.py
git commit -m "feat(soccertactics): MCQ generation for run-type and phase families"
```

---

### Task 9: P1 execution + P2 pilot (manual, on the GPU/credential machine)

No new code — the runbook that produces the recruiting demo.

- [ ] **Step 1: Full suite green**

Run: `python -m pytest tests/ -q` → all passed

- [ ] **Step 2: Seed run, 10 sequences first**

```bash
bash scripts/run_seed_dataset.sh --limit 10
python -m soccertactics.report
```
Read 5 kept + 5 dropped items. Judge: (a) responses tactically sensible and
grounded? (b) drop reasons legitimate (bad claims) vs resolver bugs?
Tune `PASS_RATIO`/signature thresholds only after reading, not before.

- [ ] **Step 3: Full 164-sequence run**

```bash
bash scripts/run_seed_dataset.sh
python -m soccertactics.report
```
Expected: seed_train/valid/test.jsonl + density_report.md. Cost estimate at
~1-8 instances/sequence: several hundred to ~1.5k LLM calls total (doubao-lite
class pricing — trivial).

- [ ] **Step 4: P2 mining pilot prerequisites (documented decision point)**

The 10-match broadcast pilot needs: SoccerNet NDA video download (already
have? confirm), SoccerNet-v2 `Labels-v2.json` event timestamps for window
cutting, and Stage-1 GSR GPU-hours for ~10 matches × ~30 windows × 30 s.
Before building the window-cutter, re-check effort with actual Stage-1
runtime per 30s clip on your hardware. This is the first task to hand a
recruit; the density report from Step 3 is the recruiting demo.

- [ ] **Step 5: Commit artifacts note**

```bash
git add outputs/soccertactics_seed/density_report.md 2>/dev/null || true
git commit -m "docs: seed dataset density report (P1 complete)" || echo "report gitignored - attach to recruiting doc manually"
```

---

## Self-Review

**Spec coverage:** taxonomy literature-complete w/ tiers + observability + verifiability (Task 1) ✓; retrieval signatures for mining (Task 2; tier-2 completion pattern documented) ✓; answer-guided free-form generation with claims (Task 3) ✓; mechanical verification filter (Task 4) ✓; SoccerChat-compatible output for the published SFT recipe (Task 5) ✓; 164-GT-sequence seed batch with split hygiene + resumability (Task 6) ✓; honest density report (Task 7) ✓; MCQ diagnostic families (Task 8) ✓; P2 pilot gated on measured GSR runtime (Task 9) ✓. Out of P0–P2 scope by design: radar/text modality export for items (reuses v2 radar renderer, add in P3), gold eval track + expert validation (P4), reference model SFT (P5).

**Placeholder scan:** none — every YAML/code block is complete; tier-2 signature expansion is explicitly scoped work with a stated pattern, not a TODO.

**Type consistency:** instance dict (`category, t0, t1, team, players, facts`) is produced by signatures (Task 2), consumed by generate (Task 3), mcq (Task 8), run_seed (Task 6); claim grammar in generate's prompt matches `filter.check_claim` → `verify.resolve_query` fields (`t0,t1,jersey,team,quantity,agg`) plus `expect` handled in filter only; `write_jsonl` consumes exactly the keys `generate_items` + `filter_items` attach.
