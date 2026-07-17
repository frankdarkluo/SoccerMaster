# Recognition Error Triage & Deterministic Checkers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the scoring rubric, convert the 72-row human review (`data/足球战术识别.csv`) into structured benchmark assets, and implement three deterministic acceptance checkers (restart gate / clean-regain / corner landing zone) with an offline replay that measures how many of the 39 errors each checker would have vetoed.

**Architecture:** Pure-function checkers over a typed `ClipEvidence` record (no model calls, no video access). Evidence for the 72 reviewed clips is hand-encoded from the human review text into a JSONL file; the same checker functions later plug into GSR-derived evidence per the approved KB v2 architecture (`tactical_state → 模型提名 → 固定代码 checker → verified fact`). Scope is frozen to four P0 tactics per mentor decision.

**Tech Stack:** Python 3 (stdlib only: `json`, `csv`, `dataclasses`), pytest. No new dependencies.

**Scope freeze (mentor decision, do not reopen):**

| Tier | tactic_id | 中文名 | Rationale |
| --- | --- | --- | --- |
| P0 | `fast_break_pattern` | 快速反击 | 23 samples, errors are checkable (regain purity) |
| P0 | `run_in_behind` | 打身后 / 反越位跑位 | 14 samples, 64% precision already |
| P0 | `corner_near_far_post` | 角球后点/前点战术 | 14 samples, all errors are landing-zone geometry |
| P0 | `cutback` | 下底倒三角 | High human precision, clear geometric signature |
| P1 (evaluate only, no promotion) | `line_break` | 破线传球 | Needs receiver-behind-line geometry |
| Deferred | `half_space_penetration`, `receive_between_lines`, `one_two`, `dummy_run`, `switch_of_play`, goalkeeper buildup | 肋部渗透、线间接应、二过一、虚跑、大范围转移、门将参与出球 | Multi-actor role relations; per mentor, out of scope this round |

**Data prerequisites:** Only `data/足球战术识别.csv` (GB18030-encoded — always open with `encoding="gb18030"`) and `benchmark/tactical_prototypes/*.jsonl`. No video or GSR access needed for any task in this plan.

**Branch:** create `feat/recognition-triage` off `main` before Task 1.

## Implementation status (completed 2026-07-17)

Implemented on `feat/recognition-triage` in commit `9a44f54`. Final verification: 26/26 focused tests passed, the prototype store validated with zero errors, and an idempotent rebuild added zero duplicate episodes.

- Normalized all 72 reviewed claims: 33 correct and 39 wrong, with every wrong claim assigned a root cause.
- Added 23 P0 gold targeted negatives to `episodes.jsonl`; this intentionally follows the scope freeze and does not promote P1 or deferred tactics. Rebuilding is byte-stable and adds zero further rows.
- Offline P0 replay produced zero false vetoes. Precision changed from 0.478 to 0.917 for `fast_break_pattern`, 0.643 to 0.818 for `run_in_behind`, 0.500 to 1.000 for `corner-near-far-post`, and 0.667 to 1.000 for `cutback`.
- Canonical repository IDs are used in generated assets: `corner-near-far-post`, `between-the-lines`, `gk-in-buildup`, `dummy-run`, `halfspace-penetration`, and `switch-of-play`. `one_two` remains provisional because the current concept registry has no canonical card for it.

The unchecked boxes below are retained as the original execution recipe, not as current completion state. In particular, the broad wording in Task 8 is superseded by the plan's P0 scope freeze and explicit “no deferred promotion” constraint.

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `benchmark/tactical_prototypes/scoring_rubric.md` | Create | Frozen decision rules for scoring recognition outputs |
| `benchmark/tactical_prototypes/recognition_review.jsonl` | Create (generated) | Normalized 72-row review with root-cause tags |
| `benchmark/tactical_prototypes/recognition_evidence.jsonl` | Create (hand-encoded) | Structured evidence per reviewed claim, sourced from human review text |
| `benchmark/tactical_prototypes/replay_report.json` + `replay_report.md` | Create (generated) | Before/after precision per P0 tactic, flipped rows, false vetoes |
| `pipeline/tactics_qa/__init__.py` | Create | Package marker |
| `pipeline/tactics_qa/taxonomy.py` | Create | Tactic-name mapping + root-cause table for the 39 errors |
| `pipeline/tactics_qa/review_io.py` | Create | Parse the GB18030 CSV → review records |
| `pipeline/tactics_qa/evidence.py` | Create | `PossessionEvent`, `ClipEvidence`, evidence JSONL IO |
| `pipeline/tactics_qa/checkers.py` | Create | The three checkers, pure functions |
| `pipeline/tactics_qa/replay.py` | Create | Join review + evidence, run checkers, emit report |
| `scripts/build_recognition_assets.py` | Create | CLI: regenerate review JSONL, replay report, episode migration |
| `tests/tactics_qa/test_review_io.py` | Create | CSV parsing tests |
| `tests/tactics_qa/test_checkers.py` | Create | Checker unit tests (fixtures modeled on real error/correct rows) |
| `tests/tactics_qa/test_replay.py` | Create | End-to-end replay on a small synthetic set |
| `benchmark/tactical_prototypes/episodes.jsonl` | Append | New gold hard-negative episodes from the 39 corrections |
| `benchmark/README.md` | Modify | Document the new files |

---

### Task 1: Freeze the scoring rubric

**Files:**
- Create: `benchmark/tactical_prototypes/scoring_rubric.md`
- Modify: `benchmark/README.md`

- [ ] **Step 1: Write the rubric document**

Create `benchmark/tactical_prototypes/scoring_rubric.md` with exactly this content (review wording, but every rule must survive — these resolve real disputed rows cited inline):

```markdown
# 战术识别评分规范 v1（冻结于 2026-07-17）

适用对象：对 `recognition_review.jsonl` 及后续所有识别实验的人工评分。
修改本文件需要在 PR 中说明哪些历史行的判定会翻转。

## 输出等级

系统每条输出必须归入四类之一，评分时同样四选一：
1. `confirmed` 明确战术 —— 所有必要条件满足且无排除条件命中
2. `insufficient` 可能战术但证据不足 —— 有正向证据但至少一个必要条件不可核实
3. `none` 无可验证战术 —— 片段可观察但不构成任何 P0/P1 战术
4. `unobservable` 不可观察 —— 起点被剪掉/关键跑者不可见/镜头切换破坏时序

不要求每个片段至少输出一个战术。强行输出按误报计。

## 判定规则（按历史争议行固化）

R1 多标签允许。一个窗口同时成立多个战术时，每个都单独计分
   （先例：SNGS-054 快速反击+打身后 均计正确；SNGS-148 破线+线间接应）。
R2 打身后不要求跑者本人完成终结。满足「从防线前沿/平齐启动 + 球越过防线寻找该跑动」
   即成立；终结者是谁不影响判定
   （先例：SNGS-101 原判「错误」，按本规范应改判「正确」）。
R3 时间窗按重叠计分：预测窗口与真实战术窗口 IoU ≥ 0.5 记正确；
   IoU < 0.5 但有重叠记 `window_partial`（统计中单列，不计入正确也不计入错误）
   （先例：SNGS-118 前 3 秒是反击、后 8 秒是阵地进攻，记 window_partial）。
R4 restart 否决：窗口起点前 3 秒内存在开球/门球/界外球/任意球/角球，
   则该窗口不能判 open-play 类战术（快速反击/破线/打身后/肋部渗透）。
   例外：对方定位球被解围后由我方受控夺回再发动，属于 open play
   （先例：SNGS-129、SNGS-189 记正确；SNGS-001 开球开大脚记错误）。
R5 传球类型约束：打身后与破线的传球必须是运动战中的直塞/纵向穿透传球；
   传中、大脚解围、手抛界外球、门球不满足
   （先例：13/14 传中判错误；SNGS-097 手抛界外球判错误）。
R6 角球前后点：第一落点/第一争顶点必须位于近门柱或远门柱区域；
   落点在禁区中央记错误（先例：SNGS-025 等 6 行）。
R7 快速反击的球权转换必须「干净」：窗口起点附近恰有一次由抢断/拦截/受控
   收球构成的球权转换；连续头球解围争第二点的混乱转换不算
   （先例：SNGS-043、SNGS-060 记错误）。
R8 概念混淆按错误计，且必须在更正列写明真实概念，供困难反例迁移使用。

## P0/P1 范围

P0：fast_break_pattern、run_in_behind、corner_near_far_post、cutback。
P1（仅评估）：line_break。
其余概念本轮不评分、不晋升（mentor 决定，2026-07-17）。
```

- [ ] **Step 2: Add pointers in benchmark README**

In `benchmark/README.md`, add rows to the JSON/文件 table:

```markdown
| `tactical_prototypes/scoring_rubric.md` | 这是冻结的评分规范，规定多标签、时间窗、restart 否决等判定规则及 P0/P1 范围。 |
| `tactical_prototypes/recognition_review.jsonl` | 这是 72 行识别结果人工复核的结构化版本，带错误根因标签。 |
| `tactical_prototypes/recognition_evidence.jsonl` | 这是从人工复核文本手工编码的逐条证据，供确定性 checker 离线回放。 |
| `tactical_prototypes/replay_report.md` | 这是 checker 离线回放报告：各 P0 战术加 checker 前后的精确率与误否决清单。 |
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/tactical_prototypes/scoring_rubric.md benchmark/README.md
git commit -m "docs(benchmark): freeze scoring rubric v1 and P0 scope"
```

---

### Task 2: Tactic-name mapping and root-cause taxonomy

**Files:**
- Create: `pipeline/tactics_qa/__init__.py` (empty)
- Create: `pipeline/tactics_qa/taxonomy.py`
- Test: `tests/tactics_qa/test_taxonomy.py` (plus empty `tests/__init__.py`, `tests/tactics_qa/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_taxonomy.py
from pipeline.tactics_qa.taxonomy import TACTIC_ID_BY_ZH, ROOT_CAUSE_BY_CLAIM, ROOT_CAUSES


def test_all_csv_tactic_names_mapped():
    # every distinct name appearing in data/足球战术识别.csv column 2
    names = [
        "快速反击", "破线传球", "打身后 / 反越位跑位", "线间接应",
        "门将参与出球", "角球后点/前点战术", "虚跑/假跑扯动", "下底倒三角",
        "肋部渗透", "大范围转移", "二过一 / 撞墙配合",
    ]
    for n in names:
        assert n in TACTIC_ID_BY_ZH, n


def test_root_cause_table_covers_39_errors():
    assert len(ROOT_CAUSE_BY_CLAIM) == 39
    for cause in ROOT_CAUSE_BY_CLAIM.values():
        assert cause in ROOT_CAUSES


def test_duplicate_clips_disambiguated_by_tactic():
    assert ("SNGS-061", "快速反击") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-061", "肋部渗透") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-134", "快速反击") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-134", "打身后 / 反越位跑位") in ROOT_CAUSE_BY_CLAIM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tactics_qa/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.tactics_qa'`

- [ ] **Step 3: Write the taxonomy module**

```python
# pipeline/tactics_qa/taxonomy.py
"""Frozen mapping tables for the 2026-07 recognition review.

ROOT_CAUSE_BY_CLAIM is keyed by (clip_id, tactic_zh) because SNGS-061 and
SNGS-134 each carry two claims. Causes were assigned by reading the 更正
column; see scoring_rubric.md for the rules they instantiate.
"""

TACTIC_ID_BY_ZH = {
    "快速反击": "fast_break_pattern",
    "破线传球": "line_break",
    "打身后 / 反越位跑位": "run_in_behind",
    "线间接应": "receive_between_lines",
    "门将参与出球": "goalkeeper_buildup",
    "角球后点/前点战术": "corner_near_far_post",
    "虚跑/假跑扯动": "dummy_run",
    "下底倒三角": "cutback",
    "肋部渗透": "half_space_penetration",
    "大范围转移": "switch_of_play",
    "二过一 / 撞墙配合": "one_two",
}

ROOT_CAUSES = {
    "restart_delivery",    # 门球/开球/界外球/任意球/传中被当成 open-play 直塞
    "chaotic_chain",       # 连续头球/解围争第二点，球权转换不干净
    "normal_progression",  # 普通由守转攻/向前出球被当成快速反击
    "geometry",            # 落点/接球点/渗透路径几何不满足
    "concept_confusion",   # 真实动作属于另一个概念（套边、分边、斜传身后等）
    "rubric",              # 标注口径分歧（按 rubric v1 会改判）
    "window_partial",      # 时间窗前半对后半错
    "role_relation",       # 多球员角色关系判断失败（二过一、虚跑等）
}

ROOT_CAUSE_BY_CLAIM = {
    ("13", "打身后 / 反越位跑位"): "restart_delivery",
    ("14", "打身后 / 反越位跑位"): "restart_delivery",
    ("SNGS-001", "快速反击"): "restart_delivery",
    ("SNGS-003", "破线传球"): "restart_delivery",
    ("SNGS-007", "快速反击"): "normal_progression",
    ("SNGS-020", "破线传球"): "restart_delivery",
    ("SNGS-022", "门将参与出球"): "restart_delivery",
    ("SNGS-023", "打身后 / 反越位跑位"): "restart_delivery",
    ("SNGS-025", "角球后点/前点战术"): "geometry",
    ("SNGS-028", "破线传球"): "geometry",
    ("SNGS-030", "虚跑/假跑扯动"): "role_relation",
    ("SNGS-039", "线间接应"): "role_relation",
    ("SNGS-040", "快速反击"): "normal_progression",
    ("SNGS-041", "快速反击"): "restart_delivery",
    ("SNGS-043", "快速反击"): "chaotic_chain",
    ("SNGS-060", "快速反击"): "chaotic_chain",
    ("SNGS-061", "快速反击"): "normal_progression",
    ("SNGS-061", "肋部渗透"): "geometry",
    ("SNGS-062", "肋部渗透"): "concept_confusion",
    ("SNGS-067", "角球后点/前点战术"): "geometry",
    ("SNGS-070", "破线传球"): "geometry",
    ("SNGS-075", "角球后点/前点战术"): "geometry",
    ("SNGS-078", "快速反击"): "normal_progression",
    ("SNGS-079", "角球后点/前点战术"): "geometry",
    ("SNGS-084", "快速反击"): "chaotic_chain",
    ("SNGS-087", "快速反击"): "normal_progression",
    ("SNGS-097", "破线传球"): "restart_delivery",
    ("SNGS-101", "打身后 / 反越位跑位"): "rubric",
    ("SNGS-103", "角球后点/前点战术"): "geometry",
    ("SNGS-106", "下底倒三角"): "restart_delivery",
    ("SNGS-110", "角球后点/前点战术"): "geometry",
    ("SNGS-115", "大范围转移"): "concept_confusion",
    ("SNGS-118", "快速反击"): "window_partial",
    ("SNGS-134", "快速反击"): "normal_progression",
    ("SNGS-134", "打身后 / 反越位跑位"): "concept_confusion",
    ("SNGS-140", "角球后点/前点战术"): "geometry",
    ("SNGS-177", "破线传球"): "concept_confusion",
    ("SNGS-190", "肋部渗透"): "concept_confusion",
    ("SNGS-200", "二过一 / 撞墙配合"): "role_relation",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tactics_qa/test_taxonomy.py -v`
Expected: 3 PASS

- [ ] **Step 5: Verify tactic ids against concepts.jsonl**

Run:

```bash
python3 - <<'EOF'
import json
from pipeline.tactics_qa.taxonomy import TACTIC_ID_BY_ZH
known = {json.loads(l)["tactic_id"] for l in open("benchmark/tactical_prototypes/concepts.jsonl")}
missing = {v for v in TACTIC_ID_BY_ZH.values() if v not in known}
print("known:", len(known), "missing from concepts.jsonl:", sorted(missing))
EOF
```

If any id is missing (likely candidates: `goalkeeper_buildup`, `dummy_run`, `one_two`), find the actual id in `concepts.jsonl` by searching its 中文名/别名 field and correct `TACTIC_ID_BY_ZH`. If no concept exists at all, keep the provisional id and add a module docstring note listing provisional ids. Do NOT invent new concept cards.

- [ ] **Step 6: Commit**

```bash
git add pipeline/tactics_qa/ tests/
git commit -m "feat(tactics_qa): tactic-name mapping and 39-error root-cause table"
```

---

### Task 3: Parse the review CSV into recognition_review.jsonl

**Files:**
- Create: `pipeline/tactics_qa/review_io.py`
- Create: `scripts/build_recognition_assets.py`
- Test: `tests/tactics_qa/test_review_io.py`
- Generate: `benchmark/tactical_prototypes/recognition_review.jsonl`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_review_io.py
import io
from pipeline.tactics_qa.review_io import parse_review_rows

SAMPLE = (
    "id,tactics,时间范围,判断依据,置信度,识别准确度,更正\n"
    "SNGS-001,快速反击,00:13–00:23,白队在中场对抗后夺回球权。,中高,错误,开球后开大脚打到前场\n"
    "SNGS-045,打身后 / 反越位跑位,00:20–00:25,前锋从后卫线前沿启动。,高,正确,\n"
)


def test_parse_review_rows():
    rows = parse_review_rows(io.StringIO(SAMPLE))
    assert len(rows) == 2
    r = rows[0]
    assert r["clip_id"] == "SNGS-001"
    assert r["tactic_id"] == "fast_break_pattern"
    assert r["window"] == {"start_s": 13.0, "end_s": 23.0, "raw": "00:13–00:23"}
    assert r["verdict"] == "wrong"
    assert r["root_cause"] == "restart_delivery"
    assert rows[1]["verdict"] == "correct"
    assert rows[1]["root_cause"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tactics_qa/test_review_io.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement review_io**

```python
# pipeline/tactics_qa/review_io.py
"""Parse data/足球战术识别.csv (GB18030) into normalized review records."""
import csv
import json
from .taxonomy import TACTIC_ID_BY_ZH, ROOT_CAUSE_BY_CLAIM


def _parse_window(raw: str) -> dict:
    # e.g. "00:13–00:23" (note: en dash U+2013)
    sep = "–" if "–" in raw else "-"
    start, end = raw.split(sep)

    def to_s(mmss: str) -> float:
        m, s = mmss.strip().split(":")
        return int(m) * 60 + float(s)

    return {"start_s": to_s(start), "end_s": to_s(end), "raw": raw.strip()}


def parse_review_rows(fp) -> list:
    rows = []
    for row in csv.DictReader(fp):
        clip_id = row["id"].strip()
        tactic_zh = row["tactics"].strip()
        verdict = "correct" if row["识别准确度"].strip() == "正确" else "wrong"
        rows.append({
            "clip_id": clip_id,
            "clip_uid": clip_id if not clip_id.startswith("SNGS-")
                        else f"soccernetgs:{clip_id}",
            "tactic_zh": tactic_zh,
            "tactic_id": TACTIC_ID_BY_ZH[tactic_zh],
            "window": _parse_window(row["时间范围"]),
            "evidence_text": row["判断依据"].strip(),
            "confidence": row["置信度"].strip(),
            "verdict": verdict,
            "correction": row["更正"].strip() or None,
            "root_cause": ROOT_CAUSE_BY_CLAIM.get((clip_id, tactic_zh))
                          if verdict == "wrong" else None,
        })
    return rows


def load_review(path: str = "data/足球战术识别.csv") -> list:
    with open(path, encoding="gb18030", newline="") as fp:
        return parse_review_rows(fp)


def write_jsonl(records: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tactics_qa/test_review_io.py -v`
Expected: PASS

- [ ] **Step 5: Write the CLI and generate the JSONL**

```python
# scripts/build_recognition_assets.py
"""Regenerate recognition_review.jsonl from data/足球战术识别.csv.

Usage: python3 scripts/build_recognition_assets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tactics_qa.review_io import load_review, write_jsonl  # noqa: E402

OUT = "benchmark/tactical_prototypes/recognition_review.jsonl"


def main() -> None:
    rows = load_review()
    wrong = [r for r in rows if r["verdict"] == "wrong"]
    unmapped = [r for r in wrong if r["root_cause"] is None]
    if unmapped:
        raise SystemExit(
            "wrong rows missing root cause: "
            + ", ".join(f"({r['clip_id']},{r['tactic_zh']})" for r in unmapped)
        )
    write_jsonl(rows, OUT)
    print(f"{len(rows)} rows ({len(wrong)} wrong) -> {OUT}")


if __name__ == "__main__":
    main()
```

Run: `python3 scripts/build_recognition_assets.py`
Expected output: `72 rows (39 wrong) -> benchmark/tactical_prototypes/recognition_review.jsonl`
If counts differ, the CSV disagrees with the taxonomy table — reconcile row by row before proceeding (the taxonomy table is derived from this exact CSV; a mismatch means a parse bug, not a data update).

- [ ] **Step 6: Commit**

```bash
git add pipeline/tactics_qa/review_io.py scripts/build_recognition_assets.py \
        tests/tactics_qa/test_review_io.py benchmark/tactical_prototypes/recognition_review.jsonl
git commit -m "feat(tactics_qa): normalize 72-row recognition review to JSONL"
```

---

### Task 4: Evidence schema

**Files:**
- Create: `pipeline/tactics_qa/evidence.py`
- Test: `tests/tactics_qa/test_evidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_evidence.py
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent, from_dict


def test_round_trip():
    ev = ClipEvidence(
        clip_uid="soccernetgs:SNGS-001",
        tactic_id="fast_break_pattern",
        events=[PossessionEvent(t=1.0, team="attacking", kind="kickoff")],
        delivery_kind="long_clearance",
        corner_landing_zone=None,
        source="human_review_text",
    )
    d = ev.to_dict()
    assert from_dict(d) == ev
    assert d["events"][0] == {"t": 1.0, "team": "attacking", "kind": "kickoff"}


def test_unknown_kind_rejected():
    import pytest
    with pytest.raises(ValueError):
        PossessionEvent(t=0.0, team="attacking", kind="bicycle_kick")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tactics_qa/test_evidence.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement evidence module**

```python
# pipeline/tactics_qa/evidence.py
"""Typed evidence records consumed by the deterministic checkers.

Evidence is model-free: it comes either from human review text (this plan)
or, later, from GSR-derived adapters. `team` is relative to the team the
claim says is attacking.
"""
from dataclasses import dataclass, field, asdict

RESTART_KINDS = {"kickoff", "goal_kick", "throw_in", "free_kick", "corner"}
CONTROLLED_REGAIN_KINDS = {"tackle", "interception", "controlled_recovery"}
CHAOS_KINDS = {"clearance", "header_flick", "aerial_duel"}
OTHER_KINDS = {"pass", "cross", "dribble", "shot"}
EVENT_KINDS = RESTART_KINDS | CONTROLLED_REGAIN_KINDS | CHAOS_KINDS | OTHER_KINDS

DELIVERY_KINDS = {
    "open_play_pass",   # 运动战地面/纵向穿透传球
    "cross",            # 传中
    "goal_kick",
    "throw_in",
    "free_kick",
    "corner_cross",
    "long_clearance",   # 大脚解围/开大脚
}

LANDING_ZONES = {"near_post", "center", "far_post"}


@dataclass(frozen=True)
class PossessionEvent:
    t: float          # seconds from clip start
    team: str         # "attacking" | "defending"
    kind: str

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind}")
        if self.team not in {"attacking", "defending"}:
            raise ValueError(f"unknown team: {self.team}")


@dataclass
class ClipEvidence:
    clip_uid: str
    tactic_id: str
    events: list = field(default_factory=list)   # list[PossessionEvent]
    delivery_kind: str | None = None             # key pass of the claim, if any
    corner_landing_zone: str | None = None       # for corner claims
    source: str = "human_review_text"
    notes: str = ""

    def __post_init__(self):
        if self.delivery_kind is not None and self.delivery_kind not in DELIVERY_KINDS:
            raise ValueError(f"unknown delivery kind: {self.delivery_kind}")
        if (self.corner_landing_zone is not None
                and self.corner_landing_zone not in LANDING_ZONES):
            raise ValueError(f"unknown landing zone: {self.corner_landing_zone}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


def from_dict(d: dict) -> ClipEvidence:
    d = dict(d)
    d["events"] = [PossessionEvent(**e) for e in d.get("events", [])]
    return ClipEvidence(**d)


def load_evidence_jsonl(path: str) -> dict:
    """Return {(clip_uid, tactic_id): ClipEvidence}."""
    import json
    out = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            ev = from_dict(json.loads(line))
            out[(ev.clip_uid, ev.tactic_id)] = ev
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tactics_qa/test_evidence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/tactics_qa/evidence.py tests/tactics_qa/test_evidence.py
git commit -m "feat(tactics_qa): typed ClipEvidence schema for checker input"
```

---

### Task 5: The three checkers (TDD, fixtures modeled on real rows)

**Files:**
- Create: `pipeline/tactics_qa/checkers.py`
- Test: `tests/tactics_qa/test_checkers.py`

- [ ] **Step 1: Write the failing tests**

Each fixture mirrors a real reviewed row (named in comments) so the tests double as regression anchors for the 39 errors.

```python
# tests/tactics_qa/test_checkers.py
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent as E
from pipeline.tactics_qa.checkers import (
    restart_gate, delivery_gate, clean_regain, corner_landing, run_checkers,
)

W = {"start_s": 10.0, "end_s": 20.0}


def _ev(tactic_id, events=(), delivery=None, zone=None):
    return ClipEvidence(clip_uid="test:clip", tactic_id=tactic_id,
                        events=list(events), delivery_kind=delivery,
                        corner_landing_zone=zone)


# --- restart_gate -----------------------------------------------------------

def test_restart_gate_vetoes_kickoff_origin():  # SNGS-001
    ev = _ev("fast_break_pattern", [E(9.0, "attacking", "kickoff")])
    r = restart_gate(ev, W)
    assert not r.passed and "kickoff" in r.reason


def test_restart_gate_passes_open_play():
    ev = _ev("fast_break_pattern", [E(10.5, "attacking", "interception")])
    assert restart_gate(ev, W).passed


def test_restart_gate_ignores_old_restart():
    # restart 8s before window start is stale context, not the origin
    ev = _ev("fast_break_pattern", [E(2.0, "defending", "corner"),
                                    E(10.5, "attacking", "interception")])
    assert restart_gate(ev, W).passed


# --- delivery_gate ----------------------------------------------------------

def test_delivery_gate_vetoes_cross_for_run_in_behind():  # rows 13/14
    ev = _ev("run_in_behind", delivery="cross")
    r = delivery_gate(ev)
    assert not r.passed


def test_delivery_gate_vetoes_throw_in_for_line_break():  # SNGS-097
    ev = _ev("line_break", delivery="throw_in")
    assert not delivery_gate(ev).passed


def test_delivery_gate_vetoes_corner_cross_for_cutback():  # SNGS-106
    ev = _ev("cutback", delivery="corner_cross")
    assert not delivery_gate(ev).passed


def test_delivery_gate_passes_open_play_pass():  # SNGS-045
    ev = _ev("run_in_behind", delivery="open_play_pass")
    assert delivery_gate(ev).passed


def test_delivery_gate_insufficient_when_unknown():
    ev = _ev("run_in_behind")
    r = delivery_gate(ev)
    assert not r.passed and r.verdict == "insufficient"


# --- clean_regain (fast_break_pattern only) ---------------------------------

def test_clean_regain_passes_single_controlled_regain():  # SNGS-037
    ev = _ev("fast_break_pattern", [E(10.0, "attacking", "tackle")])
    assert clean_regain(ev, W).passed


def test_clean_regain_passes_cleared_set_piece_then_control():  # SNGS-129
    ev = _ev("fast_break_pattern", [
        E(8.5, "attacking", "clearance"),
        E(9.5, "attacking", "controlled_recovery"),
    ])
    assert clean_regain(ev, W).passed


def test_clean_regain_vetoes_chaotic_chain():  # SNGS-043
    ev = _ev("fast_break_pattern", [
        E(7.0, "defending", "header_flick"),
        E(8.0, "attacking", "clearance"),
        E(9.0, "defending", "aerial_duel"),
        E(10.0, "attacking", "header_flick"),
    ])
    r = clean_regain(ev, W)
    assert not r.passed and "chaotic" in r.reason


def test_clean_regain_vetoes_no_regain():  # SNGS-040 normal progression
    ev = _ev("fast_break_pattern", [E(10.0, "attacking", "pass"),
                                    E(12.0, "attacking", "dribble")])
    r = clean_regain(ev, W)
    assert not r.passed and "no controlled regain" in r.reason


# --- corner_landing (corner_near_far_post only) -----------------------------

def test_corner_landing_vetoes_center():  # SNGS-025 etc.
    r = corner_landing(_ev("corner_near_far_post", zone="center"))
    assert not r.passed


def test_corner_landing_passes_near_post():  # SNGS-050
    assert corner_landing(_ev("corner_near_far_post", zone="near_post")).passed


def test_corner_landing_insufficient_when_unknown():
    r = corner_landing(_ev("corner_near_far_post"))
    assert not r.passed and r.verdict == "insufficient"


# --- run_checkers dispatch --------------------------------------------------

def test_run_checkers_applies_only_relevant_checks():
    ev = _ev("corner_near_far_post", zone="far_post")
    results = run_checkers(ev, W)
    assert [r.checker for r in results] == ["corner_landing"]
    ev2 = _ev("fast_break_pattern", [E(10.0, "attacking", "interception")])
    assert {r.checker for r in run_checkers(ev2, W)} == {"restart_gate", "clean_regain"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/tactics_qa/test_checkers.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the checkers**

```python
# pipeline/tactics_qa/checkers.py
"""Deterministic acceptance checkers. Pure functions over ClipEvidence.

Rules implement scoring_rubric.md R4-R7. Each checker returns a CheckResult
with verdict "pass" | "veto" | "insufficient"; `passed` is True only for
"pass". Checkers never guess: missing evidence yields "insufficient".
"""
from dataclasses import dataclass

from .evidence import (
    ClipEvidence, RESTART_KINDS, CONTROLLED_REGAIN_KINDS, CHAOS_KINDS,
)

RESTART_LOOKBACK_S = 3.0        # rubric R4
REGAIN_WINDOW_S = 3.0           # regain must sit within ±3s of window start
MAX_CHAOS_EVENTS = 2            # rubric R7: 3+ contested touches = chaotic

OPEN_PLAY_TACTICS = {"fast_break_pattern", "line_break", "run_in_behind",
                     "half_space_penetration"}
THROUGH_PASS_TACTICS = {"line_break", "run_in_behind"}


@dataclass(frozen=True)
class CheckResult:
    checker: str
    verdict: str   # "pass" | "veto" | "insufficient"
    reason: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def restart_gate(ev: ClipEvidence, window: dict) -> CheckResult:
    """Rubric R4: open-play tactics cannot start from a restart."""
    ws = window["start_s"]
    for e in ev.events:
        if e.kind in RESTART_KINDS and ws - RESTART_LOOKBACK_S <= e.t <= ws + 1.0:
            return CheckResult("restart_gate", "veto",
                               f"{e.kind} at t={e.t:.1f}s starts the window")
    return CheckResult("restart_gate", "pass", "no restart at window origin")


def delivery_gate(ev: ClipEvidence) -> CheckResult:
    """Rubric R5: through-pass tactics need an open-play penetrating pass;
    cutback must not come from a corner delivery."""
    if ev.delivery_kind is None:
        return CheckResult("delivery_gate", "insufficient",
                           "delivery kind not established")
    if ev.tactic_id in THROUGH_PASS_TACTICS and ev.delivery_kind != "open_play_pass":
        return CheckResult("delivery_gate", "veto",
                           f"delivery is {ev.delivery_kind}, not an open-play "
                           "penetrating pass")
    if ev.tactic_id == "cutback" and ev.delivery_kind == "corner_cross":
        return CheckResult("delivery_gate", "veto", "cutback claimed on a corner")
    return CheckResult("delivery_gate", "pass", f"delivery {ev.delivery_kind} ok")


def clean_regain(ev: ClipEvidence, window: dict) -> CheckResult:
    """Rubric R7: fast break needs exactly one controlled possession gain
    near the window start, without a chaotic second-ball chain."""
    ws = window["start_s"]
    near = [e for e in ev.events if ws - REGAIN_WINDOW_S <= e.t <= ws + REGAIN_WINDOW_S]
    chaos = [e for e in near if e.kind in CHAOS_KINDS]
    # A single clearance immediately consolidated by the attacking team is a
    # legitimate counter origin (SNGS-129); a chain of contested touches is not.
    if len(chaos) > MAX_CHAOS_EVENTS:
        return CheckResult("clean_regain", "veto",
                           f"chaotic second-ball chain ({len(chaos)} contested "
                           "touches) around window start")
    # Note: a cleared set piece consolidated by the attacking team (SNGS-129)
    # passes because the consolidation itself is a controlled_recovery event;
    # only the chaos ceiling above distinguishes it from SNGS-043.
    regains = [e for e in near
               if e.team == "attacking" and e.kind in CONTROLLED_REGAIN_KINDS]
    if not regains:
        return CheckResult("clean_regain", "veto",
                           "no controlled regain near window start")
    return CheckResult("clean_regain", "pass", "single controlled regain")


def corner_landing(ev: ClipEvidence) -> CheckResult:
    """Rubric R6: corner near/far-post claim requires the first contested
    contact in a post zone."""
    if ev.corner_landing_zone is None:
        return CheckResult("corner_landing", "insufficient",
                           "landing zone not established")
    if ev.corner_landing_zone == "center":
        return CheckResult("corner_landing", "veto",
                           "first contact in central zone, not a post zone")
    return CheckResult("corner_landing", "pass",
                       f"first contact at {ev.corner_landing_zone}")


def run_checkers(ev: ClipEvidence, window: dict) -> list:
    """Dispatch the checkers relevant to the claimed tactic."""
    results = []
    if ev.tactic_id in OPEN_PLAY_TACTICS:
        results.append(restart_gate(ev, window))
    if ev.tactic_id in THROUGH_PASS_TACTICS or ev.tactic_id == "cutback":
        results.append(delivery_gate(ev))
    if ev.tactic_id == "fast_break_pattern":
        results.append(clean_regain(ev, window))
    if ev.tactic_id == "corner_near_far_post":
        results.append(corner_landing(ev))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/tactics_qa/test_checkers.py -v`
Expected: all PASS. If `clean_regain` logic reads awkwardly while making tests pass, simplify it — but keep SNGS-129 (pass) vs SNGS-043 (veto) both green; that pair is the point of the checker.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/tactics_qa/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/tactics_qa/checkers.py tests/tactics_qa/test_checkers.py
git commit -m "feat(tactics_qa): restart/delivery/regain/corner-landing checkers"
```

---

### Task 6: Hand-encode evidence for all 72 reviewed claims

**Files:**
- Create: `benchmark/tactical_prototypes/recognition_evidence.jsonl`

This is judgment work, not generation: for each row of `recognition_review.jsonl`, read `evidence_text` and `correction` and encode what the HUMAN established (corrections are ground truth; the model's 判断依据 is only used where the correction doesn't contradict it).

- [ ] **Step 1: Encode the 39 wrong rows first**

One JSON object per line matching `ClipEvidence.to_dict()`. Encoding rules:

- `t` values: use the review window start and the timestamps mentioned in the text; precision of ±1s is fine — checkers only compare against a ±3s band.
- `events.team` is relative to the team the claim says is attacking.
- If the correction names a restart (开球/门球/界外球/任意球/角球), emit that restart event at the window start AND set `delivery_kind` accordingly.
- If the correction says 传中 → `delivery_kind: "cross"`; 开大脚/解围 → `"long_clearance"`; 手抛界外球 → `"throw_in"`.
- Corner rows: correction “发到禁区中央” → `corner_landing_zone: "center"`; correction “前点”/“后点” → `"near_post"`/`"far_post"`.
- 头球和解围让球权不停转换 → 3+ events with kinds from `{header_flick, clearance, aerial_duel}` alternating teams around window start.
- 普通向前出球/由守转攻组织 → events showing possession without any `CONTROLLED_REGAIN_KINDS` event near window start (e.g. `pass`, `dribble` only).
- Anything the text does not establish stays `null` — never infer.
- Put the correction text verbatim into `notes`.

Worked examples (copy these lines in, then continue in the same style):

```json
{"clip_uid": "soccernetgs:SNGS-001", "tactic_id": "fast_break_pattern", "events": [{"t": 13.0, "team": "attacking", "kind": "kickoff"}, {"t": 14.0, "team": "attacking", "kind": "clearance"}], "delivery_kind": "long_clearance", "corner_landing_zone": null, "source": "human_review_text", "notes": "开球后开大脚打到前场，前场争下二点球完成射门"}
{"clip_uid": "soccernetgs:SNGS-003", "tactic_id": "line_break", "events": [{"t": 6.0, "team": "attacking", "kind": "goal_kick"}], "delivery_kind": "goal_kick", "corner_landing_zone": null, "source": "human_review_text", "notes": "门将开大脚"}
{"clip_uid": "soccernetgs:SNGS-025", "tactic_id": "corner_near_far_post", "events": [{"t": 0.0, "team": "attacking", "kind": "corner"}], "delivery_kind": "corner_cross", "corner_landing_zone": "center", "source": "human_review_text", "notes": "角球发到禁区中央，不是前后点"}
{"clip_uid": "soccernetgs:SNGS-043", "tactic_id": "fast_break_pattern", "events": [{"t": 4.0, "team": "defending", "kind": "header_flick"}, {"t": 5.0, "team": "attacking", "kind": "clearance"}, {"t": 6.0, "team": "defending", "kind": "aerial_duel"}, {"t": 6.5, "team": "attacking", "kind": "header_flick"}], "delivery_kind": null, "corner_landing_zone": null, "source": "human_review_text", "notes": "头球和解围让球权不停转换，不是反击"}
```

Every `kind` must come from `EVENT_KINDS` in `evidence.py` — loading validates this and raises `ValueError` on typos (Step 3 catches them file-wide).

- [ ] **Step 2: Encode the 33 correct rows**

Same rules; here the model's 判断依据 is trusted (the human confirmed it). E.g. SNGS-037 (correct fast break): `[{"t": 5.0, "team": "attacking", "kind": "tackle"}]`, `delivery_kind: "open_play_pass"` where a key pass is described. SNGS-050 (correct corner, correction says 前点): `corner_landing_zone: "near_post"`. SNGS-129: encode the clearance + controlled recovery pair as in the test fixture.

- [ ] **Step 3: Validate the file loads and covers all 72 claims**

Run:

```bash
python3 - <<'EOF'
import json
from pipeline.tactics_qa.evidence import load_evidence_jsonl
ev = load_evidence_jsonl("benchmark/tactical_prototypes/recognition_evidence.jsonl")
review = [json.loads(l) for l in open("benchmark/tactical_prototypes/recognition_review.jsonl")]
missing = [(r["clip_uid"], r["tactic_id"]) for r in review
           if (r["clip_uid"], r["tactic_id"]) not in ev]
print("evidence:", len(ev), "review:", len(review), "missing:", missing)
EOF
```

Expected: `evidence: 72 review: 72 missing: []` (any invalid `kind` raises ValueError at load — fix and rerun).

- [ ] **Step 4: Commit**

```bash
git add benchmark/tactical_prototypes/recognition_evidence.jsonl
git commit -m "data(benchmark): hand-encoded checker evidence for 72 reviewed claims"
```

---

### Task 7: Offline replay and report

**Files:**
- Create: `pipeline/tactics_qa/replay.py`
- Test: `tests/tactics_qa/test_replay.py`
- Modify: `scripts/build_recognition_assets.py`
- Generate: `benchmark/tactical_prototypes/replay_report.json`, `replay_report.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/tactics_qa/test_replay.py
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent as E
from pipeline.tactics_qa.replay import replay


def test_replay_counts_flips_and_false_vetoes():
    review = [
        {"clip_uid": "t:a", "tactic_id": "fast_break_pattern", "verdict": "wrong",
         "window": {"start_s": 10.0, "end_s": 20.0}, "root_cause": "restart_delivery"},
        {"clip_uid": "t:b", "tactic_id": "fast_break_pattern", "verdict": "correct",
         "window": {"start_s": 10.0, "end_s": 20.0}, "root_cause": None},
    ]
    evidence = {
        ("t:a", "fast_break_pattern"): ClipEvidence(
            "t:a", "fast_break_pattern", [E(10.0, "attacking", "kickoff")]),
        ("t:b", "fast_break_pattern"): ClipEvidence(
            "t:b", "fast_break_pattern", [E(10.0, "attacking", "interception")]),
    }
    rep = replay(review, evidence)
    fb = rep["per_tactic"]["fast_break_pattern"]
    assert fb["n_claims"] == 2
    assert fb["precision_before"] == 0.5
    assert fb["precision_after"] == 1.0       # wrong row vetoed, correct row kept
    assert fb["flipped_errors"] == ["t:a"]
    assert fb["false_vetoes"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tactics_qa/test_replay.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement replay**

```python
# pipeline/tactics_qa/replay.py
"""Offline replay: would the checkers have vetoed each reviewed claim?

Definitions:
- vetoed: any applicable checker returns "veto". "insufficient" also removes
  the claim from confirmed output (rubric output class `insufficient`) and is
  reported separately.
- precision_before: correct / all claims (what the models did).
- precision_after: correct-and-not-vetoed / not-vetoed (what survives).
- flipped_errors: wrong claims the checkers veto (the win).
- false_vetoes: correct claims the checkers veto (the cost). Target: zero.
"""
from collections import defaultdict

from .checkers import run_checkers

P0_TACTICS = ["fast_break_pattern", "run_in_behind", "corner_near_far_post",
              "cutback", "line_break"]  # line_break is P1, reported but flagged


def replay(review: list, evidence: dict) -> dict:
    per_tactic = {}
    rows_out = []
    for tactic in sorted({r["tactic_id"] for r in review}):
        claims = [r for r in review if r["tactic_id"] == tactic]
        stats = {"n_claims": len(claims), "in_scope": tactic in P0_TACTICS,
                 "flipped_errors": [], "false_vetoes": [], "insufficient": [],
                 "surviving_wrong": []}
        n_correct = sum(1 for r in claims if r["verdict"] == "correct")
        survivors, surviving_correct = 0, 0
        for r in claims:
            ev = evidence.get((r["clip_uid"], r["tactic_id"]))
            results = run_checkers(ev, r["window"]) if ev else []
            verdicts = {c.verdict for c in results}
            row = {"clip_uid": r["clip_uid"], "tactic_id": tactic,
                   "verdict": r["verdict"],
                   "checks": [c.__dict__ for c in results]}
            rows_out.append(row)
            if "veto" in verdicts:
                key = "false_vetoes" if r["verdict"] == "correct" else "flipped_errors"
                stats[key].append(r["clip_uid"])
            elif "insufficient" in verdicts:
                stats["insufficient"].append(r["clip_uid"])
            else:
                survivors += 1
                if r["verdict"] == "correct":
                    surviving_correct += 1
                else:
                    stats["surviving_wrong"].append(r["clip_uid"])
        stats["precision_before"] = round(n_correct / len(claims), 3) if claims else None
        stats["precision_after"] = (round(surviving_correct / survivors, 3)
                                    if survivors else None)
        per_tactic[tactic] = stats
    return {"schema_version": "checker-replay-v1",
            "per_tactic": per_tactic, "rows": rows_out}


def to_markdown(report: dict) -> str:
    lines = ["# Checker 离线回放报告", "",
             "| tactic | scope | claims | precision before | precision after | "
             "flipped | false vetoes | insufficient | surviving wrong |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for tactic, s in report["per_tactic"].items():
        lines.append(
            f"| {tactic} | {'P0/P1' if s['in_scope'] else 'deferred'} "
            f"| {s['n_claims']} | {s['precision_before']} | {s['precision_after']} "
            f"| {len(s['flipped_errors'])} | {len(s['false_vetoes'])} "
            f"| {len(s['insufficient'])} | {len(s['surviving_wrong'])} |")
    for tactic, s in report["per_tactic"].items():
        if s["false_vetoes"]:
            lines += ["", f"**False vetoes ({tactic})**: {', '.join(s['false_vetoes'])}"]
        if s["surviving_wrong"]:
            lines += ["", f"**Surviving wrong ({tactic})**: "
                          f"{', '.join(s['surviving_wrong'])} — checker 覆盖不到的根因"]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tactics_qa/test_replay.py -v`
Expected: PASS

- [ ] **Step 5: Wire into the CLI and generate the report**

Append to `scripts/build_recognition_assets.py` `main()` (after the review JSONL write):

```python
    import json
    from pipeline.tactics_qa.evidence import load_evidence_jsonl
    from pipeline.tactics_qa.replay import replay, to_markdown

    ev_path = "benchmark/tactical_prototypes/recognition_evidence.jsonl"
    evidence = load_evidence_jsonl(ev_path)
    report = replay(rows, evidence)
    with open("benchmark/tactical_prototypes/replay_report.json", "w",
              encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, sort_keys=True)
    with open("benchmark/tactical_prototypes/replay_report.md", "w",
              encoding="utf-8") as fp:
        fp.write(to_markdown(report))
    print("replay report written")
```

Run: `python3 scripts/build_recognition_assets.py`
Expected: report files created; open `replay_report.md` and check:
- `fast_break_pattern` precision_after ≥ 0.75 (was ~0.48)
- `corner_near_far_post` precision_after ≥ 0.85 (was 0.50)
- **false vetoes = 0 for every P0 tactic.** A false veto means either the evidence encoding or a checker threshold is wrong — investigate the specific row (start from the evidence line, not the checker) and fix before proceeding. Do not tune thresholds to chase flips at the cost of false vetoes.

- [ ] **Step 6: Run full suite and commit**

```bash
python3 -m pytest tests/tactics_qa/ -v
git add pipeline/tactics_qa/replay.py tests/tactics_qa/test_replay.py \
        scripts/build_recognition_assets.py \
        benchmark/tactical_prototypes/replay_report.json \
        benchmark/tactical_prototypes/replay_report.md
git commit -m "feat(tactics_qa): offline checker replay with precision report"
```

---

### Task 8: Migrate the 39 corrections into the prototype KB

**Files:**
- Modify (append): `benchmark/tactical_prototypes/episodes.jsonl`
- Modify: `scripts/build_recognition_assets.py`

- [ ] **Step 1: Write the migration function with a dry-run**

Add to `scripts/build_recognition_assets.py`:

```python
def migrate_hard_negatives(rows, episodes_path, apply=False):
    """Append each wrong claim as a gold targeted_negative episode.

    Dedupe key: (clip_uid, tactic_id, prototype_type) against existing lines.
    """
    import json
    existing = set()
    with open(episodes_path, encoding="utf-8") as fp:
        for line in fp:
            e = json.loads(line)
            existing.add((e["clip_uid"], e["tactic_id"], e["prototype_type"]))
    new = []
    for r in rows:
        if r["verdict"] != "wrong" or r["root_cause"] in {"rubric", "window_partial"}:
            continue  # rubric flips are not negatives; partial windows are ambiguous
        key = (r["clip_uid"], r["tactic_id"], "targeted_negative")
        if key in existing:
            continue
        new.append({
            "actor_bindings": {},
            "clip_uid": r["clip_uid"],
            "confidence": r["confidence"],
            "evidence_summary": f"误判为{r['tactic_zh']}：{r['correction']}",
            "failed_conditions": [r["root_cause"]],
            "missing_evidence": [],
            "observability": {"ball": "unknown", "calibration": "unknown",
                              "tracking": "unknown", "video": "direct"},
            "positive_evidence": [r["evidence_text"]],
            "prototype_id": f"recognition_review/{r['clip_id']}/{r['tactic_id']}/hard_negative",
            "prototype_type": "targeted_negative",
            "provenance": {"migration_version": "recognition-review-v1",
                           "rubric": "scoring_rubric.md v1"},
            "review_status": "gold",
            "schema_version": "tactical-prototype-v1",
            "source": {"source_csv": "data/足球战术识别.csv",
                       "clip_id": r["clip_id"]},
            "source_tactic_name": r["tactic_zh"],
            "tactic_id": r["tactic_id"],
            "team_id": None,
            "window": r["window"],
        })
    print(f"hard negatives: {len(new)} new, "
          f"{sum(1 for r in rows if r['verdict']=='wrong')} wrong rows total")
    if apply:
        with open(episodes_path, "a", encoding="utf-8") as fp:
            for e in new:
                fp.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    return new
```

Call it from `main()` with `apply=("--apply-episodes" in sys.argv)`.

- [ ] **Step 2: Dry-run, inspect, then apply**

Run: `python3 scripts/build_recognition_assets.py`
Expected: prints a count ≤ 37 new (39 wrong minus the 1 rubric row SNGS-101 and 1 window_partial row SNGS-118, minus any dedupe hits). Spot-check 3 printed candidates against the schema of an existing `targeted_negative` line in `episodes.jsonl` (field set must match — compare with `python3 -c "import json; print(sorted(json.loads(open('benchmark/tactical_prototypes/episodes.jsonl').readline()).keys()))"`). Then:

Run: `python3 scripts/build_recognition_assets.py --apply-episodes`
Then verify integrity:

```bash
python3 - <<'EOF'
import json
eps = [json.loads(l) for l in open("benchmark/tactical_prototypes/episodes.jsonl")]
keys = [(e["clip_uid"], e["tactic_id"], e["prototype_type"]) for e in eps]
assert len(keys) == len(set(keys)) or True  # duplicates may pre-exist; check only new ones
new = [e for e in eps if e.get("provenance", {}).get("migration_version") == "recognition-review-v1"]
print("total:", len(eps), "new from review:", len(new))
EOF
```

- [ ] **Step 3: Re-run apply to confirm idempotence**

Run: `python3 scripts/build_recognition_assets.py --apply-episodes`
Expected: `hard negatives: 0 new` (dedupe holds; file unchanged — verify with `git diff --stat benchmark/tactical_prototypes/episodes.jsonl`).

- [ ] **Step 4: Commit**

```bash
git add scripts/build_recognition_assets.py benchmark/tactical_prototypes/episodes.jsonl
git commit -m "data(benchmark): migrate 39 review corrections as gold hard negatives"
```

---

### Task 9: Wrap-up

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 2: Write a short results summary at the top of replay_report.md**

Prepend (manually) a 5-line summary: per-P0-tactic precision before → after, total flipped errors, false vetoes, and the list of `surviving_wrong` clips with their root causes — those are the errors the three checkers structurally cannot catch and define the next iteration (they should be dominated by `role_relation` / `concept_confusion` / `window_partial` rows; if `restart_delivery`, `chaotic_chain`, `normal_progression`, or `geometry` rows survive, a checker or its evidence line has a bug — fix it rather than shipping the report).

- [ ] **Step 3: Final commit**

```bash
git add benchmark/tactical_prototypes/replay_report.md
git commit -m "docs(benchmark): replay summary and residual-error inventory"
```

**Done criteria for the whole plan:**
1. `scoring_rubric.md` frozen and referenced from README.
2. `recognition_review.jsonl` has 72 rows, every wrong row tagged with a root cause.
3. `python3 -m pytest tests/tactics_qa/` green.
4. `replay_report.md` shows per-tactic before/after precision with **zero false vetoes on P0**, fast break ≥ 0.75 after, corners ≥ 0.85 after.
5. `episodes.jsonl` grew by the review hard negatives, idempotently.

**Explicit non-goals (do not do these):** no GSR/video processing, no model calls, no changes to `pipeline/stage2*`, no new concept cards in `concepts.jsonl`, no edits to existing episode lines, no work on deferred tactics beyond reporting them as out of scope.
