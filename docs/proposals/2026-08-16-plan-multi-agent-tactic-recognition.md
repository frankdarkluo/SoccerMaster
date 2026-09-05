# Plan — Multi-agent tactic recognition, pilot on 37 GSR-2025 GT clips

**Date:** 2026-08-16
**Status:** implemented — see `multi-agent/pilot.py` (A+B+rule→C) and `multi-agent/c_only.py` (control)
**Goal:** raise model effort and lower human effort in tactic annotation, and prepare for automated scale-up. A working pilot is a small methodological contribution for the benchmark paper.

---

## 1. Architecture

```
                    30s clip (no GT interval — that would leak the answer)
                              │
        ┌─────────────────────┼─────────────────────┐
        │ Agent A (2 fps)     │ Agent B (4 fps)     │   A and B never see
        │ 参与者 + 空间约束     │ 时间窗 + 事件序列     │   each other's output
        │                     │ + 防守如何变化        │
        └─────────┬───────────┴──────────┬──────────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                   Agent C (adversarial)
        cites 反例 / 易混淆, must name which specific
        claim of A or B it disputes. Emits a flag only —
        it does NOT change the label.
                             │
                             ▼
              Deterministic rule layer (no LLM)
        hard gates + weighted scoring over 视频侧四要素
        ties are reported, never broken by guessing
                             │
                             ▼
                    tactic label + score + flags
                             │
                  ┌──────────┴──────────┐
                  │ Stage 2 (later)     │
                  │ Agent D — fires only│
                  │ on tie / no-match   │
                  └─────────────────────┘
```

### Settled decisions

| # | Decision | Rationale |
|---|---|---|
| Metric | **macro-F1 primary**; also report overall accuracy and lift over the 41% majority baseline | 15/37 GT clips are `counter-attack`; a system that always answers counter-attack already scores 41%. Reviewers will compute this themselves. |
| A/B coupling | **Fully independent**, joined by a shared controlled vocabulary | If B sees A's output it anchors on A's framing and their errors become correlated — the same circular-logic problem as showing GPT output to human annotators. |
| Agent C | **Flag only, no veto.** Must name the specific claim it disputes, not object in general terms | None of the 37 GT clips is "no tactic", so any veto is guaranteed to be wrong on this pilot set. Logging C as a separate field lets you score with and without applying it. |
| Rule layer | **Layered**: 1–2 hard gates per tactic + weighted scoring on the rest | The four criteria are not equally diagnostic. 事件序列 discriminates; 时间窗 is a sanity check. Strict AND across all four would match almost nothing. |
| Ties | **Reported, never broken by guessing** | A long-ball / run-in-behind tie is the measured #1 confusion pair. That is information, and it feeds distractor design for the question bank. |
| Video input | A at 2 fps, B at 4 fps, full 30 s | B needs finer sampling for second-level event ordering. Cropping to the GT interval is **cheating** — the interval is itself a prediction target (task T1). |
| Reasoning chain | **Split.** 动作 → 防守变化 goes into B's output fields; 空间 → 对进攻的作用 becomes Agent D | The first two steps are observable, the last two are interpretation. Putting interpretation inside A/B re-couples perception and labelling. |
| Tuning scope | Prompts and rule thresholds may be tuned. **The 词条表 criteria are frozen.** | Editing the criteria to raise the score changes the exam and the answer key at once. Log every threshold change and its score. |
| Pilot status | Feasibility study. Paper numbers must come from freshly verified data. | 37 clips, 1 clip = 2.7%. Tuning to 28/37 fits the prompt to 37 clips, not a method. |

### Known measurement limit — read before running Stage 2

Agent D only fires on ties and no-matches. If Stage 1 reaches 28/37, D has **9 clips** to work with, and needs to fix 5+ to be visible above noise. **Stage 2 probably cannot be measured on this pilot set.** Plan to report Stage 1 feasibility here and defer the Stage 1 vs Stage 2 ablation to the expanded dataset.

### Hook for later work

Agent D's causal chain (空间如何出现 → 对进攻的作用) is the same reasoning that tactical report generation needs. Build D's output schema so it can be reused directly as report content, not only as a tie-breaker.

---

## 2. GT set

37 clips, from `data/annotations/多模态模型战术识别.xlsx`, with Human3+Human4 collapsed to one vote (they share identical tactic *and* timestamp on 31/63 overlapping clips; every other pair matches on 0–3%).

| tactic | n | share | 视频侧四要素 |
|---|---|---|---|
| counter-attack | 15 | 41% | exists |
| long-ball | 6 | 16% | exists |
| one-two | 6 | 16% | exists |
| tactical-foul | 3 | 8% | exists |
| cutback | 2 | 5% | exists |
| run-in-behind | 2 | 5% | exists |
| line-breaking-pass | 2 | 5% | exists |
| back-line-possession-management | 1 | 3% | exists |

All 8 now carry complete video-side criteria and ≥2 counter-examples (was 3/8 on 2026-08-16). The rule layer can fire on every benchmark tactic.

Majority-class baseline **15/37 = 41%**. The 70% target (28/37) is **+13 clips** over that.

Five classes have ≤3 examples. Macro-F1 on those classes is dominated by single-clip outcomes — report per-class counts alongside, and do not over-read small-class F1.

---

## 3. Video-side criteria for the five missing tactics

> **Landed 2026-08-16.** These five are now written into `data/足球战术数据库_词条表_Grid.csv`
> (status `draft`). **The CSV is authoritative** — the shipped wording was re-aligned
> against each row's existing `可观察特征` and differs from the drafts below, notably for
> `counter-attack` (existing criteria say 1–3 finishing players, not "≥2"). The text kept
> here is the sourcing record only. A football-literate review is still outstanding.
>
> On the ≥10 m threshold in `line-breaking-pass`: metric thresholds stay in `可观察特征`,
> and the video-side column states the same constraint relationally
> ("接球点比传球点更靠近对方球门，且位于该线最深防守者身后"). This follows the table's own
> convention — 12 of the 13 previously-filled video-side entries use time units only and
> no distances. It is also verifiable rather than merely pragmatic: see §9 item 2.

Written to match the density of the existing `one-two` / `long-ball` / `run-in-behind` entries. Sources are listed so the reviewer can check each claim.

### counter-attack (快速反击) — phase: attacking_transition

| field | value |
|---|---|
| 参与者 | 夺回球权的球员、接球后向前推进的持球人、至少 2 名同步前插的队友、正在回追的防守方 |
| 空间约束 | 球权转换发生后，球向对方球门方向纵向推进并跨越至少一个三区边界；防守方在球前方的人数少于进攻方 |
| 时间窗 | 从球权转换前约 1 秒起算，到形成射门、传中或进攻中断为止，通常 6 至 20 秒 |
| 事件序列 | 断球/抢断/解围形成球权转换 → 持球人不回传、少触球直接向前 → 多名队友同步前插 → 在防守方落位完成前进入前场三区 |
| 反例① | 球队夺回球权后先向后或横向传递组织，再逐步推进。 |
| 反例① 为何不算 | 缺少「转换后立即纵向直接推进」这一必要条件，属于常规组织进攻。 |
| 反例② | 由界外球、任意球等死球重新开始后的快速推进。 |
| 反例② 为何不算 | 没有球权转换的触发时刻，防守方有时间落位。 |

Hard gate: 球权转换 + 转换后首次触球方向向前。
Source: [FIFA Training Centre — The counter-attack](https://www.fifatrainingcentre.com/en/game/game-analysis/transition-to-attacking/the-counter-attack.php) ("regains possession and immediately attacks with speed and intensity"; "as few touches as possible to play forward"; "every player in front of the player in possession must be on the front foot").

### cutback (下底倒三角) — phase: final_third

| field | value |
|---|---|
| 参与者 | 下底的持球人、接倒传的中路队友、被吸引到近门柱侧的防守者 |
| 空间约束 | 传球发起点在底线附近的边路（wing，接近或进入 penalty_area 底角）；球的落点在小禁区线到点球点之间的中路区域，且**方向朝向本方球门一侧**（即向后回做） |
| 时间窗 | 从持球人进入底线区域前约 1 秒到中路队友触球，通常 3 至 8 秒 |
| 事件序列 | 边路持球人推进至底线附近 → 防守者向球侧收缩 → 持球人不传向门前，而是向侧后方回做 → 中路后插上队友在点球点附近接球并处理 |
| 反例① | 边路球员在底线附近把球高高传向门前争顶。 |
| 反例① 为何不算 | 传球方向朝向球门而非侧后方，属于常规传中。 |
| 反例② | 球员在禁区弧顶横传给中路队友射门。 |
| 反例② 为何不算 | 发起点不在底线附近，缺少「下底」这一必要条件。 |

Hard gate: 发起点在底线附近 **且** 传球方向为向后（朝本方球门分量为正）。
Source: [Wikipedia — Cutback (football move)](https://en.wikipedia.org/wiki/Cutback_(football_move)) ("played from near or just inside the byline… passed back towards the edge of the six-yard box or penalty spot"); [The Football Analyst — Cut-Backs](https://the-footballanalyst.com/cut-backs-football-tactics-explained/); [Coach Notes — Why Cutbacks Are the Most Effective Cross Type](https://coachnotes.co.uk/blogs/coaching-tactics-1/why-cutbacks-are-the-most-effective-cross-type) (the six-yard-box-to-penalty-spot "Golden Zone").

### line-breaking-pass (破线传球) — phase: progression

| field | value |
|---|---|
| 参与者 | 传球者、被穿越的那条防守线上的至少 2 名球员、在线后或线间接球的队友 |
| 空间约束 | 传球路线在几何上穿过对方一条防守线（后卫线或中场线）的两名球员之间或其上方，且球向前推进约 10 米以上 |
| 时间窗 | 从传球前约 1 秒到接球队友完成第一次触球，通常 2 至 6 秒 |
| 事件序列 | 持球人观察 → 队友在防守线间或线后启动 → 球穿过防守线（地面或过顶） → 队友在线后完成接球并面向球门方向处理 |
| 反例① | 在防守线前方的横传或回传，队友接球后仍在同一条线之前。 |
| 反例① 为何不算 | 没有穿过任何防守线，属于线前循环。 |
| 反例② | 长传越过全部防线落向边路无人区域。 |
| 反例② 为何不算 | 没有明确的线间/线后接应人，接球点也不构成推进。 |

Hard gate: 传球路线穿过一条防守线 **且** 有明确接球人。
Source: [SkillCorner — Breaking the Lines](https://skillcorner.com/articles/gi-line-breaking) (operational definition: "intersects at least one of the opposition lines geometrically… and progresses the ball forward at least 10 metres"); [Stats Perform — How Impactful Are Line-Breaking Passes?](https://www.statsperform.com/insights/how-impactful-are-line-breaking-passes/); [FIFA Training Centre — Line breaks](https://www.fifatrainingcentre.com/en/game/tournaments/u20fwwc/group-stage-review/line-breaks.php).

**Note for the reviewer:** the 10 m threshold cannot be measured from broadcast video without homography. For the agent prompt, replace it with "球的接收点明显比传出点更靠近对方球门，且跨越了至少一条防守线". Keep the 10 m figure in the entry as the formal definition.

### tactical-foul (战术犯规) — phase: defensive_transition

| field | value |
|---|---|
| 参与者 | 刚失去球权一方的防守者、正在发起反击的持球人、裁判 |
| 空间约束 | 犯规发生在中场或对方半场，远离本方禁区；犯规时防守方多数球员位于球的后方（未回防到位） |
| 时间窗 | 从球权转换到裁判鸣哨，通常 2 至 6 秒 |
| 事件序列 | 本方丢球 → 对方持球人向前启动 → 防守者从侧面或身后拉拽/绊倒/阻挡，明显不以争球为目的 → 裁判鸣哨，进攻中断 |
| 反例① | 防守者在争抢中先触到球，随后与对手发生接触被判犯规。 |
| 反例① 为何不算 | 是争球动作，不是以中断进攻为目的的故意犯规。 |
| 反例② | 在本方禁区附近的防守犯规。 |
| 反例② 为何不算 | 位置不符合「远离本方球门、阻止转换」的特征，属于常规防守犯规。 |

Hard gate: 犯规发生在防守转换阶段（本方刚丢球）**且** 犯规明显不以争球为目的。
Source: [Goal.com — Explained: What is tactical fouling in football?](https://www.goal.com/en-us/news/explained-tactical-fouling-football/i8qlq4e2hxqc1v7rs3f0udfq9) ("loses the ball in the opponent's half and, before the counter-attack is even launched, commits a foul"); [Guide to Football — Professional fouls](https://guidetofootball.com/tactics/professional-fouls/).

### back-line-possession-management (后场倒脚控场) — phase: buildup

**Classification only — do not use for temporal grounding.** Measured inter-annotator time-window IoU is 0.46, the lowest of the eight.

| field | value |
|---|---|
| 参与者 | 后卫线球员、门将、回撤接应的后腰 |
| 空间约束 | 球在本方 defensive_third 内的后卫线之间横向或向后循环，球始终未越过中线 |
| 时间窗 | 需观察一次完整的后场循环，通常 10 至 25 秒；不适合精确到秒的区间标注 |
| 事件序列 | 从门球或后场控球开始 → 后卫之间连续横传/回传 → 对方不上抢或上抢被回避 → 球权保持在后场，节奏明显放慢 |
| 反例① | 后卫接球后直接向前场长传。 |
| 反例① 为何不算 | 没有持续的后场循环，属于直接出球。 |
| 反例② | 后场传递中出现一次纵向穿透并越过中线推进。 |
| 反例② 为何不算 | 已经进入推进阶段，不再是控场。 |

Hard gate: 连续 ≥3 次后场横传/回传 **且** 期间球未越过中线。
Source: [Total Football Analysis — The control of rhythm in build up](https://totalfootballanalysis.com/article/tactical-theory-control-of-build-up-and-how-to-train-it-tactical-analysis-tactics); [Professional Soccer Coaching — Developing Possession from the Back](https://www.professionalsoccercoaching.com/match-management/developing-possession-from-the-back).

---

## 4. Shared controlled vocabulary

A and B never see each other's output, so their outputs must be mechanically joinable. Both prompts are given the same enums, taken from the 受控词表 sheet.

```
zone_lateral   : left_wing | left_halfspace | center | right_halfspace | right_wing
zone_vertical  : defensive_third | middle_third | final_third
zone_special   : zone_14 | penalty_area | six_yard_box
team           : team_in_possession | team_out_of_possession
```

**Never refer to teams by shirt colour.** Colour is misread constantly, and it is the one field that silently breaks the join between A and B. Anchor everything to who has the ball.

---

## 5. Agent prompts

Written for `pipeline/video_models.py` (`doubao-seed-2-0-lite-260428`, native video input, strict JSON schema). All three run at `temperature=0.0`.

### Agent A — spatial perception (2 fps)

```
你是足球视频的空间感知模块。只描述你看到的画面，不要推断战术意图，不要给战术命名。

术语（只能使用这些值）：
  横向区域 zone_lateral: left_wing | left_halfspace | center | right_halfspace | right_wing
  纵向区域 zone_vertical: defensive_third | middle_third | final_third
  特殊区域 zone_special: zone_14 | penalty_area | six_yard_box
  队伍 team: team_in_possession（持球方）| team_out_of_possession（防守方）
  禁止用队服颜色指代球队。以「谁持球」为准。

对这段 30 秒视频，输出：
1. possession_timeline：球权在两队之间转换的时刻（秒）。若全程未转换，返回空数组。
2. ball_zone_track：每 5 秒一个采样点，球所在的 zone_lateral 和 zone_vertical。
3. player_counts：每 5 秒一个采样点，防守方位于其本方半场的球员数（可数出的人数，看不清则填 null）。
4. defensive_line：能否看到防守方的后卫线？若能，记录它在哪个 zone_vertical，以及是否有进攻球员位于该线之后。
5. notable_positions：任何值得记录的站位事实（例如「持球人位于底线附近的 right_wing」）。每条必须是可指认的画面事实。

对任何看不清的字段填 null，不要猜测。JSON 输出。
```

### Agent B — temporal perception + defensive response (4 fps)

```
你是足球视频的事件感知模块。描述发生了什么动作、以及防守方如何随之变化。
不要推断战术意图，不要给战术命名，不要评价这对进攻是否有利。

术语（与空间模块共用，只能使用这些值）：
  zone_lateral: left_wing | left_halfspace | center | right_halfspace | right_wing
  zone_vertical: defensive_third | middle_third | final_third
  team: team_in_possession | team_out_of_possession
  禁止用队服颜色指代球队。

对这段 30 秒视频，输出：
1. events：按时间排序的动作序列。每条包含
     t_start, t_end（秒，精确到 0.5）
     action：pass | long_pass | cross | cutback | dribble | shot | tackle | interception
             | clearance | foul | throw_in | goal_kick | corner | free_kick | header
     from_zone, to_zone（若适用）
     forward：这次动作是否使球更接近对方球门（true/false/null）
     description：一句可指认的画面描述
2. defensive_response：针对每个关键事件，防守方随之发生了什么变化。每条包含
     after_event_index：对应 events 的下标
     change：line_dropped | line_held | line_pushed_up | players_collapsed_to_ball
             | players_recovered_behind_ball | no_visible_change
     description：一句可指认的画面描述
   只描述看得见的移动，不要解释它的作用。
3. sequence_summary：用一句话按先后顺序串起主要动作。

对任何看不清的字段填 null，不要猜测。JSON 输出。
```

### Agent C — adversarial (2 fps)

C receives the clip plus the JSON from A and B, and the `反例` / `易混淆区分点` text for the tactics the rule layer scored highest.

```
你是质疑模块。你的任务不是给出战术标签，而是找出空间模块和事件模块的描述中，
哪一条具体主张与画面不符，或者不足以支撑候选战术。

给你：视频、空间模块输出、事件模块输出、以及若干候选战术的「反例」和「易混淆区分点」。

规则：
1. 每条质疑必须指向一个具体字段，格式为 agent 名 + 字段路径 + 该字段的原值。
   例：{"target": "B.events[2].action", "claimed": "long_pass", "objection": "..."}
2. 禁止泛泛而谈。「证据不足」「不够清晰」这类没有指向具体字段的质疑一律不要输出。
3. 若某个候选战术的「反例」描述与本片段实际情况吻合，明确指出是哪一条反例。
4. 若你找不到任何具体问题，返回空数组。不要为了输出而输出。

输出：objections 数组，每条包含 target / claimed / objection / matched_counter_example（可为 null）/ confidence（low|medium|high）。
```

C emits flags only. The rule layer does not read them; the evaluation script scores accuracy both with and without applying them.

---

## 6. Rule layer

Not an LLM. Reads A/B JSON, emits a scored ranking.

```python
# pipeline/tactics_qa/rule_layer.py  (sketch)

@dataclass
class Criterion:
    name: str                      # 参与者 | 空间约束 | 时间窗 | 事件序列
    check: Callable[[dict], bool]  # reads the joined A+B payload
    hard: bool = False             # failing a hard gate eliminates the tactic
    weight: float = 1.0

def score_tactic(payload: dict, criteria: list[Criterion]) -> tuple[float, dict]:
    detail = {}
    for c in criteria:
        ok = c.check(payload)
        detail[c.name] = ok
        if c.hard and not ok:
            return 0.0, detail          # hard gate failed
    earned = sum(c.weight for c in criteria if not c.hard and detail[c.name])
    total = sum(c.weight for c in criteria if not c.hard) or 1.0
    return earned / total, detail

def classify(payload: dict, table: dict[str, list[Criterion]], min_score: float = 0.5):
    scored = {t: score_tactic(payload, cs) for t, cs in table.items()}
    ranked = sorted(((s, t, d) for t, (s, d) in scored.items()), reverse=True)
    top = [(s, t, d) for s, t, d in ranked if s >= min_score]
    if not top:
        return {"label": None, "reason": "no_match", "ranked": ranked[:3]}
    if len(top) > 1 and top[0][0] == top[1][0]:
        # A tie between confusable tactics is information, not failure.
        return {"label": None, "reason": "tie", "tied": [t for _, t, _ in top if _ == top[0][0]],
                "ranked": ranked[:3]}
    return {"label": top[0][1], "score": top[0][0], "detail": top[0][2], "ranked": ranked[:3]}
```

Hard gates per tactic are the ones named in §3 and in the three existing entries. `min_score` is tunable (§1: category ② is tunable, ③ is frozen).

---

## 7. Runner

`pipeline/video_models.py` already exposes doubao with native video input and strict JSON schema. Stage 1 uses doubao for all three agents (decision Q3a). Once the loop works, swap Agent C to Codex (Q3b) — heterogeneous models are what actually breaks the correlated-error problem, but only test that after the loop runs.

```python
# scripts/run_multiagent_pilot.py  (sketch)
from pipeline.video_models import generate   # existing doubao/gemini dispatcher

def run_clip(video: Path, table, prompts) -> dict:
    a, _ = generate(prompts["A"], SCHEMA_A, video, provider="doubao", fps=2.0, temperature=0.0)
    b, _ = generate(prompts["B"], SCHEMA_B, video, provider="doubao", fps=4.0, temperature=0.0)
    payload = {"A": a, "B": b}
    verdict = classify(payload, table)
    c, _ = generate(
        prompts["C"].format(a=json.dumps(a, ensure_ascii=False),
                            b=json.dumps(b, ensure_ascii=False),
                            candidates=render_candidates(verdict["ranked"])),
        SCHEMA_C, video, provider="doubao", fps=2.0, temperature=0.0,
    )
    return {"clip": video.stem, **verdict, "objections": c.get("objections", [])}
```

Check the exact `generate` signature before wiring — the sketch assumes the dispatcher at `pipeline/video_models.py:286`.

---

## 8. Pilot protocol

**Run** all 37 clips, log one JSON line per clip with: A output, B output, C objections, rule-layer ranking and detail, predicted label, GT label.

**Score** four numbers every iteration:

| metric | why |
|---|---|
| macro-F1 | primary — forces the small classes to matter |
| overall accuracy | secondary, comparable to the 70% target |
| lift over 41% majority baseline | the number a reviewer computes anyway |
| per-class confusion | shows whether gains are real or all counter-attack |

Score twice: applying C's objections and ignoring them. The difference is C's measured contribution.

**Log every iteration** in `outputs/multiagent_pilot/iterations.csv`:

```
iter, date, changed(prompt_A|prompt_B|prompt_C|thresholds), change_summary,
macro_f1, accuracy, correct_n, ties_n, no_match_n, notes
```

This table goes in the appendix. "We reached X% after 12 iterations, here is the trajectory" is defensible; a single final number is not.

**Stopping rule.** Stop when macro-F1 stops improving for 3 consecutive iterations, or at 15 iterations, whichever comes first. Without a stopping rule, iterating on 37 clips has no natural end and the overfitting gets worse the longer it runs.

**Frozen:** the 词条表 criteria. If a criterion looks wrong during the pilot, record it as a finding — do not edit it to raise the score.

---

## 9. Open items

1. §3's five entries are in the Grid CSV (status `draft`) but still need a football-literate review. They were written from cited sources plus each row's existing `可观察特征`, not from watching the GT clips.
2. **The ≥10 m threshold is measurable on GSR-2025 after all.** `data/GSR-2025/images/SNGS-*/Labels-GameState.json` (164/200 sequences, covering 34/44 GT clip ids) carries `bbox_pitch` — per-frame pitch coordinates **in metres** for every player, goalkeeper, referee and the ball, plus `role`, `team`, `jersey`. Verified range x∈[-48.5, 24.0], y∈[-37.3, 40.0], consistent with a 105×68 m pitch. SoccerNet-AA has no equivalent. Two consequences worth acting on:
   - the relational video-side wording can be **validated** against the formal ≥10 m rule on GSR-2025 — measure true pass distance from the ball's `bbox_pitch` track and check how often the two agree
   - the "structured player positions as auxiliary input" ablation costs **zero preprocessing** on GSR-2025; the labels ship with the dataset
3. Whether `line-breaking-pass` and `halfspace-penetration` stay separate is still unsettled. Annotators cannot separate them; merging raised agreement from 33%/10% to 46%. This affects how many classes the rule layer must discriminate.
4. Stage 2 (Agent D) has no measurable room on this pilot set if Stage 1 succeeds — see §1. Decide whether to defer the ablation or deliberately hold Stage 1 back.
