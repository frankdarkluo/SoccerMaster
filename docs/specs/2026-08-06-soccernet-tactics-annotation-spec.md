# SoccerNet-Tactics 标注规范 v0.1

面向标注员与数据构造脚本。产出与 SN-VQA-2026 (`data/challenge/challenge.json`) 格式兼容的战术级四选一 QA 数据集。

---

## 0. 两阶段结构

标注员**不直接写题目**。标注员产出 clip 级的 checklist 记录,题目由脚本从 checklist 自动派生。

```
视频 clip
   ↓  人工:checklist 标注(19 项逐条判定)
annotations/*.json        ← 标注员唯一要填的东西
   ↓  脚本:题目派生 + 干扰项采样
benchmark/{train,valid,test}.json   ← SN-VQA 兼容格式
```

**为什么必须先 checklist**:四选一要求恰好一个选项成立。干扰项必须是**已核验不成立**的战术,否则一道题可能有两个正确答案。只有逐条过完 19 项、拿到完整的 present/absent/not_observable 向量,才能安全地采样干扰项。

---

## 1. 词表

### 1.1 战术标签(18 项)

`low-block, mid-block, high-press, gegenpress, pressing-triggers, buildup-structure, line-breaking-pass, switch-of-play, cutback, halfspace-penetration, overlap-underlap, counter-attack, tactical-foul, third-player-combinations, run-in-behind, one-two, third-man-run, long-ball`

> `formation-three-states` 从 QA 标签空间移除:它描述球队的阵型状态切换机制,不是单个片段内可判定的事件。保留在 KB 中作为背景概念。

### 1.2 阶段标签(5 项,采用 GenTac 分类)

`build, transition, threat, set_piece, interruption`

阶段与战术分开标注、分开评测。同一片段可有多个阶段区间。

### 1.3 干扰项专用词表(不进入 checklist)

KB 易混淆列中指向的、不在 18 项内的近似概念。这些只作为干扰项出现,不需要标注员逐条判定:

`直塞球, 推进传球, 横传球, 向前长传, 斜长传, 普通前插, 追逐解围球, 普通传接, 普通三角传递, 边路陷阱, 摆大巴, 攻转守临时回撤, 已落位的常规组织推进, 个人持球突破`

---

## 2. 标注文件格式

一个 clip 一个 JSON 对象。所有字段**强制填写**,不允许留空(不适用时显式填 `null` 或 `"not_observable"`)。

```json
{
  "clip_id": "SNT-000123",
  "source": {
    "dataset": "SN-VQA-2026",
    "material_path": "materials/q11/SoccerReplay-1988/italy_serie-a_2020-2021/2021-02-06_genoa-cfc-ssc-napoli-serie-a-2020-2021/1_05_32.mp4",
    "match_id": "2021-02-06_genoa-cfc-ssc-napoli-serie-a-2020-2021",
    "duration_s": 30.0
  },
  "annotator": "stu-03",
  "annotated_at": "2026-08-15",
  "round": 1,

  "teams": {
    "attacking_color": "white",
    "defending_color": "blue",
    "note": "按球衣主色填,不填队名、不填球衣号"
  },

  "broadcast": {
    "has_replay": false,
    "has_closeup": true,
    "has_camera_cut": true,
    "wide_shot_ratio": 0.7
  },

  "phases": [
    {"label": "transition", "start_s": 2.0, "end_s": 12.0},
    {"label": "threat",     "start_s": 12.0, "end_s": 18.5}
  ],

  "checklist": [
    {
      "tactic_id": "counter-attack",
      "verdict": "present",
      "team_color": "white",
      "anchor_s": 5.5,
      "window": [2.0, 12.0],
      "criteria_met": ["f1", "f3", "f4"],
      "confidence": "high"
    },
    {
      "tactic_id": "low-block",
      "verdict": "absent",
      "reason": "防线高度约 45m,施压起点在中圈附近,属中位而非低位"
    },
    {
      "tactic_id": "gegenpress",
      "verdict": "not_observable",
      "reason": "丢球瞬间镜头切到特写,看不到周围队友是否合围"
    }
  ],

  "notes": ""
}
```

### 字段规则

| 字段 | 规则 |
|---|---|
| `checklist` | **必须恰好 18 条**,每个 tactic_id 出现一次 |
| `verdict` | `present` / `absent` / `not_observable` 三选一 |
| `anchor_s` | 仅 present 需要。填**该战术最清楚的那一秒**,不是精确起止 |
| `window` | 仅 present 需要。粗略区间即可,评分用宽松 IoU |
| `criteria_met` | 仅 present 需要。引用 KB 该词条「可观察特征」的行号(f1..f6),**至少 2 条** |
| `reason` | `absent` 与 `not_observable` **必填**,一句话说清判据 |
| `confidence` | `high` / `medium` / `low` |

**判定顺序要求**:先看完整段视频并填 `phases`,再逐条过 18 项。不允许跳着填。

**not_observable 的含义**:该战术**可能成立但画面判断不了**(镜头切走、关键区域在画面外、遮挡)。它与 `absent`(看得见且确实不成立)是两回事,混淆会直接污染干扰项。

---

## 3. 干扰项生成规则

每题固定 **2 Hard + 1 Medium**(或 1 Hard + 2 Medium,见 §4 配额)。

| 档 | 采样来源 | 合法性保证 |
|---|---|---|
| **Hard** | (a) gold 的 KB 易混淆项,且在本 clip 标为 `absent`;或 (b) gold 的 KB 易混淆项属于「干扰项专用词表」 | (a) 人工核验;(b) KB 已记录判别依据,构造上成立 |
| **Medium** | 与 gold 同 `phase`、在本 clip 标为 `absent` 的其他战术 | 人工核验 |
| **Easy** | 与 gold 不同 `phase`、在本 clip 标为 `absent` 的战术 | 人工核验 |

**硬约束**

- `not_observable` 的战术**永不作为干扰项**。
- 同一题的四个选项必须互不相同,且只有 gold 一个成立。
- 记录每个干扰项的 `tier`,评测时分档报准确率。

---

## 4. 题目文件格式

与 `data/challenge/challenge.json` 同构,增加 `answer` 与 `meta`。**challenge split 发布时剥离这两个字段**。

```json
{
  "id": 10001,
  "Q": "In this video, between 00:02 and 00:12, which of the following holds for the team in white?",
  "materials": [
    "materials/tactics/SoccerReplay-1988/italy_serie-a_2020-2021/2021-02-06_genoa-cfc-ssc-napoli-serie-a-2020-2021/1_05_32.mp4"
  ],
  "O1": "Counter-attack",
  "O2": "Positional attack after the shape is already set",
  "O3": "Long ball",
  "O4": "Build-up structure (3+2 / 2+3)",
  "answer": "O1",
  "meta": {
    "clip_id": "SNT-000123",
    "question_type": "T1_forward",
    "gold_tactic": "counter-attack",
    "phase": "transition",
    "distractors": [
      {"option": "O2", "label": "已落位的常规组织推进", "tier": "hard", "src": "kb_confusable"},
      {"option": "O3", "label": "long-ball",  "tier": "hard",   "src": "checklist_absent"},
      {"option": "O4", "label": "buildup-structure", "tier": "medium", "src": "checklist_absent"}
    ],
    "broadcast": {"has_replay": false, "has_closeup": true, "has_camera_cut": true},
    "split_key": "2021-02-06_genoa-cfc-ssc-napoli-serie-a-2020-2021"
  }
}
```

---

## 5. 四种题型

同一份 checklist 派生四种题。目标配比见括号。

### T1_forward(45%)— 给定窗口,问哪项成立

```
Q: In this video, between 00:MM and 00:SS, which of the following holds for the team in {color}?
gold: 一个 verdict=present 的战术
```

### T2_reverse(20%)— 给定战术,问在哪个时段

```
Q: In this video, during which period does the team in {color} execute a counter-attack?
O1: 00:02-00:12   ← gold window
O2: 00:14-00:22   ← 同 clip 其他时段
O3: 00:00-00:06
O4: It does not occur in this video.
```
干扰窗口取自同一 clip 的其他时段,与 gold window 的 IoU 必须 < 0.2。

### T3_negative(15%)— 问哪项**不**成立

```
Q: In this video, between 00:MM and 00:SS, which of the following does NOT hold for the team in {color}?
gold: 一个 verdict=absent 的战术
干扰项: 三个 verdict=present 的战术(该 clip present 数 ≥3 时才能出此题)
```

### T4_abstain(20%)— 正确答案是「以上皆不成立」

```
Q: In this video, between 00:MM and 00:SS, which of the following holds for the team in {color}?
O1-O3: 三个 verdict=absent 的战术
O4: "None of the above."
answer: O4
```

> **T4 的依据**:pilot 中 44/137 (32%) 的片段专家判定无明确战术,而模型在其中 68% 上仍硬报了战术。这是现成的、高信号的弃权类,大多数 VQA benchmark 没有。

### 形式控制(脚本强制)

- 正确答案在 O1–O4 的位置分布均衡(卡方检验 p > 0.1)
- 四个选项文本长度差 < 30%
- `"None of the above."` 只在 T4 出现,且固定在 O4
- **同一 clip 派生的所有题必须整体进同一 split**
- split 按 `meta.split_key`(比赛)划分,不按 clip

---

## 6. 规模与配额

| 阶段 | 题量 | 每战术 | 目的 |
|---|---|---|---|
| Stage 0 | 200 | ~10 | 测耗时、测人-人一致率、测四选一是否被前沿模型秒杀 |
| Stage 1 | 800 | ~45 | 可投稿规模 |
| Stage 2 | 2000 | ~110 | 完整发布 |

**分层采样是强制的。** pilot 与 GenTac 的分布都证明随机抽取会得到 Zipf 分布(GenTac 最高类 1566 vs 最低类 6,260 倍)。稀有战术必须定向挖掘:用 VLM 在大 clip 池上检索候选,人工核验。**模型只负责找片段,不负责定标签。**

---

## 7. 质量控制

1. **资格测试**:用 KB 易混淆对出 10 题,答对 8 题才能开工。
2. **双标切片**:10–20% 的 clip 由两名标注员独立标,报告:
   - 逐战术 verdict 的 Cohen's κ
   - present 集合的 Jaccard
   - anchor_s 差值中位数
3. **仲裁**:双标不一致的条目由第三人裁定,裁定必须引用 KB 判据行号。
4. **Stage 0 门禁**:若 present 集合 Jaccard < 0.5,停止扩量,先修判据。

---

## 8. 开工前必须补的四件事

以下四项是 blocker,不补完标注就会返工。

### 8.1 七个战术缺易混淆项

KB 中以下 P0 词条的「易混淆」列为空,导致它们无法生成 Hard 干扰项,准确率会被系统性高估:

`high-press, buildup-structure, cutback, halfspace-penetration, overlap-underlap, tactical-foul`(以及已移除的 `formation-three-states`)

每条需补 2 个易混淆对象 + 一句判别依据。建议:

| 战术 | 建议易混淆对象 |
|---|---|
| high-press | mid-block(施压起点)、gegenpress(是否由丢球触发) |
| cutback | 普通传中(传球方向与接球点)、倒三角回做(是否到达底线) |
| halfspace-penetration | 边路突破(横向位置)、中路直塞(通道) |
| overlap-underlap | 平行推进(是否越过持球人)、换位(是否有球权关系) |
| tactical-foul | 普通犯规(是否中断了对方推进)、拼抢失败(是否有主观意图) |
| buildup-structure | 后场倒脚(是否有向前意图)、门将大脚(是否经过中场) |

### 8.2 易混淆关系需对称化

KB 中 `mid-block` 把 `high-press` 列为易混淆,但 `high-press` 没有反向列出。需要把易混淆图对称化,否则干扰项采样会有方向性偏差。

### 8.3 三条判据边界划不开

pilot 中反复出现的三向混淆:`long-ball / line-breaking-pass / run-in-behind`。一记越过防线找到插上球员的长传,三个标签同时成立。需要补互斥判别句,或明确规定这三者可以共存(则不能互为干扰项)。

### 8.4 buildup-structure 判据过薄

只有 2 条可观察特征,不足以支撑 `criteria_met` 至少引用 2 条的要求。需补到 4 条以上。

---

## 附:标注员速查

```
1. 完整看一遍 30 秒
2. 填 teams(球衣颜色)与 broadcast(有无回放/特写/切镜头)
3. 划 phases(build / transition / threat / set_piece / interruption)
4. 逐条过 18 个战术:
     看得见且成立     → present  + anchor_s + window + 至少 2 条判据行号
     看得见且不成立   → absent   + 一句理由
     看不清 / 镜头外  → not_observable + 一句理由
5. 不确定时优先选 not_observable,不要猜
6. 全部 18 条填完才算完成,不允许留空
```
