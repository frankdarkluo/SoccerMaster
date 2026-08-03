# 用 LLM inference 做基础战术解读的可行路径

## 结论

如果你的目标是**不先训练一个专门的战术模型，而是先用 LLM inference 生成“基本但可信”的战术解读**，那么最现实的路线不是照着 TacticGen / GenTac / TacEleven 去做生成模型，而是走一条更轻的路：

**SoccerMaster 现有 GSR / calibration / tracking / jersey / role / team / pitch-coordinate 管线提供“可计算的战术状态”；规则引擎先把这些状态压成少量可验证的战术事实；LLM 只负责把事实组织成中文战术解说。**

这条路和几篇论文的共同结论是一致的：它们都不是直接让模型“凭视频感觉说战术”，而是依赖**结构化上下文、球员/球队状态、事件语义、球路或目标条件**来驱动推理或生成。TacticGen 明确把条件输入写成过去轨迹、球、事件类型、时间戳和全局事件特征，并区分了“短上下文预测”和“给定球路的条件生成”；GenTac 则把战术建模为轨迹 + 离散事件，并用一组集体结构指标和攻防目标指标来评估；TacEleven 用的是“历史轨迹 + 高层指令 + 事件级文本-轨迹对 + critic”；GameSight 进一步说明，好的解说不是端到端 caption，而是**实体对齐 + 知识增强**两阶段。fileciteturn0file0 fileciteturn0file1 fileciteturn0file2 fileciteturn0file4

所以对你来说，最重要的不是先想“哪个大模型最聪明”，而是先定义：

**LLM 到底看什么格式的战术状态，哪些断言可以直接说，哪些只能保守说。**

## SoccerMaster 现有输入和代码里最有价值的部分

基于我对你上传的 SoccerMaster repo 的直接检查，**对“战术解读”最有价值的不是 SoccerMaster 的 caption / classification 头，而是它接入的 sn-gamestate 游戏状态重建链路**。当前你已经有或几乎已经有以下几类信息：

- 单目转播视频上的**检测、跟踪、分割 refinement**
- **camera calibration / pitch localization**
- 每个检测实例的 **track_id**
- **team / role / jersey number**
- **bbox_pitch** 对应的球场米制落点
- 以及你自己 superpowers 里已经开始做的 **formation / topology / relations** 这层后处理

这意味着你手里已经有了做基础战术解说最关键的“几何底座”：

**谁在场上、谁和谁是一队、他们在球场上的相对位置、谁在往哪里移动、结构是变紧凑还是被拉开。**

这比单纯的 event caption 更重要，因为“为什么这样跑”“为什么这里形成优势”“为什么这是阵地战而不是纯快攻”，核心都来自这些几何与关系量，而不是来自一句 end-to-end 文本标签。

更具体地说，SoccerMaster 现在最值得榨干的输入价值有四层。

第一层是**身份与坐标**。这层已经足够支撑大部分“观察型”战术句子：  
“右侧边路形成二打一”“后防线明显收得更低”“弱侧球员站位更宽”“中路有人从第二线前插”。  
如果没有统一到 pitch coordinates，上面这些话几乎都不可靠。

第二层是**track-level continuity**。`track_id` 比球衣号更重要，因为球衣号会丢、会错、会被遮挡，但 track 可以让你在 2–5 秒窗口里可靠地说“这个人是从身后超到持球人外侧的”，这正是 overlap / run-in-behind 这种概念的前提。

第三层是**team structure**。你现有 formation-topology 设计里已经有 block height、depth、width、hull area、lane/band counts、side overload、inter-line gaps 这类量；它们和 GenTac 用来评估 collective structure consistency 的指标高度同构。GenTac 明确把 surface area、stretch index、team width、team length、centroid displacement、Frobenius distance、sync order parameter 当成集体结构评估指标；同时还用 off-ball expected threat、width threat、depth threat、defensive shape disruption、defensive dominant region 来刻画攻防目标。也就是说，你现在的 topology/relations 方向不是旁支，而正好踩在文献正中央。fileciteturn0file1

第四层是**coarse semantic priors**。SoccerMaster 里的 caption classification / event-related模块不是没用，但它们更适合充当：
- 当前片段是 **build-up / transition / threat / interruption / set piece** 的粗先验
- 或者“这段更像 clearance / corner / ball out of play / goal / shot off target”之类的可见事件先验

它们**不应该直接承担“战术解释器”角色**。更合理的用法是：  
**event prior + relations/topology + LLM narrative**。

一句话说，SoccerMaster 现在对战术解读最有帮助的部分不是“它能不能直接生成一句高水平战术话”，而是：

> 它已经把“可验证的场上状态”这件事做到了七八成。

## 三篇里面最值得精读的是哪篇

如果你现在只能深读一篇，我建议你先读 **GenTac**。  
如果排优先级，我会排成：

**GenTac > TacticGen > TacEleven**

原因很直接。

### GenTac 最值得先读

GenTac 最贴你当前问题，因为它最像一篇“**如何把战术数字化**”的论文，而不是单纯“如何训练一个更强的生成模型”。

它对你最有用的地方有三点。

第一，它明确给了一个**可落地的任务拆法**：  
把开放比赛战术理解拆成
- 多球员轨迹预测
- tactical event grounding / forecasting
- 条件化的风格/目标模拟  
而不是一上来就要求模型能自由生成长篇高质量 commentary。fileciteturn0file1

第二，它给了你一个非常实用的**中间语义层**：  
5 个宏类、15 个细粒度 tactical events，加上结构指标和攻防目标指标。  
这对你目前最重要，因为你完全可以不学它的扩散模型，只学它的“表示层”。也就是把你的 tactical state 组织成：
- 宏观阶段：build / transition / threat / interruption / set piece
- 微观观察：switch / overload / compactness / line-break / progression / clearance / defended 等
- 结构指标：width / length / stretch / centroid / overload
- 攻防指标：depth threat / width threat / local control / defensive compactness  
这已经足够支撑 LLM 生成“基础战术解读”。fileciteturn0file1

第三，GenTac 说明了你**不一定要从零造一个大战术标注集**。它的 TacBench 把轨迹和 tactical event 绑在一起，本质上告诉你：战术理解并不一定非得从主观长文本开始，你可以先从**结构化事件空间 + 规则可测指标 + 少量人工校验**做起。fileciteturn0file1

### TacticGen 第二重要

TacticGen 更适合你拿来回答一个系统设计问题：

> 如果以后我想让系统不仅“解释刚才发生了什么”，还想“按某种战术目标去推演或重写未来几秒”，接口应该怎样设计？

它真正有价值的是**control interface**，不是 backbone 本身。  
它把 inference-time steering 很清楚地分成：
- rule-based guidance
- language-based guidance
- value-based guidance  
并且明确说条件输入包括过去轨迹、事件类型、goal difference / event outcome / timestamp 等全局事件特征，还把实际工作模式分成 predictive ball modeling 和 conditional ball modeling。fileciteturn0file0

这对你当前的启发是：

你不用做扩散模型，但你应该照它的思路，把上层接口设计成两层：
- **state facts**：当前战术状态是什么
- **guidance / objective tags**：你想让 LLM从什么角度解释它  
比如“强调宽度利用”“强调压迫效果”“强调中路人数优势”“强调边中转换”

这会让你的 LLM 输出稳定很多。

### TacEleven 第三

TacEleven 很强，但离你当前资源条件最远。  
它需要：
- 超过 100 万 text-to-trajectory pairs 的 curated dataset
- 一个 language-controlled tactical generator
- 一个 multimodal tactical critic
- 用 xG / xT / pitch control 去筛选方案  
这对“做开场可用的战术讲解 demo”来说太重了。fileciteturn0file2

它最值得借鉴的不是训练范式，而是两个想法：

- **战术可以拆成事件级 text-to-trajectory pairs**
- **critic 要评估 proposal，而不是让 generator 一步到位**

对于你的系统，这可以被轻量化成：

- 规则引擎先提出若干 tactical facts / candidates
- LLM 再从中选择值得讲的那几个，并给理由

这很像你现在 hybrid commentary 的方向，但不需要 TacEleven 那种大规模数据依赖。fileciteturn0file2

### 还有一篇你其实也该认真看

虽然你问的是“三篇”，但对你的**解说系统设计**来说，**GameSight**几乎是必须读的辅助材料。

因为 GameSight 直接回答了一个你眼前的问题：

> 做 commentary 的时候，哪些信息不能只靠视频帧直觉，必须额外补？

它发现 human 的 player/entity 对齐在 84.5% 的正确案例里都不只依赖 long-view tracking，而是结合了 line-up、历史事件、close-up、face/jersey/team cues；同时它把 commentary 设计成两阶段：先 entity alignment，再 knowledge refinement。更重要的是，它指出 live televised commentary 明显比 live text commentary 更频繁地引入内外部知识。fileciteturn0file4

这对你直接对应的结论是：

**战术解读也应该两阶段：先事实对齐，再语言增强。**

## 如果只靠 LLM inference，要给它哪些输入信息

最关键的不是“给 LLM 更多东西”，而是“给 LLM **更少但更对** 的东西”。

你现在最适合的输入不是原始视频，也不是每帧全量坐标 dump，而是一份**战术状态摘要**。这份摘要最好由规则引擎从 SoccerMaster/GSR 输出自动生成。

我建议分成四层。

### 观察层

这是 LLM 可以**用确定语气说**的部分。

必须包含：

- 时间窗口与攻击方向
- 双方可见球员和球的 pitch coordinates 摘要
- track_id、team、role、可用时再加 jersey
- 当前或最近 2–5 秒的球路
- 持球方 / 持球人候选
- 双方结构指标  
  - block height
  - block depth
  - block width
  - hull area
  - line gaps
  - lane/band counts
  - side overload
- 局部关系指标  
  - 相对 carrier 的 longitudinal / lateral offset
  - nearest defender distance
  - relative depth vs second-last defender
  - local numerical superiority near ball
  - moved-past-carrier / outside-lane / central-lane 等布尔或分数

这些量基本都能通过现有 GSR + topology + relations 层得到，或者只需要很小扩展。

### 阶段层

这是 LLM 用来决定“这段话总体属于什么情境”的。

建议至少有：

- macro phase：`build-up / transition / threat / interruption / set piece`
- possession stability
- ball progression direction
- danger zone / target zone
- 是否完成明显的 line break 或 side switch
- 是否进入 final third / box-adjacent area

这层非常重要，因为没有 phase，LLM 很容易把同样的几何形态解释错语境。GenTac 把 tactical event grounding / forecasting 明确做成一个单独层，就是因为“几何正确”不等于“语义正确”。fileciteturn0file1

### 解释层

这是允许 LLM 说“为什么值得注意”的，但必须受规则约束。

这里不要直接给结论长句，而是给：

- `candidate_concepts`
- 每个 concept 的 evidence refs
- 置信度
- 允许的 claim level

例如：

```json
{
  "concept_id": "switch_of_play",
  "evidence": ["ball.lateral_displacement>22m", "weak_side_occupancy_increase", "defensive_width_shift_delay"],
  "confidence": 0.84,
  "claim_level": "effect"
}
```

然后在 prompt 里硬性规定：

- `observation` 可以肯定说
- `effect` 只有 evidence 足够时可以肯定说
- `intention` 只能用“可能/看起来是在/似乎想要”这类保守措辞

这和你之前修订过的 claim-level 思路一致，也和 GameSight 的“先对齐再增强”一致。fileciteturn0file4turn0file5

### 约束层

这是避免 LLM 胡说的关键。

至少加入：

- visible-region / coverage
- calibration confidence
- tracking continuity
- identity confidence
- possession confidence
- unsupported regions / not observed regions

如果没有这层，LLM 天然会把“没看清”补成“推理到了”。

## 最好由规则引擎先产出的输入格式

你现在最应该新增的不是更多 prompt，而是一个中间文件，比如：

`tactical_state.json`

推荐结构如下。

```json
{
  "clip_id": "SNGS-xxxx",
  "window": {"t0": 12.0, "t1": 16.0},
  "attack_direction": "left_to_right",
  "phase": {
    "macro": "build-up",
    "possession_team": "left",
    "possession_confidence": 0.83,
    "transition_flag": false
  },
  "ball": {
    "x": 8.4,
    "y": -12.1,
    "speed": 7.3,
    "dx": 5.8,
    "dy": -1.5,
    "target_zone": "right_halfspace"
  },
  "team_states": {
    "left": {
      "block_height_m": 6.1,
      "block_width_m": 39.2,
      "block_depth_m": 24.5,
      "hull_area_m2": 512.0,
      "side_overload": 0.27,
      "line_count": 3,
      "inter_line_gaps_m": [8.9, 11.7]
    },
    "right": {
      "block_height_m": -11.8,
      "block_width_m": 31.4,
      "block_depth_m": 17.2,
      "hull_area_m2": 388.0,
      "side_overload": -0.03,
      "line_count": 2,
      "inter_line_gaps_m": [9.8]
    }
  },
  "local_relations": [
    {
      "type": "overlap_candidate",
      "runner_track_id": 27,
      "carrier_track_id": 12,
      "evidence": ["rel_x:-2.4->1.8", "outside_lane:true", "runner_disp:+7.1m"],
      "confidence": 0.81,
      "claim_level": "observation"
    }
  ],
  "semantic_priors": {
    "event_like": ["progression", "threat"],
    "goal_side_pressure": "medium"
  },
  "visibility": {
    "left_coverage": 0.89,
    "right_coverage": 0.84,
    "not_observed": ["far_left_touchline_high"]
  }
}
```

这个文件就是你给 LLM 的主输入。  
然后 LLM 只需要完成两件事：

- 从 `team_states + local_relations + semantic_priors` 中挑最值得讲的点
- 按 claim level 生成中文解说

这比“直接把 minimap 图片 + 原始轨迹 + glossary 全扔进去”更稳。

## 这些信息里哪些已经能从 SoccerMaster 现有管线得到

先说最实用的判断：

**你现在已经拥有做 observation-level tactical commentary 的 70% 输入。**

已经具备或很容易导出的：

- 球员与球场位置
- 跟踪连续性
- 球队归属
- 角色
- 球衣号
- 单目相机到 pitch coordinates 的映射
- 结构级统计的实现起点
- 事件粗先验

真正还缺的，主要是下面这几类：

- **稳定的 possession / carrier 状态**
- **ball destination / intended side** 这类球路摘要
- **local tactical relations**  
  比如 runner 相对 carrier 的 from→to 变化
- **line-breaking / switch / overload / compactness** 的规则判定
- **visibility / confidence metadata**
- 更强的 **phase segmentation**
- 如果你想讲“为什么这样站位更好”，还缺  
  - pitch control / defensive dominant region  
  - simple pass-lane openness / receiver availability  
  - local time-to-intercept 近似量

这里要强调一个很重要的现实判断：

> 你现在**不需要**作者们的训练数据，才能做出第一版基础战术解读。

你真正需要的是他们论文里已经公开的**表示方式和指标思想**，而不是他们的大模型本体。  
只有当你未来要做：
- 真实 tactical benchmark
- 学习式 tactical recognizer
- counterfactual simulator / value model  
时，才有必要去问作者要更细的数据或注释接口。fileciteturn0file0 fileciteturn0file1 fileciteturn0file2

## 需要更新到当前 specs 和 plans 吗

需要，而且我建议是**直接更新**，不是只加一段备注。

你当前的 KB 扩充 spec 已经做对了一半：它把战术概念从“写死在代码里”改成“带测试的数据”，这条方向没问题。fileciteturn0file5

但结合这几篇文章后，我认为要补的不是更多概念，而是要在 KB 前面再插一层：

**从“战术概念库”升级成“战术状态表示层 + 概念层 + 解说层”。**

### 该更新的核心点

#### 在 KB 之前新增 `tactical_state` 层

现在 spec 更偏向“如何写 concept recipe”，还不够强调“LLM 和 rules 应先共享什么中间状态”。

应该新增一个设计原则：

> 任何 tactical concept 都先从统一的 tactical_state.json 上解析，而不是直接对原始 relations 做概念级推断。

这样有三个好处：

- concepts.yaml 不会越来越像隐式 DSL
- LLM prompt 更稳定
- benchmark / evaluator / commentary 都能吃同一种状态表示

#### 吸收 GenTac 的结构指标和 phase 设计

建议把以下量正式写进 representation spec：

- surface area / hull area
- stretch / compactness proxy
- team width / length / centroid displacement
- offensive metrics：off-ball threat / width threat / depth threat 的简化近似
- defensive metrics：shape disruption / dominant region 的可计算近似
- macro phase：build / transition / threat / interruption / set piece

这些变量不一定一次都做完，但应该先进入 spec，作为 planned fields。GenTac 的价值就在于它已经证明这组量足够支撑“战术”这一层的表述。fileciteturn0file1

#### 吸收 TacticGen 的输入接口思想

建议在 plan 里显式加入两种模式：

- **observed-state commentary mode**  
  当前主模式，只解释已发生/正在发生的窗口
- **ball-conditioned what-if mode**  
  后续扩展模式，当你有人工指定或规则预测的 ball path 时，生成“如果球走向弱侧/肋部，会造成什么结构变化”的推演性解说

这不要求你训练 TacticGen，但要把接口留出来。TacticGen 把 predictive 和 conditional-ball 分开，是非常值得沿用的系统设计。fileciteturn0file0

#### 吸收 GameSight 的两阶段范式

建议把当前 commentary 生成 spec 明确改成：

- Stage A：**tactical fact selection**
- Stage B：**commentary realization + optional knowledge refinement**

并且在 schema 里加入：

- `fact_refs`
- `claim_level`
- `external_knowledge_used`
- `internal_match_context_used`

因为 GameSight 说明了：  
只做“看到什么”不够；只做“润色”会幻觉；必须先把事实对齐，再增强表达。fileciteturn0file4

## 一个我认为最适合你当前资源条件的完整计划

## 近期目标

先做一个**不用新训练数据、对大模型依赖最小、但能稳定产出基本战术解读**的 v1。

### 核心产物

- `tactical_state.json`
- `tactical_facts.json`
- `commentary_tactical.json`
- 一个小型 benchmark add-on

### 可解释范围

只先做三类：

- **phase / state**
- **structure**
- **local movement pattern**

先不做高主观度的意图和反事实。

## 表示层

### 新增 `tactical_state.json`

由现有 GSR + topology + relations 导出，不直接让 LLM碰原始轨迹。

字段分为：

- clip/window metadata
- ball state
- possession / phase
- team structure metrics
- local player-player / player-ball relations
- visibility/confidence
- semantic priors

### 必做的新规则量

你之前讨论过 query grammar 只扩四个原语，我认为这里继续成立。  
近期只补会显著改变表达能力的量：

- `track_id` 绑定
- `abs`
- `from -> to transition`
- `duration`

同时新增几个派生量：

- `ball_progress_x`
- `ball_switch_y`
- `runner_passed_carrier`
- `depth_vs_last_line`
- `local_num_adv_near_ball`
- `weak_side_occupancy_delta`

## 概念层

### v1 只做 8–12 个高可观测概念

建议第一批：

- build_up_from_back
- transition_attack
- threat_entry
- compact_block
- side_overload
- switch_of_play
- overlap_run
- run_in_behind
- halfspace_occupation
- line_breaking_progression
- local_numerical_superiority
- width_depth_stretch

建议延期：

- decoy_run
- press_resistance
- second_ball_protection
- third_man_intention
- pinning_defender

原因不是它们不重要，而是目前没有足够可靠的反事实或 defender-response 表示。

## 解说层

### 统一 claim contract

每条 tactical fact 必须包含：

- `concept_id`
- `evidence_refs`
- `confidence`
- `claim_level`
- `visibility_ok`

生成时严格限制：

- observation：可 assertive
- effect：仅 evidence 足够时 assertive
- intention：只能 hedged

### prompt 只做一件事

要求 LLM 从候选 facts 中选择 1–3 个最值得说的点，并按固定格式输出：

- Summary
- Tactical Reading
- Why it matters
- What may happen next

如果是短片段，只输出一段；如果是完整攻势，可以两段。

## 评测层

### 没有大标注集时的最小可行 benchmark

先不要追求长 free-form gold。先做：

- 50–100 个 windows
- 每个 window  
  - macro phase  
  - 1–2 个 core concept  
  - 1 个 hard negative  
  - claim level 边界
- match-disjoint split
- 输出指标：
  - concept precision
  - false tactical insertions per minute
  - overclaim rate
  - human preference on usefulness / trust

这比一开始就走 TacEleven/GenTac 那种大数据路线更符合你现在的资源。

### 如果能问 KNQ，要问什么

不是先问“能不能给我一个大战术标注集”，而是问：

- 能否提供 50–100 个典型片段
- 每个片段只标  
  - phase  
  - 最值得讲的一个 tactical point  
  - 一句自然中文说明  
- 如果可能，再给 1 个相似但不该触发的 negative

这会比要求他们做 full annotation 更容易落地。

## 最后的判断

我对你这个方向的判断是：

**完全可做，而且当前最优策略不是训练一个战术大模型，而是把 SoccerMaster/GSR 现有几何输入最大化利用起来，先做“事实化战术状态 → LLM 叙述”的架构。**

如果只选一篇最值得精读：**GenTac**。  
如果只选一个对你系统设计最有启发的额外参考：**GameSight**。  
如果只选一个当前代码里最该深挖的资产：**sn-gamestate / bbox_pitch / track_id / topology 这一整层，而不是 caption generation 头。**

更直接一点说：

> 你现在离“基础战术解读”并不远，缺的不是更大的 LLM，而是一个更干净的 `tactical_state` 表示层。它一旦建好，LLM 才真的有东西可推理。