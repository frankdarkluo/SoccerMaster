# VLM 适配层修复与冻结测试准入计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans,按 checkbox 逐任务执行。
> **Commit 政策(AGENTS.md):任何任务不得单独 commit,全计划完成后一次 commit**,或用户明确说 OK。改代码遵循 karpathy + ponytail:最小 diff,不顺手重构,不加投机抽象。

**Goal:** 修掉 dev-72 数据实测出的 vlm-claim 假否决问题(matched 子集 11 条 rejected 里 6 条人工判正确),把断言口径收紧为 confirmed-only,补齐轨迹覆盖并做 checker 激活验证,然后带着干净的系统进入 43-clip 冻结测试并产出专家评审包。

**为什么是这几件事、为什么是这个顺序(大白话):**
dev 跑完后我们知道了三件事。第一,验证层在 dev 上基本没干活——67 条里只有 3 条有轨迹数据,checker 无米下锅,所以门控前后精度几乎一样(60.7% vs 60.5%),这不是门控没用,是门控饿着。第二,让模型自报事实来顶替轨迹的那条通道(vlm-claim)出了实际问题:它否决的 11 条有人工判定的 claim 里 6 条其实是对的,一半火力打在自己人身上;病因有三种——字段表达不了"对方定位球被解围后反击"这种例外、"没报告某事实"被当成"该事实不存在"、模型幻觉出 restart。第三,corner 出现自我确认回路:模型自己说落点在前点、自己的检查就通过,39/41 条 corner 到了 possible,matched 精度只有 52%。所以顺序必然是:先修 vlm 通道(几小时,能用存量数据零成本回归)→ 把轨迹覆盖补上(v2 批次)→ 验证 checker 真的会被激活 → 才有资格进冻结测试。测试前不动模型、不动抽帧、不动阈值。

**修复的原则(一句话):** 自报证据只能凭"它明确断言了的事实"去否决;凡是"它没说"或"说得表达不清"的,一律 insufficient,不许 veto。轨迹证据不受此限(轨迹观察不到=真的没有)。

---

### Task 1: 修 vlm-claim 适配层(仅这一处代码行为变化)

**大白话:** 现在的实现把 `claimed_evidence` 直接转成事件喂给了为轨迹设计的 checker,语义错位。要改成:vlm 侧每个 checker 只有在它需要的字段被模型**明确填了**的时候才运行;而且"模型同时声称有 restart 又声称有受控夺回"(SNGS-189 那种)是规则 R4 例外的自报形态,判 insufficient 而不是 veto。

**Files:** 修改 vlm-claim → checker 的转换/调用代码(server 端,`grep -rn "claimed_evidence" pipeline/ codes/` 定位;实现函数名以实际代码为准,下面的契约和用例不变);新增回归测试文件放在 `tests/tactics_qa/test_vlm_claim_rules.py`。

- [ ] **Step 1: 先写行为规则测试(用 6 条真实案例做 fixture,期望值如下)**

行为契约:给定 `claimed_evidence` dict、tactic_id、window,返回 vlm 侧 checker 结果列表。规则:

1. `regain` 为 null → 不运行 `clean_regain` 的"无受控夺回"判据(不得因缺席 veto)。
2. `chaos: true` 单独出现 → insufficient(布尔表达不了乱战程度,轨迹侧是数触碰次数的),不 veto。
3. `restart_at_origin` 非 null 且 `regain` 也非 null(受控夺回类)→ R4 例外形态 → insufficient,不 veto。
4. `restart_at_origin` 非 null 且 `regain` 为 null → 断言式 veto **保留**(这是模型明确说"这是定位球开局"且没说夺回)。
5. `corner_landing_zone: "center"` → veto 保留;`delivery_kind` 与战术冲突(如 run_in_behind 配 cross)→ veto 保留。
6. 所有需要的字段全 null → 该 checker 不运行,整体 insufficient。

六条真实案例的期望结果(修复后):

| 案例 | claimed(实录) | 修复前 | 修复后 | 依据 |
| --- | --- | --- | --- | --- |
| SNGS-026 doubao fast_break | restart=throw_in, regain=interception | rejected | **insufficient** | 规则3 |
| SNGS-037 doubao fast_break | chaos=true, regain=interception | rejected | **insufficient** | 规则2+3(chaos 单凭布尔不否) |
| SNGS-037 gemini fast_break | chaos=false(仅此) | rejected | **insufficient** | 规则1 |
| SNGS-053 doubao run_in_behind | restart=kickoff, regain=null | rejected | **rejected(不变)** | 规则4,已知残留(该 restart 疑似幻觉,记录为 known residual,不为它破坏规则) |
| SNGS-129 doubao fast_break | chaos=true, regain=controlled_recovery | rejected | **insufficient** | 规则2+3 |
| SNGS-189 doubao fast_break | restart=corner, regain=controlled_recovery | rejected | **insufficient** | 规则3 |

另加两个反向用例防止矫枉过正:claimed restart=goal_kick 且 regain=null 的 line_break claim → 必须仍 veto;corner claim 的 corner_landing_zone=center → 必须仍 veto。

- [ ] **Step 2: 跑测试确认全红**(新测试失败、既有 82 个测试不受影响)
- [ ] **Step 3: 实现最小修改使测试全绿。** 修改范围只允许在 vlm-claim 转换/调用层;`checkers.py` 的语义一行都不许动(轨迹侧行为必须不变,拿 `tests/tactics_qa/` 全量通过作证)。
- [ ] **Step 4: 完成标准:** `python3 -m pytest tests/tactics_qa/ -v` 全绿,新增用例 8 条全绿。

### Task 2: 用存量 dev 数据离线回归(零 API 成本)

**大白话:** 67 个 agent run 里每条 candidate 的 `claimed_evidence` 都存着,不用重新问模型——把存量数据从修好的适配层重新过一遍、重算 verdict,就能量出修复的真实效果。

**Files:** 新增一个一次性脚本入口(建议挂在既有 CLI 上,如 `scripts/build_recognition_assets.py --reverdict-agent-runs`),读取 `benchmark/tactical_prototypes/agent_runs/*.json`,对每条 positive candidate 用修复后的 vlm 规则 + 原有轨迹结果重算 tier,回写各 run 文件的 `verdict` 字段,并重生成 `replay_report.md` 的 System-arm dev-72 小节。

- [ ] **Step 1: 实现重算(幂等:重复运行结果一致)。**
- [ ] **Step 2: 完成标准(硬性,达不到就回 Task 1 查):**
  - matched 口径下"rejected 且人工判正确"从 6 条降到 **恰好 1 条(SNGS-053)**;
  - 原 11 条 matched-rejected 中人工判"错误"的 5 条,修复后**不得出现在 confirmed 或 possible**(落在 rejected 还是 insufficient 均可——缺席式否决降级后,部分从 rejected 变 insufficient 是规则的正常结果);先用脚本打印这 5 条身份,并输出全部 11 条的修复前→修复后 tier 迁移表进报告;
  - SNGS-116 corner 仍是 confirmed;Missing verdict 仍为 0;
  - 报告小节更新后列出新的 gate-on/gate-off、去重口径、分层计数,并保留修复前数字作对照(一行旧值即可,方便看变化)。

### Task 3: 口径落地——confirmed-only 断言 + Sol 降级规则(只改报表和文档,不改判定逻辑)

**大白话:** dev 数据说明 possible 层精度只有 60% 上下,不能当"系统认了"的输出;它的正确角色是专家评审队列和解说里的观察式措辞。Sol 那 56 条积压不要人工去消化——那是轨迹缺失造成的洪水,不是真分歧。

- [ ] **Step 1:** 在 `scoring_rubric.md` 追加一小节(≤10 行):对外断言 = confirmed 独占;possible = 评审队列/观察式措辞,不计入"系统断言精度";测试指标定义两条线:asserted precision(confirmed)与 queue precision(confirmed+possible)。
- [ ] **Step 2:** 同节写入 Sol 规则:dev 的 56 条 pending 不处理,标记 superseded;冻结测试中如 Sol 触发率 >20%(按 clip 计),后续批次自动降级为 insufficient,不再入队。
- [ ] **Step 3: 完成标准:** rubric diff ≤ 15 行;没有任何判定代码被改动。

### Task 4: 轨迹覆盖补齐 + checker 激活验证

**大白话:** dev 最大的教训是"没有轨迹,整个验证层空转"。v2 Stage-1 批次必须跑完,而且要用几条有轨迹的 clip 实证"checker 真的醒了"——离线回放证明过轨迹证据零假否决,现在要在真系统里复现这一点。

- [ ] **Step 1:** 确认 v2 批次覆盖:43 条测试 clip 全部有 `outputs/<id>/predictions.json`;失败的列清单和原因(日志在 `/tmp/system_arm_stage1_*_v2.*`)。顺带在日志里写清 challenge 为何是 28 条而非 38−8=30(此前遗留的差 2 未解释)。
- [ ] **Step 2:** 往批次里追加 5-10 条 **dev** SNGS clip 的 Stage-1(挑人工判"正确"的 P0 行,如 SNGS-037/045/054/123/129),专门用于激活 smoke——不许动测试 clip 做调试。
- [ ] **Step 3: 激活 smoke(会花少量 API):** 对这批有轨迹的 dev clip 完整重跑系统。完成标准:
  - 每条 clip 的 candidate 出现非空 `checker_results.trajectory`;
  - 人工判"正确"的 claim 中,轨迹侧 veto 为 **0**(出现即为 bug,修完才能继续);
  - 至少出现 1 条轨迹背书的新 confirmed;
  - Sol 触发率在这批 clip 上显著低于 dev 的 56/67(预期个位数)。

### Task 5: 冻结测试与专家包

**大白话:** 前四关过了,系统才配进考场。测试按 2026-07-17 方案的 Task 6/7 执行(专家 CSV 格式、answersheet、ingest 脚本均不变),此处只列新增约束。

- [ ] **Step 1:** 冻结点记录 git diff hash;冻结后任何改动=测试作废重跑。
- [ ] **Step 2:** event_clips 0006/0007/0013/0014 曾进入 dev 调试,测试指标必须出**含/不含这 4 条的两版**;专家 CSV 里这 4 条照发(专家不知情不影响其判定,污染只影响我们的指标解读)。
- [ ] **Step 3:** 专家 CSV 中 unmatched 性质的 claim(dev 里占 64/120 的那类,人工从未审过的提名)全部进包——它们的精度是目前最大的未知数。
- [ ] **Step 4: 完成标准:** `expert_review_batch1.csv` + answersheet 生成;运行日志含冻结 hash、两版指标定义、Sol 触发率;用户拿到文件即可周日晚发出。

### Task 6: 收尾

- [ ] `python3 -m pytest tests/ -v` 全绿;报告更新;一次性 commit,message 列明:vlm 规则修复(附 6 案例前后对照)、confirmed-only 口径、Sol 处置、v2 覆盖数、冻结 hash。

---

**非目标(明确不做):** 不动抽帧策略/分辨率/提示词/模型;不做 adaptive;不修 corner 自我确认回路(possible 已降为队列,等专家数据回来再calibrate);不人工消化 56 条 Sol 积压;不追提名召回(75.9% 的缺口等专家数据定位后再攻)。

**可选、不阻塞(时间富余才做):** 同设置对 5 条 dev clip 重复跑两遍,量提名的 run-to-run 方差基线——花一次小额 token,换取以后所有 A/B 结论的解释力。
