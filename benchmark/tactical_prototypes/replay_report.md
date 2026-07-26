# Checker 离线回放报告

## 结果摘要

- P0 precision：corner-near-far-post 0.5→1.0；cutback 0.667→1.0；fast_break_pattern 0.478→0.917；run_in_behind 0.643→0.818。
- P0 共拦截 22 条既有误报，false vetoes=0。
- P0 residual：soccernetgs:SNGS-118 (window_partial)；soccernetgs:SNGS-101 (rubric)；soccernetgs:SNGS-134 (concept_confusion)。
- 其余 surviving wrong 属于 P1 或 deferred，只用于定位下一轮 checker 范围。

| tactic | scope | claims | precision before | precision after | flipped | false vetoes | insufficient | surviving wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| between-the-lines | deferred | 2 | 0.5 | 0.5 | 0 | 0 | 0 | 1 |
| corner-near-far-post | P0 | 14 | 0.5 | 1.0 | 7 | 0 | 0 | 0 |
| cutback | P0 | 3 | 0.667 | 1.0 | 1 | 0 | 0 | 0 |
| dummy-run | deferred | 1 | 0.0 | 0.0 | 0 | 0 | 0 | 1 |
| fast_break_pattern | P0 | 23 | 0.478 | 0.917 | 11 | 0 | 0 | 1 |
| gk-in-buildup | deferred | 1 | 0.0 | 0.0 | 0 | 0 | 0 | 1 |
| halfspace-penetration | deferred | 3 | 0.0 | 0.0 | 0 | 0 | 0 | 3 |
| line_break | P1 | 9 | 0.333 | 0.5 | 3 | 0 | 0 | 3 |
| one_two | deferred | 1 | 0.0 | 0.0 | 0 | 0 | 0 | 1 |
| run_in_behind | P0 | 14 | 0.643 | 0.818 | 3 | 0 | 0 | 2 |
| switch-of-play | deferred | 1 | 0.0 | 0.0 | 0 | 0 | 0 | 1 |

**Surviving wrong (between-the-lines)**: soccernetgs:SNGS-039 — checker 覆盖不到的根因

**Surviving wrong (dummy-run)**: soccernetgs:SNGS-030 — checker 覆盖不到的根因

**Surviving wrong (fast_break_pattern)**: soccernetgs:SNGS-118 — checker 覆盖不到的根因

**Surviving wrong (gk-in-buildup)**: soccernetgs:SNGS-022 — checker 覆盖不到的根因

**Surviving wrong (halfspace-penetration)**: soccernetgs:SNGS-061, soccernetgs:SNGS-062, soccernetgs:SNGS-190 — checker 覆盖不到的根因

**Surviving wrong (line_break)**: soccernetgs:SNGS-028, soccernetgs:SNGS-070, soccernetgs:SNGS-177 — checker 覆盖不到的根因

**Surviving wrong (one_two)**: soccernetgs:SNGS-200 — checker 覆盖不到的根因

**Surviving wrong (run_in_behind)**: soccernetgs:SNGS-101, soccernetgs:SNGS-134 — checker 覆盖不到的根因

**Surviving wrong (switch-of-play)**: soccernetgs:SNGS-115 — checker 覆盖不到的根因

## Contact-sheet resolution ablation

2026-07-18 在 5 条人工确认 P0 dev clip 上使用同一提示词和模型比较三档输入：SNGS-116 (corner-near-far-post)、SNGS-037 (fast_break_pattern)、SNGS-045 (run_in_behind)、SNGS-047 (cutback)、SNGS-053 (run_in_behind)。Token 成本为 Doubao/Gemini API 返回的 `total_tokens` 之和，包含 overview 和候选窗口 dense 调用。

| profile | 实际单格 | overview / dense | 提名召回 | total tokens | avg/clip | Doubao | Gemini |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 640 | 640px | 1×4×3 / 1×4×4 | 4/5 (80%) | 76,678 | 15,335.6 | 51,058 | 25,620 |
| 1024 | 1019px | 1×4×3 / 1×4×4 | 3/5 (60%) | 77,724 | 15,544.8 | 52,098 | 25,626 |
| 1080p-split | 1920×1080 | 3×2×2 / 4×2×2 | 3/5 (60%) | 150,166 | 30,033.2 | 91,573 | 58,593 |

- 640 漏掉 SNGS-047 cutback；1024 漏掉 SNGS-045 和 SNGS-047；1080p-split 漏掉 SNGS-047 和 SNGS-053。提高分辨率没有补回 cutback，且新增了漏提。
- 1024 档为使 4 列含边距的整图宽度不超过 4096px，请求值 1024px 实际生效为 1019px/格。
- 三档均使用 JPEG q3；最重的 1080p-split overview+dense 实测 base64 约 12.94MiB，低于 Gemini inline 20MB 上限。
- **冻结设置：640px/格、JPEG q3，overview 4×3、dense 4×4。** 该档在本次消融中同时取得最高召回和最低 token 成本。自适应抽帧仅作为 SNGS-047 类漏提的后续实验，不进入本轮 v1。

## Adaptive overview pilot

2026-07-18 在不改变 640px/格、overview 12 帧和 dense 设置的前提下，对比冻结 uniform sampling 与 `6 个均匀锚点 + 3 个帧差峰值的前后帧`。结果均来自相同提示词和模型的真实 API usage；仓库内 640 baseline agent runs 已逐条恢复，adaptive 结果仅保存在 `/tmp/system_arm_adaptive_pilot/`。

| paired subset | clips | uniform target hits | adaptive target hits | uniform tokens | adaptive tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| known misses (recovery) | 5 | 0/5 | 2/5 | 82,010 | 81,767 |
| known hits (retention) | 5 | 5/5 | 3/5 | 79,366 | 69,313 |
| combined | 10 | 5/10 | 5/10 | 161,376 | 151,080 |

- 找回：SNGS-031 `cutback`、SNGS-058 `run_in_behind`。
- 回归：SNGS-045、SNGS-053 的 `run_in_behind`。
- Adaptive token 成本降低 10,296（6.4%），但没有净召回收益，并产生同战术回归。
- **决定：不采用该 adaptive profile，不重跑 dev；冻结设置仍为 uniform 640。** 若未来有可靠轨迹 kick/possession timestamps，再单独评估事件驱动抽帧。

## System-arm dev-72

- Pre-fix baseline: gate-off precision 0.6071428571428571; gate-on precision 0.6052631578947368.
- Gate-off precision: 0.6071428571428571
- Gate-on precision: 0.6052631578947368
- Unique gate-off precision: 0.6111111111111112
- Unique gate-on precision: 0.5769230769230769
- Reviewed nomination recall: 0.5 (36/72)
- Correct nomination recall: 0.6666666666666666 (22/33)
- P0 correct nomination recall: 0.7586206896551724 (22/29)
- Matched claims: 56
- Unmatched claims: 64
- Missing verdict: 0

| tactic | claims | confirmed | possible | insufficient | rejected | unobservable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| corner-near-far-post | 41 | 1 | 39 | 1 | 0 | 0 |
| cutback | 4 | 0 | 4 | 0 | 0 | 0 |
| fast_break_pattern | 56 | 0 | 29 | 23 | 4 | 0 |
| run_in_behind | 19 | 0 | 9 | 7 | 3 | 0 |

Contradictions: none

### Matched-rejected tier migrations

| clip | observer | tactic | human | before | after |
| --- | --- | --- | --- | --- | --- |
| soccernetgs:SNGS-007 | gemini | fast_break_pattern | wrong | rejected | insufficient |
| soccernetgs:SNGS-023 | doubao | run_in_behind | wrong | rejected | rejected |
| soccernetgs:SNGS-026 | doubao | fast_break_pattern | correct | rejected | insufficient |
| soccernetgs:SNGS-037 | doubao | fast_break_pattern | correct | rejected | insufficient |
| soccernetgs:SNGS-037 | gemini | fast_break_pattern | correct | rejected | insufficient |
| soccernetgs:SNGS-040 | doubao | fast_break_pattern | wrong | rejected | insufficient |
| soccernetgs:SNGS-053 | doubao | run_in_behind | correct | rejected | rejected |
| soccernetgs:SNGS-061 | gemini | fast_break_pattern | wrong | rejected | insufficient |
| soccernetgs:SNGS-078 | doubao | fast_break_pattern | wrong | rejected | insufficient |
| soccernetgs:SNGS-129 | doubao | fast_break_pattern | correct | rejected | insufficient |
| soccernetgs:SNGS-189 | doubao | fast_break_pattern | correct | rejected | insufficient |

## Frozen-test provisional 20/43 (2026-07-21)

> **仅供会前优先查看，不是最终冻结测试结论。** 这是用户授权提前处理的 20/43 条：15 条 event clips 与 5 条 SoccerNetGS challenge clips；其余 23 条尚未纳入。本轮使用冻结设置（640px/格、JPEG q3、uniform overview 4×3 / dense 4×4），仅上传带时间戳 contact sheets 给 Doubao/Gemini；Sol 自动调用关闭。

- 20/20 条完成，运行失败 0；共 68 个模型提名（Doubao 51，Gemini 17）。
- 分层：confirmed 0、possible 21、insufficient 18、rejected 13、unobservable 16。没有人工专家标签，因此**不报告 precision 或召回**。
- 轨迹 checker 在 68/68 个提名上产生非空结果，说明 Stage-1 → checker 链路已被激活。
- 19/20 条触发“双模型分歧且轨迹 insufficient”，均保留为 Sol 人工复核待办，未被自动升级。这一比例远高于计划中的低个位数预期，是下一步优先诊断项。
- 战术提名分布：corner-near-far-post 28、fast_break_pattern 20、cutback 14、run_in_behind 6（按模型提名计，非去重 clip 数）。
