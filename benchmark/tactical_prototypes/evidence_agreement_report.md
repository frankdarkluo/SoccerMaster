# 自动证据一致率报告

## 结论摘要

- 本机 72 条 claim 中仅 3 条有 GSR 输出，69 条为 no_data；以下结果只是适配器诊断，不是系统泛化能力。
- checker 判定完全一致：1/3 (33.3%)。
- 自动证据 false veto：0。
- 规则阈值未为提高覆盖率而调整；缺证据保留为 auto_missing/insufficient。

### 优先感知 backlog

1. no GSR predictions.json：69 条
2. key-pass kick not detected：2 条
3. 覆盖过小，未观察到第三类原因，不补造结论。

## 逐字段一致率

| 字段 | match | mismatch | auto_missing | hand_missing | both_missing | 可比较项一致率 | 自动缺失率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| restart_at_origin | 0 | 1 | 0 | 0 | 2 | 0.0% | 0.0% |
| has_controlled_regain | 3 | 0 | 0 | 0 | 0 | 100.0% | 0.0% |
| chaotic | 3 | 0 | 0 | 0 | 0 | 100.0% | 0.0% |
| delivery_kind | 1 | 0 | 2 | 0 | 0 | 100.0% | 66.7% |
| corner_landing_zone | 1 | 0 | 0 | 0 | 2 | 100.0% | 0.0% |

### 数据审计差异

- soccernetgs:SNGS-116: restart_at_origin
- SNGS-116 的手工 corner 时间是窗口起点 3.0s，轨迹检测为实际开球 6.2s；保留原记录，需另行审计，不计作 auto_missing。

## P0 precision_after：自动证据 vs 手工证据（同一覆盖子集）

| tactic | covered claims | auto | hand | auto false vetoes |
| --- | ---: | ---: | ---: | ---: |
| fast_break_pattern | 0 | N/A | N/A | 0 |
| run_in_behind | 1 | None | 1.0 | 0 |
| corner-near-far-post | 1 | 1.0 | 1.0 | 0 |
| cutback | 0 | N/A | N/A | 0 |

## 自动证据回放明细

# Checker 离线回放报告

## 结果摘要

- P0 precision：corner-near-far-post 1.0→1.0；run_in_behind 1.0→None。
- P0 共拦截 0 条既有误报，false vetoes=0。
- P0 residual：无。
- 其余 surviving wrong 属于 P1 或 deferred，只用于定位下一轮 checker 范围。

| tactic | scope | claims | precision before | precision after | flipped | false vetoes | insufficient | surviving wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corner-near-far-post | P0 | 1 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
| line_break | P1 | 1 | 1.0 | None | 0 | 0 | 1 | 0 |
| run_in_behind | P0 | 1 | 1.0 | None | 0 | 0 | 1 | 0 |

## 手工证据回放明细（同一覆盖子集）

# Checker 离线回放报告

## 结果摘要

- P0 precision：corner-near-far-post 1.0→1.0；run_in_behind 1.0→1.0。
- P0 共拦截 0 条既有误报，false vetoes=0。
- P0 residual：无。
- 其余 surviving wrong 属于 P1 或 deferred，只用于定位下一轮 checker 范围。

| tactic | scope | claims | precision before | precision after | flipped | false vetoes | insufficient | surviving wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corner-near-far-post | P0 | 1 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
| line_break | P1 | 1 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
| run_in_behind | P0 | 1 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
