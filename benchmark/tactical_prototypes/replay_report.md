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
