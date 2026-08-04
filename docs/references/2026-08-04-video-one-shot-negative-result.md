# Video one-shot 提名：负结果记录

2026-08-04。目的：记录结论，避免以后重复尝试"给 OpenTAC 加一个正例视频当 few-shot"。

## 结论

给 Gemini 附带一段该战术的确认正例视频（FIFA Game Library 片段），再要求它对查询视频做同样的 P0 开放提名，**整体上比不给示例视频更差**。

| scope | zero-shot Top-1 | one-shot Top-1 | ΔTop-1 | zero-shot Top-3 | one-shot Top-3 | ΔTop-3 |
|---|---:|---:|---:|---:|---:|---:|
| overall (50 claims) | 0.6400 | 0.6000 | -0.0400 | 0.6600 | 0.5600 | -0.1000 |
| counter-attack | 0.7500 | 0.7083 | -0.0417 | 0.7917 | 0.5833 | -0.2083 |
| cutback | 1.0000 | 0.6667 | -0.3333 | 1.0000 | 0.6667 | -0.3333 |
| run-in-behind | 0.3571 | 0.2143 | -0.1429 | 0.3571 | 0.3571 | 0.0000 |
| line-breaking-pass | 0.6667 | 0.8889 | +0.2222 | 0.6667 | 0.7778 | +0.1111 |

只有 line-breaking-pass 一个战术有正向增益，其余全部下降或持平。Doubao 在 one-shot 条件下 0 条有效输出（视频拼接失败）。

一个改良版（video_one_shot_v2：观察阶段与提名阶段拆开、加教学证据/required/excluded 字段）覆盖率也未跑满：Gemini 17/23、Doubao 4/23，未产出可比数字。

## 产物位置

- `outputs/tactical_claim_benchmark/opentac/evaluation/shot_comparison/summary.md`（v1，50 claims，全部数字见上表）
- `outputs/tactical_claim_benchmark/opentac/evaluation/shot_comparison_v2/summary.md`（v2，23 claims，覆盖不完整）
- 原始逐 clip 产物：`outputs/tactical_claim_benchmark/opentac/phase1_video_one_shot/`、`outputs/tactical_claim_benchmark/opentac/video_one_shot_v2/`

## 代码状态

生成/评测这两条实验的代码（`run_one_shot`、`run_one_shot_v2`、`report_shot_comparison`、`report_shot_comparison_v2` 等）已从 `pipeline/tactics_qa/opentac.py` 删除（见 2026-08-04 opentac 简化）。产物 JSON 保留在 `outputs/` 下作为历史记录，但不再有代码路径可以重新生成或评分它们。

## 为什么会更差（推测，未验证）

正例视频提供了"这个机制长什么样"的额外上下文，但代价是：(a) 视频 token 预算被两段视频分摊，查询视频本身的观察精度下降；(b) 模型可能把示例的表层特征（镜头角度、球队配色、比分板）误当作判别信号迁移到查询视频。两者都未做消融，仅作为待验证假设记录。
