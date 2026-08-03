# Benchmark 数据说明

`tactical_prototypes/` 只保留当前开放战术提名与 two-stage 解说的直接输入：

| 文件 | 用途 |
|---|---|
| `source_rows.jsonl` | 冻结 67 个 clip 到原始视频路径的来源索引。 |
| `recognition_review.jsonl` | two-stage 解说使用的人工确认战术事实。 |
| `deep_commentary_exemplar_v2.json` | 第一阶段视频分析 prompt 的人工示例。 |
| `scoring_rubric.md` | 战术识别与人工复核评分规范。 |

Active 开放提名结果位于 `outputs/tactical_claim_benchmark/opentac/`；历史 closed-claim、judge queue、旧代码及可用的 superseded open-nomination 支持文件位于 `archive/tactical_claim_benchmark/`。
