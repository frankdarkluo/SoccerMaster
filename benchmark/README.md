# Benchmark 数据说明

`tactical_prototypes/` 只保留当前开放战术提名与 two-stage 解说的直接输入：

| 文件 | 用途 |
|---|---|
| `clip_manifest.jsonl` | 当前 OpenTAC 冻结的 67 个 clip 路径清单。 |
| `recognition_review.jsonl` | two-stage 解说使用的人工确认战术事实。 |
| `deep_commentary_exemplar_v2.json` | 第一阶段视频分析 prompt 的人工示例。 |
| `scoring_rubric.md` | 战术识别与人工复核评分规范。 |

Active 开放提名结果位于 `outputs/tactical_claim_benchmark/opentac/`；已退役 closed-claim 代码通过 Git 历史追溯。

完整 783 行来源标注总账保存在 `data/annotations/tactical_source_rows_full.jsonl`，不作为运行输入。
