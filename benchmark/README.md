# Benchmark 数据说明

本目录中的 JSONL 采用“一行一个 JSON 对象”的格式，JSON 则保存清单、分组、统计报告或单次运行结果。

## JSONL 文件

| 文件 | 记录数 | 一句话说明 |
| --- | ---: | --- |
| `tactical_prototypes/concepts.jsonl` | 39 | 这是战术概念词典，说明每个战术叫什么、看什么证据以及哪些相似场面必须排除。 |
| `tactical_prototypes/source_rows.jsonl` | 783 | 这是来源审计账本，无损记录每一行 `tactics.csv` 或 manifest clip 来自哪个视频、哪个数据集和哪一行。 |
| `tactical_prototypes/episodes.jsonl` | 447 | 这是原型 RAG 真正检索的案例库，保存正例、困难反例、不可观察案例及其 gold/silver 审核状态。 |
| `tactical_prototypes/recognition_review.jsonl` | 72 | 这是 72 行识别结果人工复核的结构化版本，带错误根因标签。 |
| `tactical_prototypes/recognition_evidence.jsonl` | 72 | 这是从人工复核文本手工编码的逐条证据，供确定性 checker 离线回放。 |

三者的关系是：`concepts.jsonl` 说明“战术是什么”，`source_rows.jsonl` 记录“原始标注从哪里来”，`episodes.jsonl` 提供“哪些片段可以作为该战术的案例”。

## JSON 文件

| 文件 | 一句话说明 |
| --- | --- |
| `development_clip_manifest.json` | 这是开发集使用记录，列出已经用于开发调试的片段及其数据划分和比赛编号，防止后续把它们误当独立测试数据。 |
| `manifests/soccernetgs_v1_3_integrity.json` | 这是 SoccerNetGS v1.3 完整性检查结果，记录 train/valid 应有数量以及 115 个序列的文件校验状态。 |
| `tactical_prototypes/source_groups.json` | 这是来源隔离映射，把同一比赛或同一原始视频产生的片段归为一组，避免检索库和测试集发生数据泄漏。 |
| `tactical_prototypes/migration_report.json` | 这是标注迁移总报告，汇总别名归一化、正例和审核状态计数、孤立视频以及未映射标签。 |
| `tactical_prototypes/evaluation_report.json` | 这是 source-disjoint 评估报告，记录各来源组分到 retrieval 还是 test、gold/silver 数量以及是否存在跨划分冲突。 |
| `tactical_prototypes/replay_report.json` | 这是确定性 checker 对 72 条人工复核 claim 的逐条离线回放结果及分战术精确率统计。 |
| `tactical_prototypes/agent_runs/soccernetgs__SNGS-001.json` | 这是 SNGS-001 的一次多智能体运行快照，保存 Doubao/Gemini 的独立观察、检索到的原型上下文和 Skeptic 结论。 |
| `tactical_prototypes/sol_review_bundle/disputes.json` | 这是交给 Sol 或人工复核的争议包，保存尚未解决的 Agent 分歧、相关原型上下文和复核说明。 |

## 其他文件

| 文件 | 一句话说明 |
| --- | --- |
| `tactical_prototypes/scoring_rubric.md` | 这是冻结的评分规范，规定多标签、时间窗、restart 否决等判定规则及 P0/P1 范围。 |
| `tactical_prototypes/replay_report.md` | 这是 checker 离线回放报告：各 P0 战术加 checker 前后的精确率与误否决清单。 |
