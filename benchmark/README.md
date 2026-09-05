# Benchmark 数据说明

`tactical_prototypes/` 只保留两段式解说（`pipeline/stage2b/tactical_commentary.py`）的直接输入：

| 文件 | 用途 |
|---|---|
| `recognition_review.jsonl` | 人工确认的战术事实，锁定 tag、证据文本和时间窗。 |
| `deep_commentary_exemplar_v2.json` | 第一阶段视频分析 prompt 的人工方法示例。 |

两者都由 `scripts/run_tactical_commentary.py` 读取。

multi-agent 两条臂的输入不在这里，在 `data/`：GT 是 `data/grounded_tactics/gsr2025_gt_37.csv`，
战术词条是 `data/足球战术数据库_词条表_Grid.csv`。
