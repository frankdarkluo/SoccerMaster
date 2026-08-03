# 足球战术原型 RAG 与选择性多智能体 MVP 计划

**Goal:** 在不训练模型、不修改生产 Tactical KB 的前提下，把四份人工复核
`tactics.csv` 迁移为可追溯的战术案例库，并交付原型检索、离线规则验证、
Doubao/Gemini 选择性复核、Sol 人工仲裁和主动标注闭环。

**Architecture:** 原始标注先拆为正例、针对性反例和不可观察案例；提名模型只能
读取公开概念卡和检索案例，固定规则读取原始 `tactical_state-v1` 测量；机器结果
只能进入 silver，人工复核后才能成为 gold。

**Tech Stack:** Python 3 标准库、PyYAML、FFmpeg/ffprobe、已有 OpenAI SDK、
Volcengine Ark 与 Gemini OpenAI-compatible endpoint。

## Global constraints

- 首批概念固定为 `fast_break_pattern`、`cutback`、`line_break`、
  `run_in_behind`、`overlap_run`。
- `fast_break_pattern` 与 `cutback` 仅为 prototype-only；不得修改
  `pipeline/stage2b/concepts.yaml` 或生成生产 `verified_tactical_facts`。
- 不修改任何 MP4、原始 `tactics.csv`、manifest 或战术词条表。
- 所有正标注的 `中`、`中高`、`高`均视为用户确认的 gold 正例。
- “无明确战术”不是全局负样本；只有明确指向候选的 near-miss 才生成原型。
- 外部模型只接收带时间码的抽帧，不接收完整视频；live 命令必须显式传入
  `--allow-external-upload`。
- 保留 dirty worktree；整项完成且用户明确要求前不提交 commit。

## Task 1: Freeze the unified prototype contract

- 在 `pipeline/tactical_prototype_mvp.py` 定义 `tactical-prototype-v1`、
  `tactical-source-row-v1` 与五个 MVP 概念卡。
- 原型字段必须包含全局 `clip_uid`、source group、视频路径、原始/数值时间窗、
  canonical tactic ID、三类 prototype type、review status、证据、缺失条件、actor、
  observability 和 provenance。
- 生成 `prototype_rules.yaml`，其中规则参数永不进入 Agent 的公开 prompt。

## Task 2: Migrate all current annotations

- 读取 `data/*/tactics.csv` 与对应 manifest/video；将 629 行无损写入
  `source_rows.jsonl`。
- 将 187 条正标注写为 `positive + gold`；五个 MVP 概念必须合计 88 条：
  快速反击 39、倒三角 5、破线 16、打身后 25、套边 3。
- 对明确提到 MVP 候选的“无明确战术”依据逐候选派生 silver：关键证据不清为
  `unobservable`，明确违反边界为 `targeted_negative`，两者并存时优先
  `unobservable`。
- 未指向候选的无战术行只进入来源审计；`event-clips_L3_all/video.mp4` 作为
  orphan 报告，不进入案例。
- SoccerNetGS 使用标签/完整性 manifest 的 game ID；事件切片使用 manifest 中
  原始视频哈希作为 source group。

## Task 3: Add retrieval and deterministic verification

- RAG 只从不同 source group 返回 2–3 个 gold 正例、2–3 个 gold 困难反例和
  1–2 个 gold 不可观察案例；silver 默认不进入 prompt。
- checker 读取 `tactical_state-v1`，输出严格的 `passed | failed |
  unverifiable`，并保留逐条件结果和实测值。
- 五类 checker 分别验证：转换后的直接推进、底线斜后球路、受控穿线传球、
  跑者越过可见防线、跑者从持球人身后沿外侧超越。
- tracking、ball、attack direction、关键 actor 或可见性不足时必须
  `unverifiable`。
- 可用 `verify --candidate candidate.json --state tactical_state.json --rules
  benchmark/tactical_prototypes/prototype_rules.yaml` 离线执行；结果不进入生产事实。

## Task 4: Add selective agents and active curation

- Doubao 为主观察者；有候选/弃权/低可观测性时调用 Gemini 独立复核，并固定
  抽查 10% 无候选结果。
- A/B 盲判，不互看结论；仅在分歧、时间窗低重合或规则冲突时调用一次 skeptic。
- unresolved 项导出供当前 Codex/Sol 手工复核；Sol 结果仍为 silver。
- 主动标注默认每轮 50 条：15 分歧、10 规则冲突、10 hard negative、10 稀有
  或低相似度、5 随机审计；同 source group 最多 5 条、同战术默认最多 15 条。
- 人工导入仅允许 `positive`、`targeted_negative`、`unobservable`、
  `outside_taxonomy`、`needs_adjudication`；前三类才可晋升 gold。

## Task 5: Verify the complete MVP

Run:

```bash
conda run -n tracklab python -m pytest -q tests/test_tactical_prototype_mvp.py
conda run -n tracklab python scripts/tactical_prototype_mvp.py migrate \
  --data-root data \
  --glossary data/足球战术数据库_词条表_Grid.csv \
  --out benchmark/tactical_prototypes
conda run -n tracklab python scripts/tactical_prototype_mvp.py validate \
  --root benchmark/tactical_prototypes
```

Migration must be byte-reproducible and report 629 source rows, 187 gold positives
and 88 MVP positives. Automated tests use fake providers. After offline checks pass,
run one authorized SNGS-001 frame-only smoke test against Doubao and Gemini.

## Material Passport

- Mode: implementation plan / complete MVP.
- Inputs: four local reviewed `tactics.csv` files, corresponding videos, glossary and
  existing Tactical KB v2 contracts.
- External consent: candidate-window frames may be sent to Doubao and Gemini; Sol
  receives exported dispute frames only.
- Default models: current Ark model and `gemini-3.1-flash-lite`.
- Unverified before execution: provider availability; verified only by the explicit
  single-clip live smoke test.
