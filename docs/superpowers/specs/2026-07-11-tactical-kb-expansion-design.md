# Tactical Knowledge Base (concepts.yaml) — Expansion Design

Date: 2026-07-11 · Applies to `pipeline/stage2b/concepts.yaml` (currently 9
prose-only entries; verification thresholds hard-coded in stage2b code).

## Goal

Grow the KB 9 → ~25 (v2) → ~50 (v3) concepts while keeping every entry
machine-verifiable, without a code change per concept.

## Finding: no ready-made source exists

No open tactics-glossary dataset exists (GitHub/HF checked); SoccerWiki/GOAL/
SoccerRAG are entity/stat KBs. The KB itself is therefore a contribution.
Usable raw sources:

| Source | Use | License |
|---|---|---|
| Wikipedia en+zh: Glossary of association football terms; Association football tactics | term inventory + definitions to paraphrase | CC BY-SA |
| StatsBomb open-data event spec | machine-oriented definitions ≈ ready evidence recipes (pressure, carry, cutback…) | open docs |
| Academic definitions: TacticAI, overlapping-runs GNN, pitch control/OBSO, TacSIm (2603.25199), TacEleven (2511.13326) | formal/mathematical definitions → thresholds, citable | papers |
| SoccerNet-Echoes / MatchTime commentary text | term-frequency prioritization + style reference | research use |
| Spielverlagerung / 懂球帝 / 知乎 etc. | background reading only; paraphrase facts, never copy text | copyrighted |

## Schema (decision: YAML conditions + code fallback)

Extend each entry; the `evidence` grammar is exactly what
`pipeline/relations/query.py` already resolves — no new DSL:

```yaml
- id: overlap_run
  name_zh: 套边（套上）
  name_en: overlapping run
  description: 无球队员从持球队友身后沿边路外侧超越持球人，制造边路人数优势。
  exemplar_zh: 7号从外线套上，边路瞬间形成二打一！
  exemplar_en: Number 7 overlaps down the flank — two on one!
  source: "Wikipedia: Glossary of association football terms (overlap)"
  evidence:                      # ALL conditions must hold (AND)
    - {quantity: rel_x, agg: min, op: "<", value: 0}    # starts behind carrier
    - {quantity: rel_x, agg: max, op: ">", value: 0}    # ends past carrier
    - {quantity: rel_y, agg: max, abs: true, op: ">", value: 15}
    - {quantity: speed, agg: max, op: ">", value: 5.5}

- id: decoy_run
  ...
  checker: check_decoy_run       # cross-player logic: registered Python fn
```

Rules: an entry has `evidence` (declarative, ~80% of concepts) OR `checker`
(registered in a small registry module), never neither. Loader validates:
required fields, unique ids, evidence quantities ∈ resolver vocabulary,
checker names resolvable. The LLM keyword-proposal step is unchanged; the
verification step reads recipes from the KB instead of hard-coded thresholds.
Migrate the existing 9 entries' in-code thresholds into `evidence` blocks as
part of v2.

## Expansion pipeline (per concept; KB = tested software)

1. **Inventory** — one-off script dumps the Wikipedia glossary/tactics term
   list (en+zh titles) via MediaWiki API; dedupe against existing ids.
2. **Prioritize** — one-off script counts term frequency in Echoes/MatchTime
   commentary text; add what commentators actually say, most-frequent first.
3. **Draft** — LLM drafts the full entry (definition paraphrased from the
   cited source, zh/en names, exemplars, evidence hints). Never auto-merged.
4. **Review + recipe** — Frank reviews prose and authors/adjusts the
   `evidence` conditions (the human step).
5. **Clip-validate** — pytest gate: each concept must fire on ≥2 known-positive
   windows from GT sequences and stay silent on ≥2 negatives (window refs
   stored in `tests/kb_fixtures.yaml`). No test, no merge.
6. **Release** — bump `version:` in concepts.yaml; v2 target ≈25 (the 9
   existing + underlap, third_man, one_two, cutback, buildup_from_back,
   mid_block, compactness, counter_press, wing_overload, halfspace_occupation,
   line_break, press_resistance, decoy_run, second_ball_protection,
   width_depth_stretch, transition — final list subject to step-2 ranking).

## Tooling (decision: minimal, one-off)

`scripts/kb_inventory.py` (~50 lines, MediaWiki fetch → terms.txt) and
`scripts/kb_priority.py` (~50 lines, Echoes text → term counts). Throwaway
quality, no tests, not pipeline modules. A reusable extraction module is
deferred to v3 (~50 concepts) when the benchmark taxonomy also consumes it.

## Rejected alternatives

- Full expression DSL in YAML (maintaining a language for 25 entries).
- Thresholds staying in code (every concept addition becomes a code PR;
  collaborators must read Python to contribute knowledge).
- Scraping copyrighted tactical blogs (legal risk; paraphrase-only policy).
- Polished extraction pipeline now (one-time batch of ~16 doesn't justify it).

## Risks

- Threshold tuning per concept is empirical — the clip-validation fixtures are
  the guard; expect 1-2 iterations per concept.
- zh/en term mismatches (套上 vs overlap vs 套边) — `aliases:` field optional,
  LLM prompt uses name_zh + name_en together.
- Echoes is English commentary — zh frequency signal is approximated via
  translation of the term list, acceptable for ranking.
- Evidence grammar limits (AND-only, single-player quantities) — that's what
  `checker:` is for; resist extending the grammar until ≥3 concepts need the
  same missing feature.
