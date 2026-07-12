# Tactical Knowledge Base v2 — Expansion and Verification Design

**Date:** 2026-07-11
**Status:** Approved design
**Applies to:** `pipeline/stage2b/concepts.yaml`, Stage 2B tactical proposal and verification, and the relations/topology inputs used by verified recipes

## Purpose

Expand the tactical catalog without letting catalog size increase false tactical commentary. KB v2 therefore separates broad knowledge collection from narrow production enablement:

- **Catalog:** 12–15 normalized concepts with definitions, aliases, sources, actor roles, claim policy, and maturity.
- **Candidate pool:** about six high-observability concepts receive a multi-match feasibility audit.
- **Production:** at most three concepts are enabled. Zero, one, or two is acceptable when fewer pass the gate.

The release is precision-first: missing a tactic is preferable to inserting a false one.

## Primary decisions

1. GenTac is the primary external reference for hierarchical match-phase and event context, not the canonical taxonomy of verifiable tactical concepts.
2. `tactical_state` remains concept-neutral. It contains reproducible measurements, context, quality, and provenance, not final tactical labels.
3. Doubao proposes a concept, evidence window, team, and actor `track_id` bindings. It cannot propose the acceptance test.
4. Code verifies proposals with a private, fixed, versioned recipe/checker and rebuilds a verified fact from whitelisted fields.
5. Doubao realizes Chinese commentary only from verified facts and allowed claim levels.
6. KB v2 supports observation claims only. Effect and intention remain disabled unless separately verified in a later version.
7. There is no general query DSL, production quota, per-clip threshold, or full possession-engine rewrite in KB v2.

## Verified current constraints

### Runtime

- `pipeline/stage2b/concepts.yaml` currently contains nine prose entries, while runtime code effectively consumes IDs only.
- The current model returns `evidence_queries`, predicates, and thresholds together with its tactical proposal. The verifier evaluates those model-authored tests, so the model can effectively write its own acceptance rule.
- Current relation queries cannot reliably express fixed actor pairs, ordered movement, or continuous duration.
- The current snapshot path samples at 2 Hz and removes tracks using an 80% full-clip coverage gate. That is unsuitable for short, locally visible tactical actions.
- Topology already exposes useful team-shape measurements, but KB v2 must consume only the measurements required by an audited concept.

### Local SoccerNetGS data

- The 57 local train label files are truncated and cannot be parsed as complete JSON.
- The local valid split is an empty directory skeleton.
- The 49 test clips are complete and belong to three matches: game 7 (`SNGS-116`–`131`), game 8 (`SNGS-132`–`150`), and game 11 (`SNGS-187`–`200`).
- `SNGS-116` and `SNGS-117` have already been used diagnostically. Match-disjoint evaluation therefore treats all of game 7 as contaminated development/diagnostic material.

Repairing train/valid from the authoritative SoccerNetGS distribution is a prerequisite for formal recipe development. Until then, the catalog may be completed, but no concept may claim to have passed the formal production gate.

## Taxonomy boundary

GenTac source labels remain available as a compatibility layer:

```yaml
source_labels:
  gentac:
    macro_type: transition
    subtype: progression
    confidence: 0.82
    provenance: rule_v1
```

SoccerMaster may normalize context into orthogonal fields such as possession phase, restart state, progression state, and danger state. These fields answer “what match situation is this?” and remain outside the tactical concept catalog.

The KB answers “what independently verifiable tactical pattern occurred?” Examples include overlap, switch of play, line break, or a compact block.

The initial feasibility pool is:

- `overlap_run`
- `run_in_behind`
- `switch_of_play`
- `local_numerical_superiority`
- `compact_block`
- `line_break`

Additional catalog entries may include `underlap_run`, `side_overload`, `halfspace_occupation`, `width_depth_stretch`, `one_two`, `third_man_run`, `counter_attack`, `high_press`, and `low_block`. Catalog inclusion does not imply implementation or production eligibility.

Legacy IDs are normalized during catalog review. In particular, `positional_attack` moves to match context, `depth_run` becomes an alias of `run_in_behind`, `numerical_superiority` becomes `local_numerical_superiority`, and `switch_play` becomes `switch_of_play`.

## Runtime architecture

```text
GSR / tracking / calibration / topology
                    ↓
       concept-neutral tactical_state
                    ↓
Doubao proposal {concept_id, evidence_window, team_id, actor_track_ids}
                    ↓
        private fixed recipe/checker
                    ↓
          verified_tactical_facts
                    ↓
        Doubao observation-level wording
```

### Public proposal contract

The prompt receives only public semantic fields: ID, names, aliases, observation definition, and required actor roles. Recipe implementation and thresholds are excluded.

Doubao may return only:

```yaml
concept_id: overlap_run
evidence_window: {start_s: 12.8, end_s: 18.6}
team_id: left
actors:
  carrier_track_id: 12
  runner_track_id: 27
```

Unknown fields are discarded or rejected. Model-provided `evidence_queries`, predicates, thresholds, recipe IDs, and verification results cannot affect acceptance.

### Private verification contract

The verifier:

1. validates the concept, approved window, team, and required actor bindings;
2. loads the fixed recipe/checker and parameters from the KB;
3. evaluates the candidate against raw measurements and quality gates;
4. returns `unsupported` when evidence is insufficient;
5. constructs a new verified fact from whitelisted fields when the recipe passes;
6. records the recipe version and resolved evidence.

The verifier must not copy the proposal object and add `verified: true`.

Evidence windows are independent of commentary duration and scheduling. KB v2 removes any unified five-second tactical evidence limit. Each selected recipe owns its minimum and maximum evidence duration. Commentary can start only after `decision_time_s`.

`analysis_context` may extend before or after an evidence window to resolve ambiguity, but it is not silently counted as recipe evidence. Offline lookahead delays `decision_time_s` until the last required observation is available. Final commentary slots are selected separately after that time.

### Sampling policy

- Preserve raw tracking at 25 Hz.
- Verify candidate windows against 25 Hz measurements.
- Give Doubao a compact proposal summary at approximately 12.5 Hz or lower when aggregation removes repetition.
- Generate team-structure summaries at approximately 10 Hz only when a selected recipe requires them.
- Store key timestamps, durations, resolved evidence, and quality flags in verified facts rather than copying every frame.

## Minimal data contracts

### Tactical state

```yaml
clip_id: SNGS-xxx
state_source: ground_truth  # or gsr_prediction
source_labels: {}
normalized_context: {}
measurements: {}
quality: {}
provenance: {}
```

`tactical_state` must not contain concept candidates or final concepts such as `overlap_candidate` or `likely_counter_attack`.

### Catalog entry

```yaml
- id: overlap_run
  aliases: {zh: [套边], en: [overlap]}
  observation_definition: 无球队员从持球人身后沿外侧向前超越。
  required_actor_roles: [carrier, runner]
  sources:
    - citation: "Glossary of association football terms: overlap"
      use: terminology_and_definition
  allowed_claim_levels: [observation]
  maturity: audited            # planned | audited | production
  production_enabled: false

  recipe:
    id: overlap_run
    version: 1
    checker: overlap_run
    parameters: {}
```

Rules:

- A `planned` concept may omit `recipe`.
- An `audited` concept has passed the feasibility audit and may receive a frozen recipe.
- `production_enabled: true` requires `maturity: production` and a passed production gate.
- The prompt builder explicitly projects public fields and never serializes `recipe`.
- The recipe uses a small named checker plus versioned parameters; KB v2 does not introduce a general expression language.

### Verified fact

```yaml
fact_id: fact_001
concept_id: overlap_run
recipe: overlap_run@1
state_source: gsr_prediction
evidence_window: {start_s: 12.8, end_s: 18.6}
decision_time_s: 18.6
team_id: left
actors:
  carrier_track_id: 12
  runner_track_id: 27
verified_claim_levels: [observation]
resolved_evidence: []
quality_flags: []
provenance: {}
```

Stage 2B writes under `outputs/<clip_id>/comments/`:

- `tactical_proposals.json` for proposals, rejection reasons, and audit history;
- `verified_tactical_facts.json` for facts that passed fixed recipes;
- commentary records that reference `fact_id` only.

When there is no verified fact, Stage 2B preserves ordinary event commentary.

## Catalog and feasibility workflow

### 1. Repair and validate data

Replace the truncated train labels and populate valid from the authoritative distribution. Validation must confirm parseable JSON, expected frame counts, unique sequence IDs, required `game_id`, and usable pitch-coordinate annotations.

### 2. Normalize the catalog

Create 12–15 entries with stable IDs, aliases, observation-only definitions, actor roles, source roles, and default `production_enabled: false`. Definitions must not embed unverified effects or intentions such as “creates an advantage” or “pulls away a defender.”

### 3. Audit the candidate pool

Use repaired train/valid clips only. For every candidate:

- locate at least six positive and six targeted hard-negative windows;
- cover at least two development matches;
- annotate the evidence window and required actor `track_id` values;
- record tracking, ball, calibration, and relevant structure quality;
- measure observability separately on ground-truth and predicted GSR state;
- record GT-to-GSR degradation and unsupported cases.

If the minimum evidence cannot be found, the concept remains `planned`. The audit does not lower semantic boundaries to create more positives.

### 4. Add only required capabilities

Only audited concepts receive complete recipes. If an audited concept needs a pairwise offset, ordered transition, duration, or topology quantity, implement only that bounded capability. Do not build unused grammar or expose all topology metrics pre-emptively.

### 5. Freeze and evaluate

Thresholds are global pitch-distance/time parameters calibrated on train/valid, never per clip or per match. Before opening held-out results, freeze a release manifest containing:

- KB and recipe versions;
- selected concepts and claim policy;
- precision and false-insertion gates;
- acceptable GT-to-GSR degradation;
- data-quality and abstention rules;
- model and decoding configuration.

Game 7 remains diagnostic. Games 8 and 11 remain locked, match-disjoint held-out matches. Each recipe is run once after freezing. A failed held-out recipe cannot be retuned against the same held-out matches and retain the same validation claim.

## Production gate

A concept may become production-enabled only when all of the following hold:

```yaml
production_gate:
  observed_in_multiple_matches: true
  positive_and_hard_negative_coverage: sufficient
  gt_precision: pass
  gsr_precision: pass
  false_insertions_per_minute: pass
  gt_gsr_degradation: acceptable
  evidence_traceable: true
  observation_only: true
```

Exact numerical gates are selected on development data and written to the frozen release manifest before held-out evaluation. Each concept needs at least four positive and four targeted hard-negative windows across held-out games 8 and 11, with both matches represented. Anything smaller yields `not_evaluable`, not a pass.

No category balance is imposed on the winners. If three local-action concepts are the only concepts that pass, all three may be enabled. If only two concepts pass, only two are enabled. If none pass, KB v2 ships its catalog with zero tactical concepts exposed to commentary.

## Error handling

- Malformed KB structure, duplicate IDs, invalid maturity, or an unresolved production checker fails startup validation.
- Invalid proposals are rejected and recorded without reaching the verifier.
- Missing actor, ball, calibration, or structure evidence produces `unsupported`, not a negative tactical assertion.
- Recipe exceptions produce an audited rejection and preserve direct commentary.
- Commentary referencing a missing fact ID, non-observation claim, or disabled concept fails the commentary audit.

## Testing

### Unit tests

- YAML loading and minimal validation;
- public catalog projection excludes recipes and thresholds;
- proposal whitelist and stable `track_id` actor binding;
- versioned checker dispatch and fact reconstruction;
- maturity and production gate invariants.

### CI fixtures

Each audited concept has at least two known positives and two targeted hard negatives. Fixtures are regression checks and may be used during development; they are not evidence of detector quality.

### Integration tests

- LLM-authored predicates cannot change verifier acceptance;
- insufficient evidence abstains instead of producing a fact;
- failed tactical processing preserves ordinary event commentary;
- the composer can consume verified facts but cannot consume raw proposals;
- any effect or intention wording fails the v2 audit.

### Production evaluation

Run the same frozen recipes on paired ground-truth and GSR-predicted state. Report raw TP/FP/FN/TN counts, precision, false insertions per minute, unsupported rate, GT-to-GSR degradation, actor/temporal grounding, and overclaim failures per concept and per match.

## Explicit non-goals

- Training a tactical classifier, trajectory model, critic, or counterfactual model.
- A general temporal-rule DSL or broad checker framework before repeated needs exist.
- Completing a production recipe for every catalog entry.
- A complete possession-state engine inside this KB project.
- Intention or causal-purpose commentary.
- Using GenTac labels as production detector thresholds.
- Treating CI fixtures or one inspected clip as a scientific benchmark.
- Building term-scraping, ranking, or reusable ingestion pipelines for a 12–15 concept catalog.

## Acceptance criteria

- Train/valid labels are repaired and pass the dataset integrity check before formal recipe development.
- The catalog contains 12–15 normalized entries, all disabled by default.
- The proposal model cannot author or alter verification conditions.
- Every accepted fact is traceable to a fixed recipe version, evidence window, actor bindings, measurements, and quality state.
- Candidate feasibility is assessed across multiple development matches and on both GT and GSR state.
- Only capabilities required by audited concepts are added.
- Games 8 and 11 remain untouched until the release manifest is frozen.
- At most three observation-only concepts become production-enabled, with no minimum quota.
- Any failure or unsupported evidence falls back to ordinary event commentary.
