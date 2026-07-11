# Hybrid Event-First Tactical Commentary Design

**Date:** 2026-07-10
**Status:** Revised after joint cleanup/hybrid design review

## Objective

Build Stage 2B hybrid commentary that preserves accurate, energetic visible-event narration and adds only verified tactical interpretation when it helps the viewer.

The governing rule is event first, tactics second. Tactical content is optional. A high-confidence visible event may never be lost, renamed, reversed, or crowded out.

## Product Rules

1. Direct video understanding is authoritative for what visibly happened.
2. Relations, radar, and the glossary may explain shape or intent, but may not overwrite a conflicting event.
3. Tactical density is adaptive:
   - fragmented 30-second clip: zero or one tactical sentence;
   - incomplete attack: at most one local observation;
   - one-to-two-minute clip: zero to two tactical sentences;
   - complete visible attack: up to three evidence-backed tactical sentences.
4. A short tactical clause may follow the event it explains.
5. A standalone tactical sentence is allowed only in an event-free gap of roughly four to five seconds or after a complete phase.
6. Formal terms and natural descriptions are both valid when accurate and natural.
7. Intentional silence is valid; the commentary does not cover every second.
8. The LLM selects among code-approved windows and may choose no tactical insertion.
9. Standalone tactical-only commentary mode is removed.

## Stage 2B Modes

`hybrid` is the default mode. `direct` remains an explicit baseline and fallback.

Direct mode writes:

- `comments/events.json`
- `comments/event_spine.json`
- `comments/commentary.json`

Hybrid mode writes:

- `comments/events.json`
- `comments/event_spine.json`
- `comments/commentary_direct.json`
- `comments/relations.json`
- `comments/radar/`
- `comments/tactical_candidates.json`
- `comments/commentary.json`

Stage 3 reads only `comments/commentary.json`.

## Minimal Module Structure

```text
pipeline/stage2b/
├── concepts.yaml
├── digest.py
├── events.py
├── generate.py
├── hybrid.py
├── run.py
└── video.py

pipeline/relations/
├── build.py
├── kinematics.py
├── query.py
├── radar.py
└── snapshots.py
```

- `digest.py` owns Stage 1 frame loading and the minimal tracking digest/possession helpers.
- `events.py` owns the closed event catalog and importance values.
- `generate.py` owns one ARK request path, direct observation, local visual verification, parsing, and JSON writing.
- `hybrid.py` owns candidate generation, scheduling, composition, auditing, and direct fallback.
- `relations/query.py` owns the supported query grammar and predicate evaluation.
- `concepts.yaml` is data, not executable strategy.

There is no `pipeline/tactics/`, generic LLM adapter, adapter factory, Qwen backend, OpenAI backend, or mock production backend.

## Architecture

### 1. Direct Event Observer

The observer receives the full video plus the tracking digest and produces a draft event spine and expressive direct wording at `temperature=0.7`.

Event codes come from the closed Stage 2B event catalog. The catalog includes `football.corner` and the importance values needed to identify key events.

The direct baseline is generated from this same event spine; hybrid does not run an unrelated competing event narrator.

### 2. Selective Event Verification

A second low-temperature visual pass is performed only for key or ambiguous events that are eligible for narration. Any event proposed for high confidence counts as key.

The verifier inspects a narrow temporal window and returns structured event code, timing, team, actor, outcome, and disagreement reasons. It assigns the final confidence tier.

### 3. Relations and Tactical Candidates

Stage 2B builds `relations.json` and radar PNGs from Stage 1 predictions.

The LLM proposes one to three machine-readable queries per tactical candidate. Code validates the query grammar, resolves it against relations snapshots, and computes each predicate result. The model cannot set `verified` or `predicate_passed`.

A candidate is eligible only when it has at least one valid query and every predicate passes.

### 4. Hybrid Composer

The composer receives:

- the event spine;
- direct suggested wording;
- final confidence tiers;
- verified tactical candidates;
- code-generated candidate windows;
- tactical count upper bounds.

It runs at `temperature=0.7`, preserves required event facts, and may select no tactical content.

### 5. Composition Audit

Mechanical checks cover schemas, timestamps, provenance, structured event claims, tactical verification, density bounds, and overlap.

If high-confidence suggested wording changes, a narrow low-temperature semantic equivalence check confirms event code, team/direction, actor, and outcome. This semantic check is not described as a mathematical proof; SNGS-116 remains the empirical regression gate.

### 6. Stage 3 Duration Fit

Stage 2B does not synthesize audio. Every commentary segment contains normal text plus a concise event-only fallback.

Stage 3 synthesizes normal text and measures it with FFprobe. On overflow it resynthesizes once with the fallback text. A second overflow fails explicitly. Speech is never silently truncated.

## Data Contracts

### Event Spine

`comments/event_spine.json` contains ordered events:

```json
{
  "event_id": "evt_001",
  "start_s": 0.0,
  "end_s": 5.0,
  "event_code": "football.corner",
  "player_team": "left",
  "player_jersey": "",
  "actors": ["left_team"],
  "outcome": "corner_taken",
  "confidence": "high",
  "confidence_reasons": ["directly_visible", "two_pass_agreement"],
  "suggested_wording_zh": "左侧球队准备主罚角球，禁区内双方球员密集站位。",
  "suggested_wording_en": "The team on the left prepares to take the corner.",
  "energy": "engaged",
  "verification": {
    "event_code": "football.corner",
    "midpoint_s": 2.4,
    "player_team": "left",
    "player_jersey": "",
    "outcome": "corner_taken",
    "disagreements": []
  }
}
```

Energy is exactly one of `calm`, `engaged`, `excited`, or `explosive`.

### Tactical Candidate

`comments/tactical_candidates.json` contains candidates such as:

```json
{
  "candidate_id": "tac_001",
  "window": {"start_s": 18.0, "end_s": 23.4},
  "concept_id": "low_block",
  "observation_zh": "蓝队形成紧凑的低位防守。",
  "observation_en": "The blue team forms a compact low block.",
  "why_it_matters_zh": "压缩了禁区前沿的空间。",
  "why_it_matters_en": "It compresses space in front of the box.",
  "phase_scope": "local",
  "evidence_queries": [
    {
      "query": {
        "t0": 18.0,
        "t1": 23.4,
        "team": "right",
        "quantity": "opp_line_x",
        "agg": "mean"
      },
      "predicate": {"op": "<=", "threshold": -20.0},
      "result": {"value": -24.5, "n_samples": 6},
      "predicate_passed": true
    }
  ],
  "verified": true
}
```

Supported queries use bounded `t0`/`t1`, optional `jersey` or `team`, a known quantity, and a known aggregation. Predicates use a supported comparison operator and a finite numeric threshold. Unsupported/no-data/failed results make the candidate ineligible.

### Commentary Segment

`comments/commentary.json` preserves the Stage 3 input contract:

```json
{
  "kind": "hybrid",
  "timestamp_s": 23.4,
  "end_s": 27.0,
  "text_zh": "右路迅速把球转向远端，蓝队的横向移动被拉开。",
  "text_en": "The ball is switched quickly and stretches the blue block.",
  "fallback_text_zh": "右路把球转向远端。",
  "fallback_text_en": "The ball is switched to the far side.",
  "energy": "engaged",
  "events_referenced": ["evt_005"],
  "tactical_candidates_referenced": ["tac_002"],
  "event_claims": [
    {
      "event_id": "evt_005",
      "event_code": "football.pass",
      "player_team": "left",
      "outcome": "switch_completed",
      "assertion_strength": "certain"
    }
  ]
}
```

`kind` is `event`, `hybrid`, or `tactical`. Segments are ordered and non-overlapping but need not partition the full clip.

## Confidence Policy

Version 1 uses interpretable evidence gates rather than self-reported model probability.

High confidence requires:

- matching event code across two visual judgments;
- event midpoint difference no greater than 1.0 second;
- team agreement whenever both passes name one;
- exact jersey agreement whenever a jersey is spoken;
- non-conflicting outcomes;
- a direct visible cue;
- no incompatible possession, restart, or ball-location transition in available state data.

A key event has event-catalog `importance_base >= 0.35`, begins or ends a visible phase, or is proposed for high confidence.

- High events are mandatory and fact-locked.
- Medium events with direct visual cues remain mandatory but omit disputed detail.
- Inference-only medium events are optional.
- Low events are omitted.

After a future labeled set exists, evidence gates may be calibrated numerically. The target high-confidence precision remains at least 95%.

## Scheduling

Code creates only eligible windows:

1. causal attachment to a related event;
2. standalone gap of roughly four to five seconds;
3. phase-ending summary after a complete attack.

Standalone tactics cannot overlap required event narration. Event/hybrid segments intentionally align with the events they describe.

Tactical upper bounds are not quotas. Weak tactics are never added to meet a count.

## Failure Handling

1. Retry malformed observer JSON once with schema errors.
2. Downgrade inconclusive visual verification; do not promote it.
3. Missing relations/radar data produces direct event commentary.
4. Invalid tactical queries or failed predicates remove candidates.
5. Retry invalid hybrid composition once with audit errors.
6. A second hybrid failure copies the accepted direct baseline to `comments/commentary.json`.
7. Stage 3 uses segment fallback text once on measured audio overflow.
8. A second audio overflow fails without producing or replacing a final video.

Direct fallback contains all high events and all medium events with direct visible cues.

## SNGS-116 Gate

The retained offline smoke fixture and real run must preserve:

1. left-side corner setup;
2. corner delivery and contested header when actor evidence supports it;
3. defensive clearance and transition;
4. right-side goalkeeper gathering the bouncing ball;
5. long goalkeeper restart and the visible right-side advance.

The opening corner must never be replaced by a high-press interpretation. Hybrid may add at most one verified later observation about the low block or switch of play. Zero tactical additions are valid.

## Test Policy

Only two repository tests remain:

- `tests/test_calibration_guard.py`
- `tests/test_hybrid_smoke.py`

The hybrid smoke test uses local JSON fixtures and fake ARK/TTS responses. In one offline flow it verifies:

- direct event preservation;
- code-computed tactical predicate pass/fail;
- tactical omission on no data;
- sparse scheduling and intentional gaps;
- invalid composer fallback to direct;
- normal/fallback TTS duration selection without truncation;
- the SNGS-116 corner regression.

Detailed unit tests for former detectors, adapters, relation helpers, narrators, and schema internals are removed.

## Evaluation Scope

This work does not implement the 50-clip A/B evaluation tool. The future blinded protocol remains:

- at least 50 balanced clips;
- two independent labels and one adjudicator;
- exact event-code match with ±2-second midpoint tolerance;
- event precision and key-event recall no lower in point estimate than direct;
- paired-bootstrap lower bound no worse than -2 percentage points.

For this implementation, the only real-video release gate is SNGS-116.

## Success Criteria

1. SNGS-116 begins with the corner sequence rather than unsupported pressing.
2. Required event facts survive hybrid composition.
3. Tactical claims are code-verified and may be absent.
4. Commentary retains the direct model information density and emotion.
5. Stage 3 fits speech using one event-only fallback and never truncates.
6. Direct fallback remains usable whenever hybrid evidence or composition fails.
7. No standalone tactical mode or obsolete commentary pipeline remains.
