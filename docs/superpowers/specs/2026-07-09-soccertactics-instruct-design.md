# SoccerTactics-Instruct — Design

Date: 2026-07-09 · Companion to `2026-07-09-tactical-commentary-v2-design.md`
(shares relations.py, radar renderer, concepts glossary, Pass-2 verifier).
Status: solo lead (Frank), recruiting; this spec doubles as the recruiting doc.

## What it is

Three artifacts, one pipeline:

1. **SoccerTactics-Instruct (train set — the centerpiece).** SoccerChat-format
   `(clip, query, long free-form tactical response)` pairs about positional /
   tactical "why" (阵地战站位、跑位、反击结构). Target 20–40k pairs, zh+en.
   No existing dataset covers this: SoccerChat is event-level QA; SportR and
   GameSight released nothing.
2. **Eval benchmark.** Gold free-form track (300–500 expert-validated items,
   rubric + LLM-judge) + MCQ diagnostic track + a **claim-verifiability
   metric**: extract claims from any model's free-form output, resolve them
   mechanically against the game state, report evidence precision/recall.
3. **Reference model.** QLoRA SFT of Qwen2.5-VL-3B (7B stretch) via the
   published SoccerChat ms-swift recipe on a single 24–48GB GPU; later GRPO
   using MCQ-derived verifiable rewards.

## Method core: verified state-guided synthesis

Gold trajectories + relations.json + tactical glossary → LLM writes long
grounded explanations (answer-guided, GameSight-style: the geometric facts are
known before generation) → every factual claim is mechanically verified via
the Pass-2 query grammar → unverifiable claims stripped or sample dropped.
Novelty over SoccerChat: their GPT-4o synthesis had no grounding check; ours
is verified-synthetic by construction. Generation cost is low (doubao-pro or
local Qwen3-VL); human budget goes to gold-track validation only.

## Taxonomy (top-down, literature-complete)

Anchored in the coaching-standard **four moments of the game + set pieces**:

- In possession: build-up from back, third-man combination, switch of play,
  wing overload, half-space occupation, overlap, underlap, run in behind,
  cutback, one-two, width/depth stretching, decoy run (无球牵制),
  second-ball protection (第二落点保护).
- Out of possession: high press (+ triggers), mid block, low block,
  compactness, zonal vs man marking, defensive line management / offside trap.
- Attacking transition: counter-attack, direct runners.
- Defensive transition: counter-press (Gegenpressing), rest defense.
- Set pieces: corner schemes, free kicks, long throw-ins.

The taxonomy ships complete (coverage claim). Each category documents:
definition (zh/en), observability in ≤30s clips, geometric verifiability
(which relations quantities check it), and mined instance count. Data density
is **tiered and reported honestly** — sparse categories are eval-only, never
silently cut:

| Tier | Criteria | Train pairs/cat | Gold eval/cat |
|---|---|---|---|
| 1 (~8–12 cats: passes, overlap, run-in-behind, counter, press, switch, phases) | frequent + strongly verifiable | 1k–3k | 50–100 |
| 2 (~6–8 cats: line-break, overload, blocks, counter-press) | medium | 300–1k | 30–50 |
| 3 (offside trap, throw-in schemes, rare patterns) | rare or weakly verifiable | eval-only | as mined |

## Data sourcing

- **Seed (solo-feasible, on disk):** the 164 human-annotated SoccerNetGS
  sequences — gold trajectories, zero perception noise in answer keys.
- **Scale (post-recruit):** mine SoccerNet (500 matches, NDA) with
  **event-anchored windows**: pre-cut ~30s windows around SoccerNet-v2 event
  timestamps (corners/free-kicks come pre-labeled = free set-piece retrieval),
  run Stage-1 GSR only on windows — never on full 90-minute matches (the GPU
  bottleneck). relations.json signatures then act as a tactical search index
  (possession flip + fast centroid advance → counter candidates; outside-lane
  pass-and-move → overlap candidates). ~50–100 parsed matches suffice for
  tier-1 targets (5–15 usable instances/match, 1–3 QA pairs/instance).
- Optional: SoccerTrack v2 (full-pitch, all 22 visible) for whole-team
  defensive patterns broadcast view can't verify.

## Item formats

- **Free-form (primary):** short whys ("为什么7号此时拉边？") and long per-clip
  复盘 ("解释这次反击中各球员的跑位与空间创造"). Query templates + paraphrase
  + few-shot style exemplars to avoid monotone synthetic style; small
  human-written subset mixed in.
- **MCQ (diagnostic + reward):** run-type ID, space-attribution, open-pass
  option, phase, overload localization, line-break detection. Near-miss
  distractors generated from geometry (plausible unless the model checks the
  actual quantities). Machine-checkable → GRPO-ready.
- **Three input settings per item:** broadcast clip (NDA-referenced),
  radar render (distributable), relations text (distributable). Disentangles
  perception failure from reasoning failure — no existing sports benchmark
  offers this pairing.

## Distribution & licensing

Mirror SoccerChat's split: annotations/queries/responses MIT on HF; broadcast
clips referenced by SoccerNet ID for NDA holders; radar renders + relations
text fully public. Annotations must not enable video reconstruction.

## Validation protocol

- MCQ: 15% sample checked by Frank + lab mates; report agreement; κ ≥ 0.7
  gate per family (pilot 100 items first; redesign families that miss).
- Gold free-form: authored/curated in-house, then paid pass by 1–2
  football-qualified experts (rubric: correctness of claims, tactical
  soundness, completeness). Report inter-annotator agreement.
- Blind-LLM control (question-only, no visual/state) on all MCQ families to
  detect language shortcuts.

## Phases (with effort; solo-feasible flagged)

| Phase | Work | Who | Effort |
|---|---|---|---|
| P0 | Taxonomy draft + per-category verifiability spec | Frank (solo ✓) | 1–2 wk |
| P1 | Generation+verification loop on 164 GT seqs → first ~3–5k pairs | Frank (solo ✓; needs relations.py from v2 plan) | 2–3 wk |
| P2 | 10-match mining pilot (event-anchored windows + GSR) → density report per category | Frank (solo ✓, GPU-bound) | 2 wk |
| P3 | Scale mining to 50–100 matches; silver split | recruit #1 (pipeline eng) | 4–6 wk |
| P4 | Gold eval track authoring + expert validation | Frank + recruit #2 + 1–2 paid experts | 3–4 wk |
| P5 | Reference model QLoRA SFT + eval runs + GRPO pilot | recruit #2 or Frank | 2–3 wk |
| P6 | Paper + HF release | Frank | 3–4 wk |

P0–P2 are fully solo and produce the recruiting demo: a working generation
loop plus a density table proving the benchmark is buildable.

## Risks

- Rule-leakage: models learn generation templates, not video → human-paraphrase
  subset; held-out template types in test; blind-LLM control.
- Sparse categories embarrass the coverage claim → tiered density is reported
  as a finding, not hidden; eval-only tier-3.
- Synthetic style monotony → template diversity + style exemplars + human subset.
- GSR noise in silver split → quarantine: test set from gold states only.
- Expert recruiting slips → gold track is the only externally-blocked phase;
  everything else proceeds.
- Judge circularity (LLM judges LLM prose) → claim-verifiability metric is the
  primary number; judge scores are secondary, expert-anchored.

## Rejected alternatives

- MCQ-first framing (Frank's ultimate goal is free-form generation; MCQ demoted
  to diagnostic/reward infrastructure).
- Mining-gated taxonomy (chose literature-complete + honest density instead).
- Free-form-only without mechanical verification (unverifiable synthetic data
  is the known failure mode this design exists to avoid).
- Crowdsourced validation (tactical judgment unreliable from crowd workers).
- Full-match GSR parsing (compute-infeasible; event-anchored windows instead).

## Dependencies

From the v2 pipeline plan (shared components, build first): relations.py
(Tasks 2–4), radar renderer (Task 5), concepts.yaml (Task 6 — extend into the
full taxonomy), Pass-2 query grammar/resolver (Task 9).
