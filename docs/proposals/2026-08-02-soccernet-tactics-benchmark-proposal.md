# SoccerNet-Tactics: A Benchmark for Tactic Recognition and Rule-Aware Reasoning in Broadcast Soccer Video

*Research proposal draft — Guoqing Luo, 2026-08-02. Prepared for MSRA internship discussion.*

## 1. Motivation and gap

Multimodal models can now answer factoid questions about soccer video with high accuracy — the CVPR 2026 SoccerNet VQA challenge was won at 97.6% with a task-routing + structured-evidence + single-judge pipeline. But that result covers objective questions only; the winning team themselves note that subjective, open-ended understanding (e.g., ranking event importance) remains out of scope. Meanwhile, the tactical-understanding literature approaches the problem from trajectories rather than video: GenTac/TacBench grounds 15 tactical event types in professional tracking data; TacticGen and TacEleven generate tactics conditioned on trajectories and language; GameSight studies commentary via entity alignment and knowledge refinement. **No existing benchmark asks whether a model can watch broadcast video and recognize a tactic — and justify that recognition against the tactic's defining criteria.**

This matters because tactic labels are constituted by rules, not by surface appearance. A low block and a mid block can look similar in a single frame; they are distinguished by defensive-line height, pressing onset zone, and duration. A model that names tactics without grasping these criteria will fail exactly where it matters — on confusable pairs and near-miss negatives. High-quality tactical annotation is scarce and expensive, which is precisely why the benchmark's evaluation framework must extract maximal signal per annotated item.

## 2. Core assets already built

- **Tactical knowledge base (KB)**: 46 entries compiled from FIFA Training Centre materials and tactical-analysis literature; the 19 P0 entries each carry: a one-sentence definition, 3–6 operationalized observable features (geometric/temporal/numeric — e.g., "defensive line ≤30m from own goal, ≥9 players in own half, sustained ≥60s"), 2–3 confusable concepts with one-sentence visual discriminators, positive exemplars with evidence time spans, and hard negatives with "why it doesn't count" rationales.
- **Pilot benchmark**: a 67-clip two-phase open-nomination experiment (Gemini/Doubao, 19 concept cards, 55 ground-truth claims, Top-1 claim-level accuracy protocol) — currently running.
- **Hybrid architecture**: a propose → rule-verify → realize pipeline in which VLMs nominate tactic candidates, deterministic recipes over game-state-reconstruction (GSR) outputs verify them, and claims are graded by level (observation / effect / intention).
- **GSR pipeline**: detection, tracking, camera calibration, team/role assignment, and pitch-coordinate projection on broadcast video.

## 3. Benchmark design

### 3.1 Task suite (19 P0 tactics)

- **T1 — Tactic recognition.** Clip → tactic label(s); multi-label; distractors are drawn from each tactic's KB confusable set (mid-block vs low-block, gegenpress vs high-press), not random labels.
- **T2 — Evidence grounding.** Given a tactic claim, localize the supporting time span (optionally actors). Separates "guessed the label" from "saw the mechanism"; the anti-shortcut task.
- **T3 — Rule-aware verification QA.** Given clip + claim, output a verdict plus a free-text justification grounded in the tactic's defining criteria (e.g., "not a low block: the defensive line sits at ~45m and pressing begins at the halfway line — this is a mid block"). Built directly from KB hard negatives and discriminator columns.

Full commentary generation is deliberately excluded from the benchmark core (annotation and evaluation cost); its evaluation problem is addressed through T3, below.

### 3.2 Rubric-anchored evaluation of open-ended output

T3 justifications are open-ended text — exactly the setting where *Reinforcement Learning with Rubric Anchors* (arXiv:2508.12790) replaces binary verifiable rewards with multi-dimensional, interpretable rubrics. We instantiate this per tactic, deriving each rubric anchor from a KB field:

1. correct mechanism named (from observable features);
2. correct discrimination against the relevant confusable (from discriminator columns);
3. evidence span cited and consistent with T2;
4. no overclaim beyond the verified claim level.

Because every rubric item maps to an auditable KB field, judge scores are traceable rather than holistic vibes. Judge reliability is measured, not assumed (§3.5).

### 3.3 Data

- **Primary**: SoccerNet broadcast video (public, redistributable, GSR-ready; grounds the benchmark name and a potential future challenge track).
- **Additional public sources**: SoccerReplay-1988 and X-VARS video (X-VARS used as footage for defensive/foul-adjacent scenarios only — its referee-decision task is out of scope; rule scope here is *tactical* rules, not laws of the game).
- **Where accessible**: tracking-aligned video (SkillCorner-class) for a high-precision machine-verified slice.
- FIFA Game Library clips remain the KB/exemplar layer only and never enter test data (licensing).

### 3.4 Ground-truth construction: propose → rule-verify → adjudicate

1. **Propose**: frontier VLMs nominate candidate (tactic, window, actors) triples via the two-phase open-nomination protocol (observation-first, then card-matched nomination).
2. **Rule-verify (tiered)**: where GSR confidence is high, deterministic recipes check the KB's geometric/temporal criteria on pitch coordinates (tier A: machine-verified); otherwise trained adjudicators check the KB observable-feature checklist visually (tier B: human-verified); unverifiable claims are excluded. Tier metadata ships with every claim, and the tier distribution is itself a reported finding.
3. **Adjudicate**: humans resolve disagreements against KB concept cards. A **human-only sampled slice** (annotated without model proposals) measures the recall bias of the propose stage — tactics VLMs cannot see must not silently vanish from the label distribution.

**Scale**: ~1–2k verified claims (~50–100 per tactic, positives + hard negatives), match-disjoint splits. **Annotation resources**: author + recruited football-literate students with measured inter-annotator agreement; MSRA annotation budget if available; crowdworkers gated by a KB-confusable-pair qualification quiz for scale-out.

### 3.5 Evaluation protocol

- **Frontier VLM sweep** (Gemini, GPT-class, Claude, Qwen-VL, Doubao) on T1–T3: Top-1 claim accuracy, evidence-window overlap, rubric-scored justification.
- **Anti-shortcut diagnostics**: confusable-pair confusion matrix, hard-negative false-positive rate, and "right label, wrong evidence" rate (T1 correct ∧ T2 wrong).
- **Human expert ceiling**, doubling as the **judge-reliability audit**: on the same audited slice, report human accuracy on T1/T3, judge–human agreement, and inter-judge agreement — pre-empting the standard LLM-judge-bias objection and producing the calibration data the RL phase needs.

## 4. Companion method (staged)

- **Phase 1 (in the benchmark paper)**: the hybrid propose → rule-verify → realize pipeline as a strong training-free baseline, testing whether structured rule verification beats end-to-end VLM prompting on T1–T3.
- **Phase 2 (after benchmark freeze)**: RL on an open VLM for T3 with a **hybrid reward** — verifiable verdict correctness (classic RLVR) plus rubric-anchored justification quality (the 2508.12790 extension). Reward-hacking guards: a held-out rubric split disjoint from training rewards, and human spot-audits on the eval split.

## 5. Milestones (~6 months, default allocation)

| Phase | Deliverable |
|---|---|
| M1 | Finish 67-clip pilot; use its confusion/failure data to freeze task formats and per-tactic rubrics |
| M2–M3 | Scale propose–verify–adjudicate annotation to ~1–2k claims on SoccerNet/SoccerReplay; freeze benchmark; run VLM sweep + human study |
| M4 | Benchmark paper draft |
| M5–M6 | Rubric-anchored RL against the frozen benchmark |

## 6. Risks and mitigations

- **Proposal recall bias** (model-nominated candidates skew the label pool) → human-only slice quantifies it; reported, not hidden.
- **GSR noise on broadcast** → tiered verification keeps the benchmark honest; pilot measures tier-A share early, before the design commits to it.
- **Judge bias / reward hacking** → auditable KB-derived rubrics, judge-reliability reporting, held-out rubric split, human audits.
- **Annotation throughput** (~70–170h of football-literate work) → three-source staffing plan; scale target adjustable to ~1k without changing the design.
- **Licensing** → SoccerReplay-1988 / X-VARS redistribution terms verified before any clip ships; fall back to ID/timestamp release.
