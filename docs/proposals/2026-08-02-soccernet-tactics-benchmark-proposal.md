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

### 3.4 Ground-truth construction: dual blind annotation, pooled candidates, criterion-cited arbitration

A pilot annotation pass motivates this design and illustrates the core difficulty. Over 67 clips it produced 84 claims, but 55 of those clips carry exactly one claim, only 14 of the 19 P0 tactics appear at all, and four tactics account for 77% of the claims. A 10 to 30 second soccer clip almost always contains several concurrent tactical patterns, so this distribution describes the annotation process rather than the game: free-recall annotation is dominated by visually salient tactics such as counter-attacks, while quieter patterns like half-space penetration go unrecorded. Labels of this shape can measure precision but not recall, and cannot support multi-label recognition at all. Humans and models fail in opposite directions here. Human annotators rarely assert a tactic that is not there, but they omit; VLMs rarely omit, but they confuse similar tactics. The protocol is built around that asymmetry.

1. **Human pass (checklist format, blind to model output).** Annotators judge every clip against all 19 P0 concept cards as an explicit checklist, marking each present, absent, or not observable. This converts free recall into recognition, which attacks the omission problem without any model involvement. Annotators mark an anchor timestamp where the tactic is clearest rather than precise boundaries, since human boundary agreement is poor and demanding it early manufactures spurious disagreement.
2. **Model pass (pooled, blind to human labels).** Two or more VLMs nominate candidates under the open-nomination protocol, tuned toward recall. Pooling two models beats one on coverage, and their agreement structure is a free difficulty signal: candidates both models propose are easy items, single-model candidates are hard ones.
3. **Merge.** The union is partitioned into agreement, human-only, model-only, and same-tactic-different-window.
4. **Arbitration with cited criteria.** Adjudicators resolve disagreements and set boundaries. Every accepted claim must cite which KB observable-feature it satisfies; a bare accept/reject verdict is not sufficient, because it cannot distinguish an annotator being reminded of a real tactic from being talked into a spurious one. The cited criteria also seed T3 justification data.

**Bias controls.** Ground truth pooled from human and model candidates advantages models that contributed to the pool, the pooling bias familiar from information retrieval evaluation. One model is therefore held out of the pool entirely, and its performance gap against a pooled model bounds the effect. Separately, 10 to 20% of clips are annotated independently by two humans to estimate inter-annotator agreement and the human omission rate.

**Scale**: ~1–2k verified claims (~50–100 per tactic, positives + hard negatives), match-disjoint splits. **Annotation resources**: author + recruited football-literate students with measured inter-annotator agreement; MSRA annotation budget if available; crowdworkers gated by a KB-confusable-pair qualification quiz for scale-out.

### 3.5 Evaluation protocol

- **Frontier VLM sweep** (Gemini, GPT-class, Claude, Qwen-VL, Doubao) on T1–T3: Top-1 claim accuracy, evidence-window overlap, rubric-scored justification.
- **Anti-shortcut diagnostics**: confusable-pair confusion matrix, hard-negative false-positive rate, and "right label, wrong evidence" rate (T1 correct ∧ T2 wrong).
- **Human expert ceiling**, which falls out of the §3.4 human pass rather than requiring separate annotation: it directly yields human accuracy on T1 and per-tactic human-versus-model recall, a finding in its own right. The same audited slice supplies the **judge-reliability check** (judge-human agreement and inter-judge agreement), pre-empting the standard LLM-judge-bias objection and producing the calibration data the RL phase needs.

## 4. Companion method (staged)

- **Phase 1 (in the benchmark paper)**: the hybrid propose → rule-verify → realize pipeline as a strong training-free baseline, testing whether structured rule verification beats end-to-end VLM prompting on T1–T3.
- **Phase 2 (after benchmark freeze)**: RL on an open VLM for T3 with a **hybrid reward** — verifiable verdict correctness (classic RLVR) plus rubric-anchored justification quality (the 2508.12790 extension). Reward-hacking guards: a held-out rubric split disjoint from training rewards, and human spot-audits on the eval split.

## 5. Milestones (~6 months, default allocation)

| Phase | Deliverable |
|---|---|
| M1 | Finish 67-clip pilot; validate checklist-format annotation against the current free-recall labels; freeze task formats and per-tactic rubrics |
| M2–M3 | Scale propose–verify–adjudicate annotation to ~1–2k claims on SoccerNet/SoccerReplay; freeze benchmark; run VLM sweep + human study |
| M4 | Benchmark paper draft |
| M5–M6 | Rubric-anchored RL against the frozen benchmark |

## 6. Risks and mitigations

- **Pooling bias** (models contributing to the candidate pool are advantaged, and tactics neither humans nor models spot stay invisible) → held-out model bounds the effect; reported, not hidden.
- **Checklist assumption**: the recall argument rests on checklist annotation actually outperforming free recall. This is measurable before scaling, by re-annotating a sample of the 67 pilot clips in checklist format and comparing claim density against the current 84.
- **GSR noise on broadcast** → tiered verification keeps the benchmark honest; pilot measures tier-A share early, before the design commits to it.
- **Judge bias / reward hacking** → auditable KB-derived rubrics, judge-reliability reporting, held-out rubric split, human audits.
- **Annotation throughput** (~70–170h of football-literate work) → three-source staffing plan; scale target adjustable to ~1k without changing the design.
- **Licensing** → SoccerReplay-1988 / X-VARS redistribution terms verified before any clip ships; fall back to ID/timestamp release.
