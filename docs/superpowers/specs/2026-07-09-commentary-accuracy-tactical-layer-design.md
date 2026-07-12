# Commentary Accuracy & Tactical Reasoning Layer — Design

Date: 2026-07-09 · Goal: public-facing demo (YouTube/Bilibili) first

## Problem (re-diagnosed)

The perceived "temporal video-text alignment" problem is not alignment.
Stage-2 timestamps are correct; the *content* described at those timestamps is
wrong. Two confirmed failure modes:

1. **Stage-2 action misclassification** — the doubao-lite VLM misnames actions
   from the closed 10-class menu when shown annotated frame bursts.
2. **Stage-4 hallucination** — the commentary LLM embellishes beyond
   events.json facts.
3. (Minor) Stage-5 TTS audio drifts slightly against video windows.

MatchTime is **not** the fix: it corrects noisy human-commentary timestamps in
SoccerNet *training data* via ASR+LLM preprocessing and contrastive alignment.
Our timestamps are machine-generated and already correct. Its only reusable
piece would be the contrastive aligner as an inference-time verifier — not
needed given the diagnosis.

## Decisions

### D1 — Stage-2 fix: classifier head on SoccerMaster encoder
Train a lightweight 10-class action head on the frozen SoccerMaster encoder
using public SoccerNet action-spotting labels (500 matches). The VLM is
demoted to verifier/attribution edge cases. Rationale: a specialist classifier
beats any generalist VLM on closed-menu action discrimination, fits a single
24–48GB GPU, and we already own the encoder (it was pretrained with event
classification as a downstream task).

### D2 — Stage-4 fix: fact-contract + verifier pass (no model swap)
No public commentary checkpoint is worth adopting: MatchVoice is anonymous
English captioning from raw video (can't consume events.json); GameSight has
no code; TimeSoccer release unverified. Keep an instruction-following LLM
(doubao pro tier or local Qwen3-8B) as narrator with a hard contract:
- every sentence cites `events_referenced`;
- a cheap post-hoc verifier checks each sentence against events.json/topo.json
  and strikes or rewrites unsupported claims.
Commentary = verbalizing verified facts, never watching video.

### D3 — Captioning→reasoning: tactical feature layer + entity RAG
No public KB answers positional "why" (阵地战站位/反击跑位). Tactical
explanation is derived from geometry we already compute, not retrieved from
text. Two independent modules:
- **Tactical feature layer**: from GSR/topo state compute pitch control,
  numerical superiority/overloads, run classification (overlap/underlap/
  third-man), line-breaking passes, space creation. Open references:
  Metrica sample + LaurieOnTracking, OpenSTARLab. Output = structured
  tactical facts fed into the same fact-contract narrator.
- **Entity RAG**: SoccerWiki (public on HF, 9,471 players/266 teams) + GOAL
  triples for player/team background color only.

### D4 — Stage-5: TTS pacing fix only
Duration-aware pacing (compress/pad speech to segment windows) in
pace_filter/mux. No alignment model.

## Rejected alternatives
- Importing MatchTime as alignment correction (wrong problem).
- Swapping Stage-2 VLM for a bigger generalist VLM as primary fix (wrong tool;
  kept only as optional cross-check).
- Giving the narrator raw video to "self-ground" (reintroduces hallucination).
- Adopting SoccerAgent wholesale (built for QA benchmarks, heavy integration).
- LLM reasoning over raw coordinates (unverifiable guesses).

## Priority order (demo-driven)
1. Stage-4 fact-contract + verifier (days; biggest visible-lie reduction).
2. Stage-2 SoccerMaster classifier head (1–2 weeks; needs SoccerNet label ETL).
3. Stage-5 TTS pacing.
4. Tactical feature layer (start with 2–3 cheap features: numerical
   superiority, overlap runs, line-breaking passes), then entity RAG.

## Risks
- SoccerNet action labels are broadcast-view; our candidate windows come from
  our own detector — need window-label matching tolerance during training.
- Verifier pass adds latency/cost per clip (acceptable for offline demo).
- Tactical features on noisy GSR positions can misfire; gate each feature on
  calibration/tracking confidence.
- Chinese commentary quality: SoccerNet-derived training data is English;
  narration stays LLM-generated zh, so style depends on prompt + doubao/Qwen.

## Non-goals (for now)
- Counterfactual/causal explanation, coach-level evaluation benchmark,
  real-time streaming.
