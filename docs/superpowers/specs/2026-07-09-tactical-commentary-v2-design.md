# Tactical Commentary v2 — State-to-Reasoning Design

Date: 2026-07-09 · Supersedes the Stage-2/Stage-4 sections of
`2026-07-09-commentary-accuracy-tactical-layer-design.md`.
Goal: public demo (YouTube/Bilibili); anonymous SNGS-style 30s clips.

## Motivating evidence

Frank's A/B test: doubao generating commentary directly from SoccerMaster
outputs (players, jerseys, pitch top-view, timestamps) beat the
rule-engine→typed-events→doubao path. Conclusion: typed event classification
by rules/VLM is the error-accumulation point; the LLM should do all semantics.

Literature backing the division of labor (arithmetic in code, semantics in LLM):
- TacticAI (Nature Comms 2024): relational geometry in, tactical insight out;
  experts preferred it 90% of cases.
- SportsMetrics (ACL 2024): LLMs degrade fusing long raw numeric play-by-play
  → never dump raw coordinates.
- "Automated explanation of ML models of footballing actions in words"
  (arXiv 2504.00767): numbers pre-computed, LLM verbalizes → reliable.
- GameSight (arXiv 2604.00057): two-stage; FLARE-style self-ask + LLM
  double-check gives +16.1% knowledge accuracy over plain RAG.

## Architecture

```
predictions.json (bbox_pitch, track/role/team/jersey, ball 724/750 frames)
        │
relations.py  ──►  relations.json      (measurements only, no classification)
topology/     ──►  topo.json           (existing team-shape metrics)
minimap render ──► radar frames @1-2Hz (existing minimap.py assets)
        │
Pass 1: NARRATE   doubao( radar frames + relations table + tactical glossary )
                  → draft commentary, every tactical claim cites evidence rows
Pass 2: REFINE    doubao self-asks about each claim → answers checked against
                  relations.json (time-constrained query schema) → strike or
                  quote → final commentary.json
        │
Stage 5 TTS (unchanged; pacing fix separate)
```

Retired: `classify.py` VLM action naming, typed event menu as commentary
input. `detector.py` signals remain available as optional attention anchors
but the LLM finds and names moments itself.

## relations.py (measurements only)

Input: predictions.json (`bbox_pitch` is already in pitch meters; homography
only reused to project the camera frustum → per-window visible-region mask).
Windows 2–5s, smoothing ~0.5s, players qualify only if tracked ≥80% of window.

- Per player: position, velocity vector, speed, sprint flag, displacement,
  direction vs attacking direction.
- Ball: position, speed, direction; carrier via nearest-teammate + hysteresis
  (reuse possession.py); kick anchors = ball-speed spikes (reuse detector.py).
- Player↔carrier: distance, angle, closing speed, lane relative to carrier
  (inside/outside), moved-past-carrier flag.
- Player↔opponents: nearest-defender distance, depth vs second-last defender.
- Team: defensive-line height, block width/depth/hull, third/lane counts,
  side overload, centroid velocity (reuse topology/), attackers-vs-defenders
  within radius of ball (local numerical superiority).
- Meta: visible-region mask per window; "not observed" regions declared so the
  LLM must not claim anything about them.

Output relations.json: per-window snapshot rows (~2Hz) + window aggregates.
No labels like "overlap" or "counter" — those are LLM semantics.

## Tactical concept glossary (the KB)

`pipeline/tactics/concepts.yaml`, ~30 entries v1 (target 50–150):
`id, name_zh, name_en, definition_zh, evidence_requirements (which relation
quantities must support it), exemplar_zh, exemplar_en`.
Sources for extraction (hand-curated, paraphrased): Wikipedia zh/en tactics &
formation articles (CC BY-SA); Devin Pleuler's Soccer Analytics Handbook
(open GitHub); StatsBomb open event spec as term taxonomy; formulas from
LaurieOnTracking / OpenSTARLab. No scraping of copyrighted tactical blogs.

Injection: wholesale into Pass-1 prompt (~3k tokens at 30 entries). This
matches how papers treat closed vocabularies; query-based retrieval
(GameSight-style) is reserved for large entity/stats KBs (SoccerWiki) when
real fixtures + roster_json arrive later. Rule: the LLM may only name
concepts present in the glossary and must cite evidence rows.

## Two-pass generation (paper-faithful)

- Pass 1 NARRATE: radar frame sequence (1–2Hz) + relations table + glossary +
  style rules → timestamped zh/en commentary JSON (existing schema), each
  tactical claim carrying `evidence` refs into relations.json.
- Pass 2 REFINE (GameSight/FLARE style): LLM generates explicit questions per
  claim ("was #7 outside the carrier's lane at t=6–8s?"), answers resolved
  mechanically against relations.json with time-range constraints, invalid
  claims struck/rewritten, verified knowledge quoted. Output final
  commentary.json. ~2x LLM cost, acceptable offline.

## Rejected alternatives

- Rule/VLM typed event classification as commentary input (A/B loss;
  error accumulation).
- Code-side tactical classification (overlap/counter labels in rules) —
  moved to LLM semantics per Frank's error-accumulation concern.
- Raw video frames as primary LLM input (re-introduces perception errors GSR
  solved); raw coordinate dumps (SportsMetrics failure mode).
- Embedding RAG over the concept glossary (opaque, unnecessary ≤150 entries).
- Entity RAG (SoccerWiki) for anonymous clips — retrieves nothing; deferred
  until real fixtures.
- SoccerRAG (simula/soccer-rag) — SQL retrieval over SoccerNet entity/stat
  facts ("Messi goals per season"); superseded by roster_json mapping in our
  own code. Skipped (2026-07-09 review).
- SoccerAgent as a system — orchestrator + camera tool are API-bound (GPT-4o),
  built for SoccerBench MCQ, heavy integration; not adopted. Its SoccerWiki
  database (open on HF: player/referee/team/venue profiles) is harvested as
  *data* for entity color once real fixtures + rosters arrive — it contains
  no tactical knowledge. 战术槽位 (压迫方式/弱侧调动/套边/无球牵制/第二落点
  保护/边中切换) exist in no public resource; they come from relations.py +
  glossary (decoy_run and second_ball_protection added to cover the list).

## Risks & assumptions

- Doubao-lite must handle ~30–60 radar images + tables + glossary in context;
  if not, upgrade tier or window the clip.
- Minimap rendering quality matters: radar frames must encode team colors,
  jersey numbers, ball clearly at small size.
- Pass-2 mechanical answering needs a small query grammar; keep it to
  time-ranged lookups over relations.json.
- GSR noise (jersey flicker, track breaks) still leaks through; the ≥80%
  visibility gate and visible-region mask are the guards.
- Zh commentary style consistency depends on exemplars in glossary entries.

## Testing

- Unit: synthetic trajectories → expected relation quantities (velocities,
  lanes, line heights).
- Golden: SNGS-116 / SNGS-117-2b relations.json snapshots reviewed once,
  regression-diffed after.
- Prompt-level: no concept outside glossary appears in output (string match);
  every tactical sentence has resolvable evidence refs; Pass-2 strike rate
  logged per clip as a quality metric.
- End-to-end: A/B against the current variant-B output (human judgment,
  same clips), since that is the incumbent best.

## Build order

1. relations.py (consolidate topology/ + possession.py + new per-player
   relational features) + relations.json schema.
2. Radar-frame export at 1–2Hz (reuse minimap assets).
3. concepts.yaml core-30 + Pass-1 prompt builder rewrite.
4. Pass-2 self-ask refine loop.
5. A/B eval vs variant B; then TTS pacing fix.
