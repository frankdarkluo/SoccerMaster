# System-Arm Evaluation & Expert Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
> **Commit policy (AGENTS.md override):** do NOT commit per task. One commit at the very end (Task 8), or when the user explicitly says OK. Follow karpathy-guidelines + ponytail on every code change: shortest working diff, no speculative abstractions.

**Goal:** Run the low-cost system arm (doubao-2-lite + gemini-3.1-flash-lite nomination → auto-evidence checkers → A′ three-tier verdicts → bounded Sol arbitration) on the dev-72 clips for debugging, then on a frozen 43-clip test set, and produce an expert review package ready to send Sunday evening 2026-07-20. Everything here is self-serve; the only external dependency (expert scoring) is isolated in Task 7's ingestion script, which just waits for their CSV.

**Settled decisions (do not reopen):**
- A′ asymmetric verification: checker on trajectory evidence pass → `confirmed`; checker on VLM-claimed facts pass → at most `possible`; **veto from either evidence source stands**; no usable evidence → `insufficient`; plus `unobservable`. VLM/Sol output can never mint `confirmed`.
- Models: doubao-2-lite + gemini-3.1-flash-lite nominate. Sol (chatgpt) is retained but bounded: arbitrates only structured disputes the checkers cannot resolve, chooses from fixed options, and its verdict is capped at `possible`.
- Scope: P0 tactics only — `fast_break_pattern`, `run_in_behind`, `corner_near_far_post`, `cutback`. Others: never nominated to experts (drop silently, log count).
- Two-stage eval: dev-72 = debug (threshold changes allowed ONLY here); test set = frozen (any change after freeze restarts Task 6).
- Acceptance targets on test set (measured after expert scoring returns): P0 `confirmed` precision ≥ 90%; coverage (clips with ≥1 confirmed/possible claim) ≥ 50%; abstentions spot-checked as justified.
- Evidence honesty: infer-or-abstain everywhere; never lower thresholds to raise coverage.

**Test set composition:** the 15 `event_clips` (held-out test group `source_video_sha256:ff8a…` in `evaluation_report.json`) + 28 challenge-split SNGS sequences. The repository has 36 challenge directories, not the previously estimated 38; excluding the 8 already reviewed in `data/足球战术识别.csv` gives exactly 28, so the frozen total is 43. All are no-GT / production-form. Timeline: Tasks 1–3 by Friday, Task 4 (GPU batch) runs overnight, Task 5 Saturday, Task 6 Sunday, package out Sunday evening.

---

### Task 1: Leakage fix + same-source retrieval exclusion

**Files:** modify `benchmark/tactical_prototypes/episodes.jsonl`; modify the migration function in `scripts/build_recognition_assets.py`; modify the prototype-retrieval code (server-side — locate it as the code that produced `agent_runs/*.json` `prototype_context`; search for readers of `episodes.jsonl` outside `pipeline/tactics_qa/`).

- [ ] Remove (delete the lines) the two episodes with `provenance.migration_version == "recognition-review-v1"` and `clip_uid` in `{"13", "14"}` — they come from the held-out test video (`event_clips:0013/0014`) and leak test answers into retrieval. Verify with:
  `python3 -c "import json;e=[json.loads(l) for l in open('benchmark/tactical_prototypes/episodes.jsonl')];print([x['clip_uid'] for x in e if x.get('provenance',{}).get('migration_version')=='recognition-review-v1' and x['clip_uid'] in {'13','14'}])"` → expect `[]` after the edit.
- [ ] In `migrate_hard_negatives`, before appending, resolve the row's source group via `benchmark/tactical_prototypes/source_groups.json` (event-clip rows: uid `event_clips:{clip_id:0>4}`) and skip + warn when the group's assignment in `evaluation_report.json` is `test`. Add a unit test with a fake group map asserting a test-group row is skipped.
- [ ] In the retrieval code: when selecting prototype episodes for a query clip, exclude every episode whose source group equals the query clip's group (challenge clips: group `soccernet_challenge:opaque` — the whole opaque group is excluded for challenge queries). Log how many episodes were excluded per query. Add a test if the retrieval code is testable; otherwise verify by running one challenge clip and checking the log.

### Task 2: A′ three-tier verdict layer

**Files:** create `pipeline/tactics_qa/verdict.py` (~60 lines); test `tests/tactics_qa/test_verdict.py`. This is the only new module in the plan.

- [ ] TDD a single pure function:

```python
def decide(claim_tactic_id, checker_results_trajectory, checker_results_vlm,
           observability_ok=True) -> Verdict
# Verdict = dataclass(tier, reason)  tier ∈ {"confirmed","possible","insufficient","rejected","unobservable"}
```

Rules, in order: not `observability_ok` → `unobservable`; any veto in either result list → `rejected` (reason = first veto); trajectory results exist and all pass → `confirmed`; vlm results exist and all pass → `possible`; otherwise → `insufficient`. Test each branch plus the asymmetry case (trajectory insufficient + vlm all-pass → `possible`, never `confirmed`).

### Task 3: VLM claimed-facts schema + bounded Sol

**Files:** modify the agent-runner prompt/schema (server-side, the code producing `agent_runs/*.json` observations); modify `pipeline/tactics_qa/evidence.py` only if a new `source` string needs documenting (it's free-text — just use `"vlm_claim"`); small parser in the runner.

- [ ] Extend each nominated candidate's JSON with a `claimed_evidence` block using the exact enums from `pipeline/tactics_qa/evidence.py`: `{"restart_at_origin": <RESTART_KINDS|null>, "regain": <CONTROLLED_REGAIN_KINDS|null>, "chaos": bool|null, "delivery_kind": <DELIVERY_KINDS|null>, "corner_landing_zone": <LANDING_ZONES|null>}`. Prompt instruction: fill a field only if directly observed; `null` otherwise. Parser converts this to `ClipEvidence(source="vlm_claim")` (map `restart_at_origin`/`regain` to `PossessionEvent`s at the claim window start; invalid enum values → treat as null, log).
- [ ] Sol path: trigger only when doubao and gemini disagree on tactic or decision AND the trajectory checker verdict is `insufficient`. Sol receives both candidates' `claimed_evidence`, checker results, and retrieved prototypes; must answer with one of `{support_A, support_B, neither, unobservable}` + the decisive condition. Its output feeds `decide()` as vlm-side evidence only (tier cap `possible` falls out of Task 2 automatically — no special code).
- [ ] Wire per-claim flow in the runner: nomination → auto evidence (`build_claim_evidence` from `outputs/<clip>/predictions.json` when present) → trajectory checkers → vlm checkers on `claimed_evidence` → optional Sol → `decide()`. Persist one JSON per clip in the existing `agent_runs/` schema, adding `verdict` per candidate.

### Task 4: Stage 1 batch over the test set (GPU, overnight)

**Files:** create nothing new except a clip list; drive `scripts/run_stage1.sh` in a shell loop.

- [ ] Build the clip list: 15 event-clip mp4s (the test-group video's clips — 4 are in `data/reviewed_tactics_videos/`, the rest in the event-clips dataset directory on the server per `source_rows.jsonl` `video_path`) + the 36 challenge-split dirs `codes/sn-gamestate/datasets/SoccerNetGS/challenge/SNGS-*` minus the 8 reviewed ids (derive from `data/足球战术识别.csv` challenge rows), yielding 28 challenge clips and 43 clips total.
- [ ] Loop `run_stage1.sh <clip> outputs/<id>` with `GSR_NUM_CORES/GSR_NUM_THREADS` at the repo defaults; skip when `outputs/<id>/predictions.json` already exists (idempotent re-runs). Log failures, don't abort the batch. Record the final coverage count — clips without predictions still go through the system (their claims simply cap at `possible`; that is the A′ point, not a blocker).

### Task 5: Debug run on dev-72

- [ ] Run the full system arm on the dev-72 clips (retrieval exclusion from Task 1 active; auto evidence from existing outputs where available). Produce one summary table (stdout or appended to `benchmark/tactical_prototypes/replay_report.md`, no new file): per-P0-tactic claims, tier distribution, gate-on vs gate-off precision against the existing human verdicts for claims matching xhigh's (same clip + tactic), plus the unmatched-claim count (needs fresh review — goes to experts too if capacity allows).
- [ ] Record nomination recall first: of the 72 human-reviewed dev claims, how many are re-nominated by the lite-model input path. If recall is clearly low, keep v1 frozen and report it; adaptive sampling around trajectory-detected kicks/possession changes is the next experiment, not part of this run.
- [ ] Compare against expectations: gate-on precision should approach the 91.7%-ceiling direction; zero confirmed-tier claims should contradict a human "错误" verdict whose root cause is `restart_delivery`/`chaotic_chain`/`geometry`. Each contradiction = a bug in evidence or wiring; fix before proceeding. Threshold adjustments are allowed here and must be listed in the final commit message.

### Task 6: Frozen test run + expert package (complete by Sunday 2026-07-20 evening)

- [ ] Freeze all thresholds/prompts (note the git diff hash in the run log). Run the system arm once over the test set.
- [ ] Emit the expert package as ONE file, `benchmark/tactical_prototypes/expert_review_batch1.csv` (UTF-8 with BOM so Excel opens it), one row per claim, columns exactly:
  `claim_id, clip_id, video_path, tactic_zh, tactic_id, window, system_tier, evidence_notes, 判断依据, 专家判定(正确/错误/窗口部分/无法判断), 更正说明`
  — the last two columns empty for experts to fill; `判断依据` is the system's own positive evidence text (so experts judge the claim as stated, mirroring the 足球战术识别.csv workflow they'll recognize). Include every confirmed/possible claim; include a 20% random sample of insufficient/rejected claims (labeled as such) so abstention quality gets audited too. Ship alongside the existing `scoring_rubric.md` (no new doc — tell the user to attach it).
- [ ] Also emit `benchmark/tactical_prototypes/expert_review_batch1.answersheet.jsonl` (machine copy of the same rows + full checker traces) for Task 7's join. These two files are the plan's only new artifacts besides `verdict.py`.

### Task 7: Expert-result ingestion (build now, run when their CSV returns)

- [ ] Extend `scripts/build_recognition_assets.py` with `--ingest-expert <csv>`: join on `claim_id` against the answersheet; compute and print the acceptance metrics (P0 confirmed precision, coverage, abstention audit results, per-tactic table); append expert-confirmed errors as gold hard negatives via `migrate_hard_negatives` — which now (Task 1) refuses test-group rows, so **it will rightly skip these**; instead write them to a `review_status: "quarantined_test_source"` line so they're preserved but never retrieved. Unit-test the join + one quarantine case with a 3-row fixture.

### Task 8: Cleanup + single commit

- [ ] Delete root `reference_football.csv` (md5-identical duplicate of `data/reference_football.csv`).
- [ ] Delete the two executed plan docs from July 17 (triage, auto-evidence) — git history preserves them. Keep this plan until executed, then it dies the same way.
- [ ] Add to `.gitignore`: `benchmark/tactical_prototypes/replay_report.json`, `evidence_agreement_report.json`, `recognition_evidence_auto.jsonl` (regenerable; keep the `.md` reports and the hand-encoded `recognition_evidence.jsonl` tracked). `git rm --cached` the three.
- [ ] Keep: `sol_review_bundle/` (Sol retained), `agent_runs/`, KB v2 handoff doc.
- [ ] Run `python3 -m pytest tests/tactics_qa/ -v` (all green) and one end-to-end system run on SNGS-116 as smoke. Then ONE commit of everything, message listing: leak fix, A′ layer, Sol bounding, test-set freeze hash, any dev-72 threshold changes.

**Non-goals:** no work on deferred tactics; no new evidence heuristics beyond what exists; no committing generated reports; no expert-facing docs beyond the CSV + existing rubric; no waiting on expert results — the plan ends with the package sent and the ingestion script ready.
