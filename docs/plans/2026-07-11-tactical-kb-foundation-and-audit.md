# Tactical KB v2 Foundation and Feasibility Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every code task and `superpowers:verification-before-completion` before handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the safe, disabled-by-default KB v2 foundation, repair and validate SoccerNetGS train/valid, close the model-self-verification loophole, and complete the multi-match feasibility audit that identifies every feasible candidate plus a zero-to-three-concept recipe shortlist for a later production-promotion plan.

**Architecture:** Keep `codes/sn-gamestate/datasets/SoccerNetGS` as the only canonical dataset root. Stage 2B converts 25 Hz GT or GSR measurements into a concept-neutral `tactical_state`, gives Doubao only a whitelisted public catalog and compact proposal projection, validates proposals against a strict schema, and reconstructs facts only through named code checkers. This plan deliberately ships no production recipe: all 14 concepts remain disabled until repaired train/valid data supports the audit. The audit result, not this plan, determines which zero to three checker implementations belong in the follow-up promotion plan.

**Tech Stack:** Python 3.10+, standard library, PyYAML, pytest, existing SoccerNetGS/Stage 1 data and existing ARK client. No new runtime dependency, schema framework, database, or rule DSL.

## Plan boundary

```text
read-only integrity validator
        ↓
staged train/valid repair + promotion checkpoint
        ↓
14-entry private catalog + public projection
        ↓
25 Hz concept-neutral tactical_state
        ↓
strict proposal contract + named-checker boundary
        ↓
Stage 2B facts-only commentary integration
        ↓
development annotation + feasibility audit
        ↓
0–6 feasibility-passed concepts; 0–3-concept recipe shortlist
        ↓
STOP — write the separate production-promotion plan
```

This is the complete implementation plan for the foundation and feasibility stage. It is intentionally not the production-recipe plan because recipe parameters, required relation primitives, and release gates cannot be specified honestly before the audit identifies the winning concepts.

## Global constraints

- Work in the current checkout and preserve all unrelated user changes. Never reset, restore, stash, or stage unrelated deleted/untracked docs.
- Do not modify or stage `deep-research-report.md`, `docs/papers/`, or `docs/superpowers/specs/2026-07-11-soccertactics-instruct-benchmark-addendum.md` as part of implementation unless the user separately asks.
- Remove `/tests/` from `.gitignore`; the user has explicitly recreated `tests/`. Use tracked pytest tests rather than embedded self-tests.
- Canonical data root is exactly `codes/sn-gamestate/datasets/SoccerNetGS`.
- Repair only `train/` and `valid/`. Never download over, move, relabel, or modify `test/`.
- Existing state is known: train has 57 sequence directories with 57 truncated 524,288-byte label files; valid has 58 empty sequence directories; test has 49 populated sequences.
- Use authoritative SoccerNetGS v1.3 data. The official manual instructions currently use downloader task `gamestate-2024`; do not reuse the local TrackLab wrapper's divergent `gamestate-2025` string without separately resolving that discrepancy.
- Game 7, including `SNGS-116` and `SNGS-117`, is diagnostic only. Games 8 and 11 remain unopened as held-out evaluation data until a release manifest is frozen in the follow-up plan.
- Catalog size is exactly 14 for KB v2. All entries start with `production_enabled: false`.
- Feasibility candidates are exactly: `overlap_run`, `run_in_behind`, `switch_of_play`, `local_numerical_superiority`, `compact_block`, and `line_break`.
- Production selection has a maximum of three and no minimum. Zero is a valid result.
- KB v2 permits only `observation` claims. Do not implement effect or intention claims.
- Preserve raw tracking at 25 Hz. Proposal projection may downsample to at most 12.5 Hz. Do not impose a global five-second evidence limit.
- Existing five-second windows may survive only as commentary scheduling slots. They must not constrain proposal or verified evidence windows.
- Do not implement a full possession engine, general temporal query DSL, checker class hierarchy, database, annotation UI, training pipeline, or all 14 detectors.
- Do not add a model or LLM critic to validate tactical correctness. Code checkers and human annotations are the evidence boundary.
- No test may require ARK, FFmpeg, GPU, network access, or a real SoccerNetGS clip. Use synthetic dictionaries and temporary directories.
- Data repair is an external-state checkpoint. Never promote staged data until the validator passes and the user explicitly approves the split-level move.
- Commit after each green task unless the execution request explicitly requires one final commit; in that case retain the checkpoints but defer commits.

## File map

**Create:**

- `scripts/validate_soccernetgs.py`
- `scripts/promote_soccernetgs_splits.py`
- `pipeline/stage2b/kb.py`
- `pipeline/relations/state.py`
- `pipeline/stage2b/tactics.py`
- `scripts/tactical_kb_eval.py`
- `docs/superpowers/references/2026-07-11-tactical-kb-source-audit.md`
- `tests/test_soccernetgs_integrity.py`
- `tests/test_soccernetgs_promotion.py`
- `tests/test_stage2b_kb.py`
- `tests/test_tactical_state.py`
- `tests/test_stage2b_tactics.py`
- `tests/test_stage2b_tactical_integration.py`
- `tests/test_tactical_kb_eval.py`
- `tests/fixtures/tactical_kb/*.json`, one file per feasibility-passed concept
- `benchmark/manifests/soccernetgs_v1_3_integrity.json` after successful repair
- `benchmark/development_clip_manifest.json` only after real development clips are shortlisted
- `benchmark/tactical_windows.jsonl` only after real human annotation exists
- `benchmark/feasibility_report.json` only after the development audit runs
- `benchmark/recipe_shortlist.json` only after the feasibility report is reviewed

**Modify:**

- `.gitignore`
- `pipeline/stage2b/concepts.yaml`
- `pipeline/config.py`
- `pipeline/stage2b/hybrid.py`
- `pipeline/stage2b/run.py`

**Do not modify in this plan:**

- `pipeline/relations/query.py`
- `pipeline/relations/snapshots.py`
- `pipeline/topology/`
- `scripts/run_stage1.sh`
- `scripts/run_stage2b.sh`
- any held-out data below `test/`

---

### Task 1: Track pytest and add a read-only SoccerNetGS integrity validator

**Files:**

- Modify: `.gitignore`
- Create: `scripts/validate_soccernetgs.py`
- Create: `tests/test_soccernetgs_integrity.py`

**Success criteria:** The validator detects truncated JSON, empty sequences, bad image references, missing frames, wrong dataset version, duplicate sequence IDs, and missing usable player pitch annotations without changing any dataset file. Missing ball coordinates remain a quality warning, not a corruption error.

- [ ] **Step 1: Stop ignoring the test directory**

Delete only this line from `.gitignore`:

```gitignore
/tests/
```

Do not reorder or clean unrelated ignore rules.

- [ ] **Step 2: Write the failing validator tests**

Create `tests/test_soccernetgs_integrity.py` with a helper that writes one tiny sequence containing two images, player pitch coordinates, and a ball annotation. Cover:

Import `validate_dataset` and `validate_sequence` from
`scripts.validate_soccernetgs`, then implement these seven tests:

- `test_valid_sequence_passes`
- `test_truncated_json_is_reported`
- `test_missing_frame_is_reported`
- `test_annotation_image_reference_is_checked`
- `test_dataset_detects_duplicate_sequence_id`
- `test_version_before_1_3_is_rejected`
- `test_missing_ball_is_a_quality_warning_not_an_integrity_failure`

The valid fixture must use the same essential shape as SoccerNetGS:

```python
{
    "info": {
        "version": "1.3", "game_id": "dev_game", "id": "9001",
        "name": "SNGS-9001", "frame_rate": 25, "seq_length": 2,
        "im_dir": "img1", "im_ext": ".jpg",
    },
    "images": [
        {"image_id": "1", "file_name": "000001.jpg"},
        {"image_id": "2", "file_name": "000002.jpg"},
    ],
    "annotations": [
        {
            "image_id": "1", "track_id": 10,
            "attributes": {"role": "player", "team": "left"},
            "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0},
        },
        {
            "image_id": "1", "track_id": 99,
            "attributes": {"role": "ball"},
            "bbox_pitch": {"x_bottom_middle": 1.5, "y_bottom_middle": 2.0},
        },
    ],
    "categories": [],
}
```

- [ ] **Step 3: Verify the tests fail**

Run:

```bash
python -m pytest -q tests/test_soccernetgs_integrity.py
```

Expected: import failure because `scripts/validate_soccernetgs.py` does not exist.

- [ ] **Step 4: Implement the minimal validator**

Create these interfaces in `scripts/validate_soccernetgs.py`:

```python
@dataclass(frozen=True)
class SequenceReport:
    split: str
    sequence: str
    sequence_id: str | None
    game_id: str | None
    version: str | None
    frame_count: int
    image_count: int
    annotation_count: int
    pitch_annotation_count: int
    ball_annotation_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_sequence(sequence_dir: Path, *, split: str) -> SequenceReport:
    """Read and validate one sequence without writing to it."""


def validate_dataset(
    root: Path,
    *,
    splits: tuple[str, ...],
    expected_counts: dict[str, int] | None = None,
) -> list[SequenceReport]:
    """Validate selected splits and detect duplicate info.id/name values."""
```

Use stable error codes rather than prose-only errors:

```text
missing_label
invalid_json
invalid_top_level
dataset_version_before_1_3
missing_game_id
name_mismatch
id_mismatch
bad_frame_rate
frame_count_mismatch
duplicate_image_id
duplicate_file_name
missing_frame
empty_frame
bad_annotation_image_ref
no_pitch_coordinates
duplicate_sequence_id
unexpected_sequence_count
```

Use `no_ball_coordinates` as a warning code, not an error code.

Validation rules:

- parse `Labels-GameState.json` completely;
- require top-level `info`, `images`, `annotations`, and `categories` with correct container types;
- require `info.version >= 1.3`, positive `frame_rate`, non-empty `game_id`, and folder/name/id consistency;
- require `seq_length == len(images) ==` non-empty files found under `img1/`;
- require unique `image_id` and `file_name` values and valid annotation image references;
- require at least one finite player/goalkeeper `bbox_pitch.x_bottom_middle/y_bottom_middle`;
- report ball-coordinate count and warn when it is zero; later tactical recipes must return `unsupported` when they require unavailable ball evidence;
- report clip timing inconsistencies, but do not invent a hard duration threshold;
- never treat missing tracking as proof that a tactic is absent.

Add a CLI:

```text
python scripts/validate_soccernetgs.py \
  --root PATH \
  --splits train valid \
  --expect train=57 \
  --expect valid=58 \
  [--report PATH]
```

Exit `0` only when every selected sequence and expected split count passes. The optional report is written atomically through a temporary sibling file and contains `schema_version`, validation time, root, expected counts, split summaries, and per-sequence reports.

- [ ] **Step 5: Run tests and inspect the known broken canonical root**

Run:

```bash
python -m pytest -q tests/test_soccernetgs_integrity.py
python scripts/validate_soccernetgs.py \
  --root codes/sn-gamestate/datasets/SoccerNetGS \
  --splits train valid \
  --expect train=57 \
  --expect valid=58 \
  --report /tmp/soccernetgs-before-repair.json
```

Expected:

- pytest passes;
- the real-data command exits `1`;
- report contains 57 `invalid_json` train sequences and 58 `missing_label` valid sequences;
- no file under the dataset root changes.

- [ ] **Step 6: Commit the validator checkpoint**

```bash
git add .gitignore scripts/validate_soccernetgs.py tests/test_soccernetgs_integrity.py
git commit -m "test: validate SoccerNetGS integrity"
```

---

### Task 2: Repair train/valid through a staged, user-approved promotion

**Files/data:**

- Create: `scripts/promote_soccernetgs_splits.py`
- Create: `tests/test_soccernetgs_promotion.py`
- External staging: `codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/`
- Canonical destination: `codes/sn-gamestate/datasets/SoccerNetGS/{train,valid}`
- Create after success: `benchmark/manifests/soccernetgs_v1_3_integrity.json`

**Success criteria:** Authoritative v1.3 train=57 and valid=58 pass the validator in staging and after an all-or-rollback promotion; `test/` has the same path and bytes as before. No sequence of partial shell moves can leave canonical train/valid in a mixed state.

- [ ] **Step 1: Write failing transactional-promotion tests**

Create `tests/test_soccernetgs_promotion.py` with tiny valid canonical/staging
roots and cover:

- `test_preflight_rejects_missing_or_invalid_staging_split`
- `test_preflight_rejects_existing_quarantine_destination`
- `test_success_moves_exactly_train_and_valid_and_keeps_test_untouched`
- `test_failure_after_any_move_rolls_every_prior_move_back`
- `test_post_move_validation_failure_rolls_back`
- `test_confirmation_token_is_required`

Monkeypatch the module's single private `_rename(source, destination)` helper to
raise after the first, second, third, and fourth operation; after each failure,
assert both canonical splits and both staging splits are back at their original
paths.

- [ ] **Step 2: Verify the promotion tests fail**

```bash
python -m pytest -q tests/test_soccernetgs_promotion.py
```

Expected: import failure because the promotion script does not exist.

- [ ] **Step 3: Implement one guarded promotion command**

Create `scripts/promote_soccernetgs_splits.py`:

```python
PROMOTION_SPLITS = ("train", "valid")
CONFIRMATION = "PROMOTE_TRAIN_VALID"


def promote_train_valid(
    *,
    canonical_root: Path,
    staging_root: Path,
    quarantine_root: Path,
    expected_counts: dict[str, int],
    confirmation: str,
) -> dict:
    """Preflight, validate, move both splits, or roll every move back."""
```

Implementation rules:

- require the exact confirmation token;
- require canonical/staging roots and both split directories to exist;
- require quarantine root not to exist, then create it once;
- call the Task 1 validator on both staging splits before moving anything;
- ensure staging and canonical are on the same filesystem;
- execute the four renames through one `_rename` helper and append each completed `(source, destination)` pair to a journal;
- on any rename or post-move validation exception, reverse the completed journal in strict reverse order, then re-raise;
- validate canonical train/valid after all four moves before returning success;
- write a promotion journal below quarantine only after success;
- never accept arbitrary split names and never read, move, or delete `test/`;
- never delete the quarantine.

CLI:

```text
python scripts/promote_soccernetgs_splits.py \
  --canonical-root PATH \
  --staging-root PATH \
  --quarantine-root PATH \
  --expect train=57 \
  --expect valid=58 \
  --confirm PROMOTE_TRAIN_VALID
```

- [ ] **Step 4: Run the promotion tests**

```bash
python -m pytest -q tests/test_soccernetgs_promotion.py
```

- [ ] **Step 5: Record the untouched test fingerprint**

Run:

```bash
find codes/sn-gamestate/datasets/SoccerNetGS/test -type f -print \
  | LC_ALL=C sort \
  | while IFS= read -r file; do shasum -a 256 "$file"; done \
  > /tmp/soccernetgs-test-before.sha256
```

- [ ] **Step 6: Install the operator-only downloader if necessary**

Do not add `SoccerNet` to this repository's runtime dependencies. In the environment used for the repair only:

```bash
python -m pip install SoccerNet
```

- [ ] **Step 7: Download only train/valid to staging using the official task name**

Run from the repository root:

```bash
test ! -e codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3
python -c 'from SoccerNet.Downloader import SoccerNetDownloader; d = SoccerNetDownloader(LocalDirectory="codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3"); d.downloadDataTask(task="gamestate-2024", split=["train", "valid"])'
mkdir -p codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/train
mkdir -p codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/valid
python -m zipfile -e \
  codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/gamestate-2024/train.zip \
  codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/train
python -m zipfile -e \
  codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/gamestate-2024/valid.zip \
  codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3/valid
```

If the staging root already exists, stop and inspect/remove it only with
separate approval; never mix a prior partial download into this run.

Reference: the official [SoccerNet sn-gamestate README](https://github.com/SoccerNet/sn-gamestate) specifies dataset version 1.3 and downloader task `gamestate-2024`.

- [ ] **Step 8: Gate staging before any canonical move**

Run:

```bash
python scripts/validate_soccernetgs.py \
  --root codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3 \
  --splits train valid \
  --expect train=57 \
  --expect valid=58 \
  --report /tmp/soccernetgs-staging.json
```

Expected: exit `0`; every label reports version `1.3`; no canonical file has changed.

- [ ] **Step 9: Pause for explicit promotion approval**

Show the staging report summary to the user. Do not execute the next step until the user explicitly approves replacing only canonical `train/` and `valid/`.

- [ ] **Step 10: Promote both splits through the transactional command**

After approval:

```bash
python scripts/promote_soccernetgs_splits.py \
  --canonical-root codes/sn-gamestate/datasets/SoccerNetGS \
  --staging-root codes/sn-gamestate/datasets/SoccerNetGS-repair-v1.3 \
  --quarantine-root codes/sn-gamestate/datasets/SoccerNetGS/_quarantine/repair-v1.3-prepromotion \
  --expect train=57 \
  --expect valid=58 \
  --confirm PROMOTE_TRAIN_VALID
```

The command must fail before moving anything if that fixed quarantine path
already exists. Keep the quarantine; deleting or renaming it requires a
separate later approval.

- [ ] **Step 11: Validate the development dataset index and prove test was untouched**

Run:

```bash
mkdir -p benchmark/manifests
python scripts/validate_soccernetgs.py \
  --root codes/sn-gamestate/datasets/SoccerNetGS \
  --splits train valid \
  --expect train=57 \
  --expect valid=58 \
  --report benchmark/manifests/soccernetgs_v1_3_integrity.json
find codes/sn-gamestate/datasets/SoccerNetGS/test -type f -print \
  | LC_ALL=C sort \
  | while IFS= read -r file; do shasum -a 256 "$file"; done \
  > /tmp/soccernetgs-test-after.sha256
diff -u /tmp/soccernetgs-test-before.sha256 /tmp/soccernetgs-test-after.sha256
```

Expected: validator exits `0`; the report supplies the trusted train/valid
split/clip/game mapping for development leakage checks; `diff` has no output.
Do not parse test labels here. The known locked clip ranges for games 8 and 11
come from the approved spec; semantic test validation waits until the frozen
promotion plan.

- [ ] **Step 12: Commit code/tests and only the small integrity manifest**

```bash
git add \
  scripts/promote_soccernetgs_splits.py \
  tests/test_soccernetgs_promotion.py \
  benchmark/manifests/soccernetgs_v1_3_integrity.json
git commit -m "data: record SoccerNetGS v1.3 integrity"
```

Do not stage downloaded data, archives, quarantine content, frames, or labels.

---

### Task 3: Replace the prose list with a validated private catalog and public projection

**Files:**

- Create: `pipeline/stage2b/kb.py`
- Create: `docs/superpowers/references/2026-07-11-tactical-kb-source-audit.md`
- Modify: `pipeline/stage2b/concepts.yaml`
- Create: `tests/test_stage2b_kb.py`

**Success criteria:** The catalog has 14 normalized concepts, complete source records, exact aliases and actor roles, all production-disabled; the public projection cannot leak recipes, thresholds, maturity, source records, or calibration details.

- [ ] **Step 1: Write failing catalog tests**

Create `tests/test_stage2b_kb.py` covering:

Import `CatalogError`, `load_catalog`, and `project_public_catalog` from
`pipeline.stage2b.kb`, then implement these tests:

- `test_repository_catalog_has_14_disabled_concepts`
- `test_candidate_pool_is_the_approved_six`
- `test_legacy_aliases_resolve_to_canonical_ids`
- `test_duplicate_concept_source_and_aliases_fail`
- `test_source_records_require_complete_metadata_and_sha256`
- `test_planned_concept_may_omit_recipe`
- `test_production_concept_requires_recipe_checker_and_gate`
- `test_public_projection_is_an_exact_whitelist`

The projection assertion must recurse through every returned object and fail if any of these keys appear:

```python
PRIVATE_KEYS = {
    "sources", "source_registry", "recipe", "parameters", "thresholds",
    "maturity", "production_enabled", "production_gate",
}
```

- [ ] **Step 2: Verify the tests fail**

```bash
python -m pytest -q tests/test_stage2b_kb.py
```

Expected: import failure for `pipeline.stage2b.kb`.

- [ ] **Step 3: Implement one catalog loader**

Create `pipeline/stage2b/kb.py` with no Pydantic or JSON Schema dependency:

```python
DEFAULT_CATALOG_PATH = Path(__file__).with_name("concepts.yaml")
PUBLIC_FIELDS = (
    "id", "name_zh", "name_en", "aliases",
    "observation_definition", "required_actor_roles",
)


class CatalogError(ValueError):
    pass


def load_catalog(
    path: Path = DEFAULT_CATALOG_PATH,
    *,
    checker_names: frozenset[str] | None = None,
) -> dict:
    """Load once with yaml.safe_load and validate all invariants."""


def get_concept(catalog: dict, concept_id: str) -> dict:
    """Return the exact private entry or raise CatalogError."""


def project_public_catalog(catalog: dict, *, concept_ids: set[str]) -> dict:
    """Build a new object using PUBLIC_FIELDS only; never redact a private copy."""
```

Validate:

- `catalog_version` and `public_projection_version` are non-empty strings;
- concept IDs, source IDs, and normalized aliases are unique;
- every source reference resolves;
- every `SourceRecord` has `source_id`, `title`, `authors`, `version`, `url_or_doi`, `license`, `accessed_at`, `content_hash`, and `used_for`;
- `content_hash` matches `sha256:` plus 64 lowercase hex characters;
- `allowed_claim_levels` is exactly `[observation]` in v2;
- maturity is `planned`, `audited`, or `production`;
- `production_enabled: true` requires production maturity, recipe, checker, and recorded passed gate;
- if `checker_names` is supplied, every production checker resolves;
- `feasibility_candidates` resolves to unique catalog IDs.

- [ ] **Step 4: Complete and approve the finite source audit**

Create `docs/superpowers/references/2026-07-11-tactical-kb-source-audit.md`
with exactly 14 rows: canonical ID, bilingual names/aliases, observation-only
inclusion boundary, exclusion boundary, actor fields, terminology origin, and
formalization source IDs. This is a finite review of the approved 14 concepts,
not a term-scraping pipeline.

Use these two complete external formalization records:

| Source ID | Title / authors | Version / DOI | License | Local SHA-256 | Used for |
|---|---|---|---|---|---|
| `gentac_arxiv_2604_11786_v1` | *GenTac: Generative Modeling and Forecasting of Soccer Tactics* — Jiayuan Rao, Tianlin Gui, Haoning Wu, Yanfeng Wang, Weidi Xie | `arXiv:2604.11786v1`, `https://doi.org/10.48550/arXiv.2604.11786` | CC BY-NC-SA 4.0 | `31ef3d7c6b9d2d6d65c3dd6d39984dd6ef436bd33976b95e69f87873cd2f605b` | trajectory/event-context separation and candidate team-structure variables |
| `tacticgen_arxiv_2604_18210_v1` | *TacticGen: Grounding Adaptable and Scalable Generation of Football Tactics* — Sheng Xu, Guiliang Liu, Tarak Kharrat, Yudong Luo, Mohamed Aloulou, Javier López Peña, Konstantin Sofeikov, Adam Reid, Paul Roberts, Steven Spencer, Joe Carnall, Ian McHale, Oliver Schulte, Hongyuan Zha, Wei-Shi Zheng | `arXiv:2604.18210v1`, `https://doi.org/10.48550/arXiv.2604.18210` | CC BY-NC-SA 4.0 | `0c49c07fbc4dad460c39760af854569051f6ada2d2326b7a044ee08112d1c4c2` | multi-agent movement/interaction representation |

Both records use `accessed_at: 2026-07-11`. They motivate
formalization/representation only and never calibration thresholds.

Because neither paper is a canonical glossary for all 14 terms, do not pretend
it is. The reviewed source-audit document becomes
`soccermaster_operational_definitions_v1`, with:

- title `SoccerMaster Tactical KB v2 Operational Definitions`;
- authors `[SoccerMaster project]`;
- version `2026-07-11`;
- repository URL `https://github.com/frankdarkluo/SoccerMaster/blob/main/docs/superpowers/references/2026-07-11-tactical-kb-source-audit.md`;
- license `project-internal`;
- accessed date `2026-07-11`;
- `used_for: [name, aliases, concept_boundary, actor_schema]`;
- content hash computed from the finished file with `shasum -a 256`.

All 14 concepts reference that reviewed operational-definition record under
`terminology`. Map `formalization` to one or both external records only where
the audit document states the representational connection. Pause for user
approval of this 14-row source audit before committing the catalog; unresolved
rows remain explicit and cannot enter `concepts.yaml`.

- [ ] **Step 5: Replace `concepts.yaml` with the canonical 14-entry shape**

Use top-level keys:

```yaml
catalog_version: tactical-kb-v2
public_projection_version: tactical-public-v1
source_registry: []
feasibility_candidates:
  - overlap_run
  - run_in_behind
  - switch_of_play
  - local_numerical_superiority
  - compact_block
  - line_break
concepts: []
```

The exact 14 canonical IDs are:

```text
overlap_run
run_in_behind
switch_of_play
local_numerical_superiority
compact_block
line_break
underlap_run
side_overload
halfspace_occupation
width_depth_stretch
one_two
third_man_run
high_press
low_block
```

Freeze actor fields before annotation. `same` and `opponent` are relative to
the proposal's canonical `team_id`; an empty list means the checker verifies a
team-level pattern without trusting model-selected individuals:

| Concept | Exact actor fields | Team relation |
|---|---|---|
| `overlap_run` | `carrier_track_id`, `runner_track_id` | same, same |
| `run_in_behind` | `carrier_track_id`, `runner_track_id` | same, same |
| `switch_of_play` | `passer_track_id`, `receiver_track_id` | same, same |
| `local_numerical_superiority` | `anchor_track_id` | same |
| `compact_block` | none | team-level |
| `line_break` | `passer_track_id`, `receiver_track_id` | same, same |
| `underlap_run` | `carrier_track_id`, `runner_track_id` | same, same |
| `side_overload` | `anchor_track_id` | same |
| `halfspace_occupation` | `occupant_track_id` | same |
| `width_depth_stretch` | `wide_player_track_id`, `depth_player_track_id` | same, same |
| `one_two` | `first_passer_track_id`, `wall_player_track_id` | same, same |
| `third_man_run` | `first_passer_track_id`, `link_player_track_id`, `runner_track_id` | same, same, same |
| `high_press` | `presser_track_id`, `opponent_carrier_track_id` | same, opponent |
| `low_block` | none | team-level |

Store this as public `required_actor_roles` plus private per-role team
constraints. For the six feasibility candidates, also freeze concise inclusion
and exclusion boundaries in `observation_definition`:

- `overlap_run`: same-team runner changes from behind to ahead of the carrier on the outside; exclude underlaps, already-ahead parallel support, and defensive recovery runs.
- `run_in_behind`: same-team runner moves from in front of the relevant defensive line into space behind it while the team controls or progresses the ball; exclude recovery runs and momentary noisy line crossings.
- `switch_of_play`: a controlled same-team ball transfer changes the active side of the pitch between passer and receiver; exclude clearances, deflections, and a merely diagonal pass that stays on the same side.
- `local_numerical_superiority`: within a fixed local region around the anchor, the proposal team has a sustained player-count advantage; exclude single-frame count changes and unsupported off-camera absences.
- `compact_block`: the out-of-possession team's visible inter-player width/depth and line spacing remain compact for the minimum recipe duration; exclude low visibility and a transient collapse around one duel.
- `line_break`: a controlled pass from passer to receiver moves the ball across a defined opponent line while retaining possession; exclude dribbles, clearances, and apparent crossings caused by calibration or identity errors.
- `underlap_run`: a same-team runner changes from behind to ahead of the carrier through the inside channel; exclude outside overlaps, straight parallel support, and defensive recovery runs.
- `side_overload`: the proposal team sustains a local player-count concentration in the active wide channel; exclude a momentary crowd around one duel and unsupported off-camera absences.
- `halfspace_occupation`: a same-team player occupies the defined channel between the center and touchline for a sustained interval; exclude touchline width, central-lane occupation, and transient crossings.
- `width_depth_stretch`: the same team simultaneously maintains a wide option and a depth option that expand two different spatial dimensions; exclude width-only or depth-only spacing and one-frame extrema.
- `one_two`: the first passer continues moving and receives a controlled return from the wall player within one ordered exchange; exclude two unrelated passes and reversed actor order.
- `third_man_run`: the first passer connects through a link player to a distinct third runner in an ordered three-player sequence; exclude ordinary two-player combinations and a third player with no receiving movement.
- `high_press`: the out-of-possession proposal team sustains close pressure by visible players in the opponent's build-up area; exclude one isolated chase and low-visibility/off-camera assumptions.
- `low_block`: the out-of-possession proposal team sustains a compact visible block near its own goal; exclude a transient retreat, restart setup, and conclusions requiring unseen players.

These are semantic boundaries, not detector thresholds. Metric radii,
durations, tolerances, and quality gates remain unset until development
calibration.

Normalize legacy names only as aliases:

```text
depth_run -> run_in_behind
numerical_superiority -> local_numerical_superiority
switch_play -> switch_of_play
```

Do not include `positional_attack` or `counter_attack`; they belong to normalized context. Every definition must describe only visible geometry/motion, not an effect or intention. Every entry starts as:

```yaml
allowed_claim_levels: [observation]
maturity: planned
production_enabled: false
```

Use the approved source responsibilities:

```yaml
sources:
  terminology: [source_id]
  formalization: [source_id]
  calibration: []
```

Verify the approved source-audit metadata and source-file hashes with:

```bash
shasum -a 256 docs/papers/GenTac.pdf
shasum -a 256 docs/papers/TacticGen.pdf
```

The current audited local hashes are:

```text
GenTac.pdf    31ef3d7c6b9d2d6d65c3dd6d39984dd6ef436bd33976b95e69f87873cd2f605b
TacticGen.pdf 0c49c07fbc4dad460c39760af854569051f6ada2d2326b7a044ee08112d1c4c2
```

External sources may motivate formalization only; the reviewed operational
definitions own terminology boundaries. Leave calibration empty until Task 8
creates the development audit records.

- [ ] **Step 6: Run the catalog tests**

```bash
python -m pytest -q tests/test_stage2b_kb.py
python -c 'from pipeline.stage2b.kb import load_catalog; print(len(load_catalog()["concepts"]))'
```

Expected: tests pass; command prints `14`.

- [ ] **Step 7: Commit the catalog checkpoint**

```bash
git add \
  docs/superpowers/references/2026-07-11-tactical-kb-source-audit.md \
  pipeline/stage2b/kb.py \
  pipeline/stage2b/concepts.yaml \
  tests/test_stage2b_kb.py
git commit -m "feat: add tactical KB v2 catalog contract"
```

---

### Task 4: Build the concept-neutral 25 Hz tactical state

**Files:**

- Create: `pipeline/relations/state.py`
- Create: `tests/test_tactical_state.py`

**Success criteria:** Source team labels are canonicalized without using attack direction; raw frame measurements remain at 25 Hz; public proposal state is a deterministic at-most-12.5 Hz projection; final tactical labels never enter state.

- [ ] **Step 1: Write failing state tests**

Create `tests/test_tactical_state.py` using synthetic `FrameData` objects. Cover:

Import `build_tactical_state`, `canonical_team_map`, and
`project_proposal_state` from `pipeline.relations.state`, then implement:

- `test_canonical_team_map_is_recorded_and_direction_independent`
- `test_invalid_or_partial_recorded_map_fails`
- `test_state_retains_25_hz_player_and_ball_measurements`
- `test_attack_direction_defaults_unknown_and_requires_explicit_provenance`
- `test_normalized_context_defaults_to_explicit_unknowns`
- `test_valid_source_labels_and_normalized_context_can_be_supplied`
- `test_invalid_context_enum_or_missing_provenance_fails`
- `test_state_contains_quality_and_provenance_but_no_concepts`
- `test_proposal_projection_is_at_most_12_5_hz`
- `test_proposal_events_use_canonical_team_ids_without_mutating_event_spine`

The forbidden-key check should reject keys containing `concept`, `candidate`, `overlap`, `counter_attack`, or `verified` anywhere in state.

- [ ] **Step 2: Verify the tests fail**

```bash
python -m pytest -q tests/test_tactical_state.py
```

Expected: import failure for `pipeline.relations.state`.

- [ ] **Step 3: Implement the narrow state adapter**

Create:

```python
CANONICAL_TEAM_IDS = frozenset({"team_0", "team_1"})


def canonical_team_map(
    source_labels: set[str],
    *,
    recorded: dict[str, str] | None = None,
) -> dict[str, str]:
    """Validate a recorded bijection or deterministically map two source labels."""


def build_tactical_state(
    frames: list[FrameData],
    *,
    fps: float,
    clip_id: str,
    state_source: str,
    source_team_map: dict[str, str] | None = None,
    attack_directions: dict[str, dict] | None = None,
    source_labels: dict | None = None,
    normalized_context: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """Create serializable, concept-neutral raw measurements and quality."""


def project_proposal_state(state: dict, *, max_hz: float = 12.5) -> dict:
    """Return only compact public measurements/context/quality."""


def project_proposal_events(events: list[dict], *, state: dict) -> list[dict]:
    """Return a copied event view with canonical team IDs for the tactical prompt."""
```

`state_source` is exactly `ground_truth` or `gsr_prediction`. The state contains:

```yaml
schema_version: tactical-state-v1
clip_id: SNGS-xxx
state_source: gsr_prediction
fps: 25.0
n_frames: 750
teams:
  team_0:
    source_team_label: left
    attack_direction: unknown
  team_1:
    source_team_label: right
    attack_direction: unknown
source_labels: {}
normalized_context:
  possession_phase: {team_id: null, value: unknown}
  restart_state: {value: unknown}
  progression_state: {team_id: null, value: unknown}
  danger_state: {team_id: null, value: unknown}
measurements:
  tracks: {}
  ball: []
quality: {}
provenance: {}
```

Rules:

- fallback mapping sorts the two source labels once and records `mapping_method: deterministic_source_label_sort`;
- a provided mapping must be a bijection onto `team_0/team_1`;
- never infer attack direction from source `left/right` labels; optional direction records use canonical team IDs and require `value: left_to_right|right_to_left|unknown`, `source`, and `confidence`; absent direction defaults to `unknown`;
- validate `possession_phase` against `build_up|transition|settled_attack|settled_defense|unknown`, `restart_state` against `open_play|set_piece|interruption|unknown`, `progression_state` against `stable|progressing|penetrating|unknown`, and `danger_state` against `neutral|threat|unknown`; require canonical team IDs where the field is team-relative and provenance for every non-unknown value; no phase detector is added in this plan;
- store 25 Hz per-track `frame_id`, `t`, `x`, and `y`, plus ball samples;
- quality reports track continuity, pitch-coordinate coverage, and ball coverage without producing tactic labels;
- `project_proposal_state` chooses a deterministic integer frame step at or below 12.5 Hz and never exposes private KB data;
- `project_proposal_events` rewrites team-valued event fields through the recorded source-team map, rejects unmapped labels, and leaves the event spine unchanged;
- do not build 10 Hz team structure until an audited recipe requires it.

- [ ] **Step 4: Run the state tests**

```bash
python -m pytest -q tests/test_tactical_state.py
python -m py_compile pipeline/relations/state.py
```

- [ ] **Step 5: Commit the state checkpoint**

```bash
git add pipeline/relations/state.py tests/test_tactical_state.py
git commit -m "feat: add concept-neutral tactical state"
```

---

### Task 5: Enforce a strict proposal boundary and reconstruct facts through named checkers

**Files:**

- Create: `pipeline/stage2b/tactics.py`
- Create: `tests/test_stage2b_tactics.py`

**Success criteria:** Doubao can provide only concept, proposed window, canonical team, and actors. Unknown/model-authored verification fields are rejected and logged. A checker result outside its frozen tolerance cannot produce a fact. No real production checker is implemented yet.

- [ ] **Step 1: Write failing contract tests**

Create `tests/test_stage2b_tactics.py`. Use a synthetic private catalog with one fake audited concept and inject a fake checker; never use a real recipe. Cover:

Import `process_tactical_proposals` from `pipeline.stage2b.tactics`, then
implement:

- `test_unknown_fields_are_rejected_and_logged`
- `test_model_predicate_recipe_threshold_and_verified_fields_are_rejected`
- `test_unknown_concept_team_track_and_missing_actor_are_rejected`
- `test_concept_outside_explicit_allowed_set_is_rejected`
- `test_actor_roles_are_distinct_and_follow_same_or_opponent_constraints`
- `test_checker_pass_rebuilds_a_new_fact_from_whitelisted_fields`
- `test_verified_window_cannot_escape_recipe_tolerance`
- `test_checker_failed_unsupported_and_exception_produce_no_fact`
- `test_fact_is_observation_only_and_traceable_to_recipe_and_state`
- `test_no_registered_production_checker_means_no_fact`

Include malicious proposal fields such as `evidence_queries`, `predicate`, `threshold`, `recipe`, `verified`, and `fact_id`; each whole proposal must be rejected rather than silently stripped.

- [ ] **Step 2: Verify the tests fail**

```bash
python -m pytest -q tests/test_stage2b_tactics.py
```

Expected: import failure for `pipeline.stage2b.tactics`.

- [ ] **Step 3: Implement one public processing entry point**

Create:

```python
Checker = Callable[[dict, dict, dict], dict]
CHECKERS: dict[str, Checker] = {}
PROPOSAL_FIELDS = frozenset({
    "concept_id", "proposed_window", "team_id", "actors",
})


class ProposalError(ValueError):
    pass


class VerificationError(ValueError):
    pass


def process_tactical_proposals(
    raw: object,
    *,
    catalog: dict,
    tactical_state: dict,
    duration_s: float,
    allowed_concept_ids: set[str],
    checkers: dict[str, Checker] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return one audit record per raw proposal and newly rebuilt verified facts."""
```

Keep `_normalize_proposal`, `_verify_proposal`, and `_build_fact` private. Generate `proposal_id` and `fact_id` in code; never accept them from the model.

Every raw item produces an audit record:

```yaml
proposal_id: proposal_001
clip_id: SNGS-xxx
state_source: gsr_prediction
raw: {}
normalized_proposal: null
proposal_status: accepted | rejected
reasons: []
verification_status: passed | failed | unsupported | not_run
fact_id: null
```

The checker contract is:

```python
{
    "status": "passed" | "failed" | "unsupported",
    "verified_window": {"start_s": float, "end_s": float} | None,
    "window_resolution": {
        "method": "deterministic_anchor_snap",
        "tolerance_s": float,
    },
    "decision_time_s": float | None,
    "resolved_evidence": list[dict],
    "quality_flags": list[str],
    "reason": str,
}
```

The generic layer enforces:

- exact proposal fields and valid finite clip-bounded window;
- the concept is in the explicit caller-supplied `allowed_concept_ids`; unknown concepts and concepts outside that set are rejected before checker dispatch;
- required actor keys and same/opponent team constraints are derived from the private catalog;
- known, distinct tracks satisfy those per-role constraints; team-level concepts may require an empty actor object;
- recipe/checker loaded only from the private catalog;
- verified window remains within the recipe's fixed tolerance around the proposal;
- `decision_time_s >= verified_window.end_s` and within clip bounds;
- `failed`, `unsupported`, and checker exceptions produce no fact;
- fact is a newly built whitelist object containing clip ID, proposed/verified windows, recipe version, state source, team, actors, attack direction, observation-only claim levels, evidence, quality, and provenance;
- the fact never contains raw proposal fields or a boolean `verified` marker.

Leave production `CHECKERS` empty in this plan. Tests inject a fake checker to
prove the boundary. Normal Stage 2B passes production-enabled IDs; a later
evaluation runner may pass only recipe IDs from a validated frozen manifest.
Never expose an arbitrary user/CLI concept override.

- [ ] **Step 4: Run the verifier tests**

```bash
python -m pytest -q tests/test_stage2b_tactics.py
python -m py_compile pipeline/stage2b/tactics.py
```

- [ ] **Step 5: Commit the verifier boundary**

```bash
git add pipeline/stage2b/tactics.py tests/test_stage2b_tactics.py
git commit -m "feat: enforce fixed tactical verifier boundary"
```

---

### Task 6: Integrate state, proposals, facts, and observation-only commentary into Stage 2B

**Files:**

- Modify: `pipeline/config.py`
- Modify: `pipeline/stage2b/hybrid.py`
- Modify: `pipeline/stage2b/run.py`
- Create: `tests/test_stage2b_tactical_integration.py`

**Success criteria:** Runtime no longer executes model-authored queries or predicates. Stage 2B writes state/proposal/fact artifacts, and commentary can reference only verified fact IDs. With all concepts disabled, hybrid mode preserves direct commentary and makes no tactical proposal call.

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_stage2b_tactical_integration.py` with stub calls and temporary JSON. Cover:

Implement these integration tests:

- `test_config_exposes_state_proposal_and_fact_paths`
- `test_disabled_catalog_skips_tactical_llm_call`
- `test_run_writes_empty_proposals_and_facts_and_preserves_direct`
- `test_tactical_prompt_projects_event_teams_to_canonical_ids`
- `test_composer_prompt_contains_facts_not_raw_proposals_or_private_recipe`
- `test_commentary_rejects_raw_proposal_reference`
- `test_commentary_rejects_missing_fact_disabled_concept_and_non_observation`
- `test_commentary_cannot_start_before_fact_decision_time`
- `test_checker_failure_or_unsupported_falls_back_to_direct`
- `test_commentary_slots_do_not_bound_verified_evidence_windows`

The tests must stub `observe_direct`, video duration, radar rendering, and ARK calls; they must not read a real clip or invoke FFmpeg/GPU/network.

- [ ] **Step 2: Verify the tests fail**

```bash
python -m pytest -q tests/test_stage2b_tactical_integration.py
```

- [ ] **Step 3: Replace the output contract**

In `pipeline/config.py`, replace `tactical_candidates_json` with:

```python
@property
def tactical_state_json(self) -> Path:
    return self.comments_dir / "tactical_state.json"

@property
def tactical_proposals_json(self) -> Path:
    return self.comments_dir / "tactical_proposals.json"

@property
def verified_tactical_facts_json(self) -> Path:
    return self.comments_dir / "verified_tactical_facts.json"
```

Do not retain a production fallback path to `tactical_candidates.json`.

- [ ] **Step 4: Remove self-authored verification from `hybrid.py`**

Delete:

- `copy`, `Path`, `predicate_passes`, and `resolve_query` imports used only by old candidate verification;
- `_concept_ids()`;
- `verify_candidates()`;
- `_candidate_map()`.

Rename `candidate_windows()` to `commentary_windows()` and make it accept verified facts. It may select sparse speech slots of up to five seconds, but each slot must start at or after `fact.decision_time_s`. It cannot approve, trim, or reject a fact's `proposed_window` or `verified_window`.

Change `audit_commentary` and `compose_hybrid` to consume `facts`, not
candidates, and pass the enabled concept IDs explicitly:

```python
def audit_commentary(
    segments: object,
    events: list[dict],
    facts: list[dict],
    enabled_concept_ids: set[str],
    duration_s: float,
) -> list[str]:
    """Return mechanical event/fact/claim/scheduling errors."""


def compose_hybrid(
    events: list[dict],
    direct_commentary: list[dict],
    facts: list[dict],
    enabled_concept_ids: set[str],
    windows: list[dict],
    duration_s: float,
    call: Callable = ark_chat,
) -> list[dict]:
    """Compose from facts only and return direct commentary on failure."""
```

Tactical segments use:

```yaml
tactical_facts_referenced: [fact_001]
tactical_claims:
  - fact_id: fact_001
    concept_id: overlap_run
    claim_level: observation
```

The audit requires exact fact/concept binding, membership in the explicitly
passed `enabled_concept_ids`, observation-only claim level, and commentary
start at/after `decision_time_s`. Raw proposal IDs cannot pass. Update
`_reconcile_direct` to call the audit with empty facts and an empty enabled set.

- [ ] **Step 5: Make `run.py` a narrow orchestrator**

Replace `_propose_candidates()` with:

```python
def _request_tactical_proposals(
    *,
    call: Callable,
    public_catalog: dict,
    proposal_state: dict,
    events: list[dict],
    duration_s: float,
) -> object:
    """Request JSON only; validation happens in process_tactical_proposals."""


def generate_tactical_artifacts(
    tracking_json: Path,
    *,
    clip_id: str,
    state_source: str,
    duration_s: float,
    events: list[dict],
    catalog: dict,
    allowed_concept_ids: set[str],
    call: Callable,
) -> tuple[dict, list[dict], list[dict]]:
    """Build state, request/validate proposals, and return state/audit/facts."""
```

Rules:

- `run_stage2b` loads the private catalog and derives `production_ids` only from `production_enabled: true` entries;
- `run_stage2b` skips the tactical call when `production_ids` is empty and otherwise passes that set as `allowed_concept_ids`;
- `generate_tactical_artifacts` never derives or broadens IDs; it projects, requests, and processes exactly its validated `allowed_concept_ids`;
- send only `project_public_catalog(catalog, concept_ids=allowed_concept_ids)`, compact proposal state, canonicalized event copies, and duration;
- keep source `left/right` labels only in tactical-state provenance; no team-valued field in the tactical proposal prompt may use them;
- normal `run_stage2b` calls `generate_tactical_artifacts` with its GSR `predictions.json`, `state_source="gsr_prediction"`, and production IDs;
- do not expose `allowed_concept_ids` as a Stage 2B CLI flag; a later evaluation runner obtains them only from a validated frozen manifest and invokes the same function separately for GT labels and GSR predictions;
- build/write `tactical_state.json` before proposal processing;
- write all proposal audit records to `tactical_proposals.json`;
- write only rebuilt facts to `verified_tactical_facts.json`;
- compose from facts only;
- on any KB/state/verifier/composer failure, log it and atomically preserve direct commentary;
- direct mode removes stale tactical state/proposal/fact/relations artifacts;
- update `_cache_complete` to require all three new artifacts for hybrid mode.

Do not expose the private catalog, checker parameters, or thresholds in any prompt.
Evaluation-generated facts are written below `benchmark/results/` by the later
evaluation runner and are never passed to the production commentary composer
unless the concept has independently become production-enabled.

- [ ] **Step 6: Run integration and regression checks**

```bash
python -m pytest -q \
  tests/test_stage2b_kb.py \
  tests/test_tactical_state.py \
  tests/test_stage2b_tactics.py \
  tests/test_stage2b_tactical_integration.py
python -m compileall -q pipeline/stage2b pipeline/relations
python -m pipeline.stage2b.run --help
bash -n scripts/run_stage2b.sh
rg -n "verify_candidates|evidence_queries|predicate_passes|tactical_candidates" \
  pipeline/stage2b pipeline/config.py
```

Expected: tests pass; help/shell checks pass; the final `rg` returns no runtime references.

- [ ] **Step 7: Commit Stage 2B integration**

```bash
git add \
  pipeline/config.py \
  pipeline/stage2b/hybrid.py \
  pipeline/stage2b/run.py \
  tests/test_stage2b_tactical_integration.py
git commit -m "refactor: consume verified tactical facts in stage2b"
```

---

### Task 7: Add one development annotation and feasibility-audit CLI

**Files:**

- Create: `scripts/tactical_kb_eval.py`
- Create: `tests/test_tactical_kb_eval.py`

**Success criteria:** One CLI validates the trusted development clip list and
paired GT/GSR annotations, then applies the fixed 6+6/two-match feasibility
rules. It does not implement release-manifest validation, held-out masks, or
full-clip production scoring before recipes exist.

- [ ] **Step 1: Write failing development-audit tests**

Import `audit_feasibility`, `load_annotations`, `load_dataset_index`,
`validate_annotation`, `validate_development_clip_manifest`, and
`validate_recipe_shortlist` from
`scripts.tactical_kb_eval`, then implement:

- `test_annotation_requires_match_clip_window_concept_and_actor_roles`
- `test_hard_negative_requires_type_and_target_concept`
- `test_annotation_match_id_must_equal_trusted_clip_game_mapping`
- `test_annotation_clip_must_belong_to_frozen_development_manifest`
- `test_games8_and11_clips_are_rejected_even_with_falsified_match_id`
- `test_game7_rows_are_diagnostic_and_do_not_count_toward_feasibility`
- `test_feasibility_requires_6_positive_6_hard_negative_and_2_matches`
- `test_gt_and_gsr_observability_are_reported_separately`
- `test_gt_and_gsr_actor_bindings_can_use_different_track_ids`
- `test_unresolved_gsr_actor_requires_unsupported_observability`
- `test_audit_does_not_cap_feasibility_passed_at_three`
- `test_recipe_shortlist_is_at_most_three_and_subset_of_passed_report`
- `test_recipe_shortlist_requires_exact_feasibility_report_hash`

- [ ] **Step 2: Verify the tests fail**

```bash
python -m pytest -q tests/test_tactical_kb_eval.py
```

- [ ] **Step 3: Implement the trusted development contracts**

Import the existing `pipeline.stage2b.kb` loader rather than creating another
catalog loader. Create:

```python
LOCKED_HELDOUT_CLIP_IDS = frozenset({
    *(f"SNGS-{value:03d}" for value in range(132, 151)),
    *(f"SNGS-{value:03d}" for value in range(187, 201)),
})


def load_annotations(path: Path) -> list[dict]:
    """Read JSONL, reject malformed/non-object lines, and enforce unique IDs."""


def load_dataset_index(path: Path) -> dict[tuple[str, str], dict]:
    """Build trusted train/valid clip metadata from the integrity manifest."""


def validate_development_clip_manifest(
    manifest: dict,
    *,
    dataset_index: dict[tuple[str, str], dict],
) -> list[str]:
    """Reject missing paths, metadata mismatches, and game 7/8/11 leakage."""


def validate_annotation(
    row: dict,
    *,
    catalog: dict,
    dataset_index: dict[tuple[str, str], dict],
    reviewed_clips: set[tuple[str, str]],
    purpose: str,
) -> list[str]:
    """Return deterministic schema, split, actor, window, and provenance errors."""


def audit_feasibility(
    catalog: dict,
    rows: list[dict],
    *,
    dataset_index: dict[tuple[str, str], dict],
    reviewed_clips: set[tuple[str, str]],
) -> dict:
    """Return per-concept metrics plus feasibility_passed and feasibility_failed."""


def validate_recipe_shortlist(shortlist: dict, *, feasibility_report: dict) -> list[str]:
    """Require a hash-bound zero-to-three subset of feasibility_passed."""
```

KB v2 development rows are a strict profile of the approved addendum. Where
the addendum's generic example conflicts, this profile wins:

```yaml
schema_version: tactical-kb-dev-v1
sample_id: stable-string
dataset_split: train
match_id: trusted-label-info-game-id
clip_id: SNGS-sequence-id
window: {start_s: 12.4, end_s: 16.8}
context_window: {start_s: 10.4, end_s: 18.8}
concept_id: overlap_run
label: positive
team_id: team_0
actor_bindings:
  ground_truth: {carrier_track_id: 31, runner_track_id: 18}
  gsr_prediction: {carrier_track_id: 104, runner_track_id: 97}
actor_correspondence:
  carrier: {status: reviewed_match, ground_truth_track_id: 31, gsr_prediction_track_id: 104}
  runner: {status: reviewed_match, ground_truth_track_id: 18, gsr_prediction_track_id: 97}
claim_labels: {observation: supported, effect: not_annotated, intention: not_annotated}
hard_negative: null
state_observability:
  ground_truth: direct
  gsr_prediction: direct
quality_by_state_source:
  ground_truth: {tracking: usable, ball: usable, calibration: usable, structure: not_required}
  gsr_prediction: {tracking: usable, ball: degraded, calibration: usable, structure: not_required}
kb_version: tactical-kb-v2
```

Use canonical `team_id`, never actor `team: left|right`. GT and GSR track IDs
are source-specific. An unresolved correspondence uses `status: unresolved`,
a null GSR track ID, and GSR observability `unsupported`. Development rows omit
`recipe_version`. Every `match_id` and `dataset_split` is checked against the
Task 2 manifest rather than accepted from the annotation.

Development validation accepts only train/valid clips in the trusted dataset
index and the frozen development clip manifest,
rejects `LOCKED_HELDOUT_CLIP_IDS` before self-reported metadata, rejects a
`match_id` different from trusted label metadata, and excludes any trusted
game-7 row from performance counts. It never opens test labels.

Per concept, report positive/hard-negative counts, eligible match count,
GT/GSR observable and unsupported counts, GT-to-GSR degradation,
`feasibility_passed`, failure reasons, and shortlist decision inputs. The CLI
applies the fixed 6+6/two-match minimum to all six candidates and does not cap
`feasibility_passed` at three. Task 8 records the separate human-reviewed
zero-to-three recipe shortlist.

- [ ] **Step 4: Add only development CLI subcommands**

```text
python scripts/tactical_kb_eval.py validate-development-clips --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json --clips benchmark/development_clip_manifest.json
python scripts/tactical_kb_eval.py validate-annotations --catalog pipeline/stage2b/concepts.yaml --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json --clips benchmark/development_clip_manifest.json --annotations benchmark/tactical_windows.jsonl --purpose development
python scripts/tactical_kb_eval.py audit --catalog pipeline/stage2b/concepts.yaml --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json --clips benchmark/development_clip_manifest.json --annotations benchmark/tactical_windows.jsonl --out benchmark/feasibility_report.json
python scripts/tactical_kb_eval.py validate-shortlist --report benchmark/feasibility_report.json --shortlist benchmark/recipe_shortlist.json
```

Use atomic JSON output and non-zero status for schema failures. Keep the audit
report byte-reproducible by excluding wall-clock timestamps.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest -q tests/test_tactical_kb_eval.py
python -m py_compile scripts/tactical_kb_eval.py
git add scripts/tactical_kb_eval.py tests/test_tactical_kb_eval.py
git commit -m "feat: add tactical feasibility audit tooling"
```

---

### Task 8: Build the real multi-match development audit and shortlist zero to three recipes

**Files/data:**

- Create through real clip review: `benchmark/development_clip_manifest.json`
- Create through human annotation: `benchmark/tactical_windows.jsonl`
- Create through deterministic audit: `benchmark/feasibility_report.json`
- Create through documented review: `benchmark/recipe_shortlist.json`
- Modify after audit: `pipeline/stage2b/concepts.yaml`
- Extend only for audit regressions: `tests/test_stage2b_kb.py`, `tests/test_tactical_kb_eval.py`

**Success criteria:** Every feasibility-passed concept has at least six positives and six targeted hard negatives across at least two non-game-7 development matches, with actor/window and paired GT/GSR observability. Zero to six concepts may become `audited`; a separate zero-to-three recipe shortlist controls follow-up checker work. None becomes production-enabled and no threshold is tuned on games 8/11.

- [ ] **Step 1: Inventory repaired train/valid and freeze the reviewed development clips**

Use the validator report plus label metadata to produce a review queue grouped by `game_id`. The script may list clip IDs, durations, and quality; it must not auto-label tactics.

Screen GT train/valid only, then save the clips chosen for detailed paired
review in `benchmark/development_clip_manifest.json`. The final file uses this
shape with real records:

```json
{
  "schema_version": "tactical-development-clips-v1",
  "dataset_integrity_manifest": "benchmark/manifests/soccernetgs_v1_3_integrity.json",
  "clips": []
}
```

This empty list demonstrates the container only; the saved artifact must be
non-empty and contain only reviewed real records with `dataset_split`,
`clip_id`, and label-derived `match_id`. Validate that every listed path
exists, every clip belongs to train/valid, and no `match_id` is 7, 8, or 11.
Do not use `SNGS-116/117` or any game-7 clip toward counts. Do not inspect
games 8 or 11.

Run the trusted mapping check before Stage 1:

```bash
python scripts/tactical_kb_eval.py validate-development-clips \
  --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json \
  --clips benchmark/development_clip_manifest.json
```

- [ ] **Step 2: Generate paired GSR predictions for the frozen development clips**

Use the existing Stage 1 launcher one clip at a time:

```bash
jq -r '.clips[] | [.dataset_split, .clip_id] | @tsv' \
  benchmark/development_clip_manifest.json \
  > /tmp/tactical-kb-dev-clips.tsv
while IFS=$'\t' read -r dataset_split clip_id; do
  bash scripts/run_stage1.sh \
    "codes/sn-gamestate/datasets/SoccerNetGS/${dataset_split}/${clip_id}" \
    "outputs/tactical-kb-audit/${clip_id}"
done < /tmp/tactical-kb-dev-clips.tsv
```

Do not run Stage 1 inside the evaluator and do not create predictions for
held-out games 8/11 in this task.

- [ ] **Step 3: Human-label the six-candidate pool on paired GT/GSR state**

For each candidate, inspect train/valid and create real JSONL rows conforming to the approved addendum:

- exact positive window;
- canonical `team_id`, source-specific GT/GSR actor bindings, and reviewed actor correspondence;
- targeted hard-negative type and target concept for negative rows;
- match/clip IDs and development split;
- observation claim label only;
- GT and GSR tracking/ball/calibration/structure quality;
- explicit GT and GSR observability;
- `ambiguous_exclude` for unresolved semantic cases rather than forcing a label.

Use the hard-negative taxonomy from the addendum. Random unrelated clips do not count toward the 6 hard negatives.

Every final annotation row includes `dataset_split: train|valid`, and every
clip must already appear in `benchmark/development_clip_manifest.json`.

- [ ] **Step 4: Validate annotations before counting them**

```bash
python scripts/tactical_kb_eval.py validate-annotations \
  --catalog pipeline/stage2b/concepts.yaml \
  --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json \
  --clips benchmark/development_clip_manifest.json \
  --annotations benchmark/tactical_windows.jsonl \
  --purpose development
```

Expected: exit `0`; duplicate sample IDs, invalid actor roles, missing windows, or held-out match leakage fail loudly.

- [ ] **Step 5: Run the frozen feasibility rules**

```bash
python scripts/tactical_kb_eval.py audit \
  --catalog pipeline/stage2b/concepts.yaml \
  --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json \
  --clips benchmark/development_clip_manifest.json \
  --annotations benchmark/tactical_windows.jsonl \
  --out benchmark/feasibility_report.json
```

The deterministic report records two feasibility lists:

1. Put every concept meeting 6 positives, 6 targeted hard negatives, and two eligible development matches into `feasibility_passed` (zero to six concepts).
2. Keep every other candidate in `feasibility_failed` with explicit reasons.
3. Report GSR observability, hard-negative separability, track/ball/calibration quality, and GT-to-GSR degradation as decision inputs.
4. Do not write recipe thresholds in this task.

- [ ] **Step 6: Freeze a separate zero-to-three recipe shortlist**

Review only concepts in `feasibility_passed` and create
`benchmark/recipe_shortlist.json`. It requires schema version
`tactical-recipe-shortlist-v1`, the computed SHA-256 digest of
`benchmark/feasibility_report.json` encoded as `sha256:` followed by 64
lowercase hexadecimal characters, a `concept_ids` array, and a
`decisions` array.

The saved file contains zero to three real concept IDs. Each decision records
the selected or not-selected concept, observability/quality rationale, and
reviewer/date. The user or designated football reviewer explicitly approves
the shortlist after seeing the deterministic report; the CLI does not invent a
mixed-metric score or hidden tie-break. Selecting zero is valid. Do not enforce
category diversity and do not lower a semantic boundary to fill slots. The
shortlist does not enable production and contains no recipe parameter.

```bash
python scripts/tactical_kb_eval.py validate-shortlist \
  --report benchmark/feasibility_report.json \
  --shortlist benchmark/recipe_shortlist.json
```

- [ ] **Step 7: Record development calibration provenance without enabling production**

Add two complete calibration `SourceRecord` entries because one record has one
`content_hash`:

- `soccermaster_dev_windows_v1`, whose hash is the SHA-256 of `benchmark/tactical_windows.jsonl`;
- `soccermaster_feasibility_report_v1`, whose hash is the SHA-256 of `benchmark/feasibility_report.json`.

For every concept in `feasibility_passed`:

- reference both calibration source IDs;
- set `maturity: audited`;
- keep `production_enabled: false`;
- do not add a production recipe yet.

Concepts in `feasibility_failed` remain `planned` and disabled. Concepts in
`feasibility_passed` but outside `recipe_shortlist` remain `audited` and
disabled; they simply receive no checker work in the next plan.

- [ ] **Step 8: Add regression assertions and fixtures for the real audit result**

Extend tests to assert:

- every audited concept appears in `feasibility_passed`, and every feasibility-passed concept is audited;
- `recipe_shortlist` contains no more than three feasibility-passed concepts;
- all 14 concepts remain `production_enabled: false`;
- no games 8/11 appear in the development annotation file;
- calibration hashes and source references resolve.

For every concept in `feasibility_passed`, create one JSON file named after its
canonical ID below `tests/fixtures/tactical_kb/`, containing references to exactly
two audited positives and two targeted hard negatives from train/valid. Include
sample ID, label, window, canonical team, source-specific actor bindings, and
hard-negative type, but do not copy video frames or raw tracking into Git.
These fixtures lock annotation/schema regressions only; the follow-up recipe
plan adds synthetic measurement cases for checker behavior. Add assertions that
fixtures resolve to the development annotation file and never to held-out data.

Run:

```bash
python -m pytest -q tests/test_stage2b_kb.py tests/test_tactical_kb_eval.py
```

- [ ] **Step 9: Commit only annotations, reports, catalog status, and regression assertions**

```bash
git add \
  benchmark/development_clip_manifest.json \
  benchmark/tactical_windows.jsonl \
  benchmark/feasibility_report.json \
  benchmark/recipe_shortlist.json \
  pipeline/stage2b/concepts.yaml \
  tests/fixtures/tactical_kb \
  tests/test_stage2b_kb.py \
  tests/test_tactical_kb_eval.py
git commit -m "data: freeze tactical feasibility audit"
```

Do not stage videos, images, predictions, or dataset labels.

---

### Task 9: Verify the foundation and hand off to a data-informed promotion plan

**Files:** No new production files.

**Success criteria:** All contracts and deterministic checks pass; runtime contains no model-authored verification; all concepts remain disabled; the feasibility report is sufficient to write exact checker tasks for only the recipe shortlist.

- [ ] **Step 1: Run the complete deterministic suite**

```bash
python -m pytest -q tests
python -m compileall -q pipeline scripts
python -m pipeline.stage2b.run --help
bash -n scripts/run_stage1.sh
bash -n scripts/run_stage2b.sh
git diff --check
```

- [ ] **Step 2: Re-run data and annotation gates**

```bash
python scripts/validate_soccernetgs.py \
  --root codes/sn-gamestate/datasets/SoccerNetGS \
  --splits train valid \
  --expect train=57 \
  --expect valid=58
python scripts/tactical_kb_eval.py validate-annotations \
  --catalog pipeline/stage2b/concepts.yaml \
  --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json \
  --clips benchmark/development_clip_manifest.json \
  --annotations benchmark/tactical_windows.jsonl \
  --purpose development
python scripts/tactical_kb_eval.py audit \
  --catalog pipeline/stage2b/concepts.yaml \
  --dataset-manifest benchmark/manifests/soccernetgs_v1_3_integrity.json \
  --clips benchmark/development_clip_manifest.json \
  --annotations benchmark/tactical_windows.jsonl \
  --out /tmp/feasibility-report-rerun.json
diff -u benchmark/feasibility_report.json /tmp/feasibility-report-rerun.json
python scripts/tactical_kb_eval.py validate-shortlist \
  --report benchmark/feasibility_report.json \
  --shortlist benchmark/recipe_shortlist.json
```

The report must exclude volatile timestamps or normalize them so the rerun is byte-reproducible.

- [ ] **Step 3: Audit the runtime boundary**

```bash
rg -n "verify_candidates|evidence_queries|predicate_passes|tactical_candidates|verified.?[:=].?true" \
  pipeline/stage2b pipeline/config.py
rg -n "production_enabled: true" pipeline/stage2b/concepts.yaml
rg -n '"match_id"\s*:\s*"?(8|11)"?' benchmark/tactical_windows.jsonl
```

Expected: all three searches return no matches.

- [ ] **Step 4: Review scope and dirty state**

```bash
git status --short
git diff --stat
git log --oneline --max-count=10
```

Confirm that no dataset contents, outputs, models, papers, unrelated docs, or held-out annotations are staged.

- [ ] **Step 5: Write the follow-up plan only from the frozen audit result**

Create a separate plan named:

```text
docs/superpowers/plans/2026-07-11-tactical-kb-production-promotion.md
```

That later plan must name the actual zero to three concepts in `recipe_shortlist` and, for each shortlisted concept only:

- implement the smallest required 25 Hz relation/topology primitive;
- implement one named fixed checker and versioned recipe;
- calibrate global parameters on development data;
- add 2 positive + 2 targeted-hard-negative CI fixtures;
- add a frozen-manifest evaluation runner that calls `generate_tactical_artifacts` separately for GT labels and GSR predictions using only manifest-listed recipe IDs, never a free CLI override;
- preserve `clip_id`, `state_source`, normalized proposal, actor bindings, and all rejection/unsupported reasons in per-clip proposal/fact result envelopes;
- create explicit full-clip duration/evaluable/non-evaluable masks and exhaustive-per-concept flags; unlabeled time is negative only when the mask says annotation is exhaustive;
- freeze the release manifest, including prompt/public projection versions, matching policy, gates, and model/decoding configuration;
- require at least four positives and four targeted hard negatives across held-out games 8 and 11, with both matches represented, or mark the concept `not_evaluable`;
- only then annotate/open and run full-clip held-out games 8 and 11 once;
- score every retained proposal/fact and all evaluable minutes, including proposal recall, conditional verifier precision, end-to-end recall/precision, minimum true positives, false insertions/minute, GT-to-GSR degradation, and structured actor/time grounding;
- score language overclaim separately from structured-fact scoring using retained commentary audit output;
- enable only concepts that pass; never backfill a failed concept.

If `benchmark/recipe_shortlist.json` contains zero concepts, the follow-up plan contains no checker work and records a catalog-only KB v2 release.

## Final acceptance criteria

- The canonical data root remains `codes/sn-gamestate/datasets/SoccerNetGS`.
- Repaired train=57 and valid=58 pass the version-1.3 integrity validator; test is byte-unchanged.
- The catalog contains 14 normalized concepts, all disabled by default.
- Source records are complete and grouped by terminology/formalization/calibration responsibility.
- `tactical_state` is concept-neutral, uses canonical team IDs, and preserves 25 Hz measurements.
- Doubao receives only a whitelisted public catalog and at-most-12.5 Hz proposal projection.
- Unknown proposal fields are rejected and logged; model-authored predicates/thresholds cannot execute.
- Facts are rebuilt only through a named private checker and fixed recipe contract.
- Stage 2B writes `tactical_state.json`, `tactical_proposals.json`, and `verified_tactical_facts.json`; commentary references facts only.
- No evidence window is limited by the five-second commentary slot.
- Feasibility uses repaired train/valid, at least 6 positives + 6 targeted hard negatives across at least two development matches per eligible concept, with paired GT/GSR observability.
- Game 7 is diagnostic; games 8 and 11 remain unopened.
- Zero to six concepts may become `audited`; `recipe_shortlist` contains at most three; none becomes production-enabled in this plan.
- The next production plan is written only after the frozen feasibility report and explicitly approved recipe shortlist identify the actual checker candidates.
