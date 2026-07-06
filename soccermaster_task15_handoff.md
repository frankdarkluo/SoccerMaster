# SoccerMaster Task 15 Handoff

## Suggested Skills
- `karpathy-guidelines`: keep changes surgical and evidence-based.
- `superpowers:subagent-driven-development`: the active plan requires task-by-task execution with implementer + reviewer.
- `superpowers:test-driven-development`: Task 15 still needs real acceptance evidence before commit.

## Current State
- Repo: `/Users/gluo/Desktop/SoccerMaster`
- Active plan: `docs/superpowers/plans/2026-07-05-stage2-events-rewrite.md`
- Completed through Task 14.
- Current HEAD: `e2f2a07 feat(stage2): wire detect->verify->compose->enrich flow + backend selection`
- Branch state before handoff: `main...origin/main [ahead 15]`
- Do **not** start Task 16. Continue Task 15 only.

## Task 15 Blocker
Task 15 acceptance currently fails because local `outputs/SNGS-148/predictions.json` has no ball detections:

- images: `750`
- annotations: `11020`
- category counts: `{1: 8645, 2: 457, 3: 1918}`
- role counts: `{'player': 8645, 'referee': 1918, 'goalkeeper': 457}`
- `category_id=4`: `0`
- `attributes.role == "ball"`: `0`

Current failure:

```bash
python3 -m pytest tests/test_stage2_acceptance.py -v
```

fails at:

```text
assert n_pass > 0, "still detecting zero passes"
```

The detector is ball-dependent:
- `pipeline/stage2_events/detector.py::load_frames` only sets `FrameData.ball_xy` from role `ball`.
- `pipeline/stage2_events/possession.py::possession_segments` requires `frame.ball_xy`.
- With no ball annotations, rules-only output is deterministically `0` events.

Constant tuning cannot fix this. It only changes thresholds after ball positions exist.

## Uncommitted Local State
- `tests/test_stage2_acceptance.py` exists untracked and contains the Task 15 acceptance test from the plan.
- `outputs/SNGS-148/events.json` and `outputs/SNGS-148/events_detected.json` were regenerated as 0-event files, but `outputs/` is git-ignored.
- No Task 15 commit was created.

## What Cursor Should Do Next
Cursor has remote server access and the remote server reportedly has the missing `.pklz` files. The next valid move is to regenerate SNGS-148 `predictions.json` and `homography_per_frame.json` from a real `.pklz` that contains ball detections.

Use the repo converter directly. It has no CLI. Example command from repo root:

```bash
python3 - <<'PY'
from pathlib import Path
from pipeline.stage1_inference.pklz_to_json import convert_pklz_to_json

pklz = Path("/ABS/PATH/TO/SNGS-148.pklz")  # replace with real remote path
convert_pklz_to_json(
    pklz_path=pklz,
    video_id="148",
    output_dir=Path("outputs/SNGS-148"),
    fps=25,
    sequence_name="SNGS-148",
)
print("converted", pklz)
PY
```

The `.pklz` must contain entries named:

```text
148.pkl
148_image.pkl
```

If the entries use a different video id, inspect with:

```bash
python3 - <<'PY'
import zipfile
from pathlib import Path
p = Path("/ABS/PATH/TO/SNGS-148.pklz")
with zipfile.ZipFile(p) as z:
    print("\n".join(z.namelist()[:50]))
PY
```

Then pass the matching `video_id` to `convert_pklz_to_json`.

## Validate The Regenerated Predictions
After conversion, verify ball annotations exist:

```bash
python3 - <<'PY'
import json
from collections import Counter
p = "outputs/SNGS-148/predictions.json"
data = json.load(open(p))
cat = Counter(a.get("category_id") for a in data.get("annotations", []))
roles = Counter((a.get("attributes") or {}).get("role") for a in data.get("annotations", []))
print("images", len(data.get("images", [])))
print("annotations", len(data.get("annotations", [])))
print("category_counts", dict(cat))
print("role_counts", dict(roles))
assert cat.get(4, 0) > 0 or roles.get("ball", 0) > 0, "still no ball annotations"
PY
```

## Continue Task 15
Run the plan’s Task 15 acceptance:

```bash
python3 -m pytest tests/test_stage2_acceptance.py -v
```

If it fails only on smell-test thresholds, the plan allows minimal tuning of:
- `POSSESSION_MIN_FRAMES` in `pipeline/stage2_events/possession.py`
- `GOAL_LINE_TOL_M` in `pipeline/stage2_events/detector.py`

Do not redesign detector logic unless the user explicitly changes scope.

Then refresh artifacts:

```bash
python3 -c "
from pipeline.stage2_events.detector import EventDetector, dedup_events, compose_assists, load_frames, write_events_json
from pipeline.stage2_events.enricher import enrich_events
from pipeline.stage2_events.schema import EventSchema
PRED='outputs/SNGS-148/predictions.json'
frames=load_frames(PRED)
raw=EventDetector(EventSchema(),25).detect(PRED)
write_events_json(raw,'outputs/SNGS-148/events_detected.json',{'source':'SNGS-148','fps':25})
final=enrich_events(dedup_events(compose_assists(raw)),frames)
write_events_json(final,'outputs/SNGS-148/events.json',{'source':'SNGS-148','fps':25})
print('passes:',sum(e.event_code=='football.pass' for e in final),'ints:',sum(e.event_code=='football.interception' for e in final))
"
```

Optional real VLM smoke can be skipped unless API credentials are available.

## Commit
Because `outputs/` is git-ignored, force-add the required Task 15 artifacts:

```bash
git add tests/test_stage2_acceptance.py
git add -f outputs/SNGS-148/events.json outputs/SNGS-148/events_detected.json
git commit -m "test(stage2): SNGS-148 acceptance smell-test + refreshed artifacts"
```

If the regenerated `predictions.json` must also travel with the repo for reproducibility, explicitly discuss that because it is currently ignored by `.gitignore` and Task 15’s commit list did not include it.

## Acceptance Criteria For Cursor
- `python3 -m pytest tests/test_stage2_acceptance.py -v` passes.
- Generated output prints `passes: <positive>` and a small/interception-not-dominant count.
- `outputs/SNGS-148/events.json` contains a `football.goal` around `24.0 <= timestamp_s <= 27.0` and non-null `player_jersey`.
- One commit only for Task 15 with the exact message above.
- Stop after Task 15 and report; do not proceed to Task 16.
