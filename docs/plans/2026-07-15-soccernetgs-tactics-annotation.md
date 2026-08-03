# SoccerNetGS Tactical Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a human-review candidate `tactics.csv` covering all 200 SoccerNetGS clips, after a 20-clip calibration gate, using video-first and GameState-assisted tactical annotation.

**Architecture:** Annotation is a two-pass evidence workflow rather than an automated classifier. Every clip receives a 0.5-second full-timeline review, candidate windows receive an 8–25 FPS dense review, and JSON is consulted only for identity and geometry. A temporary 20-clip calibration CSV is user-reviewed before the same rubric is applied to the remaining 180 clips.

**Tech Stack:** FFmpeg/ffprobe, Python 3 standard-library CSV/JSON parsing, SoccerNetGS MP4 and `Labels-GameState.json`, Markdown design/plan documents.

## Global Constraints

- Primary definition source: `足球战术数据库_词条表_Grid.csv`.
- Design contract: `docs/superpowers/specs/2026-07-15-soccernetgs-tactics-annotation-design.md`.
- Input scope: exactly `SNGS-001.mp4` through `SNGS-200.mp4` in `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/`.
- Final columns, in order: `id,tactics,时间范围,判断依据,置信度`.
- Positive labels require confidence `中`, `中高`, or `高`.
- `无明确战术` is an audit marker, not a negative training example.
- Video action is primary evidence; JSON may verify identity and geometry but may not invent action.
- Preserve every unrelated working-tree change.
- Do not modify MP4 files, `Labels-GameState.json`, or the tactical source CSV.
- Do not run model-training experiments.
- Do not commit during the plan; commit only after the complete annotation plan is finished and only if the user requests it.

---

### Task 1: Preflight the 200-video corpus and freeze the evidence contract

**Files:**
- Read: `足球战术数据库_词条表_Grid.csv`
- Read: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/SNGS-001.mp4` through `SNGS-200.mp4`
- Read: `codes/sn-gamestate/datasets/SoccerNetGS/{train,valid,test}/SNGS-*/Labels-GameState.json`
- Read: `docs/superpowers/specs/2026-07-15-soccernetgs-tactics-annotation-design.md`

**Interfaces:**
- Consumes: approved design and existing local media/metadata.
- Produces: proof that every required video is present and decodable; the fixed 42-label vocabulary plus `无明确战术`.

- [ ] **Step 1: Verify filename coverage**

Run:

```bash
find codes/sn-gamestate/datasets/SoccerNetGS/all_videos -maxdepth 1 -type f -name 'SNGS-*.mp4' -printf '%f\n' | sort -V | wc -l
```

Expected: `200`.

Run:

```bash
python -c "from pathlib import Path; root=Path('codes/sn-gamestate/datasets/SoccerNetGS/all_videos'); actual={p.stem for p in root.glob('SNGS-*.mp4')}; expected={f'SNGS-{i:03d}' for i in range(1,201)}; assert actual==expected, (sorted(expected-actual),sorted(actual-expected)); print('PASS: IDs 001-200')"
```

Expected: `PASS: IDs 001-200`.

- [ ] **Step 2: Verify all videos decode and have the expected basic stream**

Run:

```bash
for f in codes/sn-gamestate/datasets/SoccerNetGS/all_videos/SNGS-*.mp4; do ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,duration -of csv=p=0 "$f" >/dev/null || exit 1; done
```

Expected: exit code `0` with no undecodable file.

Run:

```bash
python -c "from pathlib import Path; import subprocess; fs=sorted(Path('codes/sn-gamestate/datasets/SoccerNetGS/all_videos').glob('SNGS-*.mp4')); ds=[float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(f)],text=True)) for f in fs]; assert len(ds)==200 and all(29.9<=d<=30.1 for d in ds), (min(ds),max(ds)); print(f'PASS: {len(ds)} videos, duration {min(ds):.3f}-{max(ds):.3f}s')"
```

Expected: a `PASS` line reporting 200 videos near 30 seconds.

- [ ] **Step 3: Verify auxiliary annotation coverage**

Run:

```bash
find codes/sn-gamestate/datasets/SoccerNetGS -mindepth 3 -maxdepth 3 -name Labels-GameState.json | wc -l
```

Expected: `164`; the remaining 36 are the approved challenge clips.

- [ ] **Step 4: Verify the source vocabulary count and approved additions**

Run:

```bash
python -c "import csv; rows=csv.DictReader(open('足球战术数据库_词条表_Grid.csv',encoding='utf-8-sig')); names={(r.get('中文名') or '').strip() for r in rows}; names={n for n in names if n and not n.startswith('中文名，')}; extras={'打身后 / 反越位跑位','二过一 / 撞墙配合','第三人跑动','边后卫内收','假九号回撤','中后卫带球推进','前场交叉换位','长传冲吊'}; assert len(names)==34 and len(names|extras)==42; print('PASS: 34 CSV + 8 approved labels')"
```

Expected: `PASS: 34 CSV + 8 approved labels`.

### Task 2: Annotate and validate the 20-clip calibration set

**Files:**
- Create: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics_calibration.csv`
- Read: the 20 calibration MP4s listed below
- Read when present: matching `Labels-GameState.json`
- Temporary only: `/tmp/soccernetgs-tactics-calibration/`

**Interfaces:**
- Consumes: Task 1 preflight and the design's two-pass rubric.
- Produces: a five-column calibration CSV representing exactly the 20 approved IDs.

- [ ] **Step 1: Create time-coded overview sheets outside the repository output**

Calibration IDs:

```text
001 006 011 016 020 021 030 041 050 059
061 062 075 104 115 117 118 130 190 200
```

For each ID, generate three 10-second sheets, each sampled at 2 FPS. Run the following with `ID` set successively to every value above:

```bash
ID=001
mkdir -p /tmp/soccernetgs-tactics-calibration/SNGS-$ID
for START in 0 10 20; do END=$((START+10)); ffmpeg -y -hide_banner -loglevel error -i codes/sn-gamestate/datasets/SoccerNetGS/all_videos/SNGS-$ID.mp4 -vf "trim=start=${START}:end=${END},fps=2,scale=320:-2,drawtext=text='%{pts\\:hms}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65,tile=5x4:padding=3:margin=3" -q:v 12 -frames:v 1 /tmp/soccernetgs-tactics-calibration/SNGS-$ID/$START-$END.jpg; done
```

Expected per clip: `0-10.jpg`, `10-20.jpg`, and `20-30.jpg`.

- [ ] **Step 2: Perform the full-timeline pass for all 20 clips**

Inspect the three sheets for every clip in chronological order. Record possession changes, attacking direction, candidate label, provisional start/end seconds, and missing hard evidence. Do not assign a label from the event class alone.

Expected: every calibration ID has either at least one candidate window or an explicit conclusion that no allowed label reaches medium confidence.

- [ ] **Step 3: Perform dense review of every candidate window**

For each provisional window, generate an 8 FPS sheet covering the candidate plus one second of context on each side. If a touch or ordering remains ambiguous, inspect the original 25 FPS neighborhood. Consult JSON only to bind team/jersey/track or verify geometry.

The dense review must explicitly check the source confusion boundary. In particular:

- cutback versus ordinary cross;
- high press versus counter-press;
- switch versus long ball;
- line-breaking pass versus ordinary forward pass;
- stable block versus temporary retreat;
- confirmed role for fullback, centre-back, and central striker labels.

Expected: every retained positive candidate has confidence at least `中` and a concrete exclusion statement.

- [ ] **Step 4: Write the calibration CSV**

Create `tactics_calibration.csv` with exactly:

```csv
id,tactics,时间范围,判断依据,置信度
```

Write one occurrence per row. Use one `无明确战术` row only when an ID has no positive row. Sort by numeric ID, start time, and tactic name.

Expected: all 20 calibration IDs appear at least once and no unapproved ID appears.

- [ ] **Step 5: Validate the calibration CSV mechanically**

Run:

```bash
python - <<'PY'
import csv
import re
from pathlib import Path

path = Path('codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics_calibration.csv')
rows = list(csv.DictReader(path.open(encoding='utf-8')))
expected_header = ['id', 'tactics', '时间范围', '判断依据', '置信度']
assert list(rows[0]) == expected_header
expected_ids = {f'SNGS-{i:03d}' for i in (1,6,11,16,20,21,30,41,50,59,61,62,75,104,115,117,118,130,190,200)}
assert {r['id'] for r in rows} == expected_ids
assert len(rows) == len({tuple(r[h] for h in expected_header) for r in rows})
assert all(r['置信度'] in {'高','中高','中','低'} for r in rows)
pat = re.compile(r'^00:(\d{2})–00:(\d{2})$')
for row in rows:
    match = pat.fullmatch(row['时间范围'])
    assert match, row
    start, end = map(int, match.groups())
    assert 0 <= start < end <= 30, row
    if row['tactics'] != '无明确战术':
        assert row['置信度'] in {'高','中高','中'}, row
print(f'PASS: {len(rows)} calibration rows, 20 IDs')
PY
```

Expected: `PASS: ... calibration rows, 20 IDs`.

- [ ] **Step 6: Re-review calibration risk rows**

Re-open every row with `中` confidence, every identity-sensitive label, and every confusion-prone label. Correct or remove any row whose hard evidence cannot be restated from the dense frames.

Expected: calibration CSV still passes Step 5 after corrections.

### Task 3: Obtain the calibration decision

**Files:**
- Review: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics_calibration.csv`

**Interfaces:**
- Consumes: validated 20-clip calibration artifact.
- Produces: explicit user approval or exact corrections to apply before full annotation.

- [ ] **Step 1: Hand the calibration file to the user**

Report the row count, label distribution, `无明确战术` count, and all medium-confidence rows. Ask the user to review the 20 clips and approve or correct the rubric.

Expected: explicit approval before Task 4.

- [ ] **Step 2: Apply user corrections and rerun validation**

Apply every requested row, timing, evidence, or confidence correction, then rerun Task 2 Step 5.

Expected: validation passes and the user-approved calibration rows become the seed of the final file.

### Task 4: Annotate the remaining 180 clips

**Files:**
- Create and incrementally complete: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics.csv`
- Read: all remaining MP4s and available GameState JSON
- Temporary only: `/tmp/soccernetgs-tactics-full/`

**Interfaces:**
- Consumes: user-approved calibration rows and unchanged annotation rubric.
- Produces: sorted candidate annotations covering every ID from 001 through 200.

- [ ] **Step 1: Seed the final CSV from approved calibration judgments**

Create `tactics.csv` with the same five-column header and copy the approved calibration rows into their sorted positions.

Expected: the file initially represents exactly the 20 calibration IDs.

- [ ] **Step 2: Process all non-calibration clips in ascending ID batches**

Use batches `001–020`, `021–040`, `041–060`, `061–080`, `081–100`, `101–120`, `121–140`, `141–160`, `161–180`, and `181–200`. Skip the 20 calibration IDs already approved.

For each remaining clip, repeat Task 2 Steps 1–3 using `/tmp/soccernetgs-tactics-full/SNGS-NNN/`, then add either all medium-or-higher positive occurrences or exactly one `无明确战术` audit row.

Expected after each batch: every processed ID appears at least once, the file remains sorted, and no unprocessed ID is represented by a fabricated placeholder.

- [ ] **Step 3: Run batch-level structural checks**

After each 20-ID batch, parse the CSV and verify unique-ID coverage for all IDs processed so far, allowed confidence values, valid time ranges, and no exact duplicates.

Expected: every batch check passes before moving to the next batch.

- [ ] **Step 4: Complete the 200-ID coverage pass**

Confirm that all 200 IDs are represented and that each clip with positive rows has no `无明确战术` row.

Expected: `len({row['id']}) == 200`.

### Task 5: Perform the final tactical audit and handoff

**Files:**
- Modify: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics.csv`
- Delete after successful final validation: `codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics_calibration.csv`

**Interfaces:**
- Consumes: complete 200-ID candidate CSV.
- Produces: verified final CSV ready for the user's full human review.

- [ ] **Step 1: Re-review all high-risk rows**

Dense-review every `中` row, all cutbacks, high presses, counter-presses, switches, long balls, line-breaking passes, low blocks, and identity-sensitive added labels. Remove or downgrade any row that fails a hard condition; if that leaves a clip with no positive row, replace its positives with one `无明确战术` row.

Expected: every surviving positive row has independently restatable visual evidence.

- [ ] **Step 2: Run the final validator**

Run:

```bash
python - <<'PY'
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

root = Path('.')
source = root / '足球战术数据库_词条表_Grid.csv'
target = root / 'codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics.csv'

with source.open(encoding='utf-8-sig') as handle:
    source_names = {(r.get('中文名') or '').strip() for r in csv.DictReader(handle)}
source_names = {n for n in source_names if n and not n.startswith('中文名，')}
extras = {
    '打身后 / 反越位跑位', '二过一 / 撞墙配合', '第三人跑动', '边后卫内收',
    '假九号回撤', '中后卫带球推进', '前场交叉换位', '长传冲吊',
}
allowed = source_names | extras | {'无明确战术'}
header = ['id', 'tactics', '时间范围', '判断依据', '置信度']

with target.open(encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == header, reader.fieldnames
    rows = list(reader)

expected_ids = {f'SNGS-{i:03d}' for i in range(1, 201)}
assert {r['id'] for r in rows} == expected_ids
assert all(r['tactics'] in allowed for r in rows)
assert all(r['置信度'] in {'高','中高','中','低'} for r in rows)
assert len(rows) == len({tuple(r[h] for h in header) for r in rows})

time_pat = re.compile(r'^00:(\d{2})–00:(\d{2})$')
start_by_row = []
by_id = defaultdict(list)
for row in rows:
    match = time_pat.fullmatch(row['时间范围'])
    assert match, row
    start, end = map(int, match.groups())
    assert 0 <= start < end <= 30, row
    if row['tactics'] != '无明确战术':
        assert row['置信度'] in {'高','中高','中'}, row
    if row['tactics'] in {'边后卫内收','假九号回撤','中后卫带球推进'}:
        assert re.search(r'\d+号', row['判断依据']), row
    start_by_row.append(start)
    by_id[row['id']].append(row)

for clip_id, clip_rows in by_id.items():
    none_rows = [r for r in clip_rows if r['tactics'] == '无明确战术']
    positive_rows = [r for r in clip_rows if r['tactics'] != '无明确战术']
    assert not (none_rows and positive_rows), clip_id
    if none_rows:
        assert len(none_rows) == 1 and none_rows[0]['时间范围'] == '00:00–00:30', clip_id

def sort_key(row):
    start = int(time_pat.fullmatch(row['时间范围']).group(1))
    return int(row['id'][-3:]), start, row['tactics']

assert rows == sorted(rows, key=sort_key)
print(f'PASS: {len(rows)} rows, 200 IDs, {len(allowed)-1} tactical labels allowed')
print('confidence:', dict(Counter(r['置信度'] for r in rows)))
print('top labels:', Counter(r['tactics'] for r in rows).most_common(10))
PY
```

Expected: a `PASS` line reporting 200 IDs, followed by confidence and label summaries.

- [ ] **Step 3: Remove the calibration artifact and verify repository scope**

Delete `tactics_calibration.csv` only after Step 2 passes. Run:

```bash
git status --short -- docs/superpowers/specs/2026-07-15-soccernetgs-tactics-annotation-design.md docs/superpowers/plans/2026-07-15-soccernetgs-tactics-annotation.md codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics.csv codes/sn-gamestate/datasets/SoccerNetGS/all_videos/tactics_calibration.csv
```

Expected: the design, plan, and final `tactics.csv` are the only task-scoped additions; the calibration file is absent.

- [ ] **Step 4: Report the handoff**

Report the final path, row count, unique-ID count, confidence distribution, tactic distribution, number of `无明确战术` audit markers, validation command result, and reminder that the user will conduct final human review.

Expected: no claim that the CSV is ground truth or that `无明确战术` rows are negative samples.
