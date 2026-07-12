# AI Football Commentary Pipeline — Design Spec

**Date:** 2026-07-01
**Status:** Draft
**Authors:** luoguoqing + AI assistant

---

## 1. Problem Statement

Given a 30-second broadcast-camera raw video of a soccer match, produce:

1. **Timestamped commentary** (Chinese + English) with event–timestamp alignment ≤ ±1 second
2. **Light-beam visual effects** on the original video highlighting key players during critical events (e.g., shots)
3. **Structured event data** following the `reference_football.csv` event ontology

The pipeline must support multiple LLM backends (local Qwen2.5-7B-VL, Doubao API, GPT-5 API) and reuse existing SoccerMaster infrastructure wherever possible.

---

## 2. Architecture Overview

**Approach: Staged Pipeline with Persistent Intermediate Artifacts**

Each stage is an independent module that reads/writes files. Stages can be run individually or chained via an orchestrator. The pipeline supports **flexible entry points** — any stage can be skipped if its output artifacts already exist.

```
clip_dir/img1/*.jpg  (raw frames)
    │
    ▼
[Stage 1: SoccerMaster Inference]          ← skip if predictions.json / Labels-GameState.json exists
    │  Output: predictions.json + homography_per_frame.json
    ▼
[Stage 2: Event Detection (rule engine)]   ← skip if events.json exists
    │  Input: predictions.json + reference_football.csv
    │  Output: events.json
    ▼
┌──────────────────────────────────────────┐
│  Stage 3 and Stage 4 run in parallel     │
├──────────────────────────────────────────┤
│ [Stage 3: Light-Beam Effects]            │
│   Input: img1/ + events.json +           │
│          predictions.json + homography   │
│   Output: annotated_video.mp4            │
│                                          │
│ [Stage 4: LLM Commentary Generation]    │
│   Input: frames/video + events.json +    │
│          topo.json                        │
│   Output: commentary.json                │
└──────────────────────────────────────────┘
```

### 2.1 Two Development Tracks (Parallel)

The pipeline supports two input tracks that can be developed and tested in parallel:

| Track | Input Source | Available Data | Stages to Run |
|-------|-------------|----------------|---------------|
| **Track 1: Test clips** | `datasets/SoccerNetGS/test/SNGS-XXX/` | Raw frames `img1/`, GT labels `Labels-GameState.json`, `clip_index.csv` events (validation) | Skip Stage 1; Stage 2→3→4 |
| **Track 2: Processed clips** | `datasets/SoccerNetGS/train/SNGS-061/` + `outputs/gsr/step_3_train_SNGS-061/` | Raw frames `img1/`, `sn-gamestate.pklz` (model predictions) | Stage 1 (pklz→JSON only)→2→3→4 |

- **Track 1** uses `Labels-GameState.json` directly as `predictions.json` (zero conversion — same format).
- **Track 2** requires `pklz_to_json.py` to convert `sn-gamestate.pklz` into `predictions.json` + `homography_per_frame.json`, then runs Stage 2→3→4.
- Both tracks render Stage 3 light beams on **raw frames** (`img1/`), not visualization videos.

### 2.2 `clip_index.csv` as Validation Benchmark

`clip_index.csv` (`test/clip_index.csv`) provides one annotated event per test clip (sample_id, event_time_sec, normalized_action). It is used **only for validation**, not as production input:

- Stage 2 runs independently to detect all events
- `clip_index.csv` validates that the detected key event matches ground truth (timestamp error ≤ ±1s)
- During early development, `clip_index.csv` events can serve as **quick fixture data** for bootstrapping Stage 3/4 before Stage 2 is fully built

Event time conversion: `event_time_sec` values with decimals are rounded to the nearest integer second (e.g., 6.293 → 6s, 6.876 → 7s).

---

## 3. Code Structure

```
pipeline/
├── __init__.py
├── run.py                        # Orchestrator: chains Stage 1→4
├── config.py                     # PipelineConfig (unified input model, see Section 8)
│
├── stage1_inference/             # Raw video → structured predictions
│   ├── __init__.py
│   ├── preprocess.py             # raw video → ffmpeg extract frames → GSR directory layout
│   ├── run_gsr.py                # Wraps SoccerMaster Steps 1-3 shell invocations
│   └── pklz_to_json.py           # pklz → Labels-GameState-compatible JSON + homography export
│
├── stage2_events/                # Event detection (rule-based)
│   ├── __init__.py
│   ├── schema.py                 # Load reference_football.csv → EventSchema registry
│   ├── detector.py               # Rule engine: ball velocity, possession, shot detection, etc.
│   └── enricher.py               # Add tag dimensions (pitch_zone, pass_direction, shot_distance)
│
├── stage3_effects/               # Visual effects on raw frames
│   ├── __init__.py
│   ├── light_beam.py             # OpenCV light-beam rendering (semi-transparent + Gaussian glow)
│   └── render.py                 # Overlay effects onto raw frames → annotated MP4
│
├── stage4_commentary/            # LLM-based commentary generation
│   ├── __init__.py
│   ├── prompt_builder.py         # Build prompt from events.json + topo.json + visual input
│   ├── llm_adapter.py            # Abstract LLMAdapter interface
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── qwen_local.py         # Local Qwen2.5-7B-VL-Instruct (video input supported)
│   │   ├── doubao_api.py         # Doubao vision API (video input supported)
│   │   └── openai_api.py         # GPT-5 API (fallback to per-second frame images)
│   └── postprocess.py            # Format LLM output → commentary.json
│
└── utils/
    ├── __init__.py
    ├── pitch.py                  # Re-export from formation_topology.pitch
    ├── video.py                  # Video I/O helpers (frame extraction, MP4 encoding)
    └── clip_index_to_events.py   # Convert clip_index.csv row → events.json fixture format
```

### Reuse from Existing Codebase

| New module | Reuses from |
|---|---|
| `pklz_to_json.py` | `compare_step3_pred_gt.py` pklz loading logic |
| `utils/pitch.py` | `formation_topology.pitch` constants + `canonicalize()` |
| `stage2_events/detector.py` | `formation_topology.possession` for nearest-player-to-ball |
| `stage4_commentary/prompt_builder.py` | `formation_topology.pipeline.analyze()` for topology context |
| `run_gsr.py` | `codes/sn-gamestate/run_train_SNGS-061.sh` invocation pattern |
| `stage3_effects/render.py` | `formation_topology.viz_topdown` rendering primitives |

---

## 4. Stage 1: SoccerMaster Inference

### 4.1 Preprocessing (`preprocess.py`)

Input: `raw_video.mp4`
Output: GSR-format directory under `codes/sn-gamestate/datasets/SoccerNetGS/sn500/`

Steps:
1. Extract frames via ffmpeg: `ffmpeg -i video.mp4 -q:v 2 img1/%06d.jpg`
2. Create `sequences_info.json` with frame count
3. Return sequence name (e.g., `SNGS-10001`)

### 4.2 GSR Pipeline (`run_gsr.py`)

Invokes SoccerMaster Steps 1→2→3 sequentially via subprocess calls:

```
Step 1: python -m tracklab.main -cn gsr_step_1_example
Step 2: bash gsr_step2_example.sh && bash merge_example.sh
Step 3: python -m tracklab.main -cn gsr_step_3_example_accelerate
```

Uses the accelerated Step 3 config (Qwen2.5-VL-7B for both JN and role) for speed.

### 4.3 pklz → JSON Converter (`pklz_to_json.py`)

Input: `sn-gamestate.pklz` from Step 3
Output:
- `predictions.json` — Labels-GameState-compatible format with `images` + `annotations` keys
- `homography_per_frame.json` — per-frame homography matrix `H` (image→pitch) and `H_inv` (pitch→image)

The converter reads the pklz via `zipfile` + `pickle` (same approach as `compare_step3_pred_gt.py`), extracts:
- `bbox_pitch` (x, y in meters)
- `bbox_image` (bounding box in pixel space)
- `role`, `team`, `jersey_number`, `track_id`
- Camera parameters / homography from image metadatas

**Output schema for `predictions.json`** — **fully compatible** with `Labels-GameState.json` format so that:
1. Downstream tools (`formation_topology`, `pitch_distances`) can consume it directly
2. Ground-truth `Labels-GameState.json` files can be used as `predictions.json` **with zero conversion** (Track 1)

```json
{
  "info": {"name": "SNGS-10001", "n_frames": 750, "fps": 25},
  "images": [
    {"image_id": "3100100001", "file_name": "000001.jpg", "width": 1920, "height": 1080,
     "is_labeled": true, "has_labeled_person": true, "has_labeled_pitch": true, "has_labeled_camera": true},
    {"image_id": "3100100002", "file_name": "000002.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": "3100100101",
      "image_id": "3100100001",
      "track_id": 3,
      "supercategory": "object",
      "category_id": 1,
      "bbox_image": {"x": 100, "y": 200, "x_center": 125.0, "y_center": 260.0, "w": 50, "h": 120},
      "bbox_pitch": {"x_bottom_left": 15.0, "y_bottom_left": -9.0,
                     "x_bottom_right": 15.6, "y_bottom_right": -8.4,
                     "x_bottom_middle": 15.3, "y_bottom_middle": -8.7},
      "attributes": {"role": "player", "team": "left", "jersey": "7"}
    }
  ],
  "categories": [
    {"supercategory": "object", "id": 1, "name": "player"},
    {"supercategory": "object", "id": 2, "name": "goalkeeper"},
    {"supercategory": "object", "id": 3, "name": "referee"},
    {"supercategory": "object", "id": 4, "name": "ball"},
    {"supercategory": "pitch", "id": 5, "name": "pitch"},
    {"supercategory": "camera", "id": 6, "name": "camera"}
  ]
}
```

**Output schema for `homography_per_frame.json`**:

```json
{
  "frames": {
    "1": {"H": [[...3x3...]], "H_inv": [[...3x3...]], "valid": true},
    "2": {"H": [[...3x3...]], "H_inv": [[...3x3...]], "valid": true}
  }
}
```

---

## 5. Stage 2: Event Detection

### 5.1 Event Schema (`schema.py`)

Loads `reference_football.csv` at init time. The CSV is the **single source of truth** for event vocabulary, detection strategy, and LLM terminology.

```python
class EventSchema:
    # --- Event registry ---
    def get_event(code) -> EventDef
        # display_name_cn/en, description, source_type, importance_base,
        # tags, trigger_notes, negative_flag

    def events_by_source_type(source_type) -> List[EventDef]
        # "model_direct" | "rule_composed" | "context_derived"

    def core_events / narrative_events / event_qualifiers -> List[EventDef]

    # --- Tag dimension registry ---
    def computable_tag_groups() -> List[str]
        # Fillable by rules: pitch_zone, shot_distance, pass_distance, pass_direction, pattern_of_play
    def visual_tag_groups() -> List[str]
        # Requires LLM visual recognition: body_part, foot_technique, ball_trajectory,
        # shot_posture, pass_height, pass_style, dribble_technique, save_technique,
        # keeper_movement, clearance_type, penalty_style, context_tag

    # --- Prompt helpers ---
    def event_definitions_for_prompt() -> str
        # All event definitions formatted for LLM context
    def tag_vocabulary_for_prompt(event_code) -> str
        # Applicable tag groups with allowed values + display_name_cn/en
```

### 5.1.1 Detection Strategy by `source_type`

The `source_type` field in the CSV dictates how each event is detected:

| source_type | Strategy | Examples |
|---|---|---|
| `model_direct` | Frame-level rule detection from pitch data | shoot, pass, clearance, tackle, pressing |
| `rule_composed` | Composed from multiple detected events in sequence | assist (= pass → goal), penalty (= foul in box), big_chance (= 1v1 + low xG distance) |
| `context_derived` | Accumulated across the 30s window (or longer for future use) | equalizer (goal + score tied), scoring_run (≥2 goals in window) |

### 5.1.2 Full CSV field utilization map

| CSV field | Used in | How |
|---|---|---|
| `event_id` / `code` | Schema, events.json | Event type identifier |
| `display_name_cn/en` | events.json, LLM prompt | Standard terminology in both languages |
| `description` | LLM prompt | Event definition for LLM context |
| `source_type` | Rule engine | Determines detection strategy (see above) |
| `importance_base` | events.json, LLM priority, light-beam trigger | Events with importance ≥ 0.5 get visual effects + tactical reasoning |
| `tags` | LLM prompt | Narrative style hints ("highlight", "clutch", "drama") affect LLM tone |
| `trigger_notes` | Rule engine | Human-readable trigger conditions guide rule implementation |
| `negative_flag` | LLM prompt | Flags own-goal etc. to adjust commentary tone |
| Tag dimension `display_name_cn/en` | LLM prompt + output | Forces LLM to use standard terms ("凌空" not "空中射门") |
| Tag `exclusive` / `non-exclusive` | LLM tag validation | Ensures mutually exclusive tags aren't co-assigned |

### 5.2 Rule Engine (`detector.py`)

Input: `predictions.json`
Output: `events.json`

#### Detectable events (from pitch-coordinate data)

| Event code | Rule logic | Key signals |
|---|---|---|
| `football.pass` | Possession switches between same-team players | possession chain + same team |
| `football.shoot` | Ball accelerates toward goal + ball near/in penalty area | ball_velocity_spike + ball_direction_to_goal |
| `football.goal` | Ball crosses goal line (x ≈ ±52.5m, y within ±3.66m) | ball position threshold |
| `football.clearance` | Ball moves rapidly away from own goal after defensive touch | ball_direction_away + high speed |
| `football.interception` | Possession switches from team A to team B without a pass | possession_break + cross-team |
| `football.tackle` | Two opposing players converge + possession changes | proximity + possession_change |
| `football.dribble` | Same player retains ball while moving significantly | sustained_possession + displacement |
| `football.pressing` | Multiple players from one team converge on ball carrier | multi-player_proximity |
| `football.save` | Ball moving toward goal + GK touches + ball doesn't cross line | ball_to_goal + GK_proximity + no_goal |
| `football.throw_in` | Ball goes out of bounds (y ≈ ±34m) | ball_position_threshold |
| `football.set_piece` | Play restart after ball out / foul (inferred from ball stationary → moving) | ball_stationary_then_moving |

#### Events requiring LLM assistance

| Event code | Why rules alone aren't enough |
|---|---|
| `football.big_chance` | Requires tactical context understanding |
| `football.foul` | Physical contact not visible in pitch coordinates |

#### Events not supported in V1

`football.red_card`, `football.yellow_card`, `football.offside`, `football.substitution`, `football.celebration` — require visual recognition beyond pitch coordinates.

#### Core algorithm

```python
class EventDetector:
    def __init__(self, schema: EventSchema, fps: int = 25):
        self.schema = schema
        self.fps = fps

    def detect(self, predictions_json_path: str) -> List[Event]:
        frames = load_per_frame_data(predictions_json_path)
        ball_positions = extract_ball_trajectory(frames)
        ball_velocities = compute_velocities(ball_positions, self.fps)
        possession_chain = compute_possession_chain(frames)  # reuses formation_topology.possession

        raw_events = []
        raw_events += self._detect_passes(possession_chain, frames)
        raw_events += self._detect_shots(ball_velocities, ball_positions, frames)
        raw_events += self._detect_clearances(ball_velocities, ball_positions, possession_chain)
        raw_events += self._detect_interceptions(possession_chain)
        raw_events += self._detect_dribbles(possession_chain, frames)
        # ... other detectors

        deduped = self._deduplicate(raw_events, min_gap_s=1.0)
        enriched = self._enrich_tags(deduped, frames)
        return sorted(enriched, key=lambda e: e.timestamp_s)
```

### 5.3 Tag Enricher (`enricher.py`)

After events are detected, enrich them with tag dimensions from `reference_football.csv`:

- **pitch_zone**: Computed from ball position (six_yard_box / inside_box / outside_box / halfway_line)
- **shot_distance**: Euclidean distance from ball to goal center
- **pass_distance**: Distance between passer and receiver
- **pass_direction**: Angle-based classification (forward / lateral / back_pass / through_ball / cross)
- **pattern_of_play**: Inferred from preceding context (open_play / fast_break / set_piece)

### 5.4 Output Format (`events.json`)

```json
{
  "video_info": {
    "source": "raw_video.mp4",
    "fps": 25,
    "duration_s": 30.0,
    "total_frames": 750
  },
  "schema_version": "v3-20260319",
  "events": [
    {
      "event_id": "evt_001",
      "timestamp_s": 3.04,
      "frame_id": 76,
      "event_code": "football.pass",
      "display_name_en": "Pass",
      "display_name_cn": "传球",
      "importance": 0.15,
      "player_jersey": 7,
      "player_team": "left",
      "target_jersey": 10,
      "target_team": "left",
      "tags": {
        "pass_distance": "short",
        "pass_direction": "forward",
        "pass_height": "ground_pass",
        "pitch_zone": "outside_box"
      },
      "confidence": 0.85,
      "description_hint": "short forward ground pass in midfield"
    },
    {
      "event_id": "evt_002",
      "timestamp_s": 8.20,
      "frame_id": 205,
      "event_code": "football.shoot",
      "display_name_en": "Shot",
      "display_name_cn": "射门",
      "importance": 0.55,
      "player_jersey": 9,
      "player_team": "right",
      "tags": {
        "shot_distance": "long_range",
        "pitch_zone": "outside_box",
        "pattern_of_play": "open_play"
      },
      "confidence": 0.92,
      "description_hint": "long-range shot from outside the box"
    }
  ]
}
```

---

## 6. Stage 3: Visual Effects (Light Beam + Tactical Topology Lines)

Stage 3 renders **two types** of visual overlays on **raw frames** (`clip_dir/img1/`):
1. **Light-beam effects** on key event players (e.g., shots, goals)
2. **Tactical topology lines** showing formation connections and predicted running paths

Light beams are rendered on **clean, raw frames** (not visualization videos with existing bbox overlays). The light beam targets are determined from structural data (`events.json` + `predictions.json`), not arbitrarily.

Reference: `Japan Strategy.mp4` by Abdullah Maythim — see saved frames in `outputs/japan_strategy_frames/`.

### 6.1 Coordinate Mapping (shared by both effects)

- Step 3 produces per-frame homography `H` (image → pitch)
- `H_inv = np.linalg.inv(H)` maps pitch → image
- Player foot point in image: `H_inv @ [pitch_x, pitch_y, 1]` (then divide by w)
- Player top: Use `bbox_image` top-y coordinate from predictions
- Pitch points (for topology lines): project arbitrary pitch coords to image via `H_inv`

### 6.2 Light-Beam Effect Design

Based on the reference video (frame_009, frame_010, frame_011), the light beam is a **perspective-correct cone/fan shape**, not a simple rectangle:

#### Visual specification

1. **Player foot marker**: Glowing circle at player's foot position in image space. Double-ring design (inner bright, outer soft glow). Color matches beam.
2. **Cone beam**: A trapezoidal shape fanning out from the player toward the target (goal for shots, receiving player for passes).
   - **Origin**: Player foot position (narrow end, width ≈ player bbox width)
   - **Target direction**: Toward goal center (for shots) or toward receiving player (for passes). Projected via `H_inv`.
   - **Fan angle**: Widens with distance (~15-20° spread)
   - **Length**: Extends to frame edge or to target position
3. **Color & transparency**: Semi-transparent gradient fill (alpha ≈ 0.3 at origin, fading to 0.05 at far end). Gaussian blur for soft glow.
4. **Color scheme by event**:

| Event type | Color (BGR) | Notes |
|---|---|---|
| `football.shoot` / `football.goal` | Yellow-green (100, 255, 200) | As in reference |
| `football.pass` (key) | Lighter green (150, 255, 150) | Subtle |
| `football.clearance` | Blue (255, 150, 50) | Defensive action |
| `football.save` | Bright green (0, 255, 100) | GK highlight |

#### Player selection per event type

Light beams highlight specific players based on event context from `events.json` + `predictions.json`:

| Event type | Players highlighted | Beam direction |
|---|---|---|
| `football.shoot` | Shooter | Shooter → goal center |
| `football.goal` | Scorer + assister (if `football.pass` precedes within 5s) | Scorer → goal; assister → scorer |
| `football.pass` (key, importance ≥ 0.3) | Passer + receiver | Passer → receiver |
| `football.clearance` | Clearing player | Player → ball direction |
| `football.save` | Goalkeeper | GK → ball direction |
| `football.tackle` | Tackler | Tackler → ball carrier |
| `football.interception` | Intercepting player | Player → ball direction |

Player positions are read from `predictions.json` annotations matching `player_jersey` + `player_team` in the event's `frame_id`.

#### Rendering algorithm (`light_beam.py`)

```python
def draw_cone_beam(frame, player_foot_img, target_img, beam_color, alpha_max=0.3):
    """
    Draw a perspective-correct cone beam from player to target direction.

    1. Compute player foot position in image (via H_inv)
    2. Compute target direction point in image (goal center or receiving player)
    3. Build trapezoid polygon: narrow at player, widening toward target
    4. Fill trapezoid on overlay with gradient alpha (bright near player, fading out)
    5. Gaussian blur the overlay for soft glow
    6. Alpha-blend overlay onto original frame
    """

def draw_foot_marker(frame, foot_xy_img, color, radius=20):
    """
    Draw glowing double-ring circle at player's feet.

    1. Inner ring: solid, bright, radius=radius
    2. Outer ring: semi-transparent, radius=radius*1.5
    3. Gaussian blur for glow
    """
```

#### Timing

- Light beam appears for ±0.5s around the event frame (~12 frames at 25fps)
- Fade-in over first 6 frames (alpha 0 → alpha_max)
- Fade-out over last 6 frames (alpha_max → 0)

### 6.3 Tactical Topology Lines Design

Based on the reference video (frame_001 through frame_005), the topology overlay includes:

#### Visual elements

1. **Player markers**: White/blue circle outlines at each player's feet (using `bbox_pitch` projected to image via `H_inv`)
2. **Formation lines (solid)**: Connect adjacent players in the same team to show spatial structure — similar to `formation_topology.viz_topdown` but projected onto the original camera view
3. **Running path lines (dashed + arrow)**: Predicted movement direction for key players
   - **Attacking players**: Dashed lines showing forward runs, through-ball paths, overlap runs
   - **Defensive players**: Dashed lines showing pressing direction, cover movements
   - Arrow head at the end indicating direction
4. **Pressing/offside line (horizontal dashed)**: A horizontal dashed line across the pitch showing the defensive line height or pressing trigger line. Computed from the depth-line analysis in `formation_topology.lines`

#### Algorithm for running path prediction

Running paths are estimated from **velocity vectors** (position change over recent frames):

```python
def predict_running_paths(predictions, frame_id, fps, horizon_s=1.0):
    """
    For each player at frame_id, compute velocity from recent N frames,
    extrapolate forward by horizon_s seconds to get predicted future position.

    Returns list of (current_pos, predicted_pos, team, role) tuples.
    Only include players with significant velocity (> 1 m/s).
    """
```

#### Rendering (`tactical_lines.py` — new file)

```python
def draw_formation_lines(frame, player_positions_img, adjacency, color, thickness=2):
    """Draw solid lines connecting adjacent players in formation."""

def draw_running_paths(frame, paths, color_attack, color_defend):
    """
    Draw dashed lines with arrow heads for predicted running directions.
    Attack paths: team in possession, forward movement
    Defend paths: team out of possession, pressing/tracking movement
    """

def draw_pressing_line(frame, line_y_pitch, H_inv, color, dash_length=20):
    """Draw horizontal dashed line at the defensive line height."""
```

#### Integration with existing `formation_topology`

- Reuse `formation_topology.lines.find_lines()` for depth-line positions (D-M-F gaps)
- Reuse `formation_topology.possession` for identifying attacking vs defending team
- Reuse `formation_topology.metrics` for block width/height to determine adjacency

### 6.4 Updated Code Structure

```
pipeline/stage3_effects/
├── __init__.py
├── light_beam.py         # Cone beam + foot marker rendering
├── tactical_lines.py     # Formation lines + running paths + pressing line (NEW)
├── render.py             # Orchestrate both effects onto original video
└── projection.py         # H_inv projection utilities (shared)
```

### 6.5 Output

`annotated_video.mp4` — same resolution/fps as input, encoded with H.264 via ffmpeg. Contains both light-beam effects on key events and tactical topology lines throughout.

---

## 7. Stage 4: LLM Commentary Generation

### 7.1 Timestamp Accuracy Strategy

**Critical design constraint:** LLM does NOT determine timestamps. The rule engine (Stage 2) provides exact timestamps; LLM only narrates.

Flow:
1. `prompt_builder.py` serializes `events.json` into a natural-language timeline
2. Prompt instructs LLM: "Use ONLY the provided timestamps. Do NOT invent new timestamps."
3. LLM generates prose commentary organized by the given timestamps
4. `postprocess.py` validates output timestamps match input events

### 7.2 Three-Step LLM Flow

For high-importance events (importance ≥ 0.5), the LLM is called **three times** with different purposes. For low-importance events, only Step 2 runs.

#### Step 1: Visual Tag Filling (LLM sees raw video/images → structured tags)

For each event with importance ≥ 0.5, the LLM watches the **raw video** clip around the event (±2s) and fills in visual tag dimensions that rules cannot compute:

```
[System]
You are a football video analyst. For the event below, watch the video clip
and fill in the visual tags. Use ONLY values from the provided vocabulary.
Output JSON only.

[Event]
t=8.2s: football.shoot by #9 (right team)

[Tag Vocabulary]  ← generated by EventSchema.tag_vocabulary_for_prompt("football.shoot")
body_part (pick one): right_foot (右脚), left_foot (左脚), head (头), other (其他)
foot_technique (pick one): instep (正脚背), inside_foot (脚弓), outside_boot (外脚背), ...
ball_trajectory (pick one): ground (地滚球), low_drive (低平球), lofted (高空球), curling (弧线球), ...
shot_posture (pick one): volley (凌空), placed_shot (推射), power_shot (大力抽射), chip (挑射), ...

[Video Clip]
<video: raw_video_t6-t10.mp4> OR <images around t=8.2s>
```

Output: `{"body_part": "right_foot", "foot_technique": "instep", "ball_trajectory": "low_drive", "shot_posture": "power_shot"}`

These visual tags are **merged into events.json** before Step 2.

#### Step 2: Commentary Generation (LLM sees topdown view + enriched events → commentary)

Now with **complete tags** (both rule-computed and LLM-filled), generate the commentary:

```
[System]
You are a professional football commentator. Generate second-by-second commentary.

RULES:
1. Use ONLY the provided timestamps. Never invent timestamps.
2. Use the EXACT terminology from the tag display names (e.g., say "大力抽射" not "用力踢")
3. For gaps between events, describe formations, positioning, build-up play.
4. Refer to players by jersey number (or name if roster provided).
5. Generate both English and Chinese commentary.
6. For events tagged "highlight" or "drama", use more excited/vivid language.
7. For events with negative_flag, adjust tone accordingly.

[Event Definitions]  ← generated by EventSchema.event_definitions_for_prompt()
football.shoot: Deliberate attempt to score a goal (射门)
football.save: Goalkeeper prevents ball from entering goal (扑救)
...

[Event Timeline with Full Tags]
t=3.0s: [football.pass] #7 → #10 (left team)
  Tags: pass_distance=short(短传), pass_direction=forward(向前传), pass_height=ground_pass(地面)
  Style: signal
t=8.2s: [football.shoot] #9 (right team)  ⚡ HIGHLIGHT
  Tags: shot_distance=long_range(远距离), pitch_zone=outside_box(禁区外),
        body_part=right_foot(右脚), foot_technique=instep(正脚背),
        ball_trajectory=low_drive(低平球), shot_posture=power_shot(大力抽射)
  Style: scoring, attacking
...

[Formation Context]
t=0-5s: Left team - height: 35m, depth: 28m, 2-line (D-M gap: 15m)
...

[Player Roster] (optional)
Left: {7: "Player A", 10: "Player B"}, Right: {9: "Player C"}

[Visual Input]
<video: topdown_view.mp4> OR <images: frame_t0.png ... frame_t29.png>
```

#### Step 3: Tactical Reasoning (separate call, see Section 7.4)

### 7.3 LLM Adapter Design

```python
class LLMAdapter(ABC):
    """Unified interface for all LLM backends."""

    @abstractmethod
    def supports_video(self) -> bool:
        """Whether this backend accepts video input directly."""

    @abstractmethod
    def generate(self, prompt: str, visual_input: Union[Path, List[Path]]) -> str:
        """
        Generate commentary.
        visual_input: video path (if supports_video) or list of image paths.
        Adapter auto-handles: if video given but not supported, extracts frames.
        """

class QwenLocalAdapter(LLMAdapter):
    """Local Qwen2.5-7B-VL-Instruct. Supports video input."""
    def supports_video(self) -> bool: return True

class DoubaoAPIAdapter(LLMAdapter):
    """Doubao Vision API. Supports video input."""
    def supports_video(self) -> bool: return True

class OpenAIAPIAdapter(LLMAdapter):
    """GPT-5 API. Fallback to per-second frame images."""
    def supports_video(self) -> bool: return False
```

Adapter selection in `config.py`:

```python
LLM_BACKEND = "openai"  # "qwen_local" | "doubao" | "openai"
```

### 7.4 Tactical Reasoning (Why Did This Goal Happen?)

For high-importance events (especially `football.goal`, `football.big_chance`, `football.shoot`), the LLM performs a **separate tactical reasoning pass** that analyzes WHY the event succeeded or failed. This is the hardest but most valuable part of the commentary.

#### Why topology + top-down view is better for reasoning

- Raw broadcast video: perspective distortion, occluded players, camera movement — hard for LLM to judge spatial relationships
- Top-down view + topo.json: clear spatial layout, exact player distances, formation gaps, pressing line height — ideal for structural reasoning

#### Input for tactical reasoning

For each high-importance event, construct a focused reasoning prompt with:

1. **Topology snapshots**: `topo.json` records for 3 time windows:
   - **T-5s** (before the attack): defensive shape, pressing line height, block compactness
   - **T-0s** (at the event): positions at the moment of the goal/shot
   - **T+2s** (after): aftermath
2. **Top-down view frames**: 5-8 frames from the top-down video spanning T-5s to T+2s (NOT the raw video)
3. **Event context**: The event from `events.json` with all tags (pitch_zone, pattern_of_play, etc.)
4. **Structured metrics delta**: Changes in key topology metrics (block height collapse, depth-line gap widening, hull area change)

#### Reasoning prompt structure

```
[System]
You are a football tactical analyst. Given the formation data and top-down view
before and during this goal, explain WHY this goal was scored.

Analyze:
1. What defensive weakness was exploited? (high line, gap between lines, wide space, etc.)
2. What attacking movement created the opportunity? (through-ball, overlap, switch of play, etc.)
3. Which specific players' positioning/movement was critical?
4. Could the defense have prevented it? How?

Use the topology metrics to support your analysis with specific numbers
(e.g., "the D-M gap widened to 18m, creating a pocket of space").

[Formation Data: T-5s]
{topo_window_before}

[Formation Data: T-0s (at goal)]
{topo_window_at_event}

[Event Details]
{event from events.json}

[Top-Down View Frames]
<images: topdown_t-5.png, topdown_t-3.png, topdown_t-1.png, topdown_t0.png, topdown_t+2.png>
```

#### Output

The tactical reasoning is stored as an additional field `tactical_analysis` in `commentary.json` for the relevant event segment:

```json
{
  "timestamp_s": 22.0,
  "end_s": 26.0,
  "text_en": "GOAL! Number 11 finishes brilliantly from close range!",
  "text_zh": "进球！11号球员近距离精彩破门！",
  "events_referenced": ["evt_008"],
  "tactical_analysis": {
    "text_en": "This goal was a product of Japan's patient build-up exploiting Brazil's high defensive line. The D-M gap widened to 18m as the midfield pushed up, creating a pocket of space behind. Number 7's through-ball split the center-backs who were 12m apart — well above the safe threshold of 8m. Number 11's diagonal run from the left channel went untracked because the right-back was drawn narrow by the overload...",
    "text_zh": "这粒进球源于日本队耐心的传控撕开了巴西队过高的防线。中场前压导致防线与中场之间的纵深间距扩大到18米，形成了身后空间。7号球员的直塞球穿透了两名中卫之间12米的间距——远超安全阈值8米。11号球员从左路的斜向跑动未被盯防，因为右后卫被局部人数优势吸引内收...",
    "key_factors": [
      "D-M gap: 18m (normal: 10-12m)",
      "CB split distance: 12m (safe: <8m)",
      "Through-ball from #7 splitting CBs",
      "Untracked diagonal run by #11",
      "Right-back drawn narrow by overload"
    ]
  }
}
```

#### Implementation notes

- Tactical reasoning is a **separate LLM call** from the main commentary generation, because it needs different input (top-down view, not raw video) and a different prompt style (analytical vs. narrative)
- Only triggered for events with `importance >= 0.5` to control API costs
- The `key_factors` list is extracted from the LLM output via structured output parsing (or a follow-up extraction call)
- This is the hardest module — expect iterative prompt tuning during Day 5

### 7.5 Output Format (`commentary.json`)

```json
{
  "video_info": {"source": "raw_video.mp4", "duration_s": 30.0},
  "model_info": {"name": "gpt-5", "backend": "openai"},
  "language": ["en", "zh"],
  "commentary": [
    {
      "timestamp_s": 0.0,
      "end_s": 3.0,
      "text_en": "The match resumes with the left team in possession, controlling the midfield...",
      "text_zh": "比赛继续，左队控制着中场球权...",
      "events_referenced": []
    },
    {
      "timestamp_s": 3.0,
      "end_s": 8.0,
      "text_en": "Number 7 picks out Number 10 with a crisp short pass, advancing the ball forward...",
      "text_zh": "7号球员一脚精准的短传找到10号，将球向前推进...",
      "events_referenced": ["evt_001"]
    },
    {
      "timestamp_s": 8.0,
      "end_s": 13.0,
      "text_en": "Oh! Number 9 unleashes a thunderous long-range strike from outside the box! The keeper dives full stretch to tip it away!",
      "text_zh": "哦！9号球员禁区外一脚大力远射！门将飞身扑救将球挡出！",
      "events_referenced": ["evt_002", "evt_003"]
    }
  ]
}
```

---

## 8. Orchestrator (`run.py`) and Configuration

### 8.1 Unified PipelineConfig

The pipeline uses a single unified configuration model. The core input is `clip_dir` (a directory containing `img1/` frames). Optional `existing_*` parameters control which stages are skipped.

```python
@dataclass
class PipelineConfig:
    # --- Core input (required) ---
    clip_dir: Path               # Parent dir containing img1/ (e.g., test/SNGS-148/)
    output_dir: Path             # All outputs written here

    # --- Optional: pre-existing intermediate artifacts (skip corresponding stages) ---
    existing_predictions_json: Optional[Path] = None   # Skip Stage 1 entirely
    existing_homography_json: Optional[Path] = None     # Skip Stage 1 homography step
    existing_pklz_path: Optional[Path] = None           # Skip GSR inference, only convert pklz→JSON
    existing_events_json: Optional[Path] = None         # Skip Stage 2

    # --- LLM settings ---
    llm_backend: str = "openai"  # "qwen_local" | "doubao" | "openai"
    roster_json: Optional[Path] = None
    languages: List[str] = field(default_factory=lambda: ["en", "zh"])

    # --- Processing options ---
    force: bool = False          # Re-run all stages even if outputs exist
    fps: int = 25
```

### 8.2 Stage Skip Logic

The orchestrator determines which stages to run based on `existing_*` parameters:

```
if existing_predictions_json AND existing_homography_json:
    → Skip Stage 1 entirely (use provided files)
elif existing_pklz_path:
    → Skip GSR inference, run only pklz→JSON conversion
else:
    → Run full Stage 1 (preprocess + GSR + pklz→JSON)

if existing_events_json:
    → Skip Stage 2 (use provided events file)
else:
    → Run Stage 2 (event detection)

Stage 3 and Stage 4 always run (unless outputs exist and --force is not set)
```

### 8.3 Example Invocations

**Track 1: Test clip with GT labels (skip Stage 1)**
```bash
python -m pipeline.run \
    --clip-dir codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148/ \
    --output-dir outputs/SNGS-148/ \
    --existing-predictions-json codes/sn-gamestate/datasets/SoccerNetGS/test/SNGS-148/Labels-GameState.json \
    --llm-backend openai
```

**Track 2: Processed clip with pklz (skip GSR inference)**
```bash
python -m pipeline.run \
    --clip-dir codes/sn-gamestate/datasets/SoccerNetGS/train/SNGS-061/ \
    --output-dir outputs/SNGS-061/ \
    --existing-pklz-path codes/sn-gamestate/outputs/gsr/step_3_train_SNGS-061/states/sn-gamestate.pklz \
    --llm-backend openai
```

**Full pipeline: new raw video**
```bash
python -m pipeline.run \
    --clip-dir outputs/my_new_clip/ \
    --output-dir outputs/my_new_clip/ \
    --llm-backend openai
```

Single-clip processing per invocation (V1). Batch processing via shell loop.

---

## 9. Success Criteria

### Minimum viable (Day 7)

- [ ] Given a 30s raw video, `python -m pipeline.run --input video.mp4` produces:
  - `predictions.json` with valid `bbox_pitch` data
  - `events.json` with detected core events using `reference_football.csv` codes
  - `annotated_video.mp4` with light-beam effects on key events
  - `commentary.json` with timestamped EN + ZH commentary
- [ ] Event timestamps in commentary match actual video events (error ≤ ±1 second)
- [ ] At least 2 LLM backends working (local Qwen + one API)

### Stretch (Day 8)

- [ ] All 3 LLM backends working
- [ ] Measurable speed improvements in SoccerMaster processing
- [ ] Tag enrichment for all supported tag dimensions

---

## 10. Work Schedule (8 days × 6 hours = 48 hours)

### Parallel Development Strategy

Development proceeds on two parallel tracks to maximize throughput:

- **Track A (Data Processing)**: Stage 1 (pklz→JSON) + Stage 2 (event detection) — uses SNGS-061 data
- **Track B (Effects + Commentary)**: Stage 3 (light beam) + Stage 4 (LLM commentary) — uses test clip SNGS-148 (goal event, t≈25s) with GT labels as predictions + `clip_index.csv` event as fixture

Track B can start immediately on Day 1 because test clips have all needed data (GT labels = predictions.json, clip_index.csv events = events_fixture.json).

| Day | Track A (Data Processing) | Track B (Effects + Commentary) | Verify |
|-----|---------------------------|-------------------------------|--------|
| **Day 1** (6h) | `pipeline/` scaffold + `config.py` + `pklz_to_json.py` (reuse `compare_step3_pred_gt.py`) + homography export | `clip_index_to_events.py` → events fixture for SNGS-148 | pklz→JSON converter working; SNGS-148 has predictions + events fixture |
| **Day 2** (6h) | Stage 2: `schema.py` + `detector.py` core rules (pass, shoot, clearance) | Stage 3: `light_beam.py` cone beam + foot marker rendering on SNGS-148 raw frames | Events detected from GT labels; light beam renders on raw frames |
| **Day 3** (6h) | Stage 2: detection tuning + `enricher.py` tag dimensions | Stage 3: tactical_lines.py + render.py (full video output) | events.json quality; annotated_video.mp4 with light beams |
| **Day 4** (6h) | Stage 1: `preprocess.py` + `run_gsr.py` (full pipeline) | Stage 4: `prompt_builder.py` + `llm_adapter.py` + first adapter | LLM generates commentary for SNGS-148 events |
| **Day 5** (6h) | Full Stage 1→2 pipeline on SNGS-061 | Stage 4: remaining adapters + tactical reasoning prompt | SNGS-061 predictions→events; 2+ LLM backends working |
| **Day 6** (6h) | Orchestrator `run.py` + unified CLI | End-to-end integration: Track A outputs feed Stage 3/4 | `python -m pipeline.run --clip-dir ...` → all outputs |
| **Day 7** (6h) | Speed optimization: batch size tuning, frame skipping, module bypass | clip_index.csv validation script | Benchmark before/after on same video |
| **Day 8** (6h) | End-to-end testing on 3+ videos + bug fixes + documentation | | Full runs without errors |

### Primary development clip

**SNGS-148** (test set, goal event at t≈25s) is the primary development target for Track B because:
- Has GT `Labels-GameState.json` (= zero-conversion `predictions.json`)
- Has `clip_index.csv` event (`goal` at 24.765s → 25s)
- Goal events are the most visually interesting for light beam + commentary development
- 750 frames (30s @ 25fps) — manageable for iteration

### Risk mitigations

- **pklz→JSON harder than expected**: Day 1 has margin; core logic is proven in `compare_step3_pred_gt.py`
- **Event detection tuning slow**: Ship with simplified rules (possession switch + ball velocity peaks), iterate later
- **Light beam timing tight**: Fallback to colored bounding-box annotation (simpler than light beam)
- **Track B blocked on fixture quality**: Can switch to manual event annotation at any time
- **Plan is flexible**: Daily review, adjust remaining days as needed

---

## 11. Constraints & Non-Goals

### Constraints
- 8 working days × 6 hours = 48 total hours
- Accuracy > speed (per mentor guidance)
- Must reuse SoccerMaster infrastructure, not reinvent

### Non-goals (out of scope for V1)
- TTS audio synthesis
- Video subtitle overlay
- Real-time / streaming processing
- Events requiring visual recognition: red/yellow card, offside, substitution, celebration
- Training new ML models
- Multi-camera support
