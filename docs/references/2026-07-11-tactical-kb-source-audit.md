# Tactical KB v2 finite source audit

**Date:** 2026-07-11
**Status:** Approved by project maintainer
**Scope:** The approved 14-concept SoccerMaster Tactical KB v2 catalog

## Responsibility boundary

This document is the reviewed operational terminology source
`soccermaster_operational_definitions_v1`. It owns names, aliases, observable
inclusion/exclusion boundaries, and actor schemas for SoccerMaster. It is not a
claim that GenTac or TacticGen is a canonical football glossary.

The external papers are used only for representational connections:

- `gentac_arxiv_2604_11786_v1` separates continuous 25 Hz multi-player
  trajectories and team-structure measurements from hierarchical event
  context. Its event labels guide context representation, not the concept
  ontology.
- `tacticgen_arxiv_2604_18210_v1` represents tactics through ordered
  multi-agent and ball trajectories and explicitly demonstrates relational,
  width, compactness, pressing, and deep-defending objectives.

Neither external source defines SoccerMaster thresholds, tolerances, durations,
quality gates, or production eligibility. Those remain unset until local
train/valid calibration. All definitions below permit observation claims only;
they do not assert effect or intention.

## Audited local source records

| Source ID | Title | Authors | Version / DOI | License | Accessed | Local PDF SHA-256 | Permitted use |
|---|---|---|---|---|---|---|---|
| `gentac_arxiv_2604_11786_v1` | *GenTac: Generative Modeling and Forecasting of Soccer Tactics* | Jiayuan Rao; Tianlin Gui; Haoning Wu; Yanfeng Wang; Weidi Xie | `arXiv:2604.11786v1`; `https://doi.org/10.48550/arXiv.2604.11786` | CC BY-NC-SA 4.0 | 2026-07-11 | `31ef3d7c6b9d2d6d65c3dd6d39984dd6ef436bd33976b95e69f87873cd2f605b` | trajectory/event-context separation; candidate team-structure variables |
| `tacticgen_arxiv_2604_18210_v1` | *TacticGen: Grounding Adaptable and Scalable Generation of Football Tactics* | Sheng Xu; Guiliang Liu; Tarak Kharrat; Yudong Luo; Mohamed Aloulou; Javier López Peña; Konstantin Sofeikov; Adam Reid; Paul Roberts; Steven Spencer; Joe Carnall; Ian McHale; Oliver Schulte; Hongyuan Zha; Wei-Shi Zheng | `arXiv:2604.18210v1`; `https://doi.org/10.48550/arXiv.2604.18210` | CC BY-NC-SA 4.0 | 2026-07-11 | `0c49c07fbc4dad460c39760af854569051f6ada2d2326b7a044ee08112d1c4c2` | multi-agent movement and interaction representation |

## Fourteen reviewed operational definitions

“Same” and “opponent” are relative to the proposal's canonical `team_id`.
Empty actor fields mean a team-level observation; they do not authorize the
model to infer unseen players.

| Canonical ID | Bilingual names and aliases | Observation-only inclusion boundary | Exclusion boundary | Exact actor fields | Team relation | Terminology origin | Formalization source IDs and representational connection |
|---|---|---|---|---|---|---|---|
| `overlap_run` | 套边、套上；overlap run, overlapping run, overlap | A same-team runner changes from behind to ahead of the carrier along the outside channel. | Exclude inside underlaps, already-ahead parallel support, and defensive recovery runs. | `carrier_track_id`, `runner_track_id` | same, same | `soccermaster_operational_definitions_v1` | `tacticgen_arxiv_2604_18210_v1`: ordered pairwise player trajectories and ball-conditioned context |
| `run_in_behind` | 前插身后、打身后；run in behind, depth run, `depth_run` | A same-team runner moves from in front of the relevant visible defensive line into space behind it while the team controls or progresses the ball. | Exclude recovery runs, unsupported off-camera line assumptions, and momentary noisy line crossings. | `carrier_track_id`, `runner_track_id` | same, same | `soccermaster_operational_definitions_v1` | `tacticgen_arxiv_2604_18210_v1`: ordered attacker/defender trajectories with ball context |
| `switch_of_play` | 弱侧转移、大范围转移；switch of play, `switch_play` | A controlled same-team ball transfer changes the active pitch side between the bound passer and receiver. | Exclude clearances, deflections, and diagonal passes that remain on the same side. | `passer_track_id`, `receiver_track_id` | same, same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: trajectory/event separation; `tacticgen_arxiv_2604_18210_v1`: player-and-ball trajectory interaction |
| `local_numerical_superiority` | 局部人数优势、局部以多打少；local numerical superiority, `numerical_superiority` | Within a fixed local region around the bound anchor, the proposal team has a sustained visible player-count advantage. | Exclude single-frame count changes and conclusions that rely on unseen or off-camera players. | `anchor_track_id` | same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: collective spatial structure; `tacticgen_arxiv_2604_18210_v1`: multi-agent interaction representation |
| `compact_block` | 紧凑防守、紧凑防守阵型；compact block, defensive compactness | The visible out-of-possession team maintains compact inter-player width, depth, and line spacing for a sustained interval. | Exclude low visibility and a transient collapse around one duel. | none | team-level | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: team width, length, stretch, and surface variables; `tacticgen_arxiv_2604_18210_v1`: defensive compactness representation |
| `line_break` | 破线传球、穿线传球；line-breaking pass, line break | A controlled pass from the bound passer to receiver moves the ball across a defined visible opponent line while possession is retained. | Exclude dribbles, clearances, and apparent crossings caused by calibration or identity errors. | `passer_track_id`, `receiver_track_id` | same, same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: trajectory-grounded event context; `tacticgen_arxiv_2604_18210_v1`: passer/receiver, ball, and defender trajectories |
| `underlap_run` | 肋部套上、内侧套上；underlap run, underlap | A same-team runner changes from behind to ahead of the carrier through the inside channel. | Exclude outside overlaps, straight parallel support, and defensive recovery runs. | `carrier_track_id`, `runner_track_id` | same, same | `soccermaster_operational_definitions_v1` | `tacticgen_arxiv_2604_18210_v1`: ordered pairwise player trajectories and ball-conditioned context |
| `side_overload` | 边路重载、边路人数集中；side overload, wide overload | The proposal team sustains a visible local player-count concentration in the active wide channel. | Exclude a momentary crowd around one duel and unsupported off-camera absences. | `anchor_track_id` | same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: collective width and spatial structure; `tacticgen_arxiv_2604_18210_v1`: multi-agent interaction and attacking spread |
| `halfspace_occupation` | 肋部占位、半空间占位；half-space occupation, halfspace occupation | A same-team player occupies the defined channel between the center and touchline for a sustained interval. | Exclude touchline width, central-lane occupation, and transient channel crossings. | `occupant_track_id` | same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: pitch-coordinate spatial structure; `tacticgen_arxiv_2604_18210_v1`: zone-conditioned agent trajectories |
| `width_depth_stretch` | 宽度纵深拉伸；width-depth stretch, width and depth | The same team simultaneously maintains a visible wide option and a depth option that expand two different spatial dimensions. | Exclude width-only or depth-only spacing and one-frame extrema. | `wide_player_track_id`, `depth_player_track_id` | same, same | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: team width, length, and stretch variables; `tacticgen_arxiv_2604_18210_v1`: attacking spread and team-width representation |
| `one_two` | 二过一、撞墙配合；one-two, give-and-go, wall pass | The first passer continues moving and receives a controlled return from the wall player within one ordered exchange. | Exclude two unrelated passes, reversed actor order, and exchanges without a controlled return. | `first_passer_track_id`, `wall_player_track_id` | same, same | `soccermaster_operational_definitions_v1` | `tacticgen_arxiv_2604_18210_v1`: ordered player-and-ball interaction trajectories |
| `third_man_run` | 第三人跑动；third-man run, third man combination | The first passer connects through a link player to a distinct third runner in an ordered three-player sequence. | Exclude ordinary two-player combinations and a third player without receiving movement. | `first_passer_track_id`, `link_player_track_id`, `runner_track_id` | same, same, same | `soccermaster_operational_definitions_v1` | `tacticgen_arxiv_2604_18210_v1`: coordinated multi-agent trajectories and role-preserving interaction |
| `high_press` | 高位逼抢；high press, high pressing | The visible out-of-possession proposal team sustains close pressure in the opponent's build-up area, including the bound presser and opponent carrier. | Exclude one isolated chase and assumptions based on low visibility or off-camera players. | `presser_track_id`, `opponent_carrier_track_id` | same, opponent | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: defensive organization and event context; `tacticgen_arxiv_2604_18210_v1`: press-carrier and multi-defender interaction representation |
| `low_block` | 低位防守、低位防守阵型；low block, deep defensive block | The visible out-of-possession proposal team sustains a compact block near its own goal. | Exclude a transient retreat, restart setup, and conclusions requiring unseen players. | none | team-level | `soccermaster_operational_definitions_v1` | `gentac_arxiv_2604_11786_v1`: defensive compactness and team-structure variables; `tacticgen_arxiv_2604_18210_v1`: compactness and deep-defending representation |

## Review outcome required

Approval means accepting these 14 operational inclusion/exclusion boundaries,
aliases, and actor schemas as the catalog terminology contract. It does not
approve any detector recipe, threshold, production concept, effect claim, or
intention claim. If any row is unresolved, that row must be revised before the
14-entry catalog replaces the legacy prose list.
