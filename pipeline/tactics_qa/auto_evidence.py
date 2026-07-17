"""Compose one ClipEvidence record from GSR trajectories."""
from .delivery_infer import classify_delivery, corner_landing_zone
from .evidence import ClipEvidence, PossessionEvent
from .gsr_io import GsrClip
from .kinematics import dead_ball_periods, detect_kicks
from .possession_timeline import team_runs, transition_events
from .restart_infer import infer_restarts

ATTACK_SIDE_MIN_SHARE = 0.6
EVENT_BAND_BEFORE_S = 5.0
EVENT_BAND_AFTER_S = 5.0
LANDING_SEARCH_S = 4.0
HOLDER_NEAR_BALL_M = 1.5


def attacking_side(runs, window) -> str | None:
    duration = {}
    for run in runs:
        overlap = max(
            0.0,
            min(window["end_s"], run["end_s"])
            - max(window["start_s"], run["start_s"]),
        )
        duration[run["team"]] = duration.get(run["team"], 0.0) + overlap
    total = sum(duration.values())
    if not total:
        return None
    team, best = max(duration.items(), key=lambda item: item[1])
    return team if best / total >= ATTACK_SIDE_MIN_SHARE else None


def _possession_segments(clip):
    from pipeline.stage2b.digest import FrameData, possession_segments

    frames, team_by_track = [], {}
    for frame in sorted(set(clip.players) | set(clip.ball)):
        players = [
            {"track_id": track, "role": "player", "team": team,
             "jersey": "", "x": x, "y": y}
            for track, team, x, y in clip.players.get(frame, [])
        ]
        frames.append(FrameData(frame, clip.ball.get(frame), players))
        for track, team, _, _ in clip.players.get(frame, []):
            if team is not None:
                team_by_track.setdefault(track, team)
    return possession_segments(frames, team_by_track)


def _attack_sign(runs, side, clip, window):
    drift = 0.0
    for run in runs:
        if run["team"] != side:
            continue
        start = max(run["start_s"], window["start_s"])
        end = min(run["end_s"], window["end_s"])
        points = [clip.ball[f] for f in sorted(clip.ball)
                  if start <= f / clip.fps <= end]
        if len(points) >= 2:
            drift += points[-1][0] - points[0][0]
    if abs(drift) < 5.0:
        return None
    return 1 if drift > 0 else -1


def _next_touch(clip, kick):
    for frame in sorted(clip.ball):
        time = frame / clip.fps
        if time <= kick.t + 0.2:
            continue
        if time > kick.t + LANDING_SEARCH_S:
            break
        bx, by = clip.ball[frame]
        if any(((px - bx) ** 2 + (py - by) ** 2) ** 0.5 <= HOLDER_NEAR_BALL_M
               for _, _, px, py in clip.players.get(frame, [])):
            return bx, by
    return None


def build_claim_evidence(clip: GsrClip, clip_uid: str, tactic_id: str,
                         window: dict) -> ClipEvidence:
    kicks = detect_kicks(clip.ball, clip.fps)
    restarts = infer_restarts(kicks, dead_ball_periods(clip.ball, clip.fps))
    runs = team_runs(_possession_segments(clip), clip.fps)
    side = attacking_side(runs, window)
    start = window["start_s"]
    events, notes = [], []
    for restart in restarts:
        if start - EVENT_BAND_BEFORE_S <= restart.t <= start + EVENT_BAND_AFTER_S:
            events.append(PossessionEvent(restart.t, "attacking", restart.kind))
            notes.append(
                f"restart {restart.kind}@{restart.t:.1f}s at {restart.xy}")
    if side is None:
        notes.append("attacking side ambiguous: no possession events emitted")
    else:
        events.extend(
            event for event in transition_events(runs, side)
            if start - EVENT_BAND_BEFORE_S <= event.t <= start + EVENT_BAND_AFTER_S
        )

    sign = _attack_sign(runs, side, clip, window) if side else None
    window_kicks = [
        kick for kick in kicks
        if start - 2.0 <= kick.t <= window["end_s"]
    ]
    if not window_kicks:
        notes.append("no key kick detected")
    delivery = None
    if window_kicks:
        key = window_kicks[0]
        delivery = classify_delivery(key, _next_touch(clip, key), sign, restarts)
        if delivery is None:
            notes.append(
                "attack direction ambiguous: no delivery emitted"
                if sign is None else "delivery geometry unobservable"
            )

    landing = None
    if tactic_id == "corner-near-far-post":
        corner = next(
            (restart for restart in restarts
             if restart.kind == "corner"
             and start - EVENT_BAND_BEFORE_S <= restart.t <= window["end_s"]),
            None,
        )
        if corner is not None:
            touch = _next_touch(clip, corner)
            landing = corner_landing_zone(corner.xy[1], touch)
            if touch is None:
                notes.append("corner landing unobservable")

    return ClipEvidence(
        clip_uid, tactic_id, sorted(events, key=lambda event: event.t),
        delivery, landing, "gsr_trajectory", "; ".join(notes))
