"""VLM verification of possession-derived events (interception, pass).

Rule engine proposes candidates (timestamps + player info); a local VLM
confirms/rejects each by watching a short clip with the actor highlighted.
Gated behind config.verify_events; fail-open on any error.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from pipeline.stage2_events.types import Event
from pipeline.utils.labels import frame_index_from_labels
from pipeline.utils.video import reencode_to_h264

log = logging.getLogger(__name__)

VERIFY_EVENT_CODES = {"football.interception", "football.pass"}
VERIFY_TEMP_DIRS = ("verify_cache", "verify_clips")


def cleanup_verify_artifacts(output_dir: Path) -> None:
    """Remove VLM verification temp dirs (clips + per-event cache). Keeps events.json and events_verification.json."""
    output_dir = Path(output_dir)
    removed: List[str] = []
    for name in VERIFY_TEMP_DIRS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
    if removed:
        log.info("Cleaned up verify temp: %s", ", ".join(removed))


def parse_verdict(raw: str) -> dict:
    """Extract the first JSON object from a VLM response; uncertain on failure."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return {"verdict": "uncertain", "reason": "unparseable"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "uncertain", "reason": "unparseable"}
    if "verdict" not in data:
        data["verdict"] = "uncertain"
    return data


def _normalize_outcome(raw) -> Optional[str]:
    """Map VLM outcome field to success|failure, or None if missing/invalid."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("success", "successful", "succeeded", "true", "yes", "1"):
        return "success"
    if s in ("failure", "failed", "fail", "unsuccessful", "false", "no", "0"):
        return "failure"
    return None


def apply_verdict(event: Event, verdict: dict) -> Optional[Event]:
    """Mutate/keep/drop an event per verdict. Returns None to drop.

    Tags written:
      verified: true | uncertain | false
      outcome:  success | failure  (only when the action is confirmed to have occurred)
    """
    if verdict.get("error"):
        event.tags["verified"] = "false"
        return event

    v = str(verdict.get("verdict") or "uncertain").lower()
    if v == "reject":
        return None
    if v == "uncertain":
        event.confidence = round(event.confidence * 0.5, 2)
        event.tags["verified"] = "uncertain"
        return event

    # confirm: action occurred; outcome says whether it succeeded.
    event.tags["verified"] = "true"
    outcome = _normalize_outcome(verdict.get("outcome"))
    if outcome is not None:
        event.tags["outcome"] = outcome
    if outcome == "failure":
        # Failed attempts are less important for commentary highlights.
        event.confidence = round(event.confidence * 0.5, 2)
        event.importance = round(event.importance * 0.5, 2)

    actor_jersey = str(verdict.get("actor_jersey") or "").strip()
    actor_team = str(verdict.get("actor_team") or "").strip()
    if actor_jersey:
        event.player_jersey = actor_jersey
    if actor_team in ("left", "right"):
        event.player_team = actor_team
    return event


def build_verify_prompt(event: Event) -> str:
    jersey = event.player_jersey or "unknown"
    team = event.player_team or "unknown"
    name = event.display_name_en
    name_cn = event.display_name_cn
    return (
        "You are a football video analyst. A tracking system flagged a candidate "
        f'"{name}" ({name_cn}) at t={event.timestamp_s:.2f}s '
        f"by the highlighted player (red box): jersey #{jersey}, {team} team.\n"
        "Watch the short clip and answer TWO separate questions:\n"
        f"1) verdict — Did this player actually ATTEMPT a {name} at this moment?\n"
        "   confirm = the action attempt happened; reject = it did not happen; "
        "uncertain = cannot tell.\n"
        f"2) outcome — If verdict is confirm, did the {name} SUCCEED?\n"
        "   success = they won/completed it (e.g. interception wins the ball; "
        "pass reaches a teammate).\n"
        "   failure = they attempted but failed (e.g. interception miss, pass "
        "intercepted/incomplete).\n"
        "   Omit outcome if verdict is reject or uncertain.\n"
        "Output ONLY JSON:\n"
        '{"verdict": "confirm"|"reject"|"uncertain", '
        '"outcome": "success"|"failure"|null, '
        '"actor_jersey": "<number or empty>", '
        '"actor_team": "left"|"right"|"", '
        '"reason": "<short>"}'
    )


def _build_bbox_index(predictions_json_path: str) -> Dict[Tuple[int, int], dict]:
    """Map (frame_num, track_id) -> bbox_image dict."""
    with open(predictions_json_path, encoding="utf-8") as f:
        data = json.load(f)
    image_id_to_frame, _ = frame_index_from_labels(data)
    index: Dict[Tuple[int, int], dict] = {}
    for ann in data.get("annotations", []):
        frame_num = image_id_to_frame.get(str(ann.get("image_id", "")))
        track_id = ann.get("track_id")
        bbox = ann.get("bbox_image")
        if frame_num is None or track_id is None or not isinstance(bbox, dict):
            continue
        index[(frame_num, int(track_id))] = bbox
    return index


def _draw_actor(frame, bbox: dict) -> None:
    x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
    w, h = int(bbox.get("w", 0)), int(bbox.get("h", 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)


def build_actor_clip(
    event: Event,
    frames_dir: Path,
    bbox_index: Dict[Tuple[int, int], dict],
    out_path: Path,
    fps: int = 25,
    window_s: float = 0.5,
) -> Optional[Path]:
    """Cut a ±window clip around event.frame_id with the actor bbox drawn."""
    frames_dir = Path(frames_dir)
    half = int(round(window_s * fps))
    start = max(1, event.frame_id - half)
    end = event.frame_id + half
    imgs = []
    for fnum in range(start, end + 1):
        fp = frames_dir / f"{fnum:06d}.jpg"
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if event.track_id is not None:
            bbox = bbox_index.get((fnum, int(event.track_id)))
            if bbox:
                _draw_actor(img, bbox)
        imgs.append(img)
    if not imgs:
        return None
    h, w = imgs[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )
    for im in imgs:
        writer.write(im)
    writer.release()
    reencode_to_h264(tmp, out_path)
    tmp.unlink(missing_ok=True)
    return out_path


def verify_events(
    events: List[Event],
    predictions_json_path: str,
    frames_dir: Path,
    output_dir: Path,
    adapter,
    fps: int = 25,
    window_s: float = 0.5,
    force: bool = False,
) -> Tuple[List[Event], List[dict]]:
    """Verify interception/pass candidates with a VLM adapter. Fail-open."""
    output_dir = Path(output_dir)
    cache_dir = output_dir / "verify_cache"
    clip_dir = output_dir / "verify_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    bbox_index = _build_bbox_index(predictions_json_path)
    verified: List[Event] = []
    audit: List[dict] = []

    for ev in events:
        if ev.event_code not in VERIFY_EVENT_CODES or ev.track_id is None:
            verified.append(ev)
            continue

        cache_path = cache_dir / f"{ev.event_id}.json"
        if cache_path.exists() and not force:
            verdict = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            clip_path = clip_dir / f"{ev.event_id}.mp4"
            built = build_actor_clip(
                ev, frames_dir, bbox_index, clip_path, fps=fps, window_s=window_s
            )
            if built is None:
                verdict = {"verdict": "uncertain", "reason": "no frames", "error": True}
            else:
                try:
                    raw = adapter.generate(build_verify_prompt(ev), built)
                    verdict = parse_verdict(raw)
                except Exception as exc:
                    # Keep a short reason only — CUDA/nvrtc dumps are huge.
                    short = str(exc).strip().splitlines()[-1][:200]
                    log.warning("verify failed for %s: %s", ev.event_id, short)
                    verdict = {
                        "verdict": "uncertain",
                        "reason": f"error: {short}",
                        "error": True,
                    }
            # Do not cache infra errors — retry should re-attempt without --force.
            if not verdict.get("error"):
                cache_path.write_text(
                    json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        kept = apply_verdict(ev, verdict)
        audit.append({
            "event_id": ev.event_id,
            "event_code": ev.event_code,
            "timestamp_s": ev.timestamp_s,
            "verdict": verdict.get("verdict"),
            "outcome": _normalize_outcome(verdict.get("outcome")),
            "reason": verdict.get("reason"),
            "kept": kept is not None,
        })
        if kept is not None:
            verified.append(kept)

    (output_dir / "events_verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return verified, audit
