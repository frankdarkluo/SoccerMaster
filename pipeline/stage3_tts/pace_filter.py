"""Merge short commentary segments for TTS.

Text is NEVER truncated (2026-07-08 decision: fitting is a speech-rate
problem — CosyVoice speed, then atempo, then spill — never a delete-words
problem). This module only merges segments shorter than MIN_SEGMENT_S with
their neighbours until each is at least TARGET_MIN_S long.
"""
from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)

MIN_SEGMENT_S = 2.0
TARGET_MIN_S = 3.0


def filter_segments(segments: List[dict]) -> List[dict]:
    """Return a new list with short segments merged. Text passes through verbatim."""
    merged = _merge_short(segments)
    log.info("Pace filter: %d segments → %d after merge", len(segments), len(merged))
    return merged


def _merge_short(segments: List[dict]) -> List[dict]:
    if not segments:
        return []

    out: list[dict] = []
    buf: dict | None = None

    for seg in segments:
        if buf is None:
            buf = _copy_seg(seg)
            continue

        buf_dur = buf["end_s"] - buf["timestamp_s"]

        if buf_dur < MIN_SEGMENT_S:
            buf = _merge_pair(buf, seg)
            continue

        seg_dur = seg["end_s"] - seg["timestamp_s"]
        if seg_dur < MIN_SEGMENT_S and buf_dur < TARGET_MIN_S:
            buf = _merge_pair(buf, seg)
            continue

        out.append(buf)
        buf = _copy_seg(seg)

    if buf is not None:
        if out and (buf["end_s"] - buf["timestamp_s"]) < MIN_SEGMENT_S:
            out[-1] = _merge_pair(out[-1], buf)
        else:
            out.append(buf)

    return out


def _copy_seg(seg: dict) -> dict:
    out = {
        "timestamp_s": seg["timestamp_s"],
        "end_s": seg["end_s"],
        "text_zh": seg.get("text_zh", ""),
        "text_en": seg.get("text_en", ""),
        "events_referenced": list(seg.get("events_referenced", [])),
    }
    if seg.get("energy"):
        out["energy"] = seg["energy"]
    return out


def _merge_pair(a: dict, b: dict) -> dict:
    joiner_zh = "，" if a.get("text_zh") and b.get("text_zh") else ""
    joiner_en = " " if a.get("text_en") and b.get("text_en") else ""
    merged = {
        "timestamp_s": a["timestamp_s"],
        "end_s": b["end_s"],
        "text_zh": a.get("text_zh", "") + joiner_zh + b.get("text_zh", ""),
        "text_en": a.get("text_en", "") + joiner_en + b.get("text_en", ""),
        "events_referenced": list(
            dict.fromkeys(a.get("events_referenced", []) + b.get("events_referenced", []))
        ),
    }
    # keep the hotter energy of the pair
    order = ("calm", "engaged", "excited", "explosive")
    energies = [e for e in (a.get("energy"), b.get("energy")) if e in order]
    if energies:
        merged["energy"] = max(energies, key=order.index)
    return merged
