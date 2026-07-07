"""Merge short commentary segments and enforce character budgets for TTS.

Rules (from grill-me session):
  - Segments shorter than MIN_SEGMENT_S are merged with adjacent ones until
    the combined duration reaches at least TARGET_MIN_S.
  - Character budget: ZH_CHARS_PER_S * (duration - GAP_S) for Chinese,
    EN_WORDS_PER_S * (duration - GAP_S) for English.
  - Text that exceeds the budget is truncated at the nearest punctuation
    boundary, falling back to a hard cut.
"""
from __future__ import annotations

import logging
import re
from typing import List

log = logging.getLogger(__name__)

MIN_SEGMENT_S = 2.0
TARGET_MIN_S = 3.0
GAP_S = 0.0
ZH_CHARS_PER_S = 5.0
EN_WORDS_PER_S = 3.3

_HIGHLIGHT_KEYWORDS_ZH = ("进球", "球进", "射门", "GOAL", "太精彩")
_HIGHLIGHT_KEYWORDS_EN = (
    "GOAL!", "GOOOOAL", "GOAL", "STRIKE!", "STRIKE",
    "fires", "finishes", "finish", "long range",
    "Wow!", "Wow",
)

# Action verbs kept when compressing English to subject+verb form.
_EN_ACTIONS = (
    "intercepts", "intercept", "clears", "clear", "shoots", "shoot",
    "finishes", "finish", "wins", "win", "reads", "read", "hits", "hit",
    "fires", "fire", "strikes", "strike", "battles", "battle",
    "advances", "advance", "makes", "make",
)


def filter_segments(segments: List[dict]) -> List[dict]:
    """Return a new list with short segments merged and text budgets enforced."""
    merged = _merge_short(segments)
    trimmed = [_enforce_budget(seg) for seg in merged]
    log.info(
        "Pace filter: %d segments → %d after merge/trim",
        len(segments), len(trimmed),
    )
    return trimmed


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
    return {
        "timestamp_s": seg["timestamp_s"],
        "end_s": seg["end_s"],
        "text_zh": seg.get("text_zh", ""),
        "text_en": seg.get("text_en", ""),
        "events_referenced": list(seg.get("events_referenced", [])),
    }


def _merge_pair(a: dict, b: dict) -> dict:
    joiner_zh = "，" if a.get("text_zh") and b.get("text_zh") else ""
    joiner_en = " " if a.get("text_en") and b.get("text_en") else ""
    return {
        "timestamp_s": a["timestamp_s"],
        "end_s": b["end_s"],
        "text_zh": a.get("text_zh", "") + joiner_zh + b.get("text_zh", ""),
        "text_en": a.get("text_en", "") + joiner_en + b.get("text_en", ""),
        "events_referenced": list(
            dict.fromkeys(a.get("events_referenced", []) + b.get("events_referenced", []))
        ),
    }


def _enforce_budget(seg: dict) -> dict:
    dur = seg["end_s"] - seg["timestamp_s"]
    usable = max(dur - GAP_S, 0.5)

    zh_max = int(usable * ZH_CHARS_PER_S)
    en_max = int(usable * EN_WORDS_PER_S)

    seg = dict(seg)
    seg["text_zh"] = _truncate_zh(seg.get("text_zh", ""), zh_max)
    seg["text_en"] = _truncate_en(seg.get("text_en", ""), en_max)
    return seg


_ZH_SENT_SPLIT = re.compile(r"(?<=[。！？])")
_ZH_CLAUSE_SPLIT = re.compile(r"(?<=[，；、,;])")


def _reorder_highlight_first(sentences: list[str]) -> list[str]:
    """Move sentences containing highlight keywords to the front."""
    hi: list[str] = []
    lo: list[str] = []
    for s in sentences:
        if any(kw in s for kw in _HIGHLIGHT_KEYWORDS_ZH):
            hi.append(s)
        else:
            lo.append(s)
    return hi + lo


def _strip_leading_punct(text: str) -> str:
    return text.lstrip("，。！？；、,!?; ")


def _truncate_zh(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    sentences = [s for s in _ZH_SENT_SPLIT.split(text) if s]
    sentences = _reorder_highlight_first(sentences)
    result = _accumulate(sentences, max_chars)
    if result:
        return _strip_leading_punct(result)
    # fall back to clause boundaries within the first sentence
    clauses = [c for c in _ZH_CLAUSE_SPLIT.split(sentences[0] if sentences else text) if c]
    result = _accumulate(clauses, max_chars)
    if result:
        return _strip_leading_punct(result.rstrip("，、,; "))
    # hard cut — reserve 1 char for trailing 。
    cut = _strip_leading_punct(text)[:max(max_chars - 1, 1)]
    return cut + "。"


def _accumulate(parts: list[str], budget: int) -> str:
    out = ""
    for p in parts:
        if len(out) + len(p) > budget:
            break
        out += p
    return out


def _en_priority_score(sentence: str) -> int:
    """Higher = keep first. Prefer action-bearing highlights over bare 'Wow!'."""
    s = sentence.strip()
    score = 0
    lower = s.lower()
    if any(kw.lower() in lower for kw in _HIGHLIGHT_KEYWORDS_EN):
        score += 10
    if re.search(r"Number\s+\d+", s, re.I):
        score += 5
    if any(re.search(rf"\b{re.escape(a)}\b", s, re.I) for a in _EN_ACTIONS):
        score += 3
    # GOAL! always tops the list; bare Wow! ranks below full action lines
    if re.search(r"GOO+AL|STRIKE", s, re.I) and len(s.split()) <= 2:
        score += 20
    elif len(s.split()) <= 2:
        score -= 4
    return score


def _reorder_highlight_first_en(sentences: list[str]) -> list[str]:
    return sorted(sentences, key=_en_priority_score, reverse=True)


def _normalize_en_verb(verb: str) -> str:
    v = verb.lower()
    if v.endswith("s") or v.endswith("es"):
        return v
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _subject_verb_en(text: str) -> str:
    """Compress to 'Number N verbs.' / 'GOAL!' when full text is too long."""
    if re.search(r"GOO+AL", text, re.I):
        return "GOAL!"
    if re.search(r"\bSTRIKE\b", text, re.I):
        return "What a strike!"

    number = re.search(r"Number\s+(\d+)", text, re.I)
    action = None
    for a in _EN_ACTIONS:
        if re.search(rf"\b{re.escape(a)}\b", text, re.I):
            action = _normalize_en_verb(a)
            break

    if number and action:
        return f"Number {number.group(1)} {action}."
    if action:
        subject = "Defender" if re.search(r"\bdefender\b", text, re.I) else "Player"
        return f"{subject} {action}."
    return ""


def _finalize_en(words: list[str]) -> str:
    truncated = " ".join(words).rstrip(",;: ")
    truncated = re.sub(r"^(but|and|or|so)\s+", "", truncated, flags=re.I)
    if truncated:
        truncated = truncated[0].upper() + truncated[1:]
    if not truncated.endswith((".", "!", "?")):
        truncated += "."
    return truncated


def _truncate_en(text: str, max_words: int) -> str:
    """English budget cut: highlight-first (C), then subject+verb (B)."""
    words = text.split()
    if len(words) <= max_words:
        return text

    # Prefer clause splits when source has no sentence punctuation.
    if re.search(r"[.!?]", text):
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    else:
        raw_sents = [s.strip() for s in re.split(r",\s*", text) if s.strip()]

    raw_sents = _reorder_highlight_first_en(raw_sents)

    # C: keep whole highlight / priority sentences that fit.
    acc_words: list[str] = []
    for sent in raw_sents:
        sw = sent.split()
        if len(acc_words) + len(sw) > max_words:
            break
        acc_words.extend(sw)
    if acc_words:
        return _finalize_en(acc_words)

    # B: compress the top-priority sentence to subject+verb.
    priority = raw_sents[0] if raw_sents else text
    sv = _subject_verb_en(priority)
    if sv:
        sv_words = sv.split()
        if len(sv_words) <= max_words:
            return sv
        return _finalize_en(sv_words[:max_words])

    return _finalize_en(words[:max_words])
