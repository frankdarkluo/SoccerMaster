"""Open tactical-tag generation and scoring for the frozen 67-clip benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from pipeline.video_models import (
    DOUBAO_COMPACT_UPLOAD_PROFILE,
    generate_json,
    model_name,
    temporary_doubao_proxies,
)

PHASES = ("phase1_direct", "phase2_observation_first")
PROMPT_VERSIONS = {
    "phase1_direct": "p0-direct-nomination-v1",
    "phase2_observation_first": "p0-observation-first-nomination-v2",
}
SCHEMA_VERSION = "p0-nomination-v1"
EXPERIMENT_ID = "p0-two-phase-nomination-v1"
ONE_SHOT_PHASE = "phase1_video_one_shot"
ONE_SHOT_PROMPT_VERSION = "p0-video-one-shot-nomination-v1"
ONE_SHOT_EXPERIMENT_ID = "p0-video-one-shot-comparison-v1"
ONE_SHOT_V2_OBSERVATION_PROMPT_VERSION = "p0-video-one-shot-v2-observation-v1"
ONE_SHOT_V2_NOMINATION_PROMPT_VERSION = "p0-video-one-shot-v2-nomination-v2"
ONE_SHOT_V2_TEACHING_VERSION = "p0-video-one-shot-v2-teaching-v1"
ONE_SHOT_V2_PHASE = "video_one_shot_v2"
ONE_SHOT_V2_SCHEMA_VERSION = "p0-video-one-shot-v2"
ONE_SHOT_V2_EXPERIMENT_ID = "p0-video-one-shot-v2-observation-locked"
ONE_SHOT_TACTICS = (
    "counter-attack",
    "cutback",
    "line-breaking-pass",
    "run-in-behind",
)
ONE_SHOT_V2_TACTICS = ("line-breaking-pass", "run-in-behind")
ONE_SHOT_EXPECTED = {
    "counter-attack": (24, 23),
    "cutback": (3, 3),
    "line-breaking-pass": (9, 9),
    "run-in-behind": (14, 14),
}
ONE_SHOT_EXEMPLARS = {
    "counter-attack": "data/FIFA Game Library/Transition to attacking/USA U-20 (2018) – Counter-attack – Attacking down the channels.mp4",
    "cutback": "data/FIFA Game Library/In possession/Bellingham (2022) — Use of wide areas — Cut-back cross.mp4",
    "line-breaking-pass": "data/FIFA Game Library/In possession/01_Line_breaks_premium.mp4",
    "run-in-behind": "data/FIFA Game Library/Position Specific/Gakpo (2022) — Role of the number 9 — Running in behind.mp4",
}
ONE_SHOT_V2_TEACHING = {
    "run-in-behind": {
        "evidence": [
            {"span": "3–8s", "observation_zh": "跑动者从最后一线平齐位置配合队友出球启动。"},
            {"span": "8–12s", "observation_zh": "跑动者进入最后一线身后空间并接应来球。"},
        ],
        "required": [
            "the runner starts in front of or level with the last defensive line",
            "the run and the teammate's pass belong to the same continuous move",
            "the target space is behind the last defensive line",
        ],
        "excluded": [
            "ordinary forward support without a coordinated pass",
            "a runner already behind the last line before the move starts",
            "chasing a clearance",
        ],
        "outcome_note": "The headed goal is illustrative, not required for this tactic.",
    },
    "line-breaking-pass": {
        "evidence": [
            {"span": "15–33s", "observation_zh": "画面标出三条防守单元线及可利用通道。"},
            {"span": "34–43s", "observation_zh": "纵向传球穿过防守单元线并到达该线后方。"},
        ],
        "required": [
            "the ball path crosses at least one clearly visible horizontal defensive line",
            "the pass reaches a teammate or usable space beyond that line",
        ],
        "excluded": [
            "mere forward progression without crossing a defensive line",
            "a lateral switch",
            "a pass around rather than through the line",
            "calling a pass line-breaking only because it is long",
        ],
        "outcome_note": "The subsequent advance is illustrative, not required for this tactic.",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_lines(value: str | None) -> list[str]:
    return [line.strip(" -•\t") for line in (value or "").splitlines() if line.strip(" -•\t")]


def load_glossary(path: Path) -> dict[str, dict[str, Any]]:
    cards = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        tactic_id = (row.get("id（英文标识）") or "").strip()
        if not tactic_id or tactic_id.startswith("英文小写连字符"):
            continue
        confusing = []
        for index in ("①", "②", "③"):
            target = (row.get(f"易混淆{index} 对象") or "").strip()
            distinction = (row.get(f"易混淆{index} 区分点") or "").strip()
            if target or distinction:
                confusing.append({"object": target, "distinction": distinction})
        cards[tactic_id] = {
            "tactic_id": tactic_id,
            "name_zh": (row.get("中文名") or "").strip(),
            "name_en": (row.get("英文名") or "").strip(),
            "priority": (row.get("优先级") or "").strip(),
            "definition": (row.get("一句话定义") or "").strip(),
            "observable_cues": split_lines(row.get("可观察特征")),
            "triggers": split_lines(row.get("触发条件")),
            "confusing": confusing,
        }
    return cards


def compact(card: dict[str, Any]) -> dict[str, Any]:
    return {key: card[key] for key in ("tactic_id", "definition", "observable_cues", "triggers", "confusing")}


def response_schema(tactic_ids: list[str], phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    evidence = {
        "type": "object",
        "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "visible_movement_zh": {"type": "string"}, "tactical_link_zh": {"type": "string"},
        },
        "required": ["start_s", "end_s", "visible_movement_zh", "tactical_link_zh"],
        "additionalProperties": False,
    }
    candidate = {
        "rank": {"type": "integer", "enum": [1, 2, 3]},
        "tactic_id": {"type": "string", "enum": tactic_ids},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason_zh": {"type": "string"},
        "matched_cues": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "evidence_spans": {"type": "array", "minItems": 1, "maxItems": 3, "items": evidence},
    }
    if phase == PHASES[1]:
        candidate["supporting_sequence_ids"] = {"type": "array", "minItems": 1, "items": {"type": "string"}}
    properties: dict[str, Any] = {
        "candidates": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "properties": candidate, "required": list(candidate), "additionalProperties": False,
        }},
        "no_nomination_reason_zh": {"type": "string"},
    }
    required = ["candidates", "no_nomination_reason_zh"]
    if phase == PHASES[1]:
        fields = ("sequence_id", "start_s", "end_s", "action_zh", "defensive_change_zh", "space_change_or_use_zh", "attacking_effect_zh", "terminal_result_zh")
        properties = {"observations": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object",
            "properties": {"sequence_id": {"type": "string"}, "start_s": {"type": "number"}, "end_s": {"type": "number"}, **{field: {"type": "string"} for field in fields[3:]}},
            "required": list(fields), "additionalProperties": False,
        }}, **properties}
        required.insert(0, "observations")
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def one_shot_v2_observation_schema() -> dict[str, Any]:
    fields: dict[str, Any] = {
        "sequence_id": {"type": "string"},
        "start_s": {"type": "number"},
        "end_s": {"type": "number"},
        "state_before_zh": {"type": "string"},
        "position_or_action_zh": {"type": "string"},
        "observed_change_zh": {"type": "string"},
        "space_origin": {"type": "string", "enum": ["created", "existing", "unclear"]},
        "space_evidence_zh": {"type": "string"},
        "opponent_constraint_zh": {"type": "string"},
        "space_change_zh": {"type": "string"},
        "functional_effect_zh": {"type": "string"},
        "terminal_result_zh": {"type": "string"},
        "mechanism_chain_zh": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "confidence_reasons": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "evidence_gaps": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "properties": {"observations": {
            "type": "array", "minItems": 1, "maxItems": 5,
            "items": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False},
        }},
        "required": ["observations"],
        "additionalProperties": False,
    }


def one_shot_v2_nomination_schema(tactic_ids: list[str]) -> dict[str, Any]:
    evidence = {
        "type": "object",
        "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}},
        "required": ["start_s", "end_s"],
        "additionalProperties": False,
    }
    candidate = {
        "type": "object",
        "properties": {
            "rank": {"type": "integer", "enum": [1, 2, 3]},
            "tactic_id": {"type": "string", "enum": tactic_ids},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "matched_cues": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "supporting_sequence_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "evidence_spans": {"type": "array", "minItems": 1, "maxItems": 3, "items": evidence},
        },
        "required": [
            "rank", "tactic_id", "confidence", "matched_cues",
            "supporting_sequence_ids", "evidence_spans",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"candidates": {"type": "array", "maxItems": 3, "items": candidate}},
        "required": ["candidates"],
        "additionalProperties": False,
    }


def build_one_shot_v2_observation_prompt(clip_alias: str, duration_s: float) -> str:
    return f"""你是足球比赛视频观察员。只观察查询视频，不识别或猜测任何战术标签。

完整观看视频，按1至5段连续过程记录可见机制。每段依次说明：动作前状态 → 关键动作或站位 → 对手可见反应 → 空间来源及证据 → 对手约束 → 空间变化 → 对进攻的可见作用 → 最终结果，再用mechanism_chain_zh概括这条可见链条。

不得输出战术ID、战术名称、战术类别或候选判断；不得从文件名、球队身份、解说、教练意图或画外信息推断。只有画面明确显示变化和空间证据时才能使用space_origin=created；证据不足时写明不清楚并列入evidence_gaps。时间必须位于视频边界内。只返回匹配schema的JSON。

Query clip: {clip_alias}
Query bounds: 0.000 to {duration_s:.3f} seconds.
OUTPUT JSON SCHEMA:
{json.dumps(one_shot_v2_observation_schema(), ensure_ascii=False)}
"""


def build_one_shot_v2_video_instructions(
    exemplar_card: dict[str, Any], teaching: dict[str, Any], clip_alias: str,
) -> tuple[str, str]:
    label = f"{exemplar_card['tactic_id']} ({exemplar_card['name_zh']}; {exemplar_card['name_en']})"
    return (
        f"REFERENCE VIDEO. This is a confirmed positive example of {label}. "
        f"Its frozen teaching card applies only to this video: {json.dumps(teaching, ensure_ascii=False)}",
        f"QUERY VIDEO ({clip_alias}). This video may or may not contain the demonstrated tactic. All nominations and evidence must describe this video only.",
    )


def build_one_shot_v2_nomination_prompt(
    clip_alias: str,
    duration_s: float,
    cards: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    return f"""You are performing observation-locked open football-tactic nomination.

The REFERENCE teaching information applies only to the reference video. The QUERY video may or may not contain the demonstrated tactic. Compare the locked query observations against ALL 19 P0 concept cards and nominate only the 0 to 3 best-supported tactics.

Every candidate must cite supporting_sequence_ids from LOCKED QUERY OBSERVATIONS. Every evidence span must be fully contained in one of its cited observations and must describe the QUERY video only. Do not rewrite or summarize observations and do not output any new factual prose. Return only tactic IDs, confidence, exact copied P0-card cues, supporting sequence IDs, and query-video time bounds. Reference-video times are never query evidence. Return JSON only.

LOCKED QUERY OBSERVATIONS:
{json.dumps(observations, ensure_ascii=False)}

Query clip: {clip_alias}
Query bounds: 0.000 to {duration_s:.3f} seconds.
P0 CONCEPT CARDS:
{json.dumps([compact(card) for card in cards], ensure_ascii=False)}

OUTPUT JSON SCHEMA:
{json.dumps(one_shot_v2_nomination_schema([card['tactic_id'] for card in cards]), ensure_ascii=False)}
"""


def validate_one_shot_v2_observations(
    payload: Any, cards: dict[str, dict[str, Any]], duration_s: float,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"observations"}:
        raise ValueError("observation response fields are not strict")
    observations = payload["observations"]
    if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
        raise ValueError("observations must contain 1 to 5 sequences")
    fields = set(one_shot_v2_observation_schema()["properties"]["observations"]["items"]["properties"])
    forbidden = {
        value.casefold()
        for card in cards.values()
        for value in (card["tactic_id"], card["name_zh"], card["name_en"])
        if value
    }
    seen = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("observation sequence fields are not strict")
        sequence_id = item["sequence_id"]
        if not isinstance(sequence_id, str) or not sequence_id or sequence_id in seen:
            raise ValueError("invalid observation sequence_id")
        seen.add(sequence_id)
        start_s, end_s = item["start_s"], item["end_s"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start_s, end_s)) or not 0 <= start_s < end_s <= duration_s + 0.001:
            raise ValueError("observation outside query bounds")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("invalid observation confidence")
        if item["space_origin"] not in {"created", "existing", "unclear"}:
            raise ValueError("invalid space_origin")
        list_fields = ("confidence_reasons", "evidence_gaps")
        if (
            any(not isinstance(item[field], list) for field in list_fields)
            or not item["confidence_reasons"]
            or any(not isinstance(value, str) for field in list_fields for value in item[field])
        ):
            raise ValueError("invalid observation reasons or gaps")
        text_fields = fields - {
            "sequence_id", "start_s", "end_s", "space_origin", "confidence",
            *list_fields,
        }
        if any(not isinstance(item[field], str) for field in text_fields):
            raise ValueError("observation text fields must be strings")
        text = " ".join(
            value
            for field, value in item.items()
            if field not in {"start_s", "end_s", "confidence"}
            for value in (value if isinstance(value, list) else [value])
            if isinstance(value, str)
        ).casefold()
        if any(term in text for term in forbidden):
            raise ValueError("observation leaks tactic name")
    return payload


def validate_one_shot_v2_nomination(
    payload: Any,
    cards: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    schema = one_shot_v2_nomination_schema(list(cards))
    candidate_schema = schema["properties"]["candidates"]["items"]
    candidate_fields = set(candidate_schema["properties"])
    evidence_fields = {"start_s", "end_s"}
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("nomination response fields are not strict")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise ValueError("nomination candidates must contain at most three items")
    by_id = {item["sequence_id"]: item for item in observations}
    seen_tactics = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
            raise ValueError("nomination candidate fields are not strict")
        rank = candidate["rank"]
        tactic_id, confidence = candidate["tactic_id"], candidate["confidence"]
        if (
            isinstance(rank, bool) or not isinstance(rank, int) or rank not in {1, 2, 3}
            or tactic_id not in cards
            or tactic_id in seen_tactics
            or isinstance(confidence, bool)
            or not isinstance(confidence, int)
            or not 0 <= confidence <= 100
        ):
            raise ValueError("invalid candidate rank, tactic, or confidence")
        # ponytail: list order defines Top-k; rank/confidence are display metadata, not evidence.
        seen_tactics.add(tactic_id)
        supporting = candidate["supporting_sequence_ids"]
        if (
            not isinstance(supporting, list)
            or not supporting
            or any(not isinstance(value, str) for value in supporting)
            or len(supporting) != len(set(supporting))
            or not set(supporting) <= set(by_id)
        ):
            raise ValueError("candidate references unknown observation")
        matched = candidate["matched_cues"]
        allowed_cues = set(cards[tactic_id]["observable_cues"] + cards[tactic_id]["triggers"])
        if (
            not isinstance(matched, list)
            or not matched
            or any(not isinstance(value, str) for value in matched)
            or not any(cue in allowed_cues for cue in matched)
        ):
            raise ValueError("candidate does not copy a cue from its P0 card")
        spans = candidate["evidence_spans"]
        if not isinstance(spans, list) or not 1 <= len(spans) <= 3:
            raise ValueError("candidate requires 1 to 3 evidence spans")
        cited = [by_id[sequence_id] for sequence_id in supporting]
        for span in spans:
            if not isinstance(span, dict) or set(span) != evidence_fields:
                raise ValueError("evidence span fields are not strict")
            start_s, end_s = span["start_s"], span["end_s"]
            if (
                any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start_s, end_s))
                or not 0 <= start_s < end_s <= duration_s + 0.001
            ):
                raise ValueError("evidence span outside query bounds")
            if not any(
                float(item["start_s"]) <= float(start_s)
                and float(end_s) <= float(item["end_s"])
                for item in cited
            ):
                raise ValueError("candidate evidence is outside its locked observation")
    return payload


def build_prompt(clip_alias: str, duration_s: float, cards: list[dict[str, Any]], phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    instruction = (
        "Compare all 19 cards directly against the video, then nominate only the 0 to 3 best-supported tactics. Do not output per-card screening rows or observations."
        if phase == PHASES[0] else
        "Before considering tactic names, write 1 to 5 purely descriptive observations. Only then compare all 19 cards and nominate 0 to 3 tactics. Observation fields must not contain tactic IDs, names, categories, or tentative tactical judgments. Every candidate must reference supporting_sequence_ids."
    )
    return f"""You are performing open football-tactic nomination from a fixed list.

Watch the complete video and read ALL 19 P0 concept cards. A supplied card does not imply that its tactic is present. Multiple tactics may coexist, and an empty nomination is valid.

{instruction}

Nominate a tactic only when its defining mechanism is visibly supported. Return at most three unique candidates in decreasing confidence order. Copy at least one matched cue exactly from its card and cite 1 to 3 decisive time spans. Write reasons and evidence in Chinese. Do not infer from filenames, prior labels, commentary, shirt identity, coach intent, another model, or unstated facts. Return JSON only.

Clip: {clip_alias}
Clip bounds: 0.000 to {duration_s:.3f} seconds.
P0 CONCEPT CARDS:
{json.dumps([compact(card) for card in cards], ensure_ascii=False)}

OUTPUT JSON SCHEMA:
{json.dumps(response_schema([card['tactic_id'] for card in cards], phase), ensure_ascii=False)}
"""


def build_one_shot_prompt(
    clip_alias: str, duration_s: float, cards: list[dict[str, Any]], exemplar_card: dict[str, Any],
) -> str:
    label = f"{exemplar_card['tactic_id']} ({exemplar_card['name_zh']}; {exemplar_card['name_en']})"
    return f"""Two videos are attached in order.
The FIRST video is a confirmed positive example of {label}.
The example label is ground truth only for the FIRST video; it is not evidence about the query.
The SECOND video is the query video and may or may not contain the demonstrated tactic.
Use the example only to calibrate the demonstrated mechanism, then compare the SECOND video against ALL 19 P0 concept cards.
All nominations, reasons, and evidence must describe the SECOND video only.
The evidence spans must refer only to the SECOND video and use its time bounds.
""" + build_prompt(clip_alias, duration_s, cards, PHASES[0])


def validate_response(payload: Any, tactic_ids: list[str], cards: dict[str, dict[str, Any]], phase: str) -> dict[str, Any]:
    expected = {"candidates", "no_nomination_reason_zh"}
    sequence_ids: set[str] = set()
    if phase == PHASES[1]:
        expected.add("observations")
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("response fields are not strict")
    if phase == PHASES[1]:
        observations = payload["observations"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
            raise ValueError("observations must contain 1 to 5 sequences")
        forbidden = {value.casefold() for card in cards.values() for value in (card["tactic_id"], card["name_zh"], card["name_en"]) if value}
        for item in observations:
            sequence_id = item.get("sequence_id")
            if not isinstance(sequence_id, str) or not sequence_id or sequence_id in sequence_ids:
                raise ValueError("invalid observation sequence_id")
            sequence_ids.add(sequence_id)
            text = " ".join(str(item.get(field, "")) for field in ("action_zh", "defensive_change_zh", "space_change_or_use_zh", "attacking_effect_zh", "terminal_result_zh")).casefold()
            if any(term in text for term in forbidden):
                raise ValueError("observation leaks tactic name")
    candidates, reason = payload["candidates"], payload["no_nomination_reason_zh"]
    if not isinstance(candidates, list) or len(candidates) > 3 or not isinstance(reason, str):
        raise ValueError("invalid candidates or reason")
    if not candidates and not reason.strip():
        raise ValueError("inconsistent no-nomination reason")
    previous, seen = 101, set()
    for index, candidate in enumerate(candidates, 1):
        confidence, tactic_id = candidate.get("confidence"), candidate.get("tactic_id")
        if candidate.get("rank") != index or tactic_id not in tactic_ids or tactic_id in seen or isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100 or confidence > previous:
            raise ValueError("invalid candidate rank, tactic, or confidence")
        previous, _ = confidence, seen.add(tactic_id)
        if not candidate.get("evidence_spans"):
            raise ValueError("candidate requires evidence")
        if phase == PHASES[1] and (not candidate.get("supporting_sequence_ids") or not set(candidate["supporting_sequence_ids"]) <= sequence_ids):
            raise ValueError("candidate references unknown observation")
    return payload


def validate_evidence_bounds(payload: dict[str, Any], duration_s: float) -> None:
    for candidate in payload["candidates"]:
        for span in candidate["evidence_spans"]:
            try:
                start_s, end_s = float(span["start_s"]), float(span["end_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("evidence span outside query bounds") from exc
            if any(isinstance(span.get(key), bool) for key in ("start_s", "end_s")) or not 0 <= start_s < end_s <= duration_s + 0.001:
                raise ValueError("evidence span outside query bounds")


def duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def frozen_sources(repo_root: Path, source_rows: Path, ground_truth: Path) -> dict[str, dict[str, Any]]:
    index = {row["clip_uid"]: {**row, "absolute_video_path": (repo_root / row["video_path"]).resolve()} for row in load_jsonl(source_rows) if row.get("video_path")}
    clip_uids = sorted({claim["clip_uid"] for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary"})
    if len(clip_uids) != 67:
        raise ValueError(f"expected 67 frozen clips, got {len(clip_uids)}")
    if missing := set(clip_uids) - set(index):
        raise ValueError(f"clips missing from source_rows: {sorted(missing)}")
    return {clip_uid: index[clip_uid] for clip_uid in clip_uids}


def build_one_shot_manifest(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for claim in claims:
        if claim.get("score_set") != "primary" or claim.get("tactic_id") not in ONE_SHOT_TACTICS:
            continue
        key = claim["tactic_id"], claim["clip_uid"]
        item = grouped.setdefault(key, {
            "tactic_id": claim["tactic_id"],
            "clip_uid": claim["clip_uid"],
            "clip_id": claim["clip_id"],
            "claims": [],
        })
        item["claims"].append({key: claim[key] for key in ("claim_id", "gt_verdict", "window")})
    manifest = [grouped[key] for key in sorted(grouped)]
    for item in manifest:
        item["claims"].sort(key=lambda claim: claim["claim_id"])
    observed = {
        tactic_id: (
            sum(len(item["claims"]) for item in manifest if item["tactic_id"] == tactic_id),
            sum(item["tactic_id"] == tactic_id for item in manifest),
        )
        for tactic_id in ONE_SHOT_TACTICS
    }
    if observed != ONE_SHOT_EXPECTED:
        raise ValueError(f"unexpected one-shot counts: {observed}")
    if len(manifest) != 49 or sum(len(item["claims"]) for item in manifest) != 50:
        raise ValueError("expected 49 one-shot pairs and 50 claims")
    return manifest


def build_one_shot_v2_manifest(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = [
        item for item in build_one_shot_manifest(claims)
        if item["tactic_id"] in ONE_SHOT_V2_TACTICS
    ]
    verdicts = {
        tactic_id: {
            verdict: sum(
                claim["gt_verdict"] == verdict
                for item in manifest if item["tactic_id"] == tactic_id
                for claim in item["claims"]
            )
            for verdict in ("present", "absent")
        }
        for tactic_id in ONE_SHOT_V2_TACTICS
    }
    if verdicts != {
        "line-breaking-pass": {"present": 3, "absent": 6},
        "run-in-behind": {"present": 10, "absent": 4},
    } or len(manifest) != 23:
        raise ValueError(f"unexpected one-shot V2 claims: {verdicts}")
    return manifest


def run(output_root: Path, glossary_path: Path, source_rows: Path, ground_truth: Path, repo_root: Path, *, phase: str, provider: str, clip_uids: Iterable[str] | None = None, force: bool = False, retry_failed: bool = False) -> dict[str, int | str]:
    cards = sorted((card for card in load_glossary(glossary_path).values() if card["priority"] == "P0"), key=lambda card: card["tactic_id"])
    if len(cards) != 19:
        raise ValueError(f"expected 19 P0 cards, got {len(cards)}")
    cards_by_id, sources = {card["tactic_id"]: card for card in cards}, frozen_sources(repo_root, source_rows, ground_truth)
    selected = set(clip_uids or sources)
    if unknown := selected - set(sources):
        raise ValueError(f"unknown clips: {sorted(unknown)}")
    counts = Counter(clips=0, calls=0, success=0, failed=0, skipped=0)
    for clip_uid in sorted(selected):
        video = sources[clip_uid]["absolute_video_path"]
        digest = sha256(video)
        prompt = build_prompt(f"clip-{digest[:12]}.mp4", duration(video), cards, phase)
        target = output_root / phase / provider / f"{clip_uid.split(':', 1)[-1]}.json"
        if target.is_file() and not force:
            current = json.loads(target.read_text(encoding="utf-8"))
            if not retry_failed or current.get("status") != "failed":
                counts.update(clips=1, skipped=1)
                continue
        try:
            payload, usage, attempts = generate_json(provider, prompt, response_schema(list(cards_by_id), phase), video_path=video)
            validate_response(payload, list(cards_by_id), cards_by_id, phase)
            result = {"status": "success", **payload}
        except Exception as exc:
            usage, attempts = [], getattr(exc, "attempts", 1)
            result = {"status": "failed", "failed_stage": "provider_request_or_validation", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        result.update({"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "phase": phase, "provider": provider, "model": model_name(provider), "clip_uid": clip_uid, "clip_sha256": digest, "prompt_version": PROMPT_VERSIONS[phase], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "temperature": 0.0, "attempts": attempts, "api_usage": usage})
        write_json(target, result)
        counts.update(clips=1, calls=1)
        counts[result["status"]] += 1
    return {"phase": phase, "provider": provider, **dict(counts)}


def run_one_shot(
    output_root: Path, glossary_path: Path, source_rows: Path, ground_truth: Path, repo_root: Path,
    *, provider: str, tactic_ids: Iterable[str] | None = None, clip_uids: Iterable[str] | None = None, force: bool = False, retry_failed: bool = False,
) -> dict[str, int | str]:
    cards = sorted((card for card in load_glossary(glossary_path).values() if card["priority"] == "P0"), key=lambda card: card["tactic_id"])
    if len(cards) != 19:
        raise ValueError(f"expected 19 P0 cards, got {len(cards)}")
    cards_by_id = {card["tactic_id"]: card for card in cards}
    manifest = build_one_shot_manifest(load_jsonl(ground_truth))
    write_jsonl(output_root / "evaluation/shot_comparison/manifest.jsonl", manifest)
    selected_tactics = set(tactic_ids or ONE_SHOT_TACTICS)
    if unknown := selected_tactics - set(ONE_SHOT_TACTICS):
        raise ValueError(f"unknown one-shot tactics: {sorted(unknown)}")
    selected_clips = set(clip_uids or (item["clip_uid"] for item in manifest))
    known_clips = {item["clip_uid"] for item in manifest}
    if unknown := selected_clips - known_clips:
        raise ValueError(f"unknown one-shot clips: {sorted(unknown)}")
    sources = frozen_sources(repo_root, source_rows, ground_truth)
    exemplars = {}
    for tactic_id in selected_tactics:
        path = (repo_root / ONE_SHOT_EXEMPLARS[tactic_id]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        exemplars[tactic_id] = path, sha256(path), duration(path)
    counts = Counter(pairs=0, calls=0, success=0, failed=0, skipped=0)
    for item in manifest:
        tactic_id, clip_uid = item["tactic_id"], item["clip_uid"]
        if tactic_id not in selected_tactics or clip_uid not in selected_clips:
            continue
        query = sources[clip_uid]["absolute_video_path"]
        exemplar, exemplar_digest, exemplar_duration = exemplars[tactic_id]
        query_digest, query_duration = sha256(query), duration(query)
        prompt = build_one_shot_prompt(f"clip-{query_digest[:12]}.mp4", query_duration, cards, cards_by_id[tactic_id])
        target = output_root / ONE_SHOT_PHASE / provider / tactic_id / f"{clip_uid.split(':', 1)[-1]}.json"
        if target.is_file() and not force:
            current = json.loads(target.read_text(encoding="utf-8"))
            if not retry_failed or current.get("status") != "failed":
                counts.update(pairs=1, skipped=1)
                continue
        try:
            payload, usage, attempts = generate_json(provider, prompt, response_schema(list(cards_by_id), PHASES[0]), video_paths=(exemplar, query))
            validate_response(payload, list(cards_by_id), cards_by_id, PHASES[0])
            validate_evidence_bounds(payload, query_duration)
            result = {"status": "success", **payload}
        except Exception as exc:
            usage, attempts = [], getattr(exc, "attempts", 1)
            result = {"status": "failed", "failed_stage": "provider_request_or_validation", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        result.update({"schema_version": SCHEMA_VERSION, "experiment_id": ONE_SHOT_EXPERIMENT_ID, "phase": ONE_SHOT_PHASE, "provider": provider, "model": model_name(provider), "clip_uid": clip_uid, "target_tactic_id": tactic_id, "query_sha256": query_digest, "query_duration_s": query_duration, "exemplar_sha256": exemplar_digest, "exemplar_duration_s": exemplar_duration, "prompt_version": ONE_SHOT_PROMPT_VERSION, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "temperature": 0.0, "attempts": attempts, "api_usage": usage})
        write_json(target, result)
        counts.update(pairs=1, calls=1)
        counts[result["status"]] += 1
    return {"phase": ONE_SHOT_PHASE, "provider": provider, **dict(counts)}



def generate_validated_json(
    provider, prompt, schema, validator, *, validation_retries=2, **kwargs,
):
    """Retry provider JSON that fails our strict local schema checks."""
    usage, attempts = [], 0
    for validation_attempt in range(validation_retries + 1):
        payload, call_usage, call_attempts = generate_json(
            provider, prompt, schema, **kwargs,
        )
        usage.extend(call_usage)
        attempts += call_attempts
        try:
            validator(payload)
        except ValueError as exc:
            if validation_attempt == validation_retries:
                exc.attempts = attempts
                exc.api_usage = usage
                exc.validation_attempts = validation_attempt + 1
                raise
            continue
        return payload, usage, attempts, validation_attempt + 1
    raise AssertionError("unreachable")

def run_one_shot_v2(
    output_root: Path, glossary_path: Path, source_rows: Path, ground_truth: Path, repo_root: Path,
    *, provider: str, tactic_ids: Iterable[str] | None = None, clip_uids: Iterable[str] | None = None,
    force: bool = False, retry_failed: bool = False,
) -> dict[str, int | str]:
    cards = sorted(
        (card for card in load_glossary(glossary_path).values() if card["priority"] == "P0"),
        key=lambda card: card["tactic_id"],
    )
    if len(cards) != 19:
        raise ValueError(f"expected 19 P0 cards, got {len(cards)}")
    cards_by_id = {card["tactic_id"]: card for card in cards}
    manifest = build_one_shot_v2_manifest(load_jsonl(ground_truth))
    write_jsonl(output_root / "evaluation/shot_comparison_v2/manifest.jsonl", manifest)

    selected_tactics = set(tactic_ids or ONE_SHOT_V2_TACTICS)
    if unknown := selected_tactics - set(ONE_SHOT_V2_TACTICS):
        raise ValueError(f"unknown one-shot V2 tactics: {sorted(unknown)}")
    selected_clips = set(clip_uids or (item["clip_uid"] for item in manifest))
    known_clips = {item["clip_uid"] for item in manifest}
    if unknown := selected_clips - known_clips:
        raise ValueError(f"unknown one-shot V2 clips: {sorted(unknown)}")

    sources = frozen_sources(repo_root, source_rows, ground_truth)
    exemplars = {}
    for tactic_id in selected_tactics:
        path = (repo_root / ONE_SHOT_EXEMPLARS[tactic_id]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        exemplars[tactic_id] = path, sha256(path), duration(path)

    model = model_name(provider)
    retries = validation_retries = 2 if provider == "gemini" or retry_failed else 0
    counts = Counter(
        pairs=0, calls=0, observation_calls=0, nomination_calls=0,
        success=0, failed=0, skipped=0,
    )
    for item in manifest:
        tactic_id, clip_uid = item["tactic_id"], item["clip_uid"]
        if tactic_id not in selected_tactics or clip_uid not in selected_clips:
            continue
        query = sources[clip_uid]["absolute_video_path"]
        exemplar, exemplar_digest, exemplar_duration = exemplars[tactic_id]
        query_digest, query_duration = sha256(query), duration(query)
        clip_alias = f"clip-{query_digest[:12]}.mp4"
        observation_prompt = build_one_shot_v2_observation_prompt(clip_alias, query_duration)
        observation_prompt_sha = hashlib.sha256(observation_prompt.encode()).hexdigest()
        teaching = ONE_SHOT_V2_TEACHING[tactic_id]
        teaching_sha = hashlib.sha256(
            json.dumps(teaching, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        target = (
            output_root / ONE_SHOT_V2_PHASE / provider / tactic_id
            / f"{clip_uid.split(':', 1)[-1]}.json"
        )
        current = None
        if target.is_file() and not force:
            current = json.loads(target.read_text(encoding="utf-8"))
            nomination_version = current.get("calls", {}).get("nomination", {}).get("prompt_version")
            if (
                current.get("status") == "success"
                and nomination_version == ONE_SHOT_V2_NOMINATION_PROMPT_VERSION
            ) or (current.get("status") == "failed" and not retry_failed):
                counts.update(pairs=1, skipped=1)
                continue

        common = {
            "schema_version": ONE_SHOT_V2_SCHEMA_VERSION,
            "experiment_id": ONE_SHOT_V2_EXPERIMENT_ID,
            "phase": ONE_SHOT_V2_PHASE,
            "provider": provider,
            "model": model,
            "clip_uid": clip_uid,
            "target_tactic_id": tactic_id,
            "query_sha256": query_digest,
            "query_duration_s": query_duration,
            "exemplar_sha256": exemplar_digest,
            "exemplar_duration_s": exemplar_duration,
            "teaching_version": ONE_SHOT_V2_TEACHING_VERSION,
            "teaching_sha256": teaching_sha,
            "temperature": 0.0,
            "upload_profile": (
                DOUBAO_COMPACT_UPLOAD_PROFILE if provider == "doubao" else "original"
            ),
        }
        reusable = bool(
            current
            and current.get("schema_version") == ONE_SHOT_V2_SCHEMA_VERSION
            and current.get("experiment_id") == ONE_SHOT_V2_EXPERIMENT_ID
            and current.get("provider") == provider
            and current.get("model") == model
            and current.get("query_sha256") == query_digest
            and current.get("calls", {}).get("observation", {}).get("prompt_version")
            == ONE_SHOT_V2_OBSERVATION_PROMPT_VERSION
            and current.get("calls", {}).get("observation", {}).get("prompt_sha256")
            == observation_prompt_sha
            and current.get("observations")
        )
        if reusable:
            try:
                validate_one_shot_v2_observations(
                    {"observations": current["observations"]}, cards_by_id, query_duration,
                )
            except (KeyError, TypeError, ValueError):
                reusable = False

        if reusable:
            record = {**current, **common}
            observations = record["observations"]
        else:
            usage: list[dict[str, Any]] = []
            attempts = 0
            validation_attempts = 0
            try:
                upload_context = temporary_doubao_proxies((query,)) if provider == "doubao" else nullcontext((query,))
                with upload_context as upload_paths:
                    payload, usage, attempts, validation_attempts = generate_validated_json(
                        provider, observation_prompt, one_shot_v2_observation_schema(),
                        lambda item: validate_one_shot_v2_observations(item, cards_by_id, query_duration),
                        video_path=upload_paths[0], retries=retries,
                        validation_retries=validation_retries,
                    )
                validate_one_shot_v2_observations(payload, cards_by_id, query_duration)
                observations = json.loads(
                    json.dumps(payload["observations"], ensure_ascii=False, sort_keys=True)
                )
                observation_call = {
                    "status": "success",
                    "prompt_version": ONE_SHOT_V2_OBSERVATION_PROMPT_VERSION,
                    "prompt_sha256": observation_prompt_sha,
                    "attempts": attempts,
                    "validation_attempts": validation_attempts,
                    "api_usage": usage,
                }
                record = {
                    **common,
                    "status": "observation_complete",
                    "observations": observations,
                    "calls": {"observation": observation_call},
                }
                write_json(target, record)
            except Exception as exc:
                usage = usage or getattr(exc, "api_usage", [])
                record = {
                    **common,
                    "status": "failed",
                    "failed_stage": "observation",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "calls": {"observation": {
                        "status": "failed",
                        "prompt_version": ONE_SHOT_V2_OBSERVATION_PROMPT_VERSION,
                        "prompt_sha256": observation_prompt_sha,
                        "attempts": attempts or getattr(exc, "attempts", 1),
                        "validation_attempts": validation_attempts or getattr(exc, "validation_attempts", 1),
                        "api_usage": usage,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }},
                }
                write_json(target, record)
                counts.update(pairs=1, calls=1, observation_calls=1, failed=1)
                continue
            counts.update(calls=1, observation_calls=1)

        nomination_prompt = build_one_shot_v2_nomination_prompt(
            clip_alias, query_duration, cards, observations,
        )
        nomination_prompt_sha = hashlib.sha256(nomination_prompt.encode()).hexdigest()
        instructions = build_one_shot_v2_video_instructions(
            cards_by_id[tactic_id], teaching, clip_alias,
        )
        instructions_sha = hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False).encode()
        ).hexdigest()
        call_sha = hashlib.sha256(
            json.dumps(
                {"video_instructions": instructions, "prompt": nomination_prompt},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        record.pop("candidates", None)
        record.pop("no_nomination_reason_zh", None)
        usage = []
        attempts = 0
        validation_attempts = 0
        try:
            upload_context = temporary_doubao_proxies((exemplar, query)) if provider == "doubao" else nullcontext((exemplar, query))
            with upload_context as upload_paths:
                payload, usage, attempts, validation_attempts = generate_validated_json(
                    provider, nomination_prompt,
                    one_shot_v2_nomination_schema(list(cards_by_id)),
                    lambda item: validate_one_shot_v2_nomination(item, cards_by_id, observations, query_duration),
                    video_paths=upload_paths, video_instructions=instructions,
                    retries=retries,
                    validation_retries=validation_retries,
                )
            validate_one_shot_v2_nomination(
                payload, cards_by_id, observations, query_duration,
            )
            for key in ("failed_stage", "error_type", "error"):
                record.pop(key, None)
            record.update({"status": "success", **payload})
            nomination_call = {
                "status": "success",
                "prompt_version": ONE_SHOT_V2_NOMINATION_PROMPT_VERSION,
                "prompt_sha256": nomination_prompt_sha,
                "video_instructions_sha256": instructions_sha,
                "call_sha256": call_sha,
                "attempts": attempts,
                "validation_attempts": validation_attempts,
                "api_usage": usage,
            }
        except Exception as exc:
            usage = usage or getattr(exc, "api_usage", [])
            record.update({
                "status": "failed",
                "failed_stage": "nomination",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
            nomination_call = {
                "status": "failed",
                "prompt_version": ONE_SHOT_V2_NOMINATION_PROMPT_VERSION,
                "prompt_sha256": nomination_prompt_sha,
                "video_instructions_sha256": instructions_sha,
                "call_sha256": call_sha,
                "attempts": attempts or getattr(exc, "attempts", 1),
                "validation_attempts": validation_attempts or getattr(exc, "validation_attempts", 1),
                "api_usage": usage,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        record.setdefault("calls", {})["nomination"] = nomination_call
        write_json(target, record)
        counts.update(pairs=1, calls=1, nomination_calls=1)
        counts[record["status"]] += 1
    return {"phase": ONE_SHOT_V2_PHASE, "provider": provider, **dict(counts)}


def claim_match(run: dict[str, Any], claim: dict[str, Any], top_k: int) -> bool:
    window = claim["window"]
    return any(candidate["tactic_id"] == claim["tactic_id"] and any(min(float(span["end_s"]), float(window["end_s"])) > max(float(span["start_s"]), float(window["start_s"])) for span in candidate["evidence_spans"]) for candidate in run["candidates"][:top_k])


def metrics(claims: list[dict[str, Any]], runs: dict[Any, dict[str, Any]], top_k: int, *, run_key: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    key = run_key or (lambda claim: claim["clip_uid"])
    rows = [(claim, "present" if claim_match(runs[key(claim)], claim, top_k) else "absent") for claim in claims if key(claim) in runs]
    correct = sum(predicted == claim["gt_verdict"] for claim, predicted in rows)
    positives = [(claim, predicted) for claim, predicted in rows if claim["gt_verdict"] == "present"]
    return {"top_k": top_k, "expected_claims": len(claims), "covered_claims": len(rows), "correct": correct, "accuracy": correct / len(claims) if len(rows) == len(claims) else None, "covered_accuracy": correct / len(rows) if rows else None, "positive_recall": sum(predicted == "present" for _, predicted in positives) / len(positives) if positives else None}


def report(output_root: Path, glossary_path: Path, ground_truth: Path) -> dict[str, Any]:
    cards = {key: value for key, value in load_glossary(glossary_path).items() if value["priority"] == "P0"}
    claims = [claim for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary" and claim.get("tactic_id") in cards]
    if len(claims) != 55:
        raise ValueError(f"expected 55 primary P0 claims, got {len(claims)}")
    conditions: dict[str, Any] = {}
    for phase in PHASES:
        conditions[phase] = {}
        for provider in ("gemini", "doubao"):
            attempted, valid = {}, {}
            for path in sorted((output_root / phase / provider).glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                attempted[payload["clip_uid"]] = payload
                if payload.get("status") != "success":
                    continue
                try:
                    keys = (["observations"] if phase == PHASES[1] else []) + ["candidates", "no_nomination_reason_zh"]
                    validate_response({key: payload[key] for key in keys}, list(cards), cards, phase)
                except (KeyError, TypeError, ValueError):
                    continue
                valid[payload["clip_uid"]] = payload
            execution = {"expected_clips": 67, "attempted_clips": len(attempted), "valid_clips": len(valid), "invalid_or_failed_clips": len(attempted) - len(valid), "missing_clips": 67 - len(attempted), "status": "complete" if len(valid) == 67 else "partial"}
            conditions[phase][provider] = {"execution": execution, "top1_claim_metrics": metrics(claims, valid, 1), "top3_claim_metrics": metrics(claims, valid, 3)}
    value = {"experiment_id": EXPERIMENT_ID, "metric_scope": "55 primary P0 claims; not exhaustive 67-video accuracy", "conditions": conditions}
    evaluation = output_root / "evaluation"
    write_json(evaluation / "summary.json", value)
    lines = ["# 两阶段 P0 开放提名报告", "", "主指标仅为 55 条 primary P0 claims 上的 claim-level accuracy，不得表述为 67 个视频的穷尽 Top-1 准确率。", "", "| phase | provider | attempted | valid | failed | Top-1 | Top-3 |", "|---|---|---:|---:|---:|---:|---:|"]
    for phase in PHASES:
        for provider in ("gemini", "doubao"):
            item, execution = conditions[phase][provider], conditions[phase][provider]["execution"]
            lines.append(f"| {phase} | {provider} | {execution['attempted_clips']} | {execution['valid_clips']} | {execution['invalid_or_failed_clips']} | {item['top1_claim_metrics']['accuracy']} | {item['top3_claim_metrics']['accuracy']} |")
    (evaluation / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def _comparison_runs(directory: Path, cards: dict[str, dict[str, Any]], *, one_shot: bool) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
    attempted, valid = {}, {}
    for path in sorted(directory.glob("*/*.json" if one_shot else "*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload.get("target_tactic_id"), payload.get("clip_uid")) if one_shot else payload.get("clip_uid")
        attempted[key] = payload
        if payload.get("status") != "success":
            continue
        try:
            validate_response({name: payload[name] for name in ("candidates", "no_nomination_reason_zh")}, list(cards), cards, PHASES[0])
            if one_shot:
                if payload["target_tactic_id"] != path.parent.name:
                    raise ValueError("one-shot tactic path mismatch")
                validate_evidence_bounds(payload, float(payload["query_duration_s"]))
        except (KeyError, TypeError, ValueError):
            continue
        valid[key] = payload
    return attempted, valid


def _accuracy_only(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("top_k", "expected_claims", "covered_claims", "correct", "accuracy")}


def _result_status(attempted: dict[Any, dict[str, Any]], valid: dict[Any, dict[str, Any]], key: Any) -> str:
    if key in valid:
        return "valid"
    payload = attempted.get(key)
    if payload is None:
        return "missing"
    return "failed" if payload.get("status") == "failed" else "invalid"


def report_shot_comparison(output_root: Path, glossary_path: Path, ground_truth: Path) -> dict[str, Any]:
    cards = {key: value for key, value in load_glossary(glossary_path).items() if value["priority"] == "P0"}
    claims = sorted((claim for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary" and claim.get("tactic_id") in ONE_SHOT_TACTICS), key=lambda claim: (claim["tactic_id"], claim["clip_uid"], claim["claim_id"]))
    manifest = build_one_shot_manifest(claims)
    evaluation = output_root / "evaluation/shot_comparison"
    write_jsonl(evaluation / "manifest.jsonl", manifest)
    expected_clips = {item["clip_uid"] for item in manifest}
    value: dict[str, Any] = {
        "experiment_id": ONE_SHOT_EXPERIMENT_ID,
        "metric_scope": "50 primary P0 claims across 49 tactic-video pairs; per-tactic video-primed open nomination, not clip accuracy",
        "expected_claims": 50,
        "expected_pairs": 49,
        "providers": {},
    }
    csv_rows = []
    one_key = lambda claim: (claim["tactic_id"], claim["clip_uid"])
    for provider in ("doubao", "gemini"):
        zero_attempted, zero_valid = _comparison_runs(output_root / PHASES[0] / provider, cards, one_shot=False)
        one_attempted, one_valid = _comparison_runs(output_root / ONE_SHOT_PHASE / provider, cards, one_shot=True)

        def comparison(selected_claims: list[dict[str, Any]]) -> dict[str, Any]:
            zero = {f"top{top_k}": _accuracy_only(metrics(selected_claims, zero_valid, top_k)) for top_k in (1, 3)}
            one = {f"top{top_k}": _accuracy_only(metrics(selected_claims, one_valid, top_k, run_key=one_key)) for top_k in (1, 3)}
            return {
                "expected_claims": len(selected_claims),
                "zero_shot": zero,
                "video_one_shot": one,
                "accuracy_delta": {
                    f"top{top_k}": one[f"top{top_k}"]["accuracy"] - zero[f"top{top_k}"]["accuracy"]
                    if one[f"top{top_k}"]["accuracy"] is not None and zero[f"top{top_k}"]["accuracy"] is not None else None
                    for top_k in (1, 3)
                },
            }

        provider_value = {
            "execution": {
                "zero_shot": {
                    "expected_clips": len(expected_clips),
                    "attempted_clips": sum(clip_uid in zero_attempted for clip_uid in expected_clips),
                    "valid_clips": sum(clip_uid in zero_valid for clip_uid in expected_clips),
                },
                "video_one_shot": {
                    "expected_pairs": 49,
                    "attempted_pairs": sum((item["tactic_id"], item["clip_uid"]) in one_attempted for item in manifest),
                    "valid_pairs": sum((item["tactic_id"], item["clip_uid"]) in one_valid for item in manifest),
                },
            },
            "overall": comparison(claims),
            "tactics": {tactic_id: comparison([claim for claim in claims if claim["tactic_id"] == tactic_id]) for tactic_id in ONE_SHOT_TACTICS},
        }
        value["providers"][provider] = provider_value
        for claim in claims:
            zero_run = zero_valid.get(claim["clip_uid"])
            one_run = one_valid.get(one_key(claim))
            zero_status = _result_status(zero_attempted, zero_valid, claim["clip_uid"])
            one_status = _result_status(one_attempted, one_valid, one_key(claim))
            row = {
                "provider": provider,
                "tactic_id": claim["tactic_id"],
                "claim_id": claim["claim_id"],
                "clip_uid": claim["clip_uid"],
                "gt_verdict": claim["gt_verdict"],
                "window_start_s": claim["window"]["start_s"],
                "window_end_s": claim["window"]["end_s"],
                "zero_shot_status": zero_status,
                "video_one_shot_status": one_status,
            }
            for top_k in (1, 3):
                zero_prediction = None if zero_run is None else "present" if claim_match(zero_run, claim, top_k) else "absent"
                one_prediction = None if one_run is None else "present" if claim_match(one_run, claim, top_k) else "absent"
                row.update({
                    f"zero_shot_top{top_k}_prediction": zero_prediction,
                    f"video_one_shot_top{top_k}_prediction": one_prediction,
                    f"zero_shot_top{top_k}_correct": None if zero_prediction is None else zero_prediction == claim["gt_verdict"],
                    f"video_one_shot_top{top_k}_correct": None if one_prediction is None else one_prediction == claim["gt_verdict"],
                })
            csv_rows.append(row)
    write_json(evaluation / "summary.json", value)
    evaluation.mkdir(parents=True, exist_ok=True)
    with (evaluation / "per_claim.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# P0 Zero-shot vs Video One-shot",
        "",
        "50条primary P0 claims；按战术提供固定FIFA示范后的开放提名，不是通用clip accuracy。",
        "",
        "| provider | scope | claims | zero valid | one-shot valid | zero Top-1 | one Top-1 | ΔTop-1 | zero Top-3 | one Top-3 | ΔTop-3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, provider_value in value["providers"].items():
        for scope, item in [("overall", provider_value["overall"]), *provider_value["tactics"].items()]:
            number = lambda result: "—" if result is None else f"{result:.4f}"
            lines.append(f"| {provider} | {scope} | {item['expected_claims']} | {item['zero_shot']['top1']['covered_claims']} | {item['video_one_shot']['top1']['covered_claims']} | {number(item['zero_shot']['top1']['accuracy'])} | {number(item['video_one_shot']['top1']['accuracy'])} | {number(item['accuracy_delta']['top1'])} | {number(item['zero_shot']['top3']['accuracy'])} | {number(item['video_one_shot']['top3']['accuracy'])} | {number(item['accuracy_delta']['top3'])} |")
    (evaluation / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def _validate_one_shot_v2_metadata(
    payload: dict[str, Any], cards: dict[str, dict[str, Any]], provider: str,
) -> None:
    tactic_id = payload["target_tactic_id"]
    if tactic_id not in ONE_SHOT_V2_TACTICS:
        raise ValueError("unknown V2 target tactic")
    query_sha = payload["query_sha256"]
    if not isinstance(query_sha, str) or len(query_sha) != 64:
        raise ValueError("missing V2 query hash")
    query_duration = float(payload["query_duration_s"])
    cards_list = sorted(cards.values(), key=lambda card: card["tactic_id"])
    teaching = ONE_SHOT_V2_TEACHING[tactic_id]
    teaching_sha = hashlib.sha256(
        json.dumps(teaching, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    alias = f"clip-{query_sha[:12]}.mp4"
    observation_prompt = build_one_shot_v2_observation_prompt(alias, query_duration)
    observations = payload["observations"]
    nomination_prompt = build_one_shot_v2_nomination_prompt(
        alias, query_duration, cards_list, observations,
    )
    instructions = build_one_shot_v2_video_instructions(
        cards[tactic_id], teaching, alias,
    )
    expected = {
        "schema_version": ONE_SHOT_V2_SCHEMA_VERSION,
        "experiment_id": ONE_SHOT_V2_EXPERIMENT_ID,
        "phase": ONE_SHOT_V2_PHASE,
        "provider": provider,
        "model": model_name(provider),
        "teaching_version": ONE_SHOT_V2_TEACHING_VERSION,
        "teaching_sha256": teaching_sha,
        "temperature": 0.0,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("V2 experiment metadata mismatch")
    observation_call = payload["calls"]["observation"]
    nomination_call = payload["calls"]["nomination"]
    expected_observation = {
        "status": "success",
        "prompt_version": ONE_SHOT_V2_OBSERVATION_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(observation_prompt.encode()).hexdigest(),
    }
    if any(observation_call.get(key) != value for key, value in expected_observation.items()):
        raise ValueError("V2 observation metadata mismatch")
    expected_nomination = {
        "status": "success",
        "prompt_version": ONE_SHOT_V2_NOMINATION_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(nomination_prompt.encode()).hexdigest(),
        "video_instructions_sha256": hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False).encode()
        ).hexdigest(),
        "call_sha256": hashlib.sha256(
            json.dumps(
                {"video_instructions": instructions, "prompt": nomination_prompt},
                ensure_ascii=False, sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    if any(nomination_call.get(key) != value for key, value in expected_nomination.items()):
        raise ValueError("V2 nomination metadata mismatch")


def _comparison_runs_v2(
    directory: Path, cards: dict[str, dict[str, Any]],
) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
    attempted, valid = {}, {}
    for path in sorted(directory.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = payload.get("target_tactic_id"), payload.get("clip_uid")
        attempted[key] = payload
        if payload.get("status") != "success":
            continue
        try:
            if payload["target_tactic_id"] != path.parent.name:
                raise ValueError("one-shot V2 tactic path mismatch")
            _validate_one_shot_v2_metadata(payload, cards, directory.name)
            query_duration = float(payload["query_duration_s"])
            observations = payload["observations"]
            validate_one_shot_v2_observations(
                {"observations": observations}, cards, query_duration,
            )
            validate_one_shot_v2_nomination(
                {"candidates": payload["candidates"]},
                cards,
                observations,
                query_duration,
            )
        except (KeyError, TypeError, ValueError):
            continue
        valid[key] = payload
    return attempted, valid


def _shot_condition(
    claims: list[dict[str, Any]],
    runs: dict[Any, dict[str, Any]],
    run_key: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    top1 = metrics(claims, runs, 1, run_key=run_key)
    top3 = metrics(claims, runs, 3, run_key=run_key)
    return {
        "valid_claims": top1["covered_claims"],
        "expected_claims": top1["expected_claims"],
        "top1_accuracy": top1["accuracy"],
        "top3_accuracy": top3["accuracy"],
    }


def _shot_delta(new: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    return {
        f"top{top_k}": (
            new[f"top{top_k}_accuracy"] - baseline[f"top{top_k}_accuracy"]
            if new[f"top{top_k}_accuracy"] is not None
            and baseline[f"top{top_k}_accuracy"] is not None
            else None
        )
        for top_k in (1, 3)
    }


def report_shot_comparison_v2(
    output_root: Path, glossary_path: Path, ground_truth: Path,
) -> dict[str, Any]:
    cards = {
        key: value
        for key, value in load_glossary(glossary_path).items()
        if value["priority"] == "P0"
    }
    all_claims = load_jsonl(ground_truth)
    manifest = build_one_shot_v2_manifest(all_claims)
    claims = sorted(
        (
            claim for claim in all_claims
            if claim.get("score_set") == "primary"
            and claim.get("tactic_id") in ONE_SHOT_V2_TACTICS
        ),
        key=lambda claim: (claim["tactic_id"], claim["clip_uid"], claim["claim_id"]),
    )
    if len(claims) != 23:
        raise ValueError(f"expected 23 one-shot V2 claims, got {len(claims)}")

    evaluation = output_root / "evaluation/shot_comparison_v2"
    write_jsonl(evaluation / "manifest.jsonl", manifest)
    pair_key = lambda claim: (claim["tactic_id"], claim["clip_uid"])
    clip_key = lambda claim: claim["clip_uid"]
    value: dict[str, Any] = {
        "experiment_id": ONE_SHOT_V2_EXPERIMENT_ID,
        "metric_scope": "23 primary P0 claims across 23 tactic-video pairs; incomplete coverage has null accuracy",
        "expected_claims": 23,
        "expected_pairs": 23,
        "providers": {},
    }
    csv_rows: list[dict[str, Any]] = []
    thresholds = {
        "line-breaking-pass": (2, 0),
        "run-in-behind": (3, 1),
    }

    for provider in ("gemini", "doubao"):
        zero_attempted, zero_valid = _comparison_runs(
            output_root / PHASES[0] / provider, cards, one_shot=False,
        )
        v2_attempted, v2_valid = _comparison_runs_v2(
            output_root / ONE_SHOT_V2_PHASE / provider, cards,
        )
        v1_attempted = v1_valid = None
        if provider == "gemini":
            v1_attempted, v1_valid = _comparison_runs(
                output_root / ONE_SHOT_PHASE / provider, cards, one_shot=True,
            )

        def comparison(selected: list[dict[str, Any]]) -> dict[str, Any]:
            zero = _shot_condition(selected, zero_valid, clip_key)
            v1 = None if v1_valid is None else _shot_condition(selected, v1_valid, pair_key)
            v2 = _shot_condition(selected, v2_valid, pair_key)
            baseline = v1 if v1 is not None else zero
            delta_name = "delta_v2_minus_v1" if v1 is not None else "delta_v2_minus_zero"
            return {
                "zero_shot": zero,
                "video_one_shot_v1": v1,
                "video_one_shot_v2": v2,
                delta_name: _shot_delta(v2, baseline),
            }

        gates = {}
        for tactic_id, (required_tp, allowed_fp) in thresholds.items():
            selected = [claim for claim in claims if claim["tactic_id"] == tactic_id]
            valid_claims = [claim for claim in selected if pair_key(claim) in v2_valid]
            true_positives = sum(
                claim["gt_verdict"] == "present"
                and claim_match(v2_valid[pair_key(claim)], claim, 1)
                for claim in valid_claims
            )
            false_positives = sum(
                claim["gt_verdict"] == "absent"
                and claim_match(v2_valid[pair_key(claim)], claim, 1)
                for claim in valid_claims
            )
            gates[tactic_id] = {
                "valid_claims": len(valid_claims),
                "expected_claims": len(selected),
                "true_positives": true_positives,
                "positive_claims": sum(claim["gt_verdict"] == "present" for claim in selected),
                "required_true_positives": required_tp,
                "false_positives": false_positives,
                "negative_claims": sum(claim["gt_verdict"] == "absent" for claim in selected),
                "allowed_false_positives": allowed_fp,
                "passed": (
                    true_positives >= required_tp and false_positives <= allowed_fp
                    if len(valid_claims) == len(selected) else None
                ),
            }

        provider_value = {
            "overall": comparison(claims),
            "tactics": {
                tactic_id: comparison(
                    [claim for claim in claims if claim["tactic_id"] == tactic_id]
                )
                for tactic_id in ONE_SHOT_V2_TACTICS
            },
            "top1_gates": gates,
        }
        value["providers"][provider] = provider_value

        for claim in claims:
            row = {
                "provider": provider,
                "tactic_id": claim["tactic_id"],
                "claim_id": claim["claim_id"],
                "clip_uid": claim["clip_uid"],
                "gt_verdict": claim["gt_verdict"],
                "window_start_s": claim["window"]["start_s"],
                "window_end_s": claim["window"]["end_s"],
            }
            conditions = {
                "zero_shot": (zero_attempted, zero_valid, claim["clip_uid"]),
                "video_one_shot_v1": (
                    None if v1_attempted is None
                    else (v1_attempted, v1_valid, pair_key(claim))
                ),
                "video_one_shot_v2": (v2_attempted, v2_valid, pair_key(claim)),
            }
            for name, condition in conditions.items():
                if condition is None:
                    row.update({
                        f"{name}_status": None,
                        f"{name}_top1_hit": None,
                        f"{name}_top3_hit": None,
                    })
                    continue
                attempted, valid, key = condition
                run = valid.get(key)
                row.update({
                    f"{name}_status": _result_status(attempted, valid, key),
                    f"{name}_top1_hit": None if run is None else claim_match(run, claim, 1),
                    f"{name}_top3_hit": None if run is None else claim_match(run, claim, 3),
                })
            csv_rows.append(row)

    write_json(evaluation / "summary.json", value)
    with (evaluation / "per_claim.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    number = lambda result: "—" if result is None else f"{result:.4f}"
    coverage = lambda result: "—" if result is None else f"{result['valid_claims']}/{result['expected_claims']}"
    lines = [
        "# P0 Zero-shot vs Video One-shot V1/V2",
        "",
        "23条primary P0 claims；覆盖不完整时accuracy和delta均为null。Doubao未运行V1。",
        "",
        "| provider | scope | zero coverage | zero Top-1 | zero Top-3 | V1 coverage | V1 Top-1 | V1 Top-3 | V2 coverage | V2 Top-1 | V2 Top-3 | delta baseline | ΔTop-1 | ΔTop-3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for provider, provider_value in value["providers"].items():
        for scope, item in [("overall", provider_value["overall"]), *provider_value["tactics"].items()]:
            zero, v1, v2 = item["zero_shot"], item["video_one_shot_v1"], item["video_one_shot_v2"]
            delta_name = "delta_v2_minus_v1" if v1 is not None else "delta_v2_minus_zero"
            lines.append(
                f"| {provider} | {scope} | {coverage(zero)} | {number(zero['top1_accuracy'])} | {number(zero['top3_accuracy'])} | "
                f"{coverage(v1)} | {number(None if v1 is None else v1['top1_accuracy'])} | {number(None if v1 is None else v1['top3_accuracy'])} | "
                f"{coverage(v2)} | {number(v2['top1_accuracy'])} | {number(v2['top3_accuracy'])} | {delta_name[6:]} | "
                f"{number(item[delta_name]['top1'])} | {number(item[delta_name]['top3'])} |"
            )
    lines.extend([
        "",
        "## V2 Top-1 gates",
        "",
        "| provider | tactic | coverage | TP / positives (min) | FP / negatives (max) | passed |",
        "|---|---|---:|---:|---:|---|",
    ])
    for provider, provider_value in value["providers"].items():
        for tactic_id, gate in provider_value["top1_gates"].items():
            passed = "—" if gate["passed"] is None else "PASS" if gate["passed"] else "FAIL"
            lines.append(
                f"| {provider} | {tactic_id} | {gate['valid_claims']}/{gate['expected_claims']} | "
                f"{gate['true_positives']}/{gate['positive_claims']} (≥{gate['required_true_positives']}) | "
                f"{gate['false_positives']}/{gate['negative_claims']} (≤{gate['allowed_false_positives']}) | {passed} |"
            )
    (evaluation / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value
