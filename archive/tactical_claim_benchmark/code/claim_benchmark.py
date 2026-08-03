"""Evidence-chain claim verification and exploratory P0 tactical recognition."""

from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from pipeline.stage2b.generate import ark_chat

LOGGER = logging.getLogger(__name__)

EXPERIMENT_ID = "tactical-evidence-chain-p0"
PROMPT_VERSION = "tactical-evidence-chain-v1"
SCHEMA_VERSION = "tactical-evidence-chain-response-v1"
DOUBAO_MODEL = "doubao-seed-2-0-lite-260428"
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TEMPERATURE = 0.0
P0_EXPERIMENT_ID = "p0-two-phase-nomination-v1"
P0_PHASES = ("phase1_direct", "phase2_observation_first")
P0_PROMPT_VERSIONS = {
    "phase1_direct": "p0-direct-nomination-v1",
    "phase2_observation_first": "p0-observation-first-nomination-v1",
}
P0_PHASE2_PROMPT_VERSION = "p0-observation-first-nomination-v2"
P0_DOUBAO_TRANSPORT_VERSION = "file-api-responses-v1"
P0_PHASE2_PROMPT_AMENDMENT = """
Observation fields are purely descriptive. In particular, terminal_result_zh must state
only the final visible event or outcome, such as a shot, goal, save, clearance, turnover,
possession retained, or the visible sequence ending. Never put a tactic ID, tactic name,
tactical category, or tentative tactical judgment in any observation field.
"""
P0_SCHEMA_VERSION = "p0-nomination-v1"
TIMEOUT_S = 1800.0
TRANSPORT_RETRIES = 2
JUDGE_SEED = 20260724
GEMINI_KEY_ENV_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2")
_GEMINI_KEY_LOCK = threading.Lock()
_GEMINI_KEY_INDEX = 0

VERDICTS = {"present", "absent", "unobservable"}
SPACE_ORIGINS = {"created", "existing", "unclear", "not_applicable"}

BASE_TACTIC_IDS = {
    "快速反击": "counter-attack",
    "破线传球": "line-breaking-pass",
    "打身后 / 反越位跑位": "run-in-behind",
    "线间接应": "between-the-lines",
    "门将参与出球": "gk-in-buildup",
    "角球后点/前点战术": "corner-near-far-post",
    "虚跑/假跑扯动": "dummy-run",
    "下底倒三角": "cutback",
    "肋部渗透": "halfspace-penetration",
    "大范围转移": "switch-of-play",
    "二过一 / 撞墙配合": "one-two",
}

# Explicit human correction phrases only. Negative mentions such as "不是反击"
# deliberately do not match this table.
CORRECTION_LABEL_PHRASES = {
    "还包含了打身后": ("run-in-behind",),
    "还包括打身后": ("run-in-behind",),
    "还包括线间接应": ("between-the-lines",),
    "还包括破线传球": ("line-breaking-pass",),
    "属于破线传球+线间接应": ("line-breaking-pass", "between-the-lines"),
    "属于破线传球＋线间接应": ("line-breaking-pass", "between-the-lines"),
    "边路二过一配合": ("one-two",),
    "边路套边": ("overlap-underlap",),
    "传球到中场脚下，中场推进至肋部打门，只是射门发生在肋部": (
        "halfspace-arriving-shot",
    ),
    "前场球员回撤到肋部接应，没有打穿防线": ("between-the-lines",),
    "斜长传至禁区打后卫身后，不是大范围转移": (
        "long-ball", "run-in-behind",
    ),
}

OLD_TACTIC_IDS = {
    "counter-attack": "fast_break_pattern",
    "line-breaking-pass": "line_break",
    "run-in-behind": "run_in_behind",
    "corner-near-far-post": "corner-near-far-post",
    "cutback": "cutback",
}

ANALYSIS_FIELDS = (
    "state_before",
    "trigger_action",
    "defensive_response",
    "space_origin",
    "space_evidence",
    "beneficiary_effect",
    "terminal_sequence",
    "supporting_cues",
    "violated_cues",
    "evidence_gaps",
    "evidence_spans",
    "verdict",
    "confidence",
    "confidence_reasons",
    "alternative_tactics",
)


class RetryableProviderError(RuntimeError):
    """A network/provider failure for which the identical request may be retried."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


def _configuration() -> dict[str, Any]:
    prompt_contract = (
        OBSERVATION_ORDER
        + json.dumps({"verdicts": sorted(VERDICTS), "space_origins": sorted(SPACE_ORIGINS)})
        + inspect.getsource(load_glossary)
        + inspect.getsource(_compact_card)
        + inspect.getsource(build_claim_prompt)
        + inspect.getsource(build_p0_prompt)
        + inspect.getsource(_analysis_schema)
        + inspect.getsource(_nonempty_strings)
        + inspect.getsource(_time_items)
        + inspect.getsource(_validate_analysis)
        + inspect.getsource(validate_claim_response)
        + inspect.getsource(validate_p0_response)
    )
    return {
        "prompt_contract_sha256": _text_sha256(prompt_contract),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "models": {"doubao": DOUBAO_MODEL, "gemini": GEMINI_MODEL},
        "temperature": TEMPERATURE,
        "timeout_s": TIMEOUT_S,
        "transport_retries": TRANSPORT_RETRIES,
    }


def _configuration_sha256() -> str:
    return _text_sha256(json.dumps(_configuration(), sort_keys=True))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        handle.write(text)
    Path(handle.name).replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip_dir(clip_uid: str) -> str:
    return clip_uid.replace(":", "__")


def _split_lines(value: str | None) -> list[str]:
    return [
        item.strip(" \t\r\n0123456789.-、")
        for item in (value or "").splitlines()
        if item.strip(" \t\r\n0123456789.-、")
    ]


def _seconds(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    return sum(part * 60**index for index, part in enumerate(reversed(parts)))


def parse_window(raw: str) -> dict[str, Any]:
    parts = re.split(r"\s*[–—−-]\s*", raw.strip())
    if len(parts) != 2:
        raise ValueError(f"invalid time window: {raw}")
    start_s, end_s = map(_seconds, parts)
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"invalid time window: {raw}")
    return {"raw": raw, "start_s": start_s, "end_s": end_s}


def load_glossary(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    cards = {}
    for row in rows:
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
            "layer": (row.get("层级 layer") or "").strip(),
            "phase": (row.get("阶段 phase") or "").strip(),
            "definition": (row.get("一句话定义") or "").strip(),
            "observable_cues": _split_lines(row.get("可观察特征")),
            "triggers": _split_lines(row.get("触发条件")),
            "confusing": confusing,
        }
    return cards


def _correction_labels(correction: str) -> list[str]:
    labels = []
    for phrase, tactic_ids in CORRECTION_LABEL_PHRASES.items():
        if phrase in correction:
            labels.extend(tactic_ids)
    return list(dict.fromkeys(labels))


def load_source_index(source_rows: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for row in _load_jsonl(source_rows):
        video = row.get("video_path")
        if not video:
            continue
        record = dict(row)
        record["absolute_video_path"] = (repo_root / video).resolve()
        index[row["clip_uid"]] = record
    return index


def prepare_gt(
    csv_path: Path,
    glossary_path: Path,
    source_rows: Path,
    output_root: Path,
    repo_root: Path,
    *,
    approved: bool = False,
) -> dict[str, Any]:
    cards = load_glossary(glossary_path)
    source_index = load_source_index(source_rows, repo_root)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 73:
        raise ValueError(f"expected 73 source claims, got {len(rows)}")

    claims: list[dict[str, Any]] = []
    audits: list[dict[str, str]] = []
    claim_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        claim_id = (row.get("claim_id") or "").strip()
        if not re.fullmatch(r"base:\d+[a-z]?", claim_id) or claim_id in claim_ids:
            raise ValueError(f"invalid or duplicate claim_id at row {row_number}: {claim_id}")
        claim_ids.add(claim_id)
        clip_id = (row.get("id") or "").strip()
        clip_uid = (
            f"event_clips:{clip_id.zfill(4)}"
            if clip_id.isdigit()
            else f"soccernetgs:{clip_id}"
        )
        if clip_uid not in source_index:
            raise ValueError(f"no source video for {clip_uid}")
        tactic_zh = (row.get("tactics") or "").strip()
        tactic_id = BASE_TACTIC_IDS.get(tactic_zh)
        if tactic_id not in cards:
            raise ValueError(f"unmapped tactic at row {row_number}: {tactic_zh}")
        accuracy = (row.get("识别准确度") or "").strip()
        if accuracy not in {"正确", "错误"}:
            raise ValueError(f"invalid review at row {row_number}: {accuracy}")
        correction = (row.get("更正") or "").strip()
        verdict = "present" if accuracy == "正确" else "absent"
        claim = {
            "claim_id": claim_id,
            "clip_id": clip_id.zfill(4) if clip_id.isdigit() else clip_id,
            "clip_uid": clip_uid,
            "tactic_id": tactic_id,
            "tactic_zh": tactic_zh,
            "gt_verdict": verdict,
            "score_set": "primary",
            "source_row": row_number,
            "window": parse_window((row.get("时间范围") or "").strip()),
            "window_status": "accepted",
            "correction": correction or None,
            "correction_label_ids": _correction_labels(correction),
        }
        claims.append(claim)
        for derived_tactic in claim["correction_label_ids"]:
            if derived_tactic not in cards:
                raise ValueError(f"unknown derived tactic: {derived_tactic}")
            if any(
                item["clip_uid"] == clip_uid and item["tactic_id"] == derived_tactic
                for item in claims
            ):
                continue
            claims.append({
                "claim_id": f"correction:{claim_id.removeprefix('base:')}:{derived_tactic}",
                "clip_id": claim["clip_id"],
                "clip_uid": clip_uid,
                "tactic_id": derived_tactic,
                "tactic_zh": cards[derived_tactic]["name_zh"],
                "gt_verdict": "present",
                "score_set": "secondary",
                "source_row": row_number,
                "window": None,
                "window_status": "not_scored",
                "correction": correction,
                "correction_label_ids": [derived_tactic],
            })
            audits.append({
                "clip_uid": clip_uid,
                "source_row": str(row_number),
                "kind": "derived_positive",
                "detail": f"{derived_tactic}: {correction}",
            })

    base = [claim for claim in claims if claim["score_set"] == "primary"]
    if len(base) != 73 or len({claim["clip_uid"] for claim in base}) != 67:
        raise ValueError("GT contract must contain 73 primary claims across 67 clips")
    s101 = [
        claim for claim in base
        if claim["clip_uid"] == "soccernetgs:SNGS-101"
        and claim["tactic_id"] == "run-in-behind"
    ]
    if len(s101) != 1 or s101[0]["gt_verdict"] != "present":
        raise ValueError("SNGS-101 run-in-behind must be present")

    gt_dir = output_root / "gt"
    gt_path = gt_dir / "gt_claims.jsonl"
    _write_jsonl(gt_path, claims)
    audit_path = gt_dir / "gt_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("clip_uid", "source_row", "kind", "detail"),
        )
        writer.writeheader()
        writer.writerows(audits)

    manifest = {
        "material_passport": {
            "id": EXPERIMENT_ID,
            "type": "multimodal_tactical_recognition_experiment",
            "status": "frozen" if approved else "prepared_for_review",
            "verification_status": "UNVERIFIED",
        },
        "schema_version": "tactical-evidence-chain-manifest-v1",
        "prompt_version": PROMPT_VERSION,
        "configuration_sha256": _configuration_sha256(),
        "gt_status": "frozen" if approved else "prepared_for_review",
        "source_csv": csv_path.relative_to(repo_root).as_posix(),
        "source_csv_sha256": _sha256(csv_path),
        "gt_claims_sha256": _sha256(gt_path),
        "glossary": glossary_path.relative_to(repo_root).as_posix(),
        "glossary_sha256": _sha256(glossary_path),
        "source_rows": source_rows.relative_to(repo_root).as_posix(),
        "models": {"doubao": DOUBAO_MODEL, "gemini": GEMINI_MODEL},
        "temperature": TEMPERATURE,
        "timeout_s": TIMEOUT_S,
        "transport_retries": TRANSPORT_RETRIES,
        "counts": {
            "primary_claims": len(base),
            "secondary_claims": len(claims) - len(base),
            "clips": len({claim["clip_uid"] for claim in base}),
            "audit_rows": len(audits),
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: card[key]
        for key in (
            "tactic_id", "definition", "observable_cues", "triggers", "confusing",
        )
    }


def _analysis_schema(tactic_ids: list[str], claim_ids: list[str] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "tactic_id": {"type": "string", "enum": tactic_ids},
        "state_before": {"type": ["string", "null"]},
        "trigger_action": {"type": ["string", "null"]},
        "defensive_response": {"type": ["string", "null"]},
        "space_origin": {"type": "string", "enum": sorted(SPACE_ORIGINS)},
        "space_evidence": {"type": ["string", "null"]},
        "beneficiary_effect": {"type": ["string", "null"]},
        "terminal_sequence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "event": {"type": "string"},
                },
                "required": ["start_s", "end_s", "event"],
                "additionalProperties": False,
            },
        },
        "supporting_cues": {"type": "array", "items": {"type": "string"}},
        "violated_cues": {"type": "array", "items": {"type": "string"}},
        "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        "evidence_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "fact": {"type": "string"},
                },
                "required": ["start_s", "end_s", "fact"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100, "multipleOf": 10},
        "confidence_reasons": {"type": "array", "items": {"type": "string"}},
        "alternative_tactics": {
            "type": "array", "maxItems": 2,
            "items": {"type": "string", "enum": tactic_ids},
        },
    }
    required = ["tactic_id", *ANALYSIS_FIELDS]
    if claim_ids is not None:
        properties = {"claim_id": {"type": "string", "enum": claim_ids}, **properties}
        required = ["claim_id", *required]
    return {
        "type": "object", "properties": properties, "required": required,
        "additionalProperties": False,
    }


def claim_response_schema(
    claims: list[dict[str, Any]],
    tactic_ids: list[str] | None = None,
) -> dict[str, Any]:
    allowed_tactics = sorted(tactic_ids or {claim["tactic_id"] for claim in claims})
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "minItems": len(claims),
                "maxItems": len(claims),
                "items": _analysis_schema(
                    allowed_tactics,
                    [claim["claim_id"] for claim in claims],
                ),
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def _p0_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "start_s": {"type": "number"},
            "end_s": {"type": "number"},
            "visible_movement_zh": {"type": "string"},
            "tactical_link_zh": {"type": "string"},
        },
        "required": [
            "start_s", "end_s", "visible_movement_zh", "tactical_link_zh",
        ],
        "additionalProperties": False,
    }


def p0_response_schema(tactic_ids: list[str], phase: str) -> dict[str, Any]:
    if phase not in P0_PHASES:
        raise ValueError(f"unknown P0 phase: {phase}")
    candidate_properties = {
        "rank": {"type": "integer", "enum": [1, 2, 3]},
        "tactic_id": {"type": "string", "enum": tactic_ids},
        "confidence": {"type": "integer", "enum": list(range(0, 101, 10))},
        "reason_zh": {"type": "string"},
        "matched_cues": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "evidence_spans": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": _p0_evidence_schema(),
        },
    }
    candidate_required = list(candidate_properties)
    if phase == "phase2_observation_first":
        candidate_properties["supporting_sequence_ids"] = {
            "type": "array", "minItems": 1, "items": {"type": "string"},
        }
        candidate_required.append("supporting_sequence_ids")
    properties: dict[str, Any] = {
        "candidates": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object",
                "properties": candidate_properties,
                "required": candidate_required,
                "additionalProperties": False,
            },
        },
        "no_nomination_reason_zh": {"type": "string"},
    }
    required = ["candidates", "no_nomination_reason_zh"]
    if phase == "phase2_observation_first":
        observation_fields = (
            "sequence_id", "start_s", "end_s", "action_zh", "defensive_change_zh",
            "space_change_or_use_zh", "attacking_effect_zh", "terminal_result_zh",
        )
        properties = {
            "observations": {
                "type": "array", "minItems": 1, "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "sequence_id": {"type": "string"},
                        "start_s": {"type": "number"},
                        "end_s": {"type": "number"},
                        **{
                            field: {"type": "string"}
                            for field in observation_fields[3:]
                        },
                    },
                    "required": list(observation_fields),
                    "additionalProperties": False,
                },
            },
            **properties,
        }
        required.insert(0, "observations")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


OBSERVATION_ORDER = """Use this order before deciding:
1. Describe the local positions and possession state before the key action.
2. Record the visible pass, run, carry, press, or restart.
3. Record the opponent's actual movement or constraint; never infer psychology.
4. Decide whether relevant space was created, already existed, is unclear, or is not applicable.
5. Identify who used the space and what playing condition it produced.
6. Compare only against the supplied definition, observable cues, triggers, and confusing distinctions.
7. Independently check restarts, crosses, clearances, player roles, and the terminal sequence.
8. Give the verdict last. A short clip that cannot establish a sustained structure is unobservable, not absent.
9. confidence must be exactly one of [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]."""


def build_claim_prompt(
    clip_uid: str,
    duration_s: float,
    claims: list[dict[str, Any]],
    cards: dict[str, dict[str, Any]],
) -> str:
    tactic_counts = Counter(claim["tactic_id"] for claim in claims)
    public_claims = [
        {
            "claim_id": claim["claim_id"],
            "tactic_id": claim["tactic_id"],
            "concept_card": _compact_card(cards[claim["tactic_id"]]),
            **(
                {
                    "evaluation_window": {
                        "start_s": claim["window"]["start_s"],
                        "end_s": claim["window"]["end_s"],
                    },
                }
                if tactic_counts[claim["tactic_id"]] > 1 else {}
            ),
        }
        for claim in claims
    ]
    prompt = f"""Watch the complete football video and independently verify each supplied tactical claim.
Do not use shirt identity, formation names, coach intent, or facts that are not directly visible.
Do not assume every clip contains a tactic. Return JSON only.

{OBSERVATION_ORDER}

Clip: {clip_uid}
Clip bounds: 0.000 to {duration_s:.3f} seconds.
Claims and public concept cards:
{json.dumps(public_claims, ensure_ascii=False)}
Allowed diagnostic alternative tactic IDs:
{json.dumps(sorted(cards), ensure_ascii=False)}

Top-level JSON must contain only the key "claims". Each claims item must contain exactly:
{json.dumps(["claim_id", "tactic_id", *ANALYSIS_FIELDS], ensure_ascii=False)}
Return exactly one analysis per claim. Evidence spans and terminal events
must stay within the clip bounds. confidence_reasons must be non-empty. alternative_tactics may only
use tactic IDs supplied in this prompt and is diagnostic, not a new formal claim.
"""
    if any(count > 1 for count in tactic_counts.values()):
        prompt += """Claims with evaluation_window refer only to that time range. Use the complete video
for context, but decide whether the tactic occurs inside the supplied window. For present or absent,
at least one decisive evidence span must overlap that claim's evaluation_window.
"""
    return prompt


DOUBAO_CLAIM_OUTPUT_CONTRACT = """
Strict JSON value contract for this provider:
- verdict must be exactly one of: present, absent, unobservable.
- space_origin must be exactly one of: created, existing, unclear, not_applicable.
- confidence must be an integer in: 0,10,20,30,40,50,60,70,80,90,100.
- supporting_cues, violated_cues, evidence_gaps, and confidence_reasons must be
  JSON arrays containing only strings. alternative_tactics must contain at most two IDs
  copied exactly from the allowed tactic ID list; use [] when no alternative is needed.
- every terminal_sequence item must be an object with exactly start_s, end_s, and event;
  every evidence_spans item must be an object with exactly start_s, end_s, and fact.
  Never put plain strings inside terminal_sequence or evidence_spans.
- state_before, trigger_action, defensive_response, space_evidence, and beneficiary_effect
  must be either a JSON string or JSON null. Do not translate enum values.
"""


def provider_claim_prompt(prompt: str, provider: str) -> str:
    return prompt if provider == "gemini" else prompt + DOUBAO_CLAIM_OUTPUT_CONTRACT

def build_p0_prompt(
    clip_alias: str,
    duration_s: float,
    p0_cards: list[dict[str, Any]],
    phase: str,
) -> str:
    if phase not in P0_PHASES:
        raise ValueError(f"unknown P0 phase: {phase}")
    cards_payload = [_compact_card(card) for card in p0_cards]
    phase_instruction = (
        "Compare all 19 cards directly against the video, then nominate only the 0 to 3 "
        "best-supported tactics. Do not output per-card screening rows, absent verdicts, or "
        "an observations field."
        if phase == "phase1_direct"
        else
        "Before considering any tactic names, extract 1 to 5 key visible sequences in the "
        "observations field. For each sequence separately record: action, actual defensive "
        "change, how space appeared or was used, attacking effect, and terminal result. "
        "Observations must not contain a tactic_id, tactic name, or tentative tactical judgment. "
        "Only after completing those observations, compare all 19 cards and nominate 0 to 3 "
        "best-supported tactics. Every candidate must reference supporting_sequence_ids."
    )
    tactic_ids = [card["tactic_id"] for card in p0_cards]
    return f"""You are performing open-vocabulary football-tactic nomination from a fixed list.

Watch the complete video and read ALL 19 supplied P0 concept cards. Inclusion of a card does not imply that the tactic is present. Multiple tactics may coexist, and no tactic may have enough visible evidence.

{phase_instruction}

A tactic may be nominated only when its defining mechanism is visibly supported. Confidence is diagnostic and used for ranking; it is not an inclusion threshold. Return at most three unique candidates, ordered by decreasing confidence. Each candidate must copy at least one matched cue exactly from that card and cite 1 to 3 decisive time spans. Write reasons and evidence descriptions in Chinese. If no tactic qualifies, return an empty candidates array and a non-empty no_nomination_reason_zh. If candidates are returned, no_nomination_reason_zh must be an empty string.

Use only visible evidence. Do not infer from filenames, prior labels, commentary, team reputation, shirt identity, coach intent, another model, or unstated facts. Do not output 19 verdict rows. Return JSON only and match the supplied schema exactly.

Clip: {clip_alias}
Clip bounds: 0.000 to {duration_s:.3f} seconds.
P0 CONCEPT CARDS:
{json.dumps(cards_payload, ensure_ascii=False)}

OUTPUT JSON SCHEMA:
{json.dumps(p0_response_schema(tactic_ids, phase), ensure_ascii=False)}
"""


def _p0_prompt_version(phase: str) -> str:
    return (
        P0_PHASE2_PROMPT_VERSION
        if phase == "phase2_observation_first"
        else P0_PROMPT_VERSIONS[phase]
    )


def _effective_p0_prompt(
    clip_alias: str,
    duration_s: float,
    p0_cards: list[dict[str, Any]],
    phase: str,
) -> str:
    prompt = build_p0_prompt(clip_alias, duration_s, p0_cards, phase)
    if phase == "phase2_observation_first":
        prompt += P0_PHASE2_PROMPT_AMENDMENT
    return prompt


def _nonempty_strings(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _time_items(value: Any, field: str, duration_s: float, text_key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    for index, item in enumerate(value):
        label = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        start, end = item.get("start_s"), item.get("end_s")
        if (
            isinstance(start, bool) or not isinstance(start, (int, float))
            or isinstance(end, bool) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or float(start) < 0 or float(end) < float(start) or float(end) > duration_s + 1e-6
        ):
            raise ValueError(f"{label} time is outside the real clip bounds")
        if not isinstance(item.get(text_key), str) or not item[text_key].strip():
            raise ValueError(f"{label}.{text_key} must be non-empty")
    return value


def _validate_analysis(
    item: Any,
    duration_s: float,
    known_tactics: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{label} must be an object")
    missing = [field for field in ("tactic_id", *ANALYSIS_FIELDS) if field not in item]
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")
    allowed = {"tactic_id", *ANALYSIS_FIELDS, "claim_id"}
    extra = set(item) - allowed
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")
    if item["tactic_id"] not in known_tactics:
        raise ValueError(f"{label}.tactic_id is unknown")
    for field in (
        "state_before", "trigger_action", "defensive_response",
        "space_evidence", "beneficiary_effect",
    ):
        if item[field] is not None and (
            not isinstance(item[field], str) or not item[field].strip()
        ):
            raise ValueError(f"{label}.{field} must be non-empty text or null")
    if item["space_origin"] not in SPACE_ORIGINS:
        raise ValueError(f"{label}.space_origin is invalid")
    if item["space_origin"] == "created" and (
        item["defensive_response"] is None or item["space_evidence"] is None
    ):
        raise ValueError(f"{label}.created requires defensive response and space evidence")
    if item["verdict"] not in VERDICTS:
        raise ValueError(f"{label}.verdict is invalid")
    confidence = item["confidence"]
    if (
        isinstance(confidence, bool) or not isinstance(confidence, int)
        or confidence < 0 or confidence > 100 or confidence % 10
    ):
        raise ValueError(f"{label}.confidence must be a multiple of 10 from 0 to 100")
    for field in ("supporting_cues", "violated_cues", "evidence_gaps"):
        _nonempty_strings(item[field], f"{label}.{field}")
    _nonempty_strings(item["confidence_reasons"], f"{label}.confidence_reasons", allow_empty=False)
    alternatives = _nonempty_strings(item["alternative_tactics"], f"{label}.alternative_tactics")
    if len(alternatives) > 2 or any(value not in known_tactics for value in alternatives):
        raise ValueError(f"{label}.alternative_tactics is invalid")
    _time_items(item["terminal_sequence"], f"{label}.terminal_sequence", duration_s, "event")
    _time_items(item["evidence_spans"], f"{label}.evidence_spans", duration_s, "fact")
    if item["verdict"] == "present":
        if not item["evidence_spans"]:
            raise ValueError(f"{label}.present requires visible evidence spans")
        if not item["trigger_action"] or not item["beneficiary_effect"] or not (
            item["defensive_response"] or item["space_evidence"]
        ):
            raise ValueError(f"{label}.present requires a complete visible mechanism chain")
    return item


def validate_claim_response(
    payload: Any,
    expected_claims: list[dict[str, Any]],
    duration_s: float,
    known_tactics: set[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise ValueError("claim response must contain only claims")
    items = payload["claims"]
    if not isinstance(items, list):
        raise ValueError("claims must be a list")
    expected = {claim["claim_id"]: claim for claim in expected_claims}
    tactic_counts = Counter(claim["tactic_id"] for claim in expected_claims)
    seen = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("claim_id") not in expected:
            raise ValueError(f"claims[{index}].claim_id is unknown")
        claim_id = item["claim_id"]
        claim = expected[claim_id]
        if claim_id in seen or item.get("tactic_id") != claim["tactic_id"]:
            raise ValueError(f"claims[{index}] is duplicated or mismatched")
        validated = _validate_analysis(
            item, duration_s, known_tactics, f"claims[{index}]",
        )
        if (
            tactic_counts[claim["tactic_id"]] > 1
            and validated["verdict"] in {"present", "absent"}
            and not any(
                min(float(span["end_s"]), float(claim["window"]["end_s"]))
                > max(float(span["start_s"]), float(claim["window"]["start_s"]))
                for span in validated["evidence_spans"]
            )
        ):
            raise ValueError(f"claims[{index}] has no evidence overlapping evaluation_window")
        seen[claim_id] = validated
    if set(seen) != set(expected):
        raise ValueError("response does not contain exactly the requested claims")
    payload["claims"] = [seen[claim["claim_id"]] for claim in expected_claims]
    return payload


def _p0_allowed_cues(card: dict[str, Any]) -> set[str]:
    cues = [*card.get("observable_cues", []), *card.get("triggers", [])]
    cues.extend(
        text
        for item in card.get("confusing", [])
        for text in (item.get("object"), item.get("distinction"))
        if text
    )
    return set(cues)


def _validate_p0_evidence(
    value: Any,
    label: str,
    duration_s: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError(f"{label} must contain 1 to 3 spans")
    fields = {"start_s", "end_s", "visible_movement_zh", "tactical_link_zh"}
    for index, span in enumerate(value):
        if not isinstance(span, dict) or set(span) != fields:
            raise ValueError(f"{label}[{index}] fields are not strict")
        _time_items([span], f"{label}[{index}]", duration_s, "visible_movement_zh")
        if not isinstance(span["tactical_link_zh"], str) or not span["tactical_link_zh"].strip():
            raise ValueError(f"{label}[{index}].tactical_link_zh must be non-empty")
    return value


def validate_p0_response(
    payload: Any,
    tactic_ids: list[str],
    duration_s: float,
    cards: dict[str, dict[str, Any]] | None = None,
    phase: str = "phase1_direct",
) -> dict[str, Any]:
    if phase not in P0_PHASES:
        raise ValueError(f"unknown P0 phase: {phase}")
    expected_top = {"candidates", "no_nomination_reason_zh"}
    if phase == "phase2_observation_first":
        expected_top.add("observations")
    if not isinstance(payload, dict) or set(payload) != expected_top:
        raise ValueError(f"P0 {phase} response fields are not strict")

    sequence_ids: set[str] = set()
    if phase == "phase2_observation_first":
        observations = payload["observations"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
            raise ValueError("observations must contain 1 to 5 sequences")
        fields = {
            "sequence_id", "start_s", "end_s", "action_zh", "defensive_change_zh",
            "space_change_or_use_zh", "attacking_effect_zh", "terminal_result_zh",
        }
        forbidden = {
            value.casefold()
            for card in (cards or {}).values()
            for value in (card.get("tactic_id"), card.get("name_zh"), card.get("name_en"))
            if value
        }
        for index, item in enumerate(observations):
            label = f"observations[{index}]"
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError(f"{label} fields are not strict")
            sequence_id = item["sequence_id"]
            if not isinstance(sequence_id, str) or not sequence_id.strip() or sequence_id in sequence_ids:
                raise ValueError(f"{label}.sequence_id is invalid or duplicated")
            sequence_ids.add(sequence_id)
            _time_items([{**item, "description": item["action_zh"]}], label, duration_s, "description")
            text_fields = fields - {"sequence_id", "start_s", "end_s"}
            for field in text_fields:
                if not isinstance(item[field], str) or not item[field].strip():
                    raise ValueError(f"{label}.{field} must be non-empty")
            observation_text = " ".join(item[field] for field in text_fields).casefold()
            if any(term in observation_text for term in forbidden):
                raise ValueError(f"{label} leaks a tactic name before nomination")

    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise ValueError("candidates must contain 0 to 3 items")
    no_reason = payload["no_nomination_reason_zh"]
    if not isinstance(no_reason, str):
        raise ValueError("no_nomination_reason_zh must be a string")
    if (not candidates and not no_reason.strip()) or (candidates and no_reason != ""):
        raise ValueError("no_nomination_reason_zh is inconsistent with candidates")

    known = set(tactic_ids)
    seen = set()
    previous_confidence = 101
    for index, item in enumerate(candidates):
        label = f"candidates[{index}]"
        fields = {
            "rank", "tactic_id", "confidence", "reason_zh", "matched_cues",
            "evidence_spans",
        }
        if phase == "phase2_observation_first":
            fields.add("supporting_sequence_ids")
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{label} fields are not strict")
        tactic_id = item["tactic_id"]
        if tactic_id not in known or tactic_id in seen:
            raise ValueError(f"{label}.tactic_id is unknown or duplicated")
        seen.add(tactic_id)
        if item["rank"] != index + 1:
            raise ValueError(f"{label}.rank must be consecutive")
        confidence = item["confidence"]
        if (
            isinstance(confidence, bool) or not isinstance(confidence, int)
            or confidence not in range(0, 101, 10) or confidence > previous_confidence
        ):
            raise ValueError(f"{label}.confidence is invalid or not sorted")
        previous_confidence = confidence
        if not isinstance(item["reason_zh"], str) or not item["reason_zh"].strip():
            raise ValueError(f"{label}.reason_zh must be non-empty")
        matched = _nonempty_strings(item["matched_cues"], f"{label}.matched_cues", allow_empty=False)
        if cards is not None and any(cue not in _p0_allowed_cues(cards[tactic_id]) for cue in matched):
            raise ValueError(f"{label} contains a cue outside its concept card")
        _validate_p0_evidence(item["evidence_spans"], f"{label}.evidence_spans", duration_s)
        if phase == "phase2_observation_first":
            references = _nonempty_strings(
                item["supporting_sequence_ids"],
                f"{label}.supporting_sequence_ids", allow_empty=False,
            )
            if any(reference not in sequence_ids for reference in references):
                raise ValueError(f"{label} references an unknown observation")
    return payload


def _json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.DOTALL)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def _canonicalize_doubao_p0_cues(
    payload: dict[str, Any], prompt: str,
) -> dict[str, Any]:
    marker = "P0 CONCEPT CARDS:\n"
    if marker not in prompt:
        return payload
    cards_text = prompt.split(marker, 1)[1].split("\n\nOUTPUT JSON SCHEMA:", 1)[0]
    cards = {card["tactic_id"]: card for card in json.loads(cards_text)}
    for candidate in payload.get("candidates", []):
        card = cards.get(candidate.get("tactic_id"))
        if not card or not isinstance(candidate.get("matched_cues"), list):
            continue
        allowed = _p0_allowed_cues(card)
        normalized = []
        for cue in candidate["matched_cues"]:
            if cue in allowed or not isinstance(cue, str):
                normalized.append(cue)
                continue
            key = cue.rstrip().rstrip("。；;")
            matches = [
                exact for exact in allowed
                if (
                    exact.rstrip().rstrip("。；;") == key
                    or exact.startswith(key + "。")
                    or ("：" in exact and exact.split("：", 1)[1] == key)
                )
            ]
            normalized.append(matches[0] if len(matches) == 1 else cue)
        candidate["matched_cues"] = normalized
    return payload


def call_doubao(
    prompt: str,
    video_path: Path,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configured_model = os.environ.get("ARK_RESPONSES_MODEL", DOUBAO_MODEL)
    if configured_model != DOUBAO_MODEL:
        raise ValueError(
            f"ARK_RESPONSES_MODEL must stay frozen at {DOUBAO_MODEL}, got {configured_model}"
        )
    if "claims" in schema.get("properties", {}):
        usage: list[dict[str, Any]] = []
        try:
            text = ark_chat(
                prompt,
                video_path=video_path,
                temperature=TEMPERATURE,
                max_tokens=12000,
                timeout_s=TIMEOUT_S,
                max_retries=0,
                usage_callback=usage.append,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tactical_response",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except Exception as exc:
            if _retryable(exc):
                raise RetryableProviderError(str(exc)) from exc
            raise
        try:
            return _json_text(text), usage
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{exc}; response excerpt: {text[:800]!r}"
            ) from exc
    api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise RuntimeError("ARK_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get(
            "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3",
        ),
        api_key=api_key,
        timeout=TIMEOUT_S,
        max_retries=0,
    )
    try:
        with video_path.open("rb") as handle:
            uploaded = client.files.create(
                file=(f"clip-{_sha256(video_path)[:12]}.mp4", handle, "video/mp4"),
                purpose="user_data",
            )
        deadline = time.monotonic() + TIMEOUT_S
        while getattr(uploaded, "status", None) == "processing":
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Doubao file processing timed out: {uploaded.id}")
            time.sleep(2.0)
            uploaded = client.files.retrieve(uploaded.id)
        if getattr(uploaded, "status", None) in {"error", "failed"}:
            raise RuntimeError(f"Doubao file processing failed: {uploaded.id}")
        response = client.responses.create(
            model=configured_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_video", "file_id": uploaded.id},
                    {"type": "input_text", "text": prompt},
                ],
            }],
            temperature=TEMPERATURE,
            max_output_tokens=12000,
            text={"format": {
                "type": "json_schema",
                "name": "tactical_response",
                "strict": True,
                "schema": schema,
            }},
        )
    except Exception as exc:
        if _retryable(exc):
            raise RetryableProviderError(str(exc)) from exc
        raise
    usage = [response.usage.model_dump(exclude_none=True)] if response.usage else []
    try:
        payload = _json_text(response.output_text)
        return _canonicalize_doubao_p0_cues(payload, prompt), usage
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{exc}; response excerpt: {response.output_text[:800]!r}"
        ) from exc


def _gemini_output_text(response: dict[str, Any]) -> str:
    texts = [
        part["text"]
        for step in response.get("steps", [])
        if step.get("type") == "model_output"
        for part in step.get("content", [])
        if part.get("type") == "text" and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise ValueError("Gemini response has no model text output")
    return "".join(texts)


def _gemini_api_keys() -> list[str]:
    return list(dict.fromkeys(
        value
        for name in GEMINI_KEY_ENV_NAMES
        if (value := os.environ.get(name, "").strip())
    ))


def _next_gemini_api_key() -> str:
    keys = _gemini_api_keys()
    if not keys:
        raise RuntimeError("no Gemini API key is set")
    try:
        offset = int(os.environ.get("GEMINI_KEY_ROTATION_OFFSET", "0"))
    except ValueError as exc:
        raise RuntimeError("GEMINI_KEY_ROTATION_OFFSET must be a non-negative integer") from exc
    if offset < 0:
        raise RuntimeError("GEMINI_KEY_ROTATION_OFFSET must be a non-negative integer")
    global _GEMINI_KEY_INDEX
    with _GEMINI_KEY_LOCK:
        api_key = keys[(offset + _GEMINI_KEY_INDEX) % len(keys)]
        _GEMINI_KEY_INDEX += 1
    return api_key


def call_gemini(
    prompt: str,
    video_path: Path,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import httpx

    api_key = _next_gemini_api_key()
    video_data = base64.b64encode(video_path.read_bytes()).decode("ascii")
    request = {
        "model": GEMINI_MODEL,
        "input": [
            {
                "type": "video",
                "data": video_data,
                "mime_type": "video/mp4",
                "resolution": "high",
            },
            {"type": "text", "text": prompt},
        ],
        "generation_config": {"temperature": TEMPERATURE},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    }
    try:
        response = httpx.post(
            os.environ.get("GEMINI_INTERACTIONS_URL", GEMINI_URL),
            headers={"x-goog-api-key": api_key},
            json=request,
            timeout=httpx.Timeout(TIMEOUT_S, connect=60.0),
        )
        if response.status_code == 429 or response.status_code >= 500:
            retry_after = getattr(response, "headers", {}).get("retry-after")
            match = re.search(r"retry in ([0-9.]+)s", response.text, re.IGNORECASE)
            raise RetryableProviderError(
                f"Gemini HTTP {response.status_code}: {response.text[:500]}",
                response.status_code,
                float(retry_after) if retry_after else float(match.group(1)) if match else None,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini HTTP {response.status_code}: {response.text[:1000]}"
            )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise RetryableProviderError(str(exc)) from exc
    body = response.json()
    usage = [body["usage"]] if isinstance(body.get("usage"), dict) else []
    return _json_text(_gemini_output_text(body)), usage


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return (
        status == 429
        or isinstance(status, int) and status >= 500
        or name in {
            "APITimeoutError", "APIConnectionError", "RateLimitError",
            "InternalServerError",
        }
    )


PROVIDER_CALLS: dict[
    str,
    Callable[[str, Path, dict[str, Any]], tuple[dict[str, Any], list[dict[str, Any]]]],
] = {"doubao": call_doubao, "gemini": call_gemini}


def call_with_transport_retries(
    call: Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    retries: int = TRANSPORT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    for attempt in range(retries + 1):
        try:
            payload, usage = call()
            return payload, usage, attempt + 1
        except RetryableProviderError as exc:
            if attempt == retries:
                exc.attempts = attempt + 1
                raise
            sleep(max(2**attempt, exc.retry_after_s or 0.0))
    raise AssertionError("unreachable")


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return _http_status(exc.__cause__) if exc.__cause__ is not None else None


def _error_record(
    provider: str,
    model: str,
    clip_uid: str,
    stage: str,
    exc: Exception,
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "provider": provider,
        "model": model,
        "clip_uid": clip_uid,
        "failed_stage": stage,
        "error_type": type(exc).__name__,
        "http_status": _http_status(exc),
        "error": str(exc)[:1000],
        "attempts": getattr(exc, "attempts", 1),
        "prompt_sha256": prompt_sha256,
        "configuration_sha256": _configuration_sha256(),
    }


def _validate_provider_payload(
    provider: str,
    payload: dict[str, Any],
    validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if provider != "doubao" or not isinstance(candidates, list):
        return validator(payload)
    confidences = [candidate.get("confidence") for candidate in candidates]
    allowed = set(range(0, 101, 10)) | {75}
    if (
        75 not in confidences
        or any(isinstance(value, bool) or value not in allowed for value in confidences)
        or confidences != sorted(confidences, reverse=True)
    ):
        return validator(payload)
    positions = [index for index, value in enumerate(confidences) if value == 75]
    for index in positions:
        candidates[index]["confidence"] = 70
    try:
        validated = validator(payload)
    finally:
        for index in positions:
            candidates[index]["confidence"] = 75
    return validated


def _run_provider(
    provider: str,
    prompt: str,
    video_path: Path,
    schema: dict[str, Any],
    validator: Callable[[Any], dict[str, Any]],
    clip_uid: str,
) -> dict[str, Any]:
    model = DOUBAO_MODEL if provider == "doubao" else GEMINI_MODEL
    prompt_hash = _text_sha256(prompt)
    try:
        payload, usage, attempts = call_with_transport_retries(
            lambda: PROVIDER_CALLS[provider](prompt, video_path, schema),
        )
    except Exception as exc:
        return _error_record(provider, model, clip_uid, "provider_request", exc, prompt_hash)
    try:
        payload = _validate_provider_payload(provider, payload, validator)
    except Exception as exc:
        error = _error_record(
            provider, model, clip_uid, "provider_validation", exc, prompt_hash,
        )
        error["response_excerpt"] = json.dumps(payload, ensure_ascii=False)[:2000]
        return error
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "provider": provider,
        "model": model,
        "clip_uid": clip_uid,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "configuration_sha256": _configuration_sha256(),
        "temperature": TEMPERATURE,
        "attempts": attempts,
        "api_usage": usage,
        **payload,
    }


def _load_frozen_gt(
    output_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("gt_status") != "frozen":
        raise ValueError("GT manifest is not frozen; run prepare-gt --approve")
    if manifest.get("configuration_sha256") != _configuration_sha256():
        raise ValueError("benchmark configuration changed; rerun prepare-gt --approve")
    paths = {
        "source_csv_sha256": repo_root / manifest["source_csv"],
        "glossary_sha256": repo_root / manifest["glossary"],
        "gt_claims_sha256": output_root / "gt/gt_claims.jsonl",
    }
    if any(manifest.get(key) != _sha256(path) for key, path in paths.items()):
        raise ValueError("GT manifest hashes no longer match the frozen inputs")
    return manifest, _load_jsonl(paths["gt_claims_sha256"])


def _run_configuration_is_current(payload: dict[str, Any]) -> bool:
    return payload.get("configuration_sha256") == _configuration_sha256()


def _run_is_current(path: Path, prompt: str, provider: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_model = DOUBAO_MODEL if provider == "doubao" else GEMINI_MODEL
    return (
        payload.get("provider") == provider
        and payload.get("model") == expected_model
        and payload.get("prompt_sha256") == _text_sha256(prompt)
        and payload.get("schema_version") == SCHEMA_VERSION
    )


def _providers_to_run(
    run_dir: Path,
    prompts: dict[str, str],
    providers: Iterable[str],
    *,
    force: bool,
    retry_failed: bool,
) -> list[str]:
    selected = tuple(providers)
    unknown = set(selected) - {"doubao", "gemini"}
    if unknown or not selected:
        raise ValueError(
            f"providers must be a non-empty subset of doubao/gemini: {sorted(unknown)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("providers must not contain duplicates")

    result = []
    for provider in selected:
        path = run_dir / f"{provider}.json"
        if force or not _run_is_current(path, prompts[provider], provider):
            result.append(provider)
            continue
        if retry_failed:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "failed":
                result.append(provider)
    return result

def run_claims(
    output_root: Path,
    glossary_path: Path,
    source_rows: Path,
    repo_root: Path,
    *,
    clip_uids: Iterable[str] | None = None,
    allow_external_upload: bool = False,
    force: bool = False,
    providers: Iterable[str] = ("doubao", "gemini"),
    retry_failed: bool = False,
) -> dict[str, int]:
    if not allow_external_upload:
        raise PermissionError("external video upload requires --allow-external-upload")
    _, claims = _load_frozen_gt(output_root, repo_root)
    cards = load_glossary(glossary_path)
    sources = load_source_index(source_rows, repo_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[claim["clip_uid"]].append(claim)
    selected = set(clip_uids or grouped)
    unknown = selected - set(grouped)
    if unknown:
        raise ValueError(f"unknown claim clips: {sorted(unknown)}")



    counts = Counter(clips=0, calls=0, success=0, failed=0, skipped=0)
    for clip_uid in sorted(selected):
        video_path = sources[clip_uid]["absolute_video_path"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        duration = video_duration(video_path)
        clip_claims = grouped[clip_uid]
        base_prompt = build_claim_prompt(clip_uid, duration, clip_claims, cards)
        if any(secret in base_prompt for secret in ("识别准确度", "更正", "判断依据")):
            raise AssertionError("GT-only fields leaked into the prompt")
        schema = claim_response_schema(clip_claims, sorted(cards))
        known = set(cards)
        validator = lambda payload: validate_claim_response(  # noqa: E731
            payload, clip_claims, duration, known,
        )
        run_dir = output_root / "runs" / _clip_dir(clip_uid)
        selected_providers = tuple(providers)
        prompts = {
            provider: provider_claim_prompt(base_prompt, provider)
            for provider in selected_providers
        }
        providers_to_run = _providers_to_run(
            run_dir, prompts, selected_providers, force=force, retry_failed=retry_failed,
        )
        counts["skipped"] += len(selected_providers) - len(providers_to_run)
        results = {}
        LOGGER.info("%s: running %s", clip_uid, ", ".join(providers_to_run) or "no providers")
        with ThreadPoolExecutor(max_workers=len(providers_to_run) or 1) as executor:
            futures = {
                executor.submit(
                    _run_provider,
                    provider,
                    prompts[provider],
                    video_path,
                    schema,
                    validator,
                    clip_uid,
                ): provider
                for provider in providers_to_run
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        for provider, result in results.items():
            _write_json(run_dir / f"{provider}.json", result)
            counts["calls"] += 1
            counts["success" if result["status"] == "success" else "failed"] += 1
            LOGGER.info("%s / %s: %s", clip_uid, provider, result["status"])
        counts["clips"] += 1
    return dict(counts)

def discover_p0_sources(
    repo_root: Path,
    source_rows: Path,
) -> dict[str, dict[str, Any]]:
    sources = load_source_index(source_rows, repo_root)
    prediction_paths = list((repo_root / "outputs").glob("*/predictions.json"))
    prediction_paths += list(
        (repo_root / "outputs/tactical-kb-audit").glob("*/predictions.json")
    )
    by_id = {}
    for path in prediction_paths:
        clip_id = path.parent.name
        by_id.setdefault(clip_id, path)
    result = {}
    for clip_id, predictions in sorted(by_id.items()):
        clip_uid = (
            f"event_clips:{clip_id}"
            if clip_id.isdigit()
            else f"soccernetgs:{clip_id}"
        )
        if clip_uid not in sources:
            raise ValueError(f"no unique source video for {clip_uid}")
        result[clip_uid] = {**sources[clip_uid], "predictions_path": predictions}
    return result


def frozen_p0_sources(
    repo_root: Path,
    source_rows: Path,
) -> dict[str, dict[str, Any]]:
    claim_root = repo_root / "outputs/tactical_claim_benchmark"
    manifest_path = claim_root / "manifest.json"
    gt_path = claim_root / "gt/gt_claims.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gt_status") != "frozen":
        raise ValueError("claim GT is not frozen")
    if manifest.get("gt_claims_sha256") != _sha256(gt_path):
        raise ValueError("frozen GT hash does not match")
    expected_source_rows = (repo_root / manifest["source_rows"]).resolve()
    if source_rows.resolve() != expected_source_rows:
        raise ValueError("source rows do not match the frozen GT manifest")
    claims = _load_jsonl(gt_path)
    clip_uids = sorted({
        claim["clip_uid"] for claim in claims if claim.get("score_set") == "primary"
    })
    if len(clip_uids) != 67:
        raise ValueError(f"expected 67 frozen P0 clips, got {len(clip_uids)}")
    sources = load_source_index(source_rows, repo_root)
    missing = set(clip_uids) - set(sources)
    if missing:
        raise ValueError(f"frozen clips are missing from source index: {sorted(missing)}")
    return {clip_uid: sources[clip_uid] for clip_uid in clip_uids}



def _p0_configuration() -> dict[str, Any]:
    contract = "".join(inspect.getsource(function) for function in (
        _p0_evidence_schema,
        p0_response_schema,
        build_p0_prompt,
        _p0_allowed_cues,
        _validate_p0_evidence,
        validate_p0_response,
    ))
    return {
        "prompt_contract_sha256": _text_sha256(contract),
        "prompt_versions": P0_PROMPT_VERSIONS,
        "schema_version": P0_SCHEMA_VERSION,
        "models": {"doubao": DOUBAO_MODEL, "gemini": GEMINI_MODEL},
        "temperature": TEMPERATURE,
        "timeout_s": TIMEOUT_S,
        "transport_retries": TRANSPORT_RETRIES,
    }


def _p0_configuration_sha256() -> str:
    return _text_sha256(json.dumps(_p0_configuration(), sort_keys=True))


def _p0_run_is_current(
    path: Path,
    prompt: str,
    provider: str,
    clip_sha256: str,
    phase: str,
    validator: Callable[[Any], dict[str, Any]] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_model = DOUBAO_MODEL if provider == "doubao" else GEMINI_MODEL
    metadata_matches = (
        payload.get("provider") == provider
        and payload.get("model") == expected_model
        and payload.get("phase") == phase
        and payload.get("prompt_sha256") == _text_sha256(prompt)
        and payload.get("prompt_version") == _p0_prompt_version(phase)
        and payload.get("schema_version") == P0_SCHEMA_VERSION
        and payload.get("configuration_sha256") == _p0_configuration_sha256()
        and payload.get("clip_sha256") == clip_sha256
        and (
            provider != "doubao"
            or payload.get("transport_version") == P0_DOUBAO_TRANSPORT_VERSION
        )
    )
    if not metadata_matches:
        return False
    if payload.get("status") == "success" and validator is not None:
        try:
            _validate_provider_payload(provider, _p0_response_body(payload, phase), validator)
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _ensure_p0_manifest(
    output_root: Path,
    glossary_path: Path,
    source_rows: Path,
    repo_root: Path,
    sources: dict[str, dict[str, Any]],
    p0_cards: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    enriched = {}
    clip_rows = []
    for clip_uid, source in sorted(sources.items()):
        video_path = source["absolute_video_path"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        digest = _sha256(video_path)
        record = {
            "clip_uid": clip_uid,
            "video_path": source.get("video_path", video_path.as_posix()),
            "neutral_alias": f"clip-{digest[:12]}.mp4",
            "sha256": digest,
            "bytes": video_path.stat().st_size,
            "duration_s": video_duration(video_path),
        }
        clip_rows.append(record)
        enriched[clip_uid] = {**source, **record}
    cards_payload = [_compact_card(card) for card in p0_cards]
    cards_text = json.dumps(cards_payload, ensure_ascii=False, sort_keys=True)
    tactic_ids = [card["tactic_id"] for card in p0_cards]
    gt_path = repo_root / "outputs/tactical_claim_benchmark/gt/gt_claims.jsonl"
    legacy_phases = {
        phase: {
            "prompt_version": P0_PROMPT_VERSIONS[phase],
            "prompt_template_sha256": _text_sha256(build_p0_prompt(
                "{{NEUTRAL_CLIP_ALIAS}}", 0.0, p0_cards, phase,
            )),
            "schema_sha256": _text_sha256(json.dumps(
                p0_response_schema(tactic_ids, phase), sort_keys=True,
            )),
        }
        for phase in P0_PHASES
    }
    expected = {
        "schema_version": "p0-nomination-manifest-v1",
        "experiment_id": P0_EXPERIMENT_ID,
        "status": "frozen",
        "frozen_inputs": {
            "clips": clip_rows,
            "clip_count": len(clip_rows),
            "tactic_ids": tactic_ids,
            "p0_cards_sha256": _text_sha256(cards_text),
            "glossary_sha256": _sha256(glossary_path),
            "source_rows_sha256": _sha256(source_rows),
            "gt_claims_sha256": _sha256(gt_path),
            "configuration_sha256": _p0_configuration_sha256(),
        },
        "phases": {
            phase: {
                "prompt_version": _p0_prompt_version(phase),
                "prompt_template_sha256": _text_sha256(_effective_p0_prompt(
                    "{{NEUTRAL_CLIP_ALIAS}}", 0.0, p0_cards, phase,
                )),
                "schema_sha256": _text_sha256(json.dumps(
                    p0_response_schema(tactic_ids, phase), sort_keys=True,
                )),
            }
            for phase in P0_PHASES
        },
        "execution": _p0_configuration(),
        "approved_amendments": {
            "2026-07-26": {
                "doubao_transport": P0_DOUBAO_TRANSPORT_VERSION,
                "doubao_transport_note": (
                    "Upload each original MP4 with Files API, wait for preprocessing, "
                    "then make one Responses API model call using its file_id."
                ),
                "doubao_parser_note": (
                    "Canonicalize only a uniquely matching concept-card cue when the "
                    "model differs solely by trailing whitespace/sentence punctuation, "
                    "or copies a unique complete first sentence ending at a period boundary, "
                    "or omits a unique structural label before a full-width colon."
                ),
                "doubao_confidence_note": (
                    "Accept confidence 75 when the candidate list remains descending; "
                    "all other non-10-point values remain invalid."
                ),
                "phase2_prompt_version": P0_PHASE2_PROMPT_VERSION,
                "phase2_prompt_amendment_sha256": _text_sha256(
                    P0_PHASE2_PROMPT_AMENDMENT,
                ),
            },
        },
    }
    path = output_root / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("frozen_inputs") != expected["frozen_inputs"]:
            raise ValueError("P0 manifest no longer matches frozen inputs/configuration")
        if existing.get("phases") not in (legacy_phases, expected["phases"]):
            raise ValueError("P0 manifest phase contracts changed")
        if existing != expected:
            _write_json(path, expected)
    else:
        _write_json(path, expected)
    _write_jsonl(output_root / "inputs/clips.jsonl", clip_rows)
    _write_json(output_root / "inputs/p0_cards.json", cards_payload)
    return enriched


def run_p0(
    output_root: Path,
    glossary_path: Path,
    source_rows: Path,
    repo_root: Path,
    *,
    phase: str = "phase1_direct",
    clip_uids: Iterable[str] | None = None,
    allow_external_upload: bool = False,
    force: bool = False,
    providers: Iterable[str] = (),
    retry_failed: bool = False,
) -> dict[str, Any]:
    if phase not in P0_PHASES:
        raise ValueError(f"unknown P0 phase: {phase}")
    if not allow_external_upload:
        raise PermissionError("external video upload requires --allow-external-upload")
    cards = load_glossary(glossary_path)
    p0_cards = sorted(
        (card for card in cards.values() if card["priority"] == "P0"),
        key=lambda card: card["tactic_id"],
    )
    if len(p0_cards) != 19:
        raise ValueError(f"expected 19 P0 tactics, got {len(p0_cards)}")
    if any(
        not card["definition"] or not card["observable_cues"] or not card["triggers"]
        for card in p0_cards
    ):
        raise ValueError("every P0 concept card requires definition, cues, and triggers")
    tactic_ids = [card["tactic_id"] for card in p0_cards]
    sources = _ensure_p0_manifest(
        output_root, glossary_path, source_rows, repo_root,
        frozen_p0_sources(repo_root, source_rows), p0_cards,
    )
    selected = set(clip_uids or sources)
    unknown = selected - set(sources)
    if unknown:
        raise ValueError(f"unknown P0 clips: {sorted(unknown)}")
    selected_providers = tuple(providers)
    if len(selected_providers) != 1 or selected_providers[0] not in {"doubao", "gemini"}:
        raise ValueError("run-p0 requires exactly one provider: doubao or gemini")

    cards_by_id = {card["tactic_id"]: card for card in p0_cards}
    cards_sha256 = _text_sha256(json.dumps(
        [_compact_card(card) for card in p0_cards], ensure_ascii=False, sort_keys=True,
    ))
    counts = Counter(clips=0, calls=0, success=0, failed=0, skipped=0)
    for clip_uid in sorted(selected):
        source = sources[clip_uid]
        video_path = source["absolute_video_path"]
        duration = source["duration_s"]
        prompt = _effective_p0_prompt(source["neutral_alias"], duration, p0_cards, phase)
        if clip_uid in prompt or str(video_path) in prompt:
            raise AssertionError("clip identity leaked into the P0 prompt")
        schema = p0_response_schema(tactic_ids, phase)
        validator = lambda payload: validate_p0_response(  # noqa: E731
            payload, tactic_ids, duration, cards_by_id, phase,
        )
        run_dir = output_root / phase / "runs" / _clip_dir(clip_uid)
        providers_to_run = []
        for provider in selected_providers:
            run_path = run_dir / f"{provider}.json"
            current = _p0_run_is_current(
                run_path, prompt, provider, source["sha256"], phase, validator,
            )
            if force or not current:
                providers_to_run.append(provider)
            elif retry_failed:
                payload = json.loads(run_path.read_text(encoding="utf-8"))
                if payload.get("status") == "failed":
                    providers_to_run.append(provider)
        counts["skipped"] += len(selected_providers) - len(providers_to_run)
        LOGGER.info("%s / %s: running %s", phase, clip_uid, ", ".join(providers_to_run) or "no providers")
        results = {
            provider: _run_provider(
                provider, prompt, video_path, schema, validator, clip_uid,
            )
            for provider in providers_to_run
        }
        for provider, result in results.items():
            result.update({
                "schema_version": P0_SCHEMA_VERSION,
                "experiment_id": P0_EXPERIMENT_ID,
                "phase": phase,
                "prompt_version": _p0_prompt_version(phase),
                "configuration_sha256": _p0_configuration_sha256(),
                "clip_sha256": source["sha256"],
                "p0_cards_sha256": cards_sha256,
            })
            if provider == "doubao":
                result["transport_version"] = P0_DOUBAO_TRANSPORT_VERSION
            _write_json(run_dir / f"{provider}.json", result)
            counts["calls"] += 1
            counts["success" if result["status"] == "success" else "failed"] += 1
            LOGGER.info("%s / %s / %s: %s", phase, clip_uid, provider, result["status"])
        counts["clips"] += 1
    return {"phase": phase, **dict(counts)}


def _run_index(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    models = {"doubao": DOUBAO_MODEL, "gemini": GEMINI_MODEL}
    for provider in ("doubao", "gemini"):
        for path in run_root.glob(f"*/{provider}.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("provider") == provider
                and payload.get("model") == models[provider]
                and payload.get("schema_version") == SCHEMA_VERSION
            ):
                result[(payload.get("clip_uid"), provider)] = payload
    return result


def _claim_item(run: dict[str, Any], claim_id: str) -> dict[str, Any] | None:
    if run.get("status") != "success":
        return None
    return next((item for item in run.get("claims", []) if item["claim_id"] == claim_id), None)


def _stable_sample(key: str, rate: float = 0.10) -> bool:
    draw = int(hashlib.sha256(f"{JUDGE_SEED}:{key}".encode()).hexdigest()[:8], 16)
    return draw % 10_000 < int(rate * 10_000)


def _frame_sheets(
    video_path: Path,
    out_dir: Path,
    duration_s: float,
    spans: list[dict[str, Any]],
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    jobs = [("overview", 0.0, duration_s, 2.0)]
    if spans:
        start = max(0.0, min(float(span["start_s"]) for span in spans) - 1.0)
        end = min(duration_s, max(float(span["end_s"]) for span in spans) + 1.0)
        jobs.append(("dense", start, end, 5.0))
    for name, start, end, fps in jobs:
        expected = max(1, math.ceil((end - start) * fps / 16))
        pattern = out_dir / f"{name}-%02d.jpg"
        filters = (
            f"trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS+{start:.3f}/TB,fps={fps},scale=640:-2,"
            "drawtext=text='%{pts\\:hms}':x=10:y=10:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.6,"
            "tile=4x4:padding=4:margin=4"
        )
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path), "-vf", filters, "-q:v", "3",
                "-frames:v", str(expected), "-y", str(pattern),
            ],
            check=True,
        )
        outputs.extend(path.as_posix() for path in sorted(out_dir.glob(f"{name}-*.jpg")))
    return outputs


def export_claim_judge_queue(
    benchmark_root: Path,
    glossary_path: Path,
    source_rows: Path,
    repo_root: Path,
) -> dict[str, int]:
    _, claims = _load_frozen_gt(benchmark_root, repo_root)
    cards = load_glossary(glossary_path)
    sources = load_source_index(source_rows, repo_root)
    runs = _run_index(benchmark_root / "runs")
    queue_root = benchmark_root / "judge_queue"
    key_rows, bundles = [], []
    for claim in (item for item in claims if item["score_set"] == "primary"):
        clip_uid, claim_id = claim["clip_uid"], claim["claim_id"]
        a = _claim_item(runs.get((clip_uid, "doubao"), {}), claim_id)
        b = _claim_item(runs.get((clip_uid, "gemini"), {}), claim_id)
        if not a or not b:
            continue
        disagreement = a["verdict"] != b["verdict"]
        if not disagreement and not _stable_sample(f"{clip_uid}:{claim_id}"):
            continue
        order = ["doubao", "gemini"]
        random.Random(f"{JUDGE_SEED}:{clip_uid}:{claim_id}").shuffle(order)
        items = {"doubao": a, "gemini": b}
        choice_a, choice_b = items[order[0]], items[order[1]]
        spans = list(choice_a["evidence_spans"]) + list(choice_b["evidence_spans"])
        bundle_id = hashlib.sha256(f"{clip_uid}:{claim_id}".encode()).hexdigest()[:16]
        frame_dir = queue_root / bundle_id / "frames"
        frames = _frame_sheets(
            sources[clip_uid]["absolute_video_path"],
            frame_dir,
            video_duration(sources[clip_uid]["absolute_video_path"]),
            spans,
        )
        bundle = {
            "schema_version": "tactical-sol-judge-bundle-v1",
            "bundle_id": bundle_id,
            "clip_uid": clip_uid,
            "claim_id": claim_id,
            "tactic_id": claim["tactic_id"],
            "reason": "disagreement" if disagreement else "agreement_audit",
            "concept_card": _compact_card(cards[claim["tactic_id"]]),
            "choice_A": choice_a,
            "choice_B": choice_b,
            "frame_paths": [
                Path(path).relative_to(repo_root).as_posix() for path in frames
            ],
            "required_output": {
                "bundle_id": bundle_id,
                "verdict": "present|absent|unobservable",
                "supported_side": "A|B|neither",
                "decisive_evidence": "non-empty text",
                "confidence": "0..100 multiple of 10",
            },
        }
        _write_json(queue_root / bundle_id / "bundle.json", bundle)
        bundles.append(bundle)
        key_rows.append({
            "bundle_id": bundle_id,
            "A": order[0],
            "B": order[1],
        })
    _write_json(benchmark_root / "judge_key.json", {
        "seed": JUDGE_SEED,
        "mappings": key_rows,
    })
    lines = ["# Tactical claim Judge queue", ""]
    for bundle in bundles:
        lines += [
            f"## {bundle['bundle_id']} — {bundle['clip_uid']} / {bundle['tactic_id']}",
            "",
            f"Reason: `{bundle['reason']}`",
            "",
            f"Bundle: `{(queue_root / bundle['bundle_id'] / 'bundle.json').as_posix()}`",
            "",
        ]
    _atomic_write(queue_root / "review.md", "\n".join(lines) + "\n")
    return {
        "bundles": len(bundles),
        "disagreements": sum(bundle["reason"] == "disagreement" for bundle in bundles),
        "agreement_audits": sum(bundle["reason"] == "agreement_audit" for bundle in bundles),
    }


def import_judge_decisions(
    benchmark_root: Path,
    reviews_path: Path,
) -> dict[str, int]:
    payload = json.loads(reviews_path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("reviews must contain a decisions list")
    bundles = {
        row["bundle_id"]: row
        for row in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in (benchmark_root / "judge_queue").glob("*/bundle.json")
        )
    }
    seen, validated = set(), []
    for index, decision in enumerate(decisions):
        bundle_id = decision.get("bundle_id") if isinstance(decision, dict) else None
        if bundle_id not in bundles or bundle_id in seen:
            raise ValueError(f"decisions[{index}] has unknown or duplicate bundle_id")
        if decision.get("verdict") not in VERDICTS:
            raise ValueError(f"decisions[{index}].verdict is invalid")
        if decision.get("supported_side") not in {"A", "B", "neither"}:
            raise ValueError(f"decisions[{index}].supported_side is invalid")
        if (
            not isinstance(decision.get("decisive_evidence"), str)
            or not decision["decisive_evidence"].strip()
        ):
            raise ValueError(f"decisions[{index}].decisive_evidence is empty")
        confidence = decision.get("confidence")
        if (
            isinstance(confidence, bool) or not isinstance(confidence, int)
            or confidence < 0 or confidence > 100 or confidence % 10
        ):
            raise ValueError(f"decisions[{index}].confidence is invalid")
        seen.add(bundle_id)
        validated.append(decision)
    _write_jsonl(benchmark_root / "judge_decisions.jsonl", validated)
    return {"imported": len(validated)}


def _judge_by_claim(benchmark_root: Path) -> dict[tuple[str, str], str]:
    ledger = benchmark_root / "judge_decisions.jsonl"
    if not ledger.is_file():
        return {}
    bundle_index = {
        row["bundle_id"]: row
        for row in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in (benchmark_root / "judge_queue").glob("*/bundle.json")
        )
    }
    return {
        (bundle_index[row["bundle_id"]]["clip_uid"], bundle_index[row["bundle_id"]]["claim_id"]):
        row["verdict"]
        for row in _load_jsonl(ledger)
        if row["bundle_id"] in bundle_index
    }


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _metrics(rows: list[tuple[str, str | None]]) -> dict[str, Any]:
    total = len(rows)
    covered = [(gt, pred) for gt, pred in rows if pred in {"present", "absent"}]
    unobservable = sum(pred == "unobservable" for _, pred in rows)
    missing = sum(pred is None for _, pred in rows)
    tp = sum(gt == pred == "present" for gt, pred in covered)
    tn = sum(gt == pred == "absent" for gt, pred in covered)
    fp = sum(gt == "absent" and pred == "present" for gt, pred in covered)
    fn = sum(gt == "present" and pred == "absent" for gt, pred in covered)
    correct = tp + tn
    positives, negatives = tp + fn, tn + fp
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    return {
        "n": total,
        "covered": len(covered),
        "coverage": round(len(covered) / total, 4) if total else None,
        "unobservable": unobservable,
        "unobservable_rate": round(unobservable / total, 4) if total else None,
        "missing": missing,
        "unobservable_or_missing": total - len(covered),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy_all": round(correct / total, 4) if total and not missing else None,
        "accuracy_all_wilson_95": _wilson(correct, total) if not missing else None,
        "accuracy_covered": round(correct / len(covered), 4) if covered else None,
        "positive_recall": round(recall, 4) if recall is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "balanced_accuracy": (
            round((recall + specificity) / 2, 4)
            if recall is not None and specificity is not None else None
        ),
    }


def _window_overlap(
    item: dict[str, Any] | None,
    window: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if item is None or window is None:
        return None
    start, end = float(window["start_s"]), float(window["end_s"])
    overlaps = [
        max(0.0, min(end, float(span["end_s"])) - max(start, float(span["start_s"])))
        for span in item["evidence_spans"]
    ]
    best = max(overlaps, default=0.0)
    return {
        "has_overlap": best > 0,
        "max_gt_window_fraction": round(best / (end - start), 4),
    }


def _fusion_ablation(
    claims: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    from pipeline.tactics_qa.auto_evidence import build_claim_evidence
    from pipeline.tactics_qa.checkers import run_checkers
    from pipeline.tactics_qa.gsr_io import load_gsr

    prediction_paths = list((repo_root / "outputs").glob("*/predictions.json"))
    prediction_paths += list(
        (repo_root / "outputs/tactical-kb-audit").glob("*/predictions.json")
    )
    predictions = {}
    for path in prediction_paths:
        predictions.setdefault(path.parent.name, path)
    selected = [
        claim for claim in claims
        if claim["clip_uid"].startswith("soccernetgs:")
        and claim["clip_id"] in predictions
    ]
    detail = {row["claim_id"]: row for row in detail_rows}
    cache = {}
    rows = []
    for claim in selected:
        try:
            clip_id = claim["clip_id"]
            if clip_id not in cache:
                cache[clip_id] = load_gsr(predictions[clip_id])
            clip = cache[clip_id]
            checker_tactic = OLD_TACTIC_IDS.get(claim["tactic_id"], claim["tactic_id"])
            evidence = build_claim_evidence(
                clip, claim["clip_uid"], checker_tactic, claim["window"],
            )
            checks = run_checkers(evidence, claim["window"])
            veto = any(check.verdict == "veto" for check in checks)
            error = None
        except Exception as exc:
            checks, veto = [], False
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
        source = detail[claim["claim_id"]]
        provider_results = {}
        for provider in ("doubao", "gemini"):
            before = source[provider]
            after = "absent" if before == "present" and veto else before
            provider_results[provider] = {"before": before, "after": after}
        rows.append({
            "clip_uid": claim["clip_uid"],
            "claim_id": claim["claim_id"],
            "tactic_id": claim["tactic_id"],
            "gt": claim["gt_verdict"],
            "checker_tactic_id": checker_tactic,
            "checks": [check.__dict__ for check in checks],
            "veto": veto,
            "error": error,
            "providers": provider_results,
        })
    systems = {}
    for provider in ("doubao", "gemini"):
        before_rows = [
            (row["gt"], row["providers"][provider]["before"]) for row in rows
        ]
        after_rows = [
            (row["gt"], row["providers"][provider]["after"]) for row in rows
        ]
        systems[provider] = {
            "before": _metrics(before_rows),
            "after": _metrics(after_rows),
            "changed": sum(before != after for (_, before), (_, after) in zip(before_rows, after_rows)),
            "corrected": sum(
                before != gt and after == gt
                for (gt, before), (_, after) in zip(before_rows, after_rows)
            ),
            "harmed": sum(
                before == gt and after != gt
                for (gt, before), (_, after) in zip(before_rows, after_rows)
            ),
        }
    return {
        "status": "complete" if not any(row["error"] for row in rows) else "partial",
        "scope": "secondary_diagnostic_only",
        "fusion_rule": "trajectory veto changes provider present to absent; no other verdict changes",
        "overlap_claims": len(rows),
        "systems": systems,
        "rows": rows,
    }


def generate_claim_report(
    benchmark_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    _, all_claims = _load_frozen_gt(benchmark_root, repo_root)
    claims = [claim for claim in all_claims if claim["score_set"] == "primary"]
    runs = _run_index(benchmark_root / "runs")
    expected_clips = {claim["clip_uid"] for claim in claims}
    expected_run_keys = {
        (clip_uid, provider)
        for clip_uid in expected_clips for provider in ("doubao", "gemini")
    }
    judge = _judge_by_claim(benchmark_root)
    provider_rows = {"doubao": [], "gemini": [], "consensus": [], "sol_final": []}
    detail_rows = []
    for claim in claims:
        clip_uid, claim_id, gt = claim["clip_uid"], claim["claim_id"], claim["gt_verdict"]
        items = {
            provider: _claim_item(runs.get((clip_uid, provider), {}), claim_id)
            for provider in ("doubao", "gemini")
        }
        verdicts = {
            provider: item["verdict"] if item else None
            for provider, item in items.items()
        }
        consensus = (
            verdicts["doubao"]
            if verdicts["doubao"] is not None and verdicts["doubao"] == verdicts["gemini"]
            else None
        )
        judge_verdict = judge.get((clip_uid, claim_id))
        final = judge_verdict if judge_verdict is not None else consensus
        for provider in ("doubao", "gemini"):
            provider_rows[provider].append((gt, verdicts[provider]))
        provider_rows["consensus"].append((gt, consensus))
        provider_rows["sol_final"].append((gt, final))
        detail_rows.append({
            "clip_uid": clip_uid,
            "claim_id": claim_id,
            "tactic_id": claim["tactic_id"],
            "gt": gt,
            **verdicts,
            "consensus": consensus,
            "judge": judge_verdict,
            "sol_final": final,
            "window_overlap": {
                provider: _window_overlap(item, claim["window"])
                for provider, item in items.items()
            },
            "alternative_matches_correction": {
                provider: (
                    bool(set(item["alternative_tactics"]) & set(claim["correction_label_ids"]))
                    if item and claim["correction_label_ids"] else None
                )
                for provider, item in items.items()
            },
        })

    per_tactic = {}
    for tactic_id in sorted({claim["tactic_id"] for claim in claims}):
        tactic_rows = [row for row in detail_rows if row["tactic_id"] == tactic_id]
        systems = {
            system: _metrics([(row["gt"], row[system]) for row in tactic_rows])
            for system in ("doubao", "gemini", "consensus", "sol_final")
        }
        per_tactic[tactic_id] = {
            **systems["sol_final"],
            "systems": systems,
            "sample_status": (
                "insufficient_sample" if len(tactic_rows) < 5 else "measured"
            ),
        }

    paired = [
        row for row in detail_rows
        if row["doubao"] is not None and row["gemini"] is not None
    ]
    disagreements = [row for row in paired if row["doubao"] != row["gemini"]]
    queue_bundles = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (benchmark_root / "judge_queue").glob("*/bundle.json")
    ]
    alternative_diagnostic = {}
    window_diagnostic = {}
    for provider in ("doubao", "gemini"):
        alternative_values = [
            row["alternative_matches_correction"][provider]
            for row in detail_rows
            if row["alternative_matches_correction"][provider] is not None
        ]
        overlap_values = [
            row["window_overlap"][provider]
            for row in detail_rows
            if row["window_overlap"][provider] is not None
        ]
        alternative_diagnostic[provider] = {
            "opportunities": len(alternative_values),
            "matches": sum(alternative_values),
            "match_rate": (
                round(sum(alternative_values) / len(alternative_values), 4)
                if alternative_values else None
            ),
        }
        window_diagnostic[provider] = {
            "claims_with_result": len(overlap_values),
            "any_overlap": sum(value["has_overlap"] for value in overlap_values),
            "overlap_rate": (
                round(
                    sum(value["has_overlap"] for value in overlap_values)
                    / len(overlap_values),
                    4,
                ) if overlap_values else None
            ),
        }

    provider_execution = {}
    for provider in ("doubao", "gemini"):
        expected_keys = {(clip_uid, provider) for clip_uid in expected_clips}
        attempted = expected_keys & set(runs)
        successful = {
            key for key in attempted if runs[key].get("status") == "success"
        }
        valid_claims = sum(row[provider] is not None for row in detail_rows)
        provider_execution[provider] = {
            "expected_video_runs": len(expected_keys),
            "attempted_video_runs": len(attempted),
            "successful_video_runs": len(successful),
            "failed_video_runs": len(attempted - successful),
            "missing_video_runs": len(expected_keys - attempted),
            "expected_claim_outputs": len(claims),
            "valid_claim_outputs": valid_claims,
            "missing_claim_outputs": len(claims) - valid_claims,
            "status": "complete" if valid_claims == len(claims) else "partial",
        }
    execution_complete = all(
        values["status"] == "complete" for values in provider_execution.values()
    )

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED" if execution_complete else "PARTIAL",
            "version_label": "tactical_claim_report_v1",
        },
        "experiment_id": EXPERIMENT_ID,
        "development_set_only": True,
        "primary_claims": len(claims),
        "execution": {
            "status": "complete" if execution_complete else "partial",
            "expected_provider_runs": len(expected_run_keys),
            "current_provider_runs": len(set(runs) & expected_run_keys),
            "missing_provider_runs": len(expected_run_keys - set(runs)),
            "expected_provider_claim_outputs": len(claims) * 2,
            "valid_provider_claim_outputs": sum(
                values["valid_claim_outputs"] for values in provider_execution.values()
            ),
            "providers": provider_execution,
        },
        "systems": {name: _metrics(rows) for name, rows in provider_rows.items()},
        "per_tactic": per_tactic,
        "provider_diagnostics": {
            "alternative_tactic_matches_correction": alternative_diagnostic,
            "gt_window_overlap": window_diagnostic,
            "paired_claims": len(paired),
            "disagreements": len(disagreements),
            "disagreement_rate": (
                round(len(disagreements) / len(paired), 4) if paired else None
            ),
        },
        "judge": {
            "queue_bundles": len(queue_bundles),
            "trigger_rate": (
                round(len(queue_bundles) / len(paired), 4) if paired else None
            ),
            "decisions_imported": len(judge),
            "consensus_overturns": sum(
                row["judge"] is not None and row["consensus"] is not None
                and row["judge"] != row["consensus"] for row in detail_rows
            ),
            "differs_from_doubao": sum(
                row["judge"] is not None and row["doubao"] is not None
                and row["judge"] != row["doubao"] for row in detail_rows
            ),
            "differs_from_gemini": sum(
                row["judge"] is not None and row["gemini"] is not None
                and row["judge"] != row["gemini"] for row in detail_rows
            ),
            "unresolved_disagreements": sum(
                row["doubao"] is not None and row["gemini"] is not None
                and row["doubao"] != row["gemini"] and row["sol_final"] is None
                for row in detail_rows
            ),
        },
        "rows": detail_rows,
    }
    reports = benchmark_root / "reports"
    _write_json(reports / "claim_metrics.json", report)
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "per_tactic.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "tactic_id", "system", "sample_status", "n", "covered", "coverage",
            "unobservable", "unobservable_rate", "missing", "tp", "tn", "fp", "fn",
            "accuracy_all", "accuracy_all_wilson_95", "accuracy_covered", "balanced_accuracy",
            "positive_recall", "specificity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tactic_id, values in per_tactic.items():
            for system, metrics in values["systems"].items():
                writer.writerow({
                    "tactic_id": tactic_id,
                    "system": system,
                    "sample_status": values["sample_status"],
                    **{key: metrics.get(key) for key in fields[3:]},
                })

    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Verification Status: {'ANALYZED' if execution_complete else 'PARTIAL'}",
        "- Version Label: tactical_claim_report_v1",
        "",
        "# 战术 Claim 开发集评测",
        "",
        "本报告只描述当前开发集表现，不代表未见视频泛化能力。",
        "",
        "| System | Accuracy | Balanced accuracy | Positive recall | Specificity | Coverage | Unobservable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["systems"].items():
        lines.append(
            f"| {name} | {values['accuracy_all']} | "
            f"{values['balanced_accuracy']} | {values['positive_recall']} | "
            f"{values['specificity']} | {values['coverage']} | "
            f"{values['unobservable_rate']} |"
        )
    _atomic_write(reports / "claim_metrics.md", "\n".join(lines) + "\n")
    _atomic_write(
        reports / "confusion_analysis.md",
        "# Confusion analysis\n\n"
        + "\n".join(
            f"- {tactic}: TP={row['tp']}, TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, "
            f"status={row['sample_status']}"
            for tactic, row in per_tactic.items()
        )
        + "\n",
    )
    _write_json(
        reports / "fusion_ablation.json",
        _fusion_ablation(claims, detail_rows, repo_root),
    )
    _write_json(reports / "historical_reference.json", {
        "status": "not_strict_ab",
        "source": "benchmark/tactical_prototypes/agent_runs",
        "note": "Historical runs used contact sheets and a different open-nomination task.",
    })
    return report


def _p0_response_body(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    keys = ["candidates", "no_nomination_reason_zh"]
    if phase == "phase2_observation_first":
        keys.insert(0, "observations")
    return {key: payload[key] for key in keys if key in payload}


def _p0_runs_for_report(
    output_root: Path,
    phase: str,
    provider: str,
    tactic_ids: list[str],
    cards: dict[str, dict[str, Any]],
    clip_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, int]:
    runs = {}
    attempted = 0
    invalid = 0
    for clip_uid, clip in clip_rows.items():
        path = output_root / phase / "runs" / _clip_dir(clip_uid) / f"{provider}.json"
        if not path.is_file():
            continue
        attempted += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_model = DOUBAO_MODEL if provider == "doubao" else GEMINI_MODEL
        if (
            payload.get("status") != "success"
            or payload.get("provider") != provider
            or payload.get("model") != expected_model
            or payload.get("phase") != phase
            or payload.get("prompt_version") != _p0_prompt_version(phase)
            or payload.get("schema_version") != P0_SCHEMA_VERSION
            or payload.get("configuration_sha256") != _p0_configuration_sha256()
            or payload.get("clip_sha256") != clip["sha256"]
            or (
                provider == "doubao"
                and payload.get("transport_version") != P0_DOUBAO_TRANSPORT_VERSION
            )
        ):
            invalid += 1
            continue
        try:
            _validate_provider_payload(
                provider,
                _p0_response_body(payload, phase),
                lambda value: validate_p0_response(
                    value, tactic_ids, float(clip["duration_s"]), cards, phase,
                ),
            )
        except ValueError:
            invalid += 1
            continue
        runs[clip_uid] = payload
    return runs, attempted, invalid


def _candidate_matches_claim(
    run: dict[str, Any],
    claim: dict[str, Any],
    top_k: int,
) -> bool:
    window = claim["window"]
    return any(
        candidate["tactic_id"] == claim["tactic_id"]
        and any(
            min(float(span["end_s"]), float(window["end_s"]))
            > max(float(span["start_s"]), float(window["start_s"]))
            for span in candidate["evidence_spans"]
        )
        for candidate in run["candidates"][:top_k]
    )


def _p0_claim_metrics(
    claims: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for claim in claims:
        run = runs.get(claim["clip_uid"])
        if run is None:
            continue
        tactic_nominated = any(
            candidate["tactic_id"] == claim["tactic_id"]
            for candidate in run["candidates"][:top_k]
        )
        predicted_present = _candidate_matches_claim(run, claim, top_k)
        predicted = "present" if predicted_present else "absent"
        rows.append({
            "claim_id": claim["claim_id"],
            "clip_uid": claim["clip_uid"],
            "tactic_id": claim["tactic_id"],
            "gt_verdict": claim["gt_verdict"],
            "predicted_verdict": predicted,
            "tactic_nominated": tactic_nominated,
            "evidence_window_overlap": predicted_present,
            "correct": predicted == claim["gt_verdict"],
            "top_k": top_k,
        })
    correct = sum(row["correct"] for row in rows)
    complete = len(rows) == len(claims)
    positives = [row for row in rows if row["gt_verdict"] == "present"]
    expected_positives = sum(claim["gt_verdict"] == "present" for claim in claims)
    positive_hits = sum(row["predicted_verdict"] == "present" for row in positives)
    nominated = [row for row in rows if row["tactic_nominated"]]
    by_tactic = {}
    for tactic_id in sorted({claim["tactic_id"] for claim in claims}):
        expected_n = sum(claim["tactic_id"] == tactic_id for claim in claims)
        subset = [row for row in rows if row["tactic_id"] == tactic_id]
        tactic_correct = sum(row["correct"] for row in subset)
        by_tactic[tactic_id] = {
            "expected_n": expected_n,
            "covered_n": len(subset),
            "correct": tactic_correct,
            "accuracy": (
                tactic_correct / expected_n if len(subset) == expected_n else None
            ),
            "covered_accuracy": tactic_correct / len(subset) if subset else None,
        }
    return {
        "top_k": top_k,
        "expected_claims": len(claims),
        "covered_claims": len(rows),
        "correct": correct,
        "accuracy": correct / len(claims) if complete else None,
        "covered_accuracy": correct / len(rows) if rows else None,
        "wilson_95": _wilson(correct, len(claims)) if complete else None,
        "covered_wilson_95": _wilson(correct, len(rows)),
        "positive_claims": expected_positives,
        "covered_positive_claims": len(positives),
        "positive_hits": positive_hits,
        "positive_recall": (
            positive_hits / expected_positives if complete and expected_positives else None
        ),
        "covered_positive_recall": (
            positive_hits / len(positives) if positives else None
        ),
        "tactic_nominated_claims": len(nominated),
        "nominations_with_window_overlap": sum(
            row["evidence_window_overlap"] for row in nominated
        ),
        "evidence_window_overlap_rate": (
            sum(row["evidence_window_overlap"] for row in nominated) / len(nominated)
            if nominated else None
        ),
        "by_tactic": by_tactic,
    }, rows


def _p0_candidate_diagnostics(
    runs: dict[str, dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    counts = [len(run["candidates"]) for run in runs.values()]
    tactics = Counter(
        candidate["tactic_id"]
        for run in runs.values()
        for candidate in run["candidates"]
    )
    confidences = [
        candidate["confidence"]
        for run in runs.values()
        for candidate in run["candidates"]
    ]
    diagnostics = {
        "valid_clips": len(runs),
        "empty_clips": sum(count == 0 for count in counts),
        "empty_rate": sum(count == 0 for count in counts) / len(counts) if counts else None,
        "mean_candidates": sum(counts) / len(counts) if counts else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "nominations_by_tactic": dict(sorted(tactics.items())),
    }
    if phase == "phase2_observation_first":
        observations = [item for run in runs.values() for item in run["observations"]]
        candidates = [item for run in runs.values() for item in run["candidates"]]
        observation_fields = (
            "action_zh", "defensive_change_zh", "space_change_or_use_zh",
            "attacking_effect_zh", "terminal_result_zh",
        )
        complete = sum(
            all(isinstance(item.get(field), str) and item[field].strip() for field in observation_fields)
            for item in observations
        )
        valid_references = sum(
            bool(item.get("supporting_sequence_ids"))
            for item in candidates
        )
        diagnostics["observations"] = {
            "sequences": len(observations),
            "five_dimension_field_complete_rate": (
                complete / len(observations) if observations else None
            ),
            "candidate_reference_valid_rate": (
                valid_references / len(candidates) if candidates else None
            ),
        }
    return diagnostics


def _p0_provider_agreement(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clips = sorted(set(first) & set(second))
    top1_matches = 0
    ranked_matches = 0
    jaccards = []
    for clip_uid in clips:
        a = [item["tactic_id"] for item in first[clip_uid]["candidates"]]
        b = [item["tactic_id"] for item in second[clip_uid]["candidates"]]
        top1_matches += (a[0] if a else None) == (b[0] if b else None)
        ranked_matches += a == b
        union = set(a) | set(b)
        jaccards.append(len(set(a) & set(b)) / len(union) if union else 1.0)
    return {
        "paired_clips": len(clips),
        "top1_agreement": top1_matches / len(clips) if clips else None,
        "ranked_list_agreement": ranked_matches / len(clips) if clips else None,
        "mean_top3_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
    }


def generate_p0_report(
    output_root: Path,
    glossary_path: Path,
) -> dict[str, Any]:
    cards = load_glossary(glossary_path)
    p0_cards = sorted(
        (card for card in cards.values() if card["priority"] == "P0"),
        key=lambda card: card["tactic_id"],
    )
    tactic_ids = [card["tactic_id"] for card in p0_cards]
    cards_by_id = {card["tactic_id"]: card for card in p0_cards}
    clip_rows = {
        row["clip_uid"]: row
        for row in _load_jsonl(output_root / "inputs/clips.jsonl")
    }
    gt_path = output_root.parent / "gt/gt_claims.jsonl"
    claims = [
        claim for claim in _load_jsonl(gt_path)
        if claim.get("score_set") == "primary" and claim.get("tactic_id") in set(tactic_ids)
    ]
    if len(claims) != 55:
        raise ValueError(f"expected 55 primary P0 claims, got {len(claims)}")

    all_runs: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    conditions = {}
    detail_rows = []
    for phase in P0_PHASES:
        conditions[phase] = {}
        for provider in ("gemini", "doubao"):
            runs, attempted, invalid = _p0_runs_for_report(
                output_root, phase, provider, tactic_ids, cards_by_id, clip_rows,
            )
            all_runs[(phase, provider)] = runs
            top1, top1_rows = _p0_claim_metrics(claims, runs, 1)
            top3, top3_rows = _p0_claim_metrics(claims, runs, 3)
            for row in top1_rows:
                detail_rows.append({"phase": phase, "provider": provider, **row})
            conditions[phase][provider] = {
                "execution": {
                    "expected_clips": len(clip_rows),
                    "attempted_clips": attempted,
                    "valid_clips": len(runs),
                    "invalid_or_failed_clips": invalid,
                    "missing_clips": len(clip_rows) - attempted,
                    "status": "complete" if len(runs) == len(clip_rows) else "partial",
                },
                "top1_claim_metrics": top1,
                "top3_claim_metrics": top3,
                "candidates": _p0_candidate_diagnostics(runs, phase),
            }
        conditions[phase]["provider_agreement"] = _p0_provider_agreement(
            all_runs[(phase, "gemini")], all_runs[(phase, "doubao")],
        )

    phase_changes = {}
    for provider in ("gemini", "doubao"):
        _, first_rows = _p0_claim_metrics(claims, all_runs[(P0_PHASES[0], provider)], 1)
        _, second_rows = _p0_claim_metrics(claims, all_runs[(P0_PHASES[1], provider)], 1)
        first = {row["claim_id"]: row for row in first_rows}
        second = {row["claim_id"]: row for row in second_rows}
        paired = sorted(set(first) & set(second))
        before = sum(first[key]["correct"] for key in paired)
        after = sum(second[key]["correct"] for key in paired)
        complete = len(paired) == len(claims)
        phase1_confidence = conditions[P0_PHASES[0]][provider]["candidates"]["mean_confidence"]
        phase2_confidence = conditions[P0_PHASES[1]][provider]["candidates"]["mean_confidence"]
        confidence_change = (
            phase2_confidence - phase1_confidence
            if phase1_confidence is not None and phase2_confidence is not None else None
        )
        phase_changes[provider] = {
            "paired_claims": len(paired),
            "phase1_accuracy": before / len(paired) if complete else None,
            "phase2_accuracy": after / len(paired) if complete else None,
            "absolute_change": (after - before) / len(paired) if complete else None,
            "relative_change": (after - before) / before if complete and before else None,
            "covered_phase1_accuracy": before / len(paired) if paired else None,
            "covered_phase2_accuracy": after / len(paired) if paired else None,
            "covered_absolute_change": (after - before) / len(paired) if paired else None,
            "phase1_mean_candidate_confidence": phase1_confidence,
            "phase2_mean_candidate_confidence": phase2_confidence,
            "mean_candidate_confidence_change": confidence_change if complete else None,
            "covered_mean_candidate_confidence_change": confidence_change,
            "wrong_to_correct": [
                key for key in paired if not first[key]["correct"] and second[key]["correct"]
            ],
            "correct_to_wrong": [
                key for key in paired if first[key]["correct"] and not second[key]["correct"]
            ],
        }

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "experiment_id": P0_EXPERIMENT_ID,
        },
        "metric_scope": "55 primary P0 claims; not exhaustive 67-video Top-1 accuracy",
        "conditions": conditions,
        "phase_changes": phase_changes,
    }
    reports = output_root / "reports"
    _write_json(reports / "summary.json", report)
    _write_jsonl(reports / "top1_claim_details.jsonl", detail_rows)
    lines = [
        "## Material Passport", "", "- Origin Skill: experiment-agent",
        "- Origin Mode: run", "- Verification Status: ANALYZED", "",
        "# 两阶段 P0 开放提名报告", "",
        "主指标仅为 55 条 primary P0 claims 上的 Top-1 claim-level accuracy；",
        "不得表述为 67 个视频的穷尽 Top-1 准确率。", "",
        "| phase | provider | valid clips | Top-1 claim accuracy | Top-3 claim accuracy | positive recall@1 | positive recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for phase in P0_PHASES:
        for provider in ("gemini", "doubao"):
            value = conditions[phase][provider]
            lines.append(
                f"| {phase} | {provider} | {value['execution']['valid_clips']} | "
                f"{value['top1_claim_metrics']['accuracy']} | "
                f"{value['top3_claim_metrics']['accuracy']} | "
                f"{value['top1_claim_metrics']['positive_recall']} | "
                f"{value['top3_claim_metrics']['positive_recall']} |"
            )
    _atomic_write(reports / "summary.md", "\n".join(lines) + "\n")
    return report
