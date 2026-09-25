"""Index, inspect, and analyse the local FIFA Game Library."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from pipeline.tactics_qa.opentac import compact, load_glossary, sha256
from pipeline.utils.video import get_video_info
from pipeline.video_models import generate_json, model_name

EXPECTED_VIDEOS = 626
EXPECTED_UNIQUE_VIDEOS = 575
EXPECTED_DUPLICATE_VIDEOS = 51
PROMPT_VERSION = "fifa-source-label-analysis-v2"
SCHEMA_VERSION = "fifa-game-library-analysis-v2"
DOUBAO_UPLOAD_PROFILE = "480p12-crf28-v1"
FOLDERS = (
    "In possession",
    "Position Specific",
    "Set Plays",
    "Transition to attacking",
    "Transition to defending",
    "out of possession",
)
INDEX_FIELDS = (
    "video_path",
    "source_folder",
    "filename",
    "duration_s",
    "sha256",
    "duplicate_of",
    "raw_label",
    "tactic_id",
    "label_status",
    "description_zh",
    "description_source",
    "canonical_video_path",
    "analysis_result_path",
    "source_label",
    "label_kind",
    "label_role",
    "mapping_status",
    "evidence_spans_json",
)
REVIEW_FIELDS = ("source_label", "proposed_tactic_id", "decision", "note")
APPROVED: dict[str, tuple[str, ...]] = {
    "In possession/01_Line_breaks_premium.mp4": ("line-breaking-pass",),
    "In possession/Bellingham (2022) — Use of wide areas — Cut-back cross.mp4": ("cutback",),
    "In possession/Salem(2023) — Combination play — Switching play.mp4": ("switch-of-play",),
    "Position Specific/Nuno Mendes (2025) - Full-back - Inverted position.mp4": ("inverted-full-back",),
    "Position Specific/Gakpo (2022) — Role of the number 9 — Running in behind.mp4": ("run-in-behind",),
    "Position Specific/Sofia Fuente (2022) — Goalkeeper — Build-up.mp4": ("gk-in-buildup",),
    "Set Plays/Al Hilal (2025) — Corner — zonal system.mp4": ("set-piece-zonal",),
    "Set Plays/Seattle Sounders (2025) — Corner — Hybrid system.mp4": ("hybrid-marking",),
    "Set Plays/Sweden (2023) – Corner – Front post.mp4": ("corner-near-far-post",),
    "Transition to attacking/Australia U-20 (2025) — Transition to attack — Breaking lines.mp4": ("line-breaking-pass",),
    "Transition to attacking/Brazil U-20 (2018) – Counter-attack – Run in behind.mp4": ("counter-attack", "run-in-behind"),
    "Transition to attacking/Japan (2018) – Medium block –Counter-attack.mp4": ("counter-attack", "mid-block"),
    "Transition to defending/Chelsea FC(2025) — Transition to defending — Counter pressing.mp4": ("gegenpress",),
    "Transition to defending/Germany (2023) – Recovering – Counter-press.mp4": ("gegenpress",),
    "Transition to defending/Spain (2023) – Counter-pressing – Winning the ball in the final third.mp4": ("gegenpress",),
    "out of possession/Borussia Dortmund (2025) — High-block — Pressing.mp4": ("high-press",),
    "out of possession/Japan U20 (2018) – Team organisation – Medium block.mp4": ("mid-block",),
    "out of possession/Uruguay (2023) — Keeping it compact — Defending in a low-block.mp4": ("low-block",),
}
APPROVED_SOURCE_LABELS: dict[str, tuple[str, ...]] = {
    "In possession/01_Line_breaks_premium.mp4": ("Line breaks",),
    "In possession/Bellingham (2022) — Use of wide areas — Cut-back cross.mp4": ("Cut-back cross",),
    "In possession/Salem(2023) — Combination play — Switching play.mp4": ("Switching play",),
    "Position Specific/Nuno Mendes (2025) - Full-back - Inverted position.mp4": ("Inverted position",),
    "Position Specific/Gakpo (2022) — Role of the number 9 — Running in behind.mp4": ("Running in behind",),
    "Position Specific/Sofia Fuente (2022) — Goalkeeper — Build-up.mp4": ("Build-up",),
    "Set Plays/Al Hilal (2025) — Corner — zonal system.mp4": ("zonal system",),
    "Set Plays/Seattle Sounders (2025) — Corner — Hybrid system.mp4": ("Hybrid system",),
    "Set Plays/Sweden (2023) – Corner – Front post.mp4": ("Front post",),
    "Transition to attacking/Australia U-20 (2025) — Transition to attack — Breaking lines.mp4": ("Breaking lines",),
    "Transition to attacking/Brazil U-20 (2018) – Counter-attack – Run in behind.mp4": ("Counter-attack", "Run in behind"),
    "Transition to attacking/Japan (2018) – Medium block –Counter-attack.mp4": ("Counter-attack", "Medium block"),
    "Transition to defending/Chelsea FC(2025) — Transition to defending — Counter pressing.mp4": ("Counter pressing",),
    "Transition to defending/Germany (2023) – Recovering – Counter-press.mp4": ("Counter-press",),
    "Transition to defending/Spain (2023) – Counter-pressing – Winning the ball in the final third.mp4": ("Counter-pressing",),
    "out of possession/Borussia Dortmund (2025) — High-block — Pressing.mp4": ("High-block",),
    "out of possession/Japan U20 (2018) – Team organisation – Medium block.mp4": ("Medium block",),
    "out of possession/Uruguay (2023) — Keeping it compact — Defending in a low-block.mp4": ("Defending in a low-block",),
}
LABEL_KINDS = ("tactic", "role", "technique", "event", "outcome", "context")
LABEL_ROLES = ("primary", "secondary", "context")
PATTERN_THRESHOLDS = {
    "player_ring": 0.42,
    "attack_defense_colors": 0.42,
    "topology_lines": 0.67,
    "striped_zone": 0.45,
    "arrow_path": 0.46,
    "vertical_beam": 0.44,
    "freeze_or_slow_motion": 0.96,
    "top_down_transition": 0.58,
    "text_label": 0.48,
}
STYLE_REFERENCES = {
    "player_ring": "In possession/01_Line_breaks_premium.mp4",
    "attack_defense_colors": "In possession/01_Line_breaks_premium.mp4",
    "arrow_path": "Position Specific/Ødegaard (2026) — Build up — Offering to receive.mp4",
    "vertical_beam": "Position Specific/Japan (2018) – Playmaker – Attacking open spaces.mp4",
    "topology_lines": "In possession/01_Line_breaks_premium.mp4",
    "striped_zone": "In possession/Argentina (2026) — Creating overloads — Switching play.mp4",
}

VERTICAL_BEAM_EVIDENCE = (
    (
        "Position Specific/Japan (2018) – Playmaker – Attacking open spaces.mp4",
        9.0,
        "01_japan_playmaker_09.00.jpg",
    ),
    ("Position Specific/Playmaker analysis 1.mp4", 5.0, "02_playmaker_analysis_1_05.00.jpg"),
    ("Position Specific/Playmaker analysis 2.mp4", 5.0, "03_playmaker_analysis_2_05.00.jpg"),
    ("Position Specific/Playmaker analysis 4.mp4", 5.0, "04_playmaker_analysis_4_05.00.jpg"),
    ("Position Specific/Taylor (2022) – Running with the ball – Breaking lines.mp4", 12.0, "05_taylor_breaking_lines_12.00.jpg"),
)
ARROW_PATH_EVIDENCE = (
    (
        "Position Specific/Ødegaard (2026) — Build up — Offering to receive.mp4",
        48.5,
        "01_odegaard_offering_to_receive_48.50.jpg",
    ),
    (
        "In possession/Algeria (2021) – Receiving between lines – Run in behind.mp4",
        19.0,
        "02_algeria_run_in_behind_19.00.jpg",
    ),
    (
        "In possession/Argentina (2026) — Creating overloads — Switching play.mp4",
        7.0,
        "03_argentina_switching_play_07.00.jpg",
    ),
)
STRIPED_ZONE_EVIDENCE = (
    (
        "In possession/1_Goal_Analysis_-_Goals_Outside_Penalty_Area_premium.mp4",
        11.0,
        "5954d4903b_11.00_1.000.jpg",
    ),
    (
        "In possession/Argentina (2026) — Creating overloads — Switching play.mp4",
        7.0,
        "02_argentina_overload_07.00.jpg",
    ),
    (
        "Position Specific/Taylor (2022) – Running with the ball – Breaking lines.mp4",
        9.0,
        "03_taylor_breaking_lines_09.00.jpg",
    ),
)
TOPOLOGY_LINE_EVIDENCE = (
    (
        "Position Specific/Ødegaard (2026) — Build up — Offering to receive.mp4",
        48.5,
        "01_odegaard_offering_to_receive_48.50.jpg",
    ),
    (
        "In possession/01_Line_breaks_premium.mp4",
        20.0,
        "02_line_breaks_premium_20.00.jpg",
    ),
    (
        "Position Specific/Taylor (2022) – Running with the ball – Breaking lines.mp4",
        9.0,
        "03_taylor_breaking_lines_09.00.jpg",
    ),
    (
        "In possession/Colombia (2022) – Breaking lines – Dribbling.mp4",
        24.0,
        "04_colombia_breaking_lines_24.00.jpg",
    ),
)


def _write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields))
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, handle.getvalue(), encoding="utf-8-sig")


def _videos(library_root: Path) -> list[Path]:
    missing = [name for name in FOLDERS if not (library_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing FIFA folders: {missing}")
    return sorted(
        (path for name in FOLDERS for path in (library_root / name).rglob("*.mp4")),
        key=lambda path: path.relative_to(library_root).as_posix(),
    )


def _raw_label(path: Path) -> str:
    value = re.sub(r"[_]+", " ", path.stem)
    return re.sub(r"\s+", " ", value).strip()


def _source_labels(path: Path) -> list[str]:
    value = re.sub(r"^\d+\s+", "", _raw_label(path))
    value = re.sub(r"\s+premium$", "", value, flags=re.IGNORECASE)
    labels = [
        part.strip()
        for part in re.split(r"\s*[—–]\s*|\s+-\s+", value)
        if part.strip()
    ]
    return list(dict.fromkeys(labels or [value]))


def _label_kind(label: str, *, concept_position: bool = False) -> str:
    lowered = unicodedata.normalize("NFC", label).casefold()
    if lowered in {
        "bright",
        "girelli",
        "horan",
        "inout of possession",
        "paredes",
        "popp",
        "rolfö",
    }:
        return "context"
    if re.search(
        r"rest defen[cs]e|recovering shape|between (?:the )?lines|"
        r"switch ?ofplay|verticalprogressiveplay|playermovementsandrotations|"
        r"effectiveuseoflateralareas|ballsbehindthedefence|"
        r"building from the back|third-player combinations|cut[ -]?back(?: cross)?|"
        r"short corner|defen[cs]e[- ]splitting pass|"
        r"wing play full[ -]?back|centre[ -]?back building|"
        r"attacking space|reaction after loss|verticality after regaining|"
        r"closing passing lines|forcing a long pass|"
        r"(?:run|running|play) in behind|attacking (?:wide|central) areas|"
        r"creating (?:width|depth)|use of (?:wide|lateral) areas|"
        r"balls? in behind|\bin[- ]behind\b|counter[ -]?attack\w*|countering|"
        r"\b(?:press\w*|(?:high|mid|medium|low|defensive|compact)[ -]?block|"
        r"build(?:ing)?[ -]?up|break\w*|switch\w*|"
        r"compact\w*|mark\w*|overload\w*|width|depth|system)\b",
        lowered,
    ):
        return "tactic"
    if re.search(
        r"\b(goalkeep\w*|centre[ -]?back|center[ -]?back|full[ -]?backs?|"
        r"wing[ -]?back|centre[ -]?forward|number 9|no\. 9|striker|winger|"
        r"midfielder|playmaker|role)\b",
        lowered,
    ):
        return "role"
    if re.search(
        r"\b(transition|restart|corner|free[ -]?kicks?|goal[ -]?kicks?|"
        r"penalty[ -]?kicks?|throw(?:s|[ -]?ins?)?|recover(?:y|ing)|"
        r"reaction after loss|regain\w*|possession loss)\b",
        lowered,
    ):
        return "event"
    if re.search(
        r"\b(goals?|scor\w*|shot|shoot\w*|sav(?:e\w*|ing)|chance|outcome|"
        r"winning|finish\w*|assist)\b",
        lowered,
    ):
        return "outcome"
    if re.search(
        r"\b(pass\w*|cross\w*|head\w*|dribbl\w*|receiv\w*|carry\w*|"
        r"tak\w+ on|tackl\w*|scann\w*|technique|first touch|through ball|"
        r"flick-on|long kick|kick from hands|running with (?:and without )?the ball|"
        r"positioning|step-in|decision-making|anticipation|distribution|1v1|"
        r"quick reaction|intercept\w*|closing the angle|technical ability|"
        r"in-swing|outswing|explosiveness|blocking|x-block|defending the area)\b",
        lowered,
    ):
        return "technique"
    if re.fullmatch(
        r"(?:final third|left channel|outside the box|front zone|back post|"
        r"near post|far post|deep position|high position)",
        lowered,
    ):
        return "context"
    return "tactic" if concept_position else "context"


def _display_path(relative: str) -> str:
    return f"FIFA Game Library/{relative}"


def _analysis_result_path(relative: str) -> str:
    return f"analysis/{Path(relative).with_suffix('.json').as_posix()}"


def prepare_index(
    library_root: Path,
    output_csv: Path,
    *,
    expected_videos: int | None = EXPECTED_VIDEOS,
) -> dict[str, int]:
    """Create one row per physical video/title concept."""
    videos = _videos(library_root)
    if expected_videos is not None and len(videos) != expected_videos:
        raise ValueError(f"expected {expected_videos} FIFA videos, got {len(videos)}")
    relative = {path: path.relative_to(library_root).as_posix() for path in videos}
    if missing := set(APPROVED) - set(relative.values()):
        raise FileNotFoundError(f"approved FIFA videos missing: {sorted(missing)}")

    metadata = {}
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in videos:
        digest = sha256(path)
        rel = relative[path]
        info = get_video_info(path)
        metadata[path] = (digest, info["duration_s"])
        paths_by_hash[digest].append(rel)
    canonical_by_hash = {
        digest: min(paths, key=lambda rel: (rel not in APPROVED, rel))
        for digest, paths in paths_by_hash.items()
    }
    duplicate_videos = sum(
        canonical_by_hash[digest] != relative[path]
        for path, (digest, _) in metadata.items()
    )
    if expected_videos == EXPECTED_VIDEOS:
        if len(paths_by_hash) != EXPECTED_UNIQUE_VIDEOS:
            raise ValueError(
                f"expected {EXPECTED_UNIQUE_VIDEOS} unique FIFA videos, "
                f"got {len(paths_by_hash)}"
            )
        if duplicate_videos != EXPECTED_DUPLICATE_VIDEOS:
            raise ValueError(
                f"expected {EXPECTED_DUPLICATE_VIDEOS} duplicate FIFA paths, "
                f"got {duplicate_videos}"
            )

    previous = {}
    if output_csv.is_file():
        with output_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_label"):
                    previous[(row.get("video_path", ""), row["source_label"])] = row
    rows = []
    for path in videos:
        rel = relative[path]
        digest, duration_s = metadata[path]
        labels = _source_labels(path)
        exact = dict(zip(APPROVED_SOURCE_LABELS.get(rel, ()), APPROVED.get(rel, ())))
        if rel in APPROVED and set(exact.values()) != set(APPROVED[rel]):
            raise ValueError(f"approved source-label mapping is incomplete: {rel}")
        if missing_labels := set(exact) - set(labels):
            raise ValueError(
                f"approved source labels not found in title {rel}: {sorted(missing_labels)}"
            )
        exact_role = {
            tactic_id: "primary" if index == 0 else "secondary"
            for index, tactic_id in enumerate(APPROVED.get(rel, ()))
        }
        kinds = [
            "tactic" if label in exact else _label_kind(label, concept_position=index > 0)
            for index, label in enumerate(labels)
        ]
        first_non_context = next(
            (index for index, kind in enumerate(kinds) if kind != "context"), None,
        )
        canonical = canonical_by_hash[digest]
        for index, (label, kind) in enumerate(zip(labels, kinds)):
            tactic_id = exact.get(label, "")
            if tactic_id:
                role = exact_role[tactic_id]
                mapping_status = "direct_exact"
                label_status = "source_gt_reviewed"
            else:
                role = (
                    "context"
                    if kind == "context"
                    else "primary" if index == first_non_context else "secondary"
                )
                mapping_status = "candidate_new" if kind == "tactic" else "not_applicable"
                label_status = "source_gt_unmapped" if kind == "tactic" else "source_gt"
            old = previous.get((_display_path(rel), label), {})
            if old.get("sha256") == digest and not tactic_id and old.get("tactic_id"):
                tactic_id = old["tactic_id"]
                mapping_status = old.get("mapping_status", "model_mapped_unreviewed")
            rows.append({
                "video_path": _display_path(rel),
                "source_folder": rel.split("/", 1)[0],
                "filename": path.name,
                "duration_s": f"{duration_s:.3f}",
                "sha256": digest,
                "duplicate_of": _display_path(canonical) if canonical != rel else "",
                "raw_label": _raw_label(path),
                "tactic_id": tactic_id,
                "label_status": label_status,
                "description_zh": old.get("description_zh", ""),
                "description_source": old.get("description_source", ""),
                "canonical_video_path": _display_path(canonical),
                "analysis_result_path": _analysis_result_path(canonical),
                "source_label": label,
                "label_kind": kind,
                "label_role": role,
                "mapping_status": mapping_status,
                "evidence_spans_json": old.get("evidence_spans_json", ""),
            })
    _write_csv(output_csv, rows, INDEX_FIELDS)
    return {
        "videos": len(videos),
        "rows": len(rows),
        "unique_hashes": len(paths_by_hash),
        "approved_videos": len(APPROVED),
        "approved_mappings": sum(map(len, APPROVED.values())),
        "duplicate_videos": duplicate_videos,
    }


def _span_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "start_s": {"type": "number"},
            "end_s": {"type": "number"},
            "observation_zh": {"type": "string"},
        },
        "required": ["start_s", "end_s", "observation_zh"],
        "additionalProperties": False,
    }


def analysis_schema(source_labels: list[str], tactic_ids: list[str]) -> dict[str, Any]:
    normalized_ids = ["", *tactic_ids]
    source_item = {
        "type": "object",
        "properties": {
            "source_label": {"type": "string", "enum": source_labels},
            "label_kind": {"type": "string", "enum": list(LABEL_KINDS)},
            "label_role": {"type": "string", "enum": list(LABEL_ROLES)},
            "normalized_tactic_id": {"type": "string", "enum": normalized_ids},
            "description_zh": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "evidence_gaps": {"type": "array", "items": {"type": "string"}},
            "evidence_spans": {
                "type": "array", "minItems": 0, "maxItems": 4,
                "items": _span_schema(),
            },
        },
        "required": [
            "source_label", "label_kind", "label_role", "normalized_tactic_id",
            "description_zh", "confidence", "evidence_spans", "evidence_gaps",
        ],
        "additionalProperties": False,
    }
    event = {
        "type": "object",
        "properties": {
            "event_id": {"type": "integer", "minimum": 1},
            "start_s": {"type": "number"},
            "end_s": {"type": "number"},
            "phase": {
                "type": "string",
                "enum": [
                    "in_possession", "out_of_possession", "transition_attacking",
                    "transition_defending", "set_piece", "unknown",
                ],
            },
            "action_zh": {"type": "string"},
            "space_change_zh": {"type": "string"},
            "result_zh": {"type": "string"},
        },
        "required": [
            "event_id", "start_s", "end_s", "phase", "action_zh",
            "space_change_zh", "result_zh",
        ],
        "additionalProperties": False,
    }
    suggestion = {
        "type": "object",
        "properties": {
            "suggested_label_zh": {"type": "string"},
            "normalized_tactic_id": {"type": "string", "enum": normalized_ids},
            "evidence_event_ids": {"type": "array", "items": {"type": "integer"}},
            "rationale_zh": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "suggested_label_zh", "normalized_tactic_id", "evidence_event_ids",
            "rationale_zh", "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "video_summary_zh": {"type": "string"},
            "source_labels": {
                "type": "array", "minItems": len(source_labels),
                "maxItems": len(source_labels), "items": source_item,
            },
            "action_events": {"type": "array", "maxItems": 30, "items": event},
            "model_suggestions": {"type": "array", "maxItems": 8, "items": suggestion},
        },
        "required": ["video_summary_zh", "source_labels", "action_events", "model_suggestions"],
        "additionalProperties": False,
    }


def build_prompt(
    clip_alias: str,
    duration_s: float,
    source_inputs: list[dict[str, str]],
    source_titles: list[str],
    cards: list[dict[str, Any]],
) -> str:
    supplied = [
        {
            "source_label": item["source_label"],
            "label_kind_hint": item["label_kind"],
            "label_role_hint": item["label_role"],
            "source_ground_truth_tactic_id": item.get("source_tactic_id", ""),
        }
        for item in source_inputs
    ]
    return f"""你是足球战术视频证据分析员。完整视频已作为输入，上传文件名为中性哈希名。

source_labels 是 FIFA 来源标题中显式出现的原始概念，属于来源 GT，但不保证每项都是战术：也可能是位置角色、技术动作、事件、结果或背景。非空 source_ground_truth_tactic_id 是人工确认映射，不得改写。

要求：
1. source_labels 必须逐项原样返回且只返回这些项；保留给定的 label_kind_hint 和 label_role_hint。只有 context 可在画面与标题都明确时重分类；不得把 role、technique、event 或 outcome 升级成战术。映射到给定词条 ID 或空字符串。
2. 每项用中文描述画面中实际可见的动作、对手约束、空间变化和结果，给出 0–100 的 confidence；证据不足可留空 evidence_spans，并写入 evidence_gaps。
   多个战术标签共存时，尽量为每项分别给出证据时间段；单战术视频不强制时间段。
3. 另按时间顺序提取 action_events。时间只能在 0–{duration_s:.3f} 秒。
4. 标题之外观察到的战术只放 model_suggestions；可映射现有 ID，无法映射时 ID 留空并给出中文候选名。
5. 不得编造身份、号码、阵型、意图或画外结果。只返回严格 JSON。

中性视频别名：{clip_alias}
完整官方标题：
{json.dumps(source_titles, ensure_ascii=False)}
来源标签与人工映射：
{json.dumps(supplied, ensure_ascii=False)}
规范词条卡：
{json.dumps([compact(card) for card in cards], ensure_ascii=False)}
"""


def _validate_span(span: Any, duration_s: float) -> None:
    if not isinstance(span, dict) or set(span) != {"start_s", "end_s", "observation_zh"}:
        raise ValueError("evidence span fields are not strict")
    start, end = span["start_s"], span["end_s"]
    if (
        not isinstance(span["observation_zh"], str)
        or isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
        or not math.isfinite(start) or not math.isfinite(end)
        or start < 0 or end < start or end > duration_s + 0.05
    ):
        raise ValueError("evidence span is outside the video")


def validate_analysis(
    payload: Any,
    source_inputs: list[dict[str, str]],
    tactic_ids: list[str],
    duration_s: float,
) -> dict[str, Any]:
    top_fields = {"video_summary_zh", "source_labels", "action_events", "model_suggestions"}
    if not isinstance(payload, dict) or set(payload) != top_fields:
        raise ValueError("analysis response fields are not strict")
    if not isinstance(payload["video_summary_zh"], str):
        raise ValueError("video_summary_zh must be a string")
    expected = {
        unicodedata.normalize("NFC", item["source_label"]): item
        for item in source_inputs
    }
    source_results = payload["source_labels"]
    if not isinstance(source_results, list) or len(source_results) != len(expected):
        raise ValueError("source-label count does not match input")
    source_fields = {
        "source_label", "label_kind", "label_role", "normalized_tactic_id",
        "description_zh", "confidence", "evidence_spans", "evidence_gaps",
    }
    returned = set()
    valid_tactics = {"", *tactic_ids}
    for item in source_results:
        if not isinstance(item, dict) or set(item) != source_fields:
            raise ValueError("source-label fields are not strict")
        label = unicodedata.normalize("NFC", item["source_label"])
        if label not in expected or label in returned:
            raise ValueError("source labels do not match input")
        returned.add(label)
        item["source_label"] = expected[label]["source_label"]
        source = expected[label]
        item["label_kind"] = source["label_kind"]
        item["label_role"] = source["label_role"]
        if source["label_kind"] != "tactic":
            item["normalized_tactic_id"] = ""
        if (
            item["label_kind"] not in LABEL_KINDS
            or item["label_role"] not in LABEL_ROLES
            or item["normalized_tactic_id"] not in valid_tactics
            or not isinstance(item["description_zh"], str)
            or isinstance(item["confidence"], bool)
            or not isinstance(item["confidence"], int)
            or not 0 <= item["confidence"] <= 100
            or not isinstance(item["evidence_gaps"], list)
            or not all(isinstance(gap, str) for gap in item["evidence_gaps"])
        ):
            raise ValueError("source-label classification is invalid")
        if item["label_kind"] != "tactic" and item["normalized_tactic_id"]:
            raise ValueError("non-tactic source label cannot map to a tactic")
        if source["label_kind"] != "context" and (
            item["label_kind"] != source["label_kind"]
            or item["label_role"] != source["label_role"]
        ):
            raise ValueError("source GT label kind or role was changed")
        locked = source.get("source_tactic_id", "")
        if locked and item["normalized_tactic_id"] != locked:
            raise ValueError("source GT tactic mapping was changed")
        spans = item["evidence_spans"]
        if not isinstance(spans, list) or len(spans) > 4:
            raise ValueError("source-label evidence must contain at most four spans")
        for span in spans:
            _validate_span(span, duration_s)
    if returned != set(expected):
        raise ValueError("source labels do not match input")

    event_fields = {
        "event_id", "start_s", "end_s", "phase", "action_zh",
        "space_change_zh", "result_zh",
    }
    valid_phases = {
        "in_possession", "out_of_possession", "transition_attacking",
        "transition_defending", "set_piece", "unknown",
    }
    event_ids = set()
    if not isinstance(payload["action_events"], list):
        raise ValueError("action_events must be an array")
    for event in payload["action_events"]:
        if not isinstance(event, dict) or set(event) != event_fields:
            raise ValueError("action-event fields are not strict")
        event_id = event["event_id"]
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 1:
            raise ValueError("action-event ID is invalid")
        if event_id in event_ids:
            raise ValueError("action-event IDs must be unique")
        event_ids.add(event_id)
        _validate_span({
            "start_s": event["start_s"], "end_s": event["end_s"],
            "observation_zh": event["action_zh"],
        }, duration_s)
        if (
            event["phase"] not in valid_phases
            or not isinstance(event["space_change_zh"], str)
            or not isinstance(event["result_zh"], str)
        ):
            raise ValueError("action-event content is invalid")

    suggestion_fields = {
        "suggested_label_zh", "normalized_tactic_id", "evidence_event_ids",
        "rationale_zh", "confidence",
    }
    if not isinstance(payload["model_suggestions"], list):
        raise ValueError("model_suggestions must be an array")
    for item in payload["model_suggestions"]:
        confidence = item.get("confidence") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict) or set(item) != suggestion_fields
            or not isinstance(item["suggested_label_zh"], str)
            or item["normalized_tactic_id"] not in valid_tactics
            or not isinstance(item["rationale_zh"], str)
            or isinstance(confidence, bool) or not isinstance(confidence, int)
            or not 0 <= confidence <= 100
            or not isinstance(item["evidence_event_ids"], list)
            or not all(event_id in event_ids for event_id in item["evidence_event_ids"])
        ):
            raise ValueError("model suggestion is invalid")
    return payload
def _index_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(INDEX_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"FIFA index missing fields: {sorted(missing)}")
        return list(reader)


def _grouped_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_video_path"]].append(row)
    for canonical, items in grouped.items():
        if not canonical.startswith("FIFA Game Library/"):
            raise ValueError(f"invalid canonical video path: {canonical}")
        rel = canonical[len("FIFA Game Library/"):]
        if len({item["sha256"] for item in items}) != 1:
            raise ValueError(f"inconsistent hashes for {canonical}")
        if {item["analysis_result_path"] for item in items} != {_analysis_result_path(rel)}:
            raise ValueError(f"inconsistent analysis result path for {canonical}")
    return grouped


def _source_inputs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label: dict[str, dict[str, str]] = {}
    role_rank = {"context": 0, "secondary": 1, "primary": 2}
    for row in rows:
        label = row["source_label"]
        tactic_id = row["tactic_id"] if row["mapping_status"] in {"direct_exact", "reviewed_exact"} else ""
        current = by_label.get(label)
        if current is None:
            by_label[label] = {
                "source_label": label,
                "source_tactic_id": tactic_id,
                "label_kind": row["label_kind"],
                "label_role": row["label_role"],
            }
            continue
        if tactic_id and current["source_tactic_id"] not in {"", tactic_id}:
            raise ValueError(f"conflicting source GT mappings for {label}")
        current["source_tactic_id"] = tactic_id or current["source_tactic_id"]
        if role_rank[row["label_role"]] > role_rank[current["label_role"]]:
            current["label_role"] = row["label_role"]
    return list(by_label.values())


def _source_ground_truth(source_inputs: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "source_label": item["source_label"],
            "label_kind": "tactic" if item["source_tactic_id"] else item["label_kind"],
            "label_role": item["label_role"],
            "tactic_id": item["source_tactic_id"],
        }
        for item in source_inputs
    ]


def _legacy_result(
    path: Path,
    source_inputs: list[dict[str, str]],
    digest: str,
    video_path: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    legacy = json.loads(path.read_text(encoding="utf-8"))
    if legacy.get("clip_sha256") != digest:
        raise ValueError(f"legacy result hash mismatch: {path}")
    if legacy.get("status") != "success":
        return None
    by_tactic = {
        item.get("tactic_id"): item
        for item in legacy.get("analyses", []) if isinstance(item, dict)
    }
    source_results, events = [], []
    for item in source_inputs:
        tactic_id = item["source_tactic_id"]
        analysis = by_tactic.get(tactic_id, {})
        spans = analysis.get("evidence_spans", []) if tactic_id else []
        source_results.append({
            "source_label": item["source_label"],
            "label_kind": "tactic" if tactic_id else item["label_kind"],
            "label_role": item["label_role"],
            "normalized_tactic_id": tactic_id,
            "description_zh": analysis.get("positive_description_zh", ""),
            "confidence": int(analysis.get("confidence", 0)),
            "evidence_gaps": analysis.get("evidence_gaps", []),
            "evidence_spans": spans,
            "mapping_status": (
                "direct_exact" if tactic_id
                else "candidate_new" if item["label_kind"] == "tactic"
                else "not_applicable"
            ),
        })
        for span in spans:
            events.append({
                "event_id": len(events) + 1,
                "start_s": span["start_s"],
                "end_s": span["end_s"],
                "phase": "unknown",
                "action_zh": span["observation_zh"],
                "space_change_zh": analysis.get("space_change_zh", ""),
                "result_zh": analysis.get("result_zh", ""),
            })
    return {
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "prompt_version": legacy.get("prompt_version", ""),
        "prompt_sha256": legacy.get("prompt_sha256", ""),
        "provider": "gemini",
        "model": legacy.get("model", model_name("gemini")),
        "video_path": video_path,
        "clip_sha256": digest,
        "source_ground_truth": _source_ground_truth(source_inputs),
        "video_summary_zh": legacy.get("video_summary_zh", ""),
        "source_labels": source_results,
        "action_events": events,
        "model_suggestions": [],
        "temperature": legacy.get("temperature", 0.0),
        "attempts": legacy.get("attempts"),
        "api_usage": legacy.get("api_usage", []),
        "provider_failures": [],
        "reused_from": path.as_posix(),
    }

def migrate_results(
    index_csv: Path,
    output_root: Path,
    *,
    force: bool = False,
    clip_shas: Iterable[str] | None = None,
) -> dict[str, int]:
    grouped = _grouped_rows(_index_rows(index_csv))
    requested = set(clip_shas or ())
    known = {items[0]["sha256"] for items in grouped.values()}
    if unknown := requested - known:
        raise ValueError(f"unknown clip SHA256: {sorted(unknown)}")
    if requested:
        grouped = {
            canonical: items for canonical, items in grouped.items()
            if items[0]["sha256"] in requested
        }
    counts = Counter(
        clips=len(grouped), migrated_success=0, created_pending=0,
        existing=0, recovered=0,
    )
    for canonical, rows in grouped.items():
        rel = canonical[len("FIFA Game Library/"):]
        digest = rows[0]["sha256"]
        source_inputs = _source_inputs(rows)
        expected_labels = {item["source_label"] for item in source_inputs}
        expected_ground_truth = _source_ground_truth(source_inputs)
        source_titles = list(dict.fromkeys(Path(row["filename"]).stem for row in rows))
        target = output_root / _analysis_result_path(rel)
        current = None
        if target.is_file():
            try:
                current = json.loads(target.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if not force:
                    raise ValueError(f"invalid analysis result; use force to recover: {target}") from exc
            if current is not None:
                current_labels = {
                    item.get("source_label")
                    for item in current.get("source_labels", []) if isinstance(item, dict)
                }
                stale = (
                    current.get("clip_sha256") != digest
                    or current.get("schema_version") != SCHEMA_VERSION
                    or current_labels != expected_labels
                )
                metadata_stale = (
                    current.get("video_path") != canonical
                    or current.get("source_titles") != source_titles
                    or current.get("source_ground_truth") != expected_ground_truth
                )
                if not stale and not metadata_stale:
                    counts["existing"] += 1
                    continue
                if not stale and current.get("status") == "success":
                    current.update({
                        "video_path": canonical,
                        "source_titles": source_titles,
                        "source_ground_truth": expected_ground_truth,
                    })
                    _write_json(target, current)
                    counts["existing"] += 1
                    continue
                if current.get("status") != "pending" and not force:
                    raise ValueError(f"stale analysis result; use force to recover: {target}")
            counts["recovered"] += 1
        legacy = _legacy_result(
            output_root / "gemini" / f"{Path(rel).stem}.json",
            source_inputs,
            digest,
            canonical,
        ) if rel in APPROVED else None
        if legacy is not None:
            legacy["source_titles"] = source_titles
            _write_json(target, legacy)
            counts["migrated_success"] += 1
            continue
        _write_json(target, {
            "status": "pending",
            "schema_version": SCHEMA_VERSION,
            "video_path": canonical,
            "clip_sha256": digest,
            "source_titles": source_titles,
            "source_ground_truth": _source_ground_truth(source_inputs),
            "source_labels": [
                {"source_label": item["source_label"]} for item in source_inputs
            ],
            "model_suggestions": [],
        })
        counts["created_pending"] += 1
    return dict(counts)


@contextmanager
def _hard_timeout(seconds: float):
    def expired(_signum, _frame):
        raise TimeoutError(f"provider call exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _doubao_proxy(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vf", "scale=-2:480,fps=12", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-maxrate", "800k", "-bufsize", "1600k",
            "-movflags", "+faststart", str(target),
        ],
        check=True,
        timeout=300,
    )


def _preflight_gemini_keys() -> None:
    names = ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2")
    values = [os.environ.get(name, "").strip() for name in names]
    if not all(values) or len(set(values)) != 3:
        raise RuntimeError("run requires three distinct Gemini API keys")
    if not os.environ.get("ARK_API_KEY", "").strip():
        raise RuntimeError("run requires ARK_API_KEY for Doubao fallback")


def _active_cards(glossary_path: Path) -> dict[str, dict[str, Any]]:
    cards = load_glossary(glossary_path)
    with glossary_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        allowed = {
            (row.get("id（英文标识）") or "").strip()
            for row in rows
            if (row.get("优先级") or "").strip() in {"P0", "P1", "P2"}
            and (row.get("状态") or "").strip()
            not in {"candidate_unreviewed", "unverified"}
        }
    return {
        tactic_id: card for tactic_id, card in cards.items() if tactic_id in allowed
    }
def _enrich_source_results(
    payload: dict[str, Any],
    source_inputs: list[dict[str, str]],
) -> None:
    inputs = {item["source_label"]: item for item in source_inputs}
    for result in payload["source_labels"]:
        source = inputs[result["source_label"]]
        if source["source_tactic_id"]:
            result["mapping_status"] = "direct_exact"
        elif result["normalized_tactic_id"]:
            result["mapping_status"] = "model_mapped_unreviewed"
        elif result["label_kind"] == "tactic":
            result["mapping_status"] = "candidate_new"
        else:
            result["mapping_status"] = "not_applicable"


def run_library(
    library_root: Path,
    index_csv: Path,
    glossary_path: Path,
    output_root: Path,
    *,
    force: bool = False,
    retry_failed: bool = False,
    clip_shas: Iterable[str] | None = None,
) -> dict[str, int]:
    rows = _index_rows(index_csv)
    grouped = _grouped_rows(rows)
    requested = set(clip_shas or ())
    known = {items[0]["sha256"] for items in grouped.values()}
    if unknown := requested - known:
        raise ValueError(f"unknown clip SHA256: {sorted(unknown)}")
    if requested:
        grouped = {
            canonical: items for canonical, items in grouped.items()
            if items[0]["sha256"] in requested
        }
    migrate_results(
        index_csv,
        output_root,
        force=force,
        clip_shas=requested,
    )
    cards_by_id = _active_cards(glossary_path)
    tactic_ids = sorted(cards_by_id)
    cards = [cards_by_id[tactic_id] for tactic_id in tactic_ids]
    counts = Counter(
        clips=len(grouped), api_calls=0, gemini_calls=0, doubao_calls=0,
        success=0, failed=0, skipped=0,
    )
    preflight_done = False
    for canonical, source_rows in grouped.items():
        rel = canonical[len("FIFA Game Library/"):]
        video = library_root / rel
        if not video.is_file():
            raise FileNotFoundError(video)
        digest = source_rows[0]["sha256"]
        if sha256(video) != digest:
            raise ValueError(f"video hash mismatch: {canonical}")
        target = output_root / _analysis_result_path(rel)
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("clip_sha256") != digest:
            raise ValueError(f"analysis result hash mismatch: {target}")
        source_inputs = _source_inputs(source_rows)
        current_labels = {
            item.get("source_label")
            for item in current.get("source_labels", []) if isinstance(item, dict)
        }
        expected_labels = {item["source_label"] for item in source_inputs}
        if current_labels != expected_labels:
            raise ValueError(f"analysis source labels changed; rerun migration: {target}")
        status = current.get("status")
        if not force and (
            status == "success" or status == "failed" and not retry_failed
        ):
            counts["skipped"] += 1
            continue
        if not preflight_done:
            _preflight_gemini_keys()
            preflight_done = True
        locked = {
            item["source_tactic_id"] for item in source_inputs if item["source_tactic_id"]
        }
        if missing := locked - set(cards_by_id):
            raise ValueError(f"source GT tactics missing from glossary: {sorted(missing)}")
        duration_s = float(source_rows[0]["duration_s"])
        source_titles = list(dict.fromkeys(Path(row["filename"]).stem for row in source_rows))
        prompt = build_prompt(
            f"clip-{digest[:12]}.mp4", duration_s,
            source_inputs, source_titles, cards,
        )
        failures, result = [], None
        for provider in ("gemini", "doubao"):
            provider_error = None
            for validation_retry in range(2):
                counts["api_calls"] += 1
                counts[f"{provider}_calls"] += 1
                try:
                    with tempfile.TemporaryDirectory(prefix="fifa-upload-") as directory:
                        alias = Path(directory) / f"clip-{digest[:16]}.mp4"
                        if provider == "doubao":
                            _doubao_proxy(video, alias)
                        else:
                            os.symlink(video.resolve(), alias)
                        kwargs = {"video_path": alias}
                        if provider == "doubao":
                            kwargs["retries"] = 0
                        with _hard_timeout(300 if provider == "gemini" else 240):
                            payload, usage, attempts = generate_json(
                                provider,
                                prompt,
                                analysis_schema(
                                    [item["source_label"] for item in source_inputs], tactic_ids,
                                ),
                                **kwargs,
                            )
                    validate_analysis(payload, source_inputs, tactic_ids, duration_s)
                    _enrich_source_results(payload, source_inputs)
                    result = {
                        "status": "success", **payload,
                        "provider": provider, "model": model_name(provider),
                        "attempts": attempts, "api_usage": usage,
                        "validation_retries": validation_retry,
                        "upload_profile": (
                            "original" if provider == "gemini"
                            else DOUBAO_UPLOAD_PROFILE
                        ),
                    }
                    counts["success"] += 1
                    break
                except Exception as exc:
                    provider_error = exc
                    if validation_retry == 0 and isinstance(exc, ValueError):
                        continue
                    break
            if result is not None:
                break
            assert provider_error is not None
            failures.append({
                "provider": provider,
                "model": model_name(provider),
                "error_type": type(provider_error).__name__,
                "error": str(provider_error)[:1000],
            })
        if result is None:
            result = {
                "status": "failed",
                "failed_stage": "provider_request_or_validation",
                "provider": "doubao",
                "model": model_name("doubao"),
                "error_type": failures[-1]["error_type"],
                "error": failures[-1]["error"],
                "source_labels": [
                    {"source_label": item["source_label"]} for item in source_inputs
                ],
                "model_suggestions": [],
            }
            counts["failed"] += 1
        result.update({
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "video_path": canonical,
            "clip_sha256": digest,
            "source_titles": source_titles,
            "source_ground_truth": _source_ground_truth(source_inputs),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "temperature": 0.0,
            "provider_failures": failures,
        })
        _write_json(target, result)
    return dict(counts)


def _result_description(result: dict[str, Any], source_label: str) -> tuple[str, str]:
    if result.get("status") != "success":
        return "", ""
    for item in result.get("source_labels", []):
        text = str(item.get("description_zh", "")).strip()
        if item.get("source_label") == source_label and 30 <= len(text) <= 120:
            return text, str(result.get("provider", ""))
    return "", ""


def _approved_reviews(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(REVIEW_FIELDS) - set(reader.fieldnames or ()):
            raise ValueError("mapping review has invalid columns")
        rows = list(reader)
    if invalid := {row["decision"] for row in rows} - {"approve", "reject"}:
        raise ValueError(f"invalid mapping review decisions: {sorted(invalid)}")
    return {
        (row["source_label"].strip(), row["proposed_tactic_id"].strip())
        for row in rows if row["decision"] == "approve"
    }


def _extend_grid_header(header: list[str]) -> None:
    for index in (5, 6):
        for suffix in ("比赛", "描述", "来源"):
            name = f"正例{chr(0x245F + index)} {suffix}"
            if name not in header:
                header.append(name)


def _example_identity(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    team = re.split(r"\s*[—–]\s*|\s+-\s+|_", stem, maxsplit=1)[0]
    team = re.sub(r"\s*[（(](?:19|20)\d{2}[）)]\s*$", "", team).strip().casefold()
    edition = re.search(r"\b(?:19|20)\d{2}\b", stem)
    return team, edition.group(0) if edition else ""


def apply_grid(
    index_csv: Path,
    glossary_path: Path,
    output_root: Path,
    review_path: Path | None = None,
) -> dict[str, int]:
    rows = _index_rows(index_csv)
    approved_reviews = _approved_reviews(review_path)
    results = {}
    rank_meta: dict[tuple[str, str], tuple[bool, int, int, int]] = {}
    filled = model_mapped = reviewed_mapped = 0
    for row in rows:
        result_path = output_root / row["analysis_result_path"]
        if result_path not in results:
            results[result_path] = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file() else None
            )
        result = results[result_path]
        if result is None:
            continue
        if result.get("clip_sha256") != row["sha256"]:
            raise ValueError(f"analysis result hash mismatch: {result_path}")
        source_result = next((
            item for item in result.get("source_labels", [])
            if item.get("source_label") == row["source_label"]
        ), None)
        if source_result is None or result.get("status") != "success":
            continue
        gaps = source_result.get("evidence_gaps", [])
        spans = source_result.get("evidence_spans", [])
        rank_meta[(row["video_path"], row["source_label"])] = (
            not bool(spans),
            len(gaps),
            -len(spans),
            -int(source_result.get("confidence", 0)),
        )
        row["evidence_spans_json"] = json.dumps(
            source_result.get("evidence_spans", []),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        reviewed = (
            source_result.get("mapping_status") == "model_mapped_unreviewed"
            and (
                row["source_label"],
                source_result.get("normalized_tactic_id", ""),
            ) in approved_reviews
        )
        if row["mapping_status"] not in {"direct_exact", "reviewed_exact"}:
            row["tactic_id"] = source_result.get("normalized_tactic_id", "")
            row["mapping_status"] = (
                "reviewed_exact" if reviewed else source_result["mapping_status"]
            )
            if reviewed:
                row["label_status"] = "source_gt_reviewed"
                reviewed_mapped += 1
            else:
                model_mapped += row["mapping_status"] == "model_mapped_unreviewed"
        text, provider = _result_description(result, row["source_label"])
        if text:
            row["description_zh"], row["description_source"] = text, provider
            filled += 1
    _write_csv(index_csv, rows, INDEX_FIELDS)

    with glossary_path.open(encoding="utf-8-sig", newline="") as handle:
        grid = list(csv.reader(handle))
    header = grid[0]
    _extend_grid_header(header)
    for row in grid[1:]:
        row.extend([""] * (len(header) - len(row)))
    by_tactic = {row[1]: row for row in grid[1:] if len(row) > 1 and row[1]}
    slots = [
        tuple(header.index(f"正例{marker} {suffix}") for suffix in ("比赛", "描述", "来源"))
        for marker in ("①", "②", "③", "④", "⑤", "⑥")
    ]
    inserted = grid_full_skipped = duplicate_sha_skipped = 0
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in rows:
        if (
            item["mapping_status"] not in {"direct_exact", "reviewed_exact"}
            or not item["description_zh"]
        ):
            continue
        example_key = (item["tactic_id"], item["sha256"])
        current = unique.get(example_key)
        item_rank = rank_meta.get(
            (item["video_path"], item["source_label"]),
            (True, 99, 0, 0),
        )
        if current is None:
            unique[example_key] = item
        else:
            duplicate_sha_skipped += 1
            current_rank = rank_meta.get(
                (current["video_path"], current["source_label"]),
                (True, 99, 0, 0),
            )
            if item_rank < current_rank:
                unique[example_key] = item

    by_candidate_tactic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in unique.values():
        by_candidate_tactic[item["tactic_id"]].append(item)
    for tactic_id, candidates in by_candidate_tactic.items():
        grid_row = by_tactic.get(tactic_id)
        if grid_row is None:
            raise ValueError(f"tactic missing from Grid: {tactic_id}")
        existing_games = {
            grid_row[slot[0]].strip() for slot in slots if grid_row[slot[0]].strip()
        }
        candidates = [
            item for item in candidates
            if Path(item["filename"]).stem not in existing_games
        ]
        open_slots = [
            slot for slot in slots if not any(grid_row[index].strip() for index in slot)
        ]
        existing_identities = {_example_identity(game) for game in existing_games}
        used_teams = {team for team, _ in existing_identities if team}
        used_editions = {edition for _, edition in existing_identities if edition}
        while candidates and open_slots:
            item = min(
                candidates,
                key=lambda candidate: (
                    *rank_meta.get(
                        (candidate["video_path"], candidate["source_label"]),
                        (True, 99, 0, 0),
                    ),
                    _example_identity(candidate["filename"])[0] in used_teams,
                    not _example_identity(candidate["filename"])[1]
                    or _example_identity(candidate["filename"])[1] in used_editions,
                    float(candidate["duration_s"]),
                    candidate["video_path"],
                ),
            )
            candidates.remove(item)
            game_i, description_i, source_i = open_slots.pop(0)
            game = Path(item["filename"]).stem
            grid_row[game_i] = game
            grid_row[description_i] = item["description_zh"]
            grid_row[source_i] = (
                f"FIFA Training Centre Game Library｜{item['video_path']}"
            )
            spans = json.loads(item["evidence_spans_json"] or "[]")
            if spans:
                times = "; ".join(
                    f"{span['start_s']:.1f}–{span['end_s']:.1f}s" for span in spans
                )
                grid_row[source_i] += f"｜证据 {times}"
            team, edition = _example_identity(game)
            used_teams.add(team)
            if edition:
                used_editions.add(edition)
            inserted += 1
        grid_full_skipped += len(candidates)
    matrix = io.StringIO(newline="")
    csv.writer(matrix).writerows(grid)
    _write_text(glossary_path, matrix.getvalue(), encoding="utf-8-sig")
    return {
        "descriptions_filled": filled,
        "model_mappings_recorded": model_mapped,
        "reviewed_mappings_recorded": reviewed_mapped,
        "grid_mappings_inserted": inserted,
        "grid_full_skipped": grid_full_skipped,
        "duplicate_sha_skipped": duplicate_sha_skipped,
    }


def report(
    index_csv: Path,
    output_root: Path,
    glossary_path: Path | None = None,
) -> dict[str, Any]:
    rows = _index_rows(index_csv)
    grouped = _grouped_rows(rows)
    statuses, providers, mappings = Counter(), Counter(), Counter()
    index_candidate_counts = Counter(
        row["source_label"] for row in rows if row["mapping_status"] == "candidate_new"
    )
    candidate_source_counts, suggestion_counts = Counter(), Counter()
    action_events = model_suggestions = attempted = files_present = 0
    tactic_labels = tactic_labels_with_spans = 0
    failed_records, evidence_rows = [], []
    for canonical, source_rows in grouped.items():
        path = output_root / source_rows[0]["analysis_result_path"]
        if not path.is_file():
            continue
        files_present += 1
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("clip_sha256") != source_rows[0]["sha256"]:
            raise ValueError(f"analysis result hash mismatch: {path}")
        status = result.get("status", "unknown")
        statuses[status] += 1
        if status != "pending":
            attempted += 1
        if result.get("provider"):
            providers[result["provider"]] += 1
        if status == "failed":
            failed_records.append({
                "video_path": canonical,
                "provider": result.get("provider", ""),
                "error_type": result.get("error_type", ""),
                "error": result.get("error", ""),
            })
        action_events += len(result.get("action_events", []))
        suggestions = result.get("model_suggestions", [])
        model_suggestions += len(suggestions)
        suggestion_counts.update(
            (item.get("suggested_label_zh", ""), item.get("normalized_tactic_id", ""))
            for item in suggestions if isinstance(item, dict)
        )
        source_labels = [
            item for item in result.get("source_labels", []) if isinstance(item, dict)
        ]
        mappings.update(
            item["mapping_status"] for item in source_labels if item.get("mapping_status")
        )
        for item in source_labels:
            if item.get("mapping_status") == "candidate_new":
                candidate_source_counts[item.get("source_label", "")] += 1
            spans = item.get("evidence_spans", [])
            if item.get("label_kind") == "tactic":
                tactic_labels += 1
                tactic_labels_with_spans += bool(spans)
            for span in spans:
                evidence_rows.append({
                    "video_path": canonical,
                    "source_label": item.get("source_label", ""),
                    "tactic_id": item.get("normalized_tactic_id", ""),
                    "mapping_status": item.get("mapping_status", ""),
                    "start_s": span.get("start_s", ""),
                    "end_s": span.get("end_s", ""),
                    "observation_zh": span.get("observation_zh", ""),
                })
    candidate_source_counts = candidate_source_counts or index_candidate_counts
    candidate_grid_entries = []
    if glossary_path is not None:
        with glossary_path.open(encoding="utf-8-sig", newline="") as handle:
            candidate_grid_entries = [
                {
                    "tactic_id": row.get("id（英文标识）", ""),
                    "name_zh": row.get("中文名", ""),
                    "name_en": row.get("英文名", ""),
                    "source": row.get("参考来源", ""),
                }
                for row in csv.DictReader(handle)
                if row.get("状态") == "candidate_unreviewed"
            ]
    candidate_labels = [
        {"origin": "source_gt", "source_label": label, "count": count}
        for label, count in candidate_source_counts.most_common()
    ] + [
        {
            "origin": "model_suggestion",
            "suggested_label_zh": label,
            "normalized_tactic_id": tactic_id,
            "count": count,
        }
        for (label, tactic_id), count in suggestion_counts.most_common()
    ]
    summary = {
        "physical_videos": len({row["video_path"] for row in rows}),
        "unique_hashes": len(grouped),
        "source_label_rows": len(rows),
        "source_ground_truth_mappings": sum(
            row["mapping_status"] in {"direct_exact", "reviewed_exact"} for row in rows
        ),
        "expected_analyses": len(grouped),
        "files_present": files_present,
        "attempted": attempted,
        "pending": statuses["pending"],
        "missing": len(grouped) - files_present,
        "statuses": dict(sorted(statuses.items())),
        "providers": dict(sorted(providers.items())),
        "source_label_mapping_statuses": dict(sorted(mappings.items())),
        "index_mapping_statuses": dict(sorted(Counter(
            row["mapping_status"] for row in rows
        ).items())),
        "evidence": {
            "tactic_labels": tactic_labels,
            "with_spans": tactic_labels_with_spans,
            "without_spans": tactic_labels - tactic_labels_with_spans,
            "spans": len(evidence_rows),
        },
        "action_events": action_events,
        "model_suggestions": model_suggestions,
        "failed_records": failed_records,
        "candidate_labels": candidate_labels,
        "candidate_grid_entries": candidate_grid_entries,
        "candidate_source_labels": [
            {"source_label": label, "count": count}
            for label, count in candidate_source_counts.most_common()
        ],
    }
    _write_json(output_root / "report.json", summary)
    evidence = io.StringIO(newline="")
    writer = csv.DictWriter(evidence, fieldnames=(
        "video_path", "source_label", "tactic_id", "mapping_status",
        "start_s", "end_s", "observation_zh",
    ))
    writer.writeheader()
    writer.writerows(evidence_rows)
    _write_text(output_root / "evidence_spans.csv", evidence.getvalue())
    lines = [
        "# FIFA Game Library analysis report", "",
        f"- physical videos: {summary['physical_videos']}",
        f"- unique hashes / expected analyses: {summary['unique_hashes']}",
        f"- attempted / pending / missing: {attempted} / {summary['pending']} / {summary['missing']}",
        f"- statuses: {json.dumps(summary['statuses'], ensure_ascii=False)}",
        f"- providers: {json.dumps(summary['providers'], ensure_ascii=False)}",
        f"- result mappings: {json.dumps(summary['source_label_mapping_statuses'], ensure_ascii=False)}",
        f"- reviewed index mappings: {json.dumps(summary['index_mapping_statuses'], ensure_ascii=False)}",
        f"- tactic evidence: {tactic_labels_with_spans}/{tactic_labels} labels, {len(evidence_rows)} spans",
        f"- failed / candidate labels: {len(failed_records)} / {sum(candidate_source_counts.values())}",
        f"- action events / model suggestions: {action_events} / {model_suggestions}", "",
        "## Candidate Grid entries", "",
        "| tactic_id | 中文名 | source |", "|---|---|---|",
        *[
            f"| {item['tactic_id']} | {item['name_zh']} | {item['source']} |"
            for item in candidate_grid_entries
        ],
        "", "## Top source labels still unmapped", "",
        "| source label | videos |", "|---|---:|",
        *[
            f"| {item['source_label'].replace('|', chr(92) + '|')} | {item['count']} |"
            for item in summary["candidate_source_labels"][:20]
        ],
        "", "Full evidence spans: `evidence_spans.csv`.", "",
    ]
    _write_text(output_root / "report.md", "\n".join(lines))
    return summary


def _visual_scores(frame, previous, scene_delta: float = 0.0):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    small = cv2.resize(frame, (480, max(2, round(height * 480 / width))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    saturated = cv2.inRange(hsv, (0, 85, 90), (179, 255, 255))
    green = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    accent_mask = cv2.bitwise_and(saturated, cv2.bitwise_not(green))
    green_ratio = float(np.mean(green > 0))
    contours, _ = cv2.findContours(
        accent_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
    )
    ring_candidates = 0
    ring_mask = np.zeros_like(saturated)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        pad = max(w, h)
        context = green[
            max(0, y - pad):min(small.shape[0], y + h + pad),
            max(0, x - pad):min(small.shape[1], x + w + pad),
        ]
        green_context = float(np.mean(context > 0)) if context.size else 0.0
        aspect = w / max(h, 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        if (
            8 <= area <= 0.006 * small.shape[0] * small.shape[1]
            and 1.15 <= aspect <= 5.0
            and green_context >= 0.30
            and circularity >= 0.20
        ):
            ring_candidates += 1
            cv2.ellipse(
                ring_mask,
                (x + w // 2, y + h // 2),
                (max(2, w // 2 + 2), max(2, h // 2 + 2)),
                0, 0, 360, 255, -1,
            )
    ring_score = min(1.0, ring_candidates / 4)
    ring_selector = ring_mask > 0
    primary_colors = []
    ring_accent = ring_selector & (accent_mask > 0)
    ring_hues = hsv[:, :, 0][ring_accent]
    hue_bins = Counter(int(value) // 15 for value in ring_hues)
    for hue_bin, count in hue_bins.most_common(3):
        if count < max(5, len(ring_hues) // 20):
            continue
        selector = ring_accent & (hsv[:, :, 0] // 15 == hue_bin)
        primary_colors.append(np.median(small[selector], axis=0))
    grass_pixels = small[green > 0]
    green_ring = small[ring_selector & (green > 0)]
    if len(grass_pixels) and len(green_ring):
        grass_color = np.median(grass_pixels, axis=0)
        distinct_green = green_ring[
            np.linalg.norm(green_ring.astype(float) - grass_color, axis=1) >= 35
        ]
        if len(distinct_green) >= 5:
            primary_colors.append(np.median(distinct_green, axis=0))
    team_palette = []
    for color in primary_colors:
        if all(np.linalg.norm(color - known) >= 35 for known in team_palette):
            team_palette.append(color)
    team_palette = [[int(value) for value in color] for color in team_palette[:3]]
    white_ring = small[
        ring_selector & (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 180)
    ]
    ring_white = (
        [int(value) for value in np.median(white_ring, axis=0)]
        if len(white_ring) else None
    )
    palette_score = ring_score if len(team_palette) >= 2 else 0.0

    edges = cv2.Canny(gray, 70, 170)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35,
        minLineLength=max(18, small.shape[1] // 18), maxLineGap=10,
    )
    colored_angles, colored_lengths, colored_midpoints, colored_segments, line_widths = [], [], [], [], []
    distance = cv2.distanceTransform(accent_mask, cv2.DIST_L2, 3)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
            mask = np.zeros_like(saturated)
            context = np.zeros_like(saturated)
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
            cv2.line(context, (x1, y1), (x2, y2), 255, 11)
            accent_coverage = float(np.mean(accent_mask[mask > 0] > 0))
            green_context = float(np.mean(green[context > 0] > 0))
            if accent_coverage > 0.32 and green_context > 0.35:
                colored_angles.append(angle)
                colored_lengths.append(length)
                colored_midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
                colored_segments.append(((x1, y1), (x2, y2)))
                widths = distance[(mask > 0) & (distance > 0)]
                if widths.size:
                    line_widths.append(float(np.median(widths) * 2))
    colored_lines = len(colored_angles)
    long_lines = sum(
        length > small.shape[1] * 0.12 for length in colored_lengths
    )
    topology_score = min(1.0, colored_lines / 6)
    angle_bins = Counter(round(angle / 10) * 10 for angle in colored_angles)
    dominant_angle, parallel = (angle_bins.most_common(1)[0] if angle_bins else (None, 0))
    parallel_points = [
        point for point, angle in zip(colored_midpoints, colored_angles)
        if round(angle / 10) * 10 == dominant_angle
    ]
    if dominant_angle is not None and len(parallel_points) > 1:
        normal = np.radians(dominant_angle + 90)
        offsets = sorted(x * np.cos(normal) + y * np.sin(normal) for x, y in parallel_points)
        gaps = [right - left for left, right in zip(offsets, offsets[1:]) if right - left >= 2]
    else:
        gaps = []
    striped_score = min(1.0, parallel / 10) * min(1.0, colored_lines / 6)
    arrowheads = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 12 <= area <= 0.004 * small.shape[0] * small.shape[1]:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.06 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        pad = max(w, h)
        context = green[
            max(0, y - pad):min(small.shape[0], y + h + pad),
            max(0, x - pad):min(small.shape[1], x + w + pad),
        ]
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        endpoint_distance = min(
            (
                np.hypot(center[0] - point[0], center[1] - point[1])
                for segment in colored_segments for point in segment
            ),
            default=float("inf"),
        )
        if (
            3 <= len(polygon) <= 4
            and cv2.isContourConvex(polygon)
            and context.size
            and np.mean(context > 0) >= 0.30
            and endpoint_distance <= max(8, 1.5 * max(w, h))
        ):
            arrowheads += 1
    arrow_score = (
        min(1.0, long_lines / 3)
        * min(1.0, colored_lines / 3)
        * min(1.0, arrowheads / 2)
    )

    brightness = gray.astype(float).mean(axis=0)
    beam_contrast = max(
        0.0, float(np.percentile(brightness, 98) - np.median(brightness)),
    )
    bright = gray > max(180, float(np.median(gray)) + 35)
    beam_continuity = float(np.max(np.mean(bright, axis=0)))
    beam_score = (
        min(1.0, beam_contrast / 50)
        * min(1.0, beam_continuity / 0.45)
        * min(1.0, green_ratio / 0.55)
    )
    top_down_score = min(1.0, max(0.0, green_ratio - 0.80) / 0.12)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 110, 110), (12, 255, 255)),
        cv2.inRange(hsv, (165, 110, 110), (179, 255, 255)),
    )
    red_contours, _ = cv2.findContours(
        red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_area = small.shape[0] * small.shape[1]
    label_boxes = 0
    for contour in red_contours:
        x, y, w, h = cv2.boundingRect(contour)
        pad = max(4, h)
        context = green[
            max(0, y - pad):min(small.shape[0], y + h + pad),
            max(0, x - pad):min(small.shape[1], x + w + pad),
        ]
        green_context = float(np.mean(context > 0)) if context.size else 0.0
        if (
            0.0005 * frame_area <= w * h <= 0.08 * frame_area
            and w >= 1.5 * h
            and green_context >= 0.20
        ):
            label_boxes += 1
    text_score = min(1.0, label_boxes / 2)

    freeze_score = 0.0
    scene_change = 0.0
    if previous is not None and previous.shape == small.shape:
        difference = float(np.mean(cv2.absdiff(small, previous))) / 255
        freeze_score = max(0.0, 1 - difference * 14)
        scene_change = min(1.0, difference * 5)
    accent_pixels = small[accent_mask > 0]
    accent = (
        [int(value) for value in np.median(accent_pixels, axis=0)]
        if len(accent_pixels) else [255, 255, 255]
    )
    scores = {
        "player_ring": ring_score,
        "attack_defense_colors": palette_score,
        "topology_lines": topology_score,
        "striped_zone": striped_score,
        "arrow_path": arrow_score,
        "vertical_beam": beam_score,
        "freeze_or_slow_motion": freeze_score,
        "top_down_transition": top_down_score * min(1.0, scene_delta / 0.18),
        "text_label": text_score,
    }
    metrics = {
        "accent_bgr": accent,
        "team_primary_colors_bgr": team_palette,
        "team_color_assignment": "unverified",
        "ring_white_bgr": ring_white,
        "ring_candidate_count": ring_candidates,
        "accent_area_ratio": round(float(np.mean(accent_mask > 0)), 4),
        "green_ratio": round(green_ratio, 4),
        "colored_line_count": colored_lines,
        "colored_line_mean_length_px_at_480w": (
            round(float(np.mean(colored_lengths)), 2) if colored_lengths else None
        ),
        "line_width_px_at_480w": (
            round(float(np.median(line_widths)), 2) if line_widths else None
        ),
        "dominant_line_angle_deg": dominant_angle,
        "parallel_line_count": parallel,
        "parallel_spacing_px_at_480w": (
            round(float(np.median(gaps)), 2) if gaps else None
        ),
        "overlay_alpha_estimate": None,
        "overlay_alpha_note": "not identifiable from composited pixels",
        "arrowhead_candidate_count": arrowheads,
        "red_label_box_count": label_boxes,
        "top_down_view_score": round(top_down_score, 4),
        "scene_change_score": round(scene_change, 4),
    }
    return small, scores, metrics


def visual_scan(
    library_root: Path,
    output_root: Path,
    *,
    expected_videos: int | None = EXPECTED_VIDEOS,
    sample_hz: float = 2.0,
    evidence_cap: int = 3,
) -> dict[str, int]:
    """Stream every video once; retain only compact detections and evidence."""
    import cv2
    import numpy as np

    videos = _videos(library_root)
    if expected_videos is not None and len(videos) != expected_videos:
        raise ValueError(f"expected {expected_videos} FIFA videos, got {len(videos)}")
    rows: list[dict[str, Any]] = []
    evidence: dict[str, list[tuple[float, str, float, dict[str, Any]]]] = defaultdict(list)
    reference_metrics: dict[str, tuple[float, dict[str, Any]]] = {}
    total_expected = total_decoded = total_sampled = scene_change_samples = 0
    for video in videos:
        rel = video.relative_to(library_root).as_posix()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open FIFA video: {rel}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, round(fps / sample_hz))
        previous, previous_scene = None, None
        frame_number = decoded_frames = sampled_frames = 0
        last_scene_candidate = -stride
        best: dict[str, tuple[float, float, dict[str, Any]]] = {}
        hits, periodic_hits = Counter(), Counter()
        first_hit: dict[str, float] = {}
        last_hit: dict[str, float] = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            tiny = cv2.cvtColor(
                cv2.resize(frame, (96, max(2, round(frame.shape[0] * 96 / frame.shape[1])))),
                cv2.COLOR_BGR2GRAY,
            )
            scene_delta = (
                float(np.mean(cv2.absdiff(tiny, previous_scene))) / 255
                if previous_scene is not None else 0.0
            )
            previous_scene = tiny
            periodic = frame_number % stride == 0
            scene_candidate = (
                scene_delta >= 0.18
                and frame_number - last_scene_candidate >= max(1, stride // 2)
            )
            if not periodic and not scene_candidate:
                frame_number += 1
                continue
            if scene_candidate:
                last_scene_candidate = frame_number
            sampled_frames += 1
            scene_change_samples += int(scene_candidate and not periodic)
            timestamp = frame_number / fps
            previous, scores, metrics = _visual_scores(frame, previous, scene_delta)
            metrics = {
                **metrics,
                "sample_reason": "periodic" if periodic else "scene_change",
                "scene_delta": round(scene_delta, 4),
            }
            for pattern, score in scores.items():
                if score > best.get(pattern, (-1, 0, {}))[0]:
                    best[pattern] = (score, timestamp, metrics)
                if score >= PATTERN_THRESHOLDS[pattern]:
                    hits[pattern] += 1
                    periodic_hits[pattern] += int(periodic)
                    first_hit.setdefault(pattern, timestamp)
                    last_hit[pattern] = timestamp
                if STYLE_REFERENCES.get(pattern) == rel:
                    current = reference_metrics.get(pattern)
                    if current is None or score > current[0]:
                        reference_metrics[pattern] = (score, metrics)
            frame_number += 1
        capture.release()
        if not decoded_frames or not sampled_frames:
            raise RuntimeError(f"decoded no usable frames from FIFA video: {rel}")
        tolerance = max(2, round(expected_frames * 0.01))
        if expected_frames and decoded_frames + tolerance < expected_frames:
            raise RuntimeError(
                f"FIFA video ended early: {rel} "
                f"({decoded_frames}/{expected_frames} frames)"
            )
        total_expected += expected_frames or decoded_frames
        total_decoded += decoded_frames
        total_sampled += sampled_frames
        for pattern, (score, timestamp, metrics) in best.items():
            item = (score, rel, timestamp, metrics)
            heap = evidence[pattern]
            if len(heap) < evidence_cap:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
            minimum_hits = 1 if pattern == "top_down_transition" else 2
            if hits[pattern] < minimum_hits:
                continue
            row_metrics = {
                **metrics,
                "hit_count": hits[pattern],
                "periodic_hit_count": periodic_hits[pattern],
                "first_hit_s": round(first_hit[pattern], 3),
                "last_hit_s": round(last_hit[pattern], 3),
                "estimated_duration_s": round(periodic_hits[pattern] / sample_hz, 3),
                "decoded_frames": decoded_frames,
                "expected_frames": expected_frames or None,
                "decode_ratio": (
                    round(decoded_frames / expected_frames, 4) if expected_frames else None
                ),
            }
            rows.append({
                "video_path": f"FIFA Game Library/{rel}",
                "source_folder": rel.split("/", 1)[0],
                "pattern": pattern,
                "timestamp_s": f"{timestamp:.3f}",
                "score": f"{score:.4f}",
                "status": "unverified",
                "metrics_json": json.dumps(row_metrics, ensure_ascii=False, sort_keys=True),
            })

    visual_root = output_root / "visual_patterns"
    evidence_root = visual_root / "evidence"
    shutil.rmtree(evidence_root, ignore_errors=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    fields = (
        "video_path", "source_folder", "pattern", "timestamp_s",
        "score", "status", "metrics_json",
    )
    _write_csv(visual_root / "detections.csv", rows, fields)
    for pattern, confirmed in {
        "vertical_beam": VERTICAL_BEAM_EVIDENCE,
        "arrow_path": ARROW_PATH_EVIDENCE,
        "striped_zone": STRIPED_ZONE_EVIDENCE,
        "topology_lines": TOPOLOGY_LINE_EVIDENCE,
    }.items():
        evidence[pattern] = [
            (1.0, rel, timestamp, {"evidence_filename": filename})
            for rel, timestamp, filename in confirmed
        ]
    for pattern, candidates in evidence.items():
        directory = evidence_root / pattern
        directory.mkdir(parents=True, exist_ok=True)
        for score, rel, timestamp, metrics in sorted(candidates, reverse=True):
            capture = cv2.VideoCapture(str(library_root / rel))
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                continue
            token = hashlib.sha256(rel.encode()).hexdigest()[:10]
            filename = metrics.get("evidence_filename", f"{token}_{timestamp:.2f}_{score:.3f}.jpg")
            cv2.imwrite(str(directory / filename), frame)

    counts = Counter(row["pattern"] for row in rows)
    summary = [
        "# FIFA Game Library visual-pattern scan",
        "",
        f"- Videos scanned and decoded: {len(videos)}",
        f"- Frames decoded: {total_decoded}/{total_expected}; analysed: {total_sampled}",
        f"- Sampling: scene changes plus {sample_hz:g} fps; streamed, no full-frame cache",
        f"- Extra scene-change samples: {scene_change_samples}",
        "- Detection status: unverified; semantic topology is not inferred",
        "- Evidence: top review candidates; an image may be below its detection threshold",
        "",
        "| Pattern | Candidate clips | Threshold |",
        "|---|---:|---:|",
        *[
            f"| {pattern} | {counts[pattern]} | {PATTERN_THRESHOLDS[pattern]:.2f} |"
            for pattern in PATTERN_THRESHOLDS
        ],
        "",
    ]
    (visual_root / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    style = {
        "schema_version": "fifa-visual-style-v1",
        "source": "qualitative rules confirmed from user-selected FIFA reference frames",
        "patterns": {
            pattern: {
                "status": "qualitative_rule_user_confirmed",
                "reference_video": f"FIFA Game Library/{STYLE_REFERENCES[pattern]}",
                "measurement_status": "automated_unverified",
                "detector_score": round(score, 4),
                "detector_candidate": score >= PATTERN_THRESHOLDS[pattern],
                "measurements": metrics,
            }
            for pattern, (score, metrics) in reference_metrics.items()
        },
        "stage4": {
            "status": "provisional_rule_engine",
            "parameter_source": "hand_tuned from user-confirmed reference frames",
            "attacker_color_bgr": [70, 190, 80],
            "defender_color_bgr": [60, 70, 220],
            "beam_color_bgr": [90, 230, 215],
            "white_color_bgr": [245, 245, 245],
            "inter_line_color_bgr": [220, 160, 60],
            "ring_radius": 20,
            "ring_thickness": 4,
            "line_alpha": 0.72,
            "zone_alpha": 0.16,
            "stripe_spacing": 16,
            "beam_alpha": 0.28,
        },
    }
    for pattern in ("vertical_beam", "arrow_path", "striped_zone", "topology_lines"):
        style["patterns"].setdefault(pattern, {
            "status": "qualitative_rule_user_confirmed",
            "reference_video": f"FIFA Game Library/{STYLE_REFERENCES[pattern]}",
            "measurement_status": "reference_frames_user_confirmed_measurements_unverified",
            "detector_score": None,
            "detector_candidate": None,
            "measurements": {},
        })
    style["patterns"]["vertical_beam"].update({
        "measurement_status": "reference_frames_user_confirmed_measurements_unverified",
        "reference_frames": [
            {
                "video_path": f"FIFA Game Library/{rel}",
                "timestamp_s": timestamp,
            }
            for rel, timestamp, *_ in VERTICAL_BEAM_EVIDENCE
        ],
        "semantic_role": "spotlight the creator or ball carrier behind a decisive attacking action",
        "selection_rule": (
            "football.assist, high-importance football.pass, or line-breaking ball carry; "
            "do not select the goal scorer by default"
        ),
        "visual_notes": "soft warm yellow-green shaft, diffuse edges, elliptical ground glow",
    })
    style["patterns"]["arrow_path"].update({
        "measurement_status": "reference_frames_user_confirmed_measurements_unverified",
        "reference_frames": [
            {
                "video_path": f"FIFA Game Library/{rel}",
                "timestamp_s": timestamp,
            }
            for rel, timestamp, *_ in ARROW_PATH_EVIDENCE
        ],
        "semantic_role": "guide a pass or run toward its intended receiving point or free space",
        "selection_rule": (
            "draw only when structured evidence supplies a start and intended destination; "
            "the arrow tip marks the receiving point"
        ),
        "visual_notes": (
            "thin white solid or dashed line with a clear arrowhead; "
            "route around players and keep the field readable"
        ),
    })
    style["patterns"]["striped_zone"].update({
        "measurement_status": "reference_frames_user_confirmed_measurements_unverified",
        "reference_frames": [
            {
                "video_path": f"FIFA Game Library/{rel}",
                "timestamp_s": timestamp,
            }
            for rel, timestamp, *_ in STRIPED_ZONE_EVIDENCE
        ],
        "semantic_role": "mark a tactically useful free area or defending unit",
        "visual_notes": (
            "pair stripes with the defending team's labelled forward, midfield, and defensive "
            "lines when explaining how the attacker breaks the block"
        ),
    })
    style["patterns"]["topology_lines"].update({
        "measurement_status": "reference_frames_user_confirmed_measurements_unverified",
        "reference_frames": [
            {"video_path": f"FIFA Game Library/{rel}", "timestamp_s": timestamp}
            for rel, timestamp, *_ in TOPOLOGY_LINE_EVIDENCE
        ],
        "semantic_role": "show only stable, visible tactical units rather than connecting every player",
        "selection_rule": (
            "use at least two stable members for attacking forward and midfield lines or "
            "the defending defensive line; keep uncertain groups as data only"
        ),
        "visual_notes": "thin consecutive links; striped bands cover selected units or inter-line space only",
    })
    _write_json(visual_root / "style_profile.json", style)
    return {
        "videos_scanned": len(videos),
        "frames_expected": total_expected,
        "frames_decoded": total_decoded,
        "frames_analysed": total_sampled,
        "scene_change_samples": scene_change_samples,
        "detections": len(rows),
        "evidence_images": sum(
            1 for path in evidence_root.rglob("*.jpg")
        ),
    }
