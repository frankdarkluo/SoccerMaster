"""Offline tactical prototype migration, retrieval, verification, and curation."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import os
import math
import random
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from pipeline.tactics_qa.auto_evidence import build_claim_evidence
from pipeline.tactics_qa.checkers import CheckResult, run_checkers
from pipeline.tactics_qa.gsr_io import load_gsr
from pipeline.tactics_qa.verdict import decide

from pipeline.tactics_qa.evidence import (
    CONTROLLED_REGAIN_KINDS, DELIVERY_KINDS, LANDING_ZONES, RESTART_KINDS,
    ClipEvidence, PossessionEvent,
)


SCHEMA_VERSION = "tactical-prototype-v1"
MIGRATION_VERSION = "tactical-prototype-migration-v1"
MVP_TACTICS = {
    "fast_break_pattern",
    "cutback",
    "line_break",
    "run_in_behind",
    "overlap_run",
}

SYSTEM_ARM_TACTICS = {"fast_break_pattern", "run_in_behind", "corner-near-far-post", "cutback"}
CONTACT_SHEET_PROFILES = {"640": (640, False), "1024": (1019, False), "1080p-split": (1920, True)}
LOGGER = logging.getLogger(__name__)

DATASETS = (
    ("soccernetgs", "SoccerNetGS_videos"),
    ("event_clips", "event-clips_L3_all"),
    ("canada_bosnia", "event-clips_canada_vs_bosnia_L3_all"),
    ("germany_paraguay", "event-clips_Germany_Paraguay_L3_all"),
)
MANIFEST_ONLY_DATASETS = (
    ("ivory_ecuador", "event-clips_ivory_ecuador_L3_all"),
)
ALL_DATASETS = DATASETS + MANIFEST_ONLY_DATASETS

LABEL_IDS = {
    "快速反击": "fast_break_pattern",
    "下底倒三角": "cutback",
    "破线传球": "line_break",
    "打身后 / 反越位跑位": "run_in_behind",
    "套边与内切套上": "overlap_run",
    "角球后点/前点战术": "corner_near_far_post",
    "角球前点 / 后点战术": "corner_near_far_post",
    "虚跑/假跑扯动": "dummy_run",
    "虚跑 / 假跑扯动": "dummy_run",
    "线间接应": "receive_between_lines",
    "门将参与出球": "goalkeeper_build_up",
    "肋部渗透": "half_space_penetration",
    "大范围转移": "switch_of_play",
    "二过一 / 撞墙配合": "one_two",
    "宽度与纵深": "width_and_depth",
    "区域防守": "zonal_defending",
    "定位球区域防守": "set_piece_zonal_defending",
    "反抢 / 就地反抢": "counterpress",
    "中后卫带球推进": "centre_back_carry",
    "大力手抛球 / 长距离界外球": "long_throw",
    "角球围堵门将": "corner_goalkeeper_block",
    "逼抢触发器": "pressing_trigger",
    "混合盯防": "mixed_marking",
    "长传冲吊": "long_ball",
    "后场倒脚控场": "defensive_circulation",
}

TARGET_ALIASES = {
    "fast_break_pattern": ("快速反击", "反击"),
    "cutback": ("下底倒三角", "倒三角"),
    "line_break": ("破线传球", "破线"),
    "run_in_behind": ("打身后", "反越位"),
    "overlap_run": ("套边", "套上", "外侧超越"),
}

UNOBSERVABLE_MARKERS = (
    "无法确认", "不能确认", "不清", "不够清", "未展示", "看不到", "缺失", "不足以",
    "无法判断", "不可见", "不能认定", "无法可靠", "时序不清", "镜头", "没有被连续展示",
)

CONCEPT_CARDS = {
    "fast_break_pattern": {
        "name_zh": "快速反击",
        "definition": "明确夺回球权后立即纵向推进，并伴随进攻队员前插与对手回追。",
        "exclusions": ["既有稳定控球后的推进", "片段从反击中段开始", "固定开球"],
    },
    "cutback": {
        "name_zh": "下底倒三角",
        "definition": "进攻者到达边路底线附近后，将球回做至禁区内后方接应点。",
        "exclusions": ["普通高球传中", "沿门前平扫", "未到达底线区域"],
    },
    "line_break": {
        "name_zh": "破线传球",
        "definition": "受控传球越过一条稳定可见的对手防线，并由线后队友控制。",
        "exclusions": ["盘带越线", "解围", "沿防线外侧绕行"],
    },
    "run_in_behind": {
        "name_zh": "打身后 / 反越位跑位",
        "definition": "进攻跑者因自身前插从防线身前进入防线身后。",
        "exclusions": ["跑者原本在线后", "防守回追", "仅由防线前压造成表观越线"],
    },
    "corner-near-far-post": {
        "name_zh": "角球后点/前点战术",
        "definition": "角球落向明确的近门柱或远门柱区域，并有进攻者攻击该落点。",
        "exclusions": ["落点在禁区中央", "开放比赛传中", "落点不可见"],
    },
    "overlap_run": {
        "name_zh": "套边",
        "definition": "同队跑者从持球人身后沿外侧通道持续前插并到达身前。",
        "exclusions": ["平行接应", "内侧套上", "防守回追"],
    },
}

DEFAULT_RULES = {
    "schema_version": "tactical-prototype-rules-v1",
    "private": True,
    "common": {"min_track_continuity": 0.5, "min_pitch_coverage": 0.5, "min_ball_coverage": 0.5},
    "overlap_run": {"min_longitudinal_change": 1.0, "min_lateral_separation": 2.0},
    "run_in_behind": {"min_line_crossing": 1.0},
    "line_break": {"min_line_crossing": 1.0, "max_receive_distance": 8.0},
    "cutback": {"bottom_zone_x": 35.0, "min_backward_change": 1.0},
    "fast_break_pattern": {"max_transition_seconds": 10.0, "min_forward_change": 5.0},
}


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _seconds(token: str) -> float:
    parts = token.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"invalid time: {token}")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time: {token}") from exc
    return sum(value * 60 ** index for index, value in enumerate(reversed(values)))


def parse_window(raw: str, duration_s: float | None = None) -> dict[str, Any]:
    parts = re.split(r"\s*[–—−-]\s*", raw.strip())
    if len(parts) != 2:
        raise ValueError(f"invalid time window: {raw}")
    start_s, end_s = map(_seconds, parts)
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"invalid time window: {raw}")
    if duration_s is not None and end_s > duration_s:
        if end_s - duration_s > 1.0 + 1e-9:
            raise ValueError(f"window exceeds video by more than one second: {raw}")
        end_s = duration_s
    return {"raw": raw, "start_s": start_s, "end_s": end_s}


def _load_manifests(data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id, dirname in ALL_DATASETS:
        manifest_path = data_root / dirname / "manifest.json"
        clips: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            for clip in json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]:
                clips[str(clip["file"]).split("_", 1)[0]] = clip
        result[dataset_id] = clips
    return result


def _manifest_source_group(clip: dict[str, Any], dataset_id: str) -> str:
    match = re.search(r"/clips/([0-9a-f]{64})/", str(clip.get("video_clip", "")))
    return f"source_video_sha256:{match.group(1)}" if match else f"dataset:{dataset_id}"


def _soccernet_groups(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    integrity = repo_root / "benchmark/manifests/soccernetgs_v1_3_integrity.json"
    if integrity.is_file():
        payload = json.loads(integrity.read_text(encoding="utf-8"))
        records = payload.get("records", payload if isinstance(payload, list) else [])
        for record in records:
            clip_id = record.get("clip_id") or record.get("sequence") or record.get("id")
            game_id = record.get("game_id")
            if clip_id and game_id is not None:
                result[str(clip_id)] = f"soccernet_game:{game_id}"
    labels_root = repo_root / "codes/sn-gamestate/datasets/SoccerNetGS"
    for label in labels_root.glob("*/*/Labels-GameState.json"):
        try:
            payload = json.loads(label.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        game_id = payload.get("info", {}).get("game_id")
        if game_id is not None:
            result.setdefault(label.parent.name, f"soccernet_game:{game_id}")
    return result


def _video_for(folder: Path, clip_id: str, manifest_clip: dict[str, Any] | None) -> Path | None:
    if manifest_clip:
        path = folder / manifest_clip["file"]
        return path if path.is_file() else None
    candidates = sorted(folder.glob(f"{clip_id}*.mp4"))
    return candidates[0] if candidates else None


def _duration(clip: dict[str, Any] | None) -> float | None:
    if not clip:
        return 30.0
    start, end = clip.get("start_ms"), clip.get("end_ms")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(end - start)
    return None


def _manifest_source_records(
    data_root: Path, dataset_id: str, dirname: str, repo_root: Path,
) -> list[dict[str, Any]]:
    folder = data_root / dirname
    manifest_path = folder / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"manifest has no clips list: {manifest_path}")
    if payload.get("exported_count") not in (None, len(clips)):
        raise ValueError(f"manifest exported_count mismatch: {manifest_path}")
    records = []
    for row_number, clip in enumerate(clips, start=1):
        clip_id = str(clip["file"]).split("_", 1)[0]
        video_path = folder / clip["file"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        records.append({
            "schema_version": "tactical-source-row-v1",
            "dataset_id": dataset_id,
            "source_group_id": _manifest_source_group(clip, dataset_id),
            "source_csv": None,
            "source_manifest": manifest_path.relative_to(repo_root).as_posix(),
            "source_row_number": row_number,
            "clip_id": clip_id,
            "clip_uid": f"{dataset_id}:{clip_id}",
            "event_id": str(clip.get("id")) if clip.get("id") is not None else None,
            "video_path": video_path.relative_to(repo_root).as_posix(),
            "row": clip,
        })
    if len({row["clip_uid"] for row in records}) != len(records):
        raise ValueError(f"duplicate clip IDs in manifest: {manifest_path}")
    return records


def register_manifest_source(
    root: Path | str, data_root: Path | str, dataset_id: str, dirname: str,
    repo_root: Path | str | None = None,
) -> dict[str, int]:
    root, data_root = Path(root), Path(data_root).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else data_root.parent
    records = _manifest_source_records(data_root, dataset_id, dirname, repo_root)
    source_path = root / "source_rows.jsonl"
    existing = load_jsonl(source_path)
    rows = [row for row in existing if row["dataset_id"] != dataset_id] + records
    rows.sort(key=lambda row: (row["dataset_id"], row["source_row_number"]))
    _write_jsonl(source_path, rows)

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["source_group_id"]].append(row["clip_uid"])
    _write_json(root / "source_groups.json",
                {key: sorted(set(values)) for key, values in sorted(groups.items())})

    report_path = root / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_row_count"] = len(rows)
    registered = dict(report.get("registered_manifest_sources", {}))
    registered[dataset_id] = len(records)
    report["registered_manifest_sources"] = dict(sorted(registered.items()))
    _write_json(report_path, report)
    return {"registered": len(records), "source_row_count": len(rows)}


def _video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _tactic_id(label: str) -> str:
    if label in LABEL_IDS:
        return LABEL_IDS[label]
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if ascii_slug:
        return ascii_slug
    return "source_label_" + hashlib.sha256(label.encode()).hexdigest()[:12]


def _candidate_mentions(reason: str) -> list[str]:
    return sorted(tactic for tactic, aliases in TARGET_ALIASES.items() if any(alias in reason for alias in aliases))


def _source_phase(clip: dict[str, Any] | None, video_path: Path | None) -> str | None:
    if clip and clip.get("event_type"):
        return str(clip["event_type"])
    if video_path:
        match = re.search(r"_L3_(.+?)_\d+p\d+-", video_path.name)
        if match:
            return match.group(1)
    return None


def _episode(
    *, dataset_id: str, clip_id: str, event_id: str | None, source_group_id: str,
    video_path: Path | None, source_csv: Path, row_number: int, window: dict[str, Any],
    tactic_id: str, source_tactic_name: str, prototype_type: str, review_status: str,
    confidence: str, reason: str, phase: str | None,
) -> dict[str, Any]:
    prototype_id = f"{dataset_id}/{clip_id}/{row_number}/{tactic_id}"
    missing = [marker for marker in UNOBSERVABLE_MARKERS if marker in reason]
    return {
        "schema_version": SCHEMA_VERSION,
        "prototype_id": prototype_id,
        "clip_uid": f"{dataset_id}:{clip_id}",
        "source": {
            "dataset_id": dataset_id,
            "source_group_id": source_group_id,
            "clip_id": clip_id,
            "event_id": event_id,
            "video_path": video_path.as_posix() if video_path else None,
            "source_csv": source_csv.as_posix(),
            "source_row_number": row_number,
            "phase": phase,
        },
        "window": window,
        "tactic_id": tactic_id,
        "source_tactic_name": source_tactic_name,
        "prototype_type": prototype_type,
        "review_status": review_status,
        "confidence": confidence,
        "evidence_summary": reason,
        "positive_evidence": [reason] if prototype_type == "positive" else [],
        "failed_conditions": [reason] if prototype_type == "targeted_negative" else [],
        "missing_evidence": missing if prototype_type == "unobservable" else [],
        "team_id": None,
        "actor_bindings": {},
        "observability": {"video": "direct", "tracking": "unknown", "ball": "unknown", "calibration": "unknown"},
        "provenance": {"migration_version": MIGRATION_VERSION},
    }


def _read_glossary(glossary: Path) -> list[dict[str, str]]:
    with glossary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("中文名") and not row.get("优先级", "").startswith("下拉选择")]


def migrate(data_root: Path | str, glossary: Path | str, out: Path | str, repo_root: Path | str | None = None) -> dict[str, Any]:
    data_root, glossary, out = Path(data_root).resolve(), Path(glossary).resolve(), Path(out)
    repo_root = Path(repo_root).resolve() if repo_root else data_root.parent
    out.mkdir(parents=True, exist_ok=True)
    manifests = _load_manifests(data_root)
    soccernet_groups = _soccernet_groups(repo_root)
    source_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    source_groups: dict[str, list[str]] = defaultdict(list)
    mapped_videos: set[Path] = set()
    aliases: dict[str, str] = {}

    for dataset_id, dirname in DATASETS:
        folder = data_root / dirname
        source_csv = folder / "tactics.csv"
        with source_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=1):
                clip_id = row["id"].strip()
                manifest_clip = manifests[dataset_id].get(clip_id)
                video_path = _video_for(folder, clip_id, manifest_clip)
                if video_path:
                    mapped_videos.add(video_path.resolve())
                source_group = (
                    soccernet_groups.get(clip_id, "soccernet_challenge:opaque")
                    if dataset_id == "soccernetgs"
                    else _manifest_source_group(manifest_clip or {}, dataset_id)
                )
                source_groups[source_group].append(f"{dataset_id}:{clip_id}")
                try:
                    window = parse_window(row["时间范围"], _duration(manifest_clip))
                except ValueError:
                    # Source timestamps can differ slightly from the muxed duration.
                    # Probe only exceptional rows and still enforce the one-second rule.
                    if not video_path:
                        raise
                    window = parse_window(row["时间范围"], _video_duration(video_path))
                event_id = str(manifest_clip.get("id")) if manifest_clip else None
                source_record = {
                    "schema_version": "tactical-source-row-v1",
                    "dataset_id": dataset_id,
                    "source_group_id": source_group,
                    "source_csv": source_csv.relative_to(repo_root).as_posix(),
                    "source_row_number": row_number,
                    "clip_id": clip_id,
                    "clip_uid": f"{dataset_id}:{clip_id}",
                    "event_id": event_id,
                    "video_path": video_path.relative_to(repo_root).as_posix() if video_path else None,
                    "row": row,
                }
                source_rows.append(source_record)
                label, reason = row["tactics"].strip(), row["判断依据"].strip()
                common = dict(
                    dataset_id=dataset_id, clip_id=clip_id, event_id=event_id,
                    source_group_id=source_group,
                    video_path=video_path.relative_to(repo_root) if video_path else None,
                    source_csv=source_csv.relative_to(repo_root), row_number=row_number,
                    window=window, confidence=row["置信度"].strip(), reason=reason,
                    phase=_source_phase(manifest_clip, video_path),
                )
                if label != "无明确战术":
                    tactic_id = _tactic_id(label)
                    aliases[label] = tactic_id
                    episodes.append(_episode(tactic_id=tactic_id, source_tactic_name=label,
                                             prototype_type="positive", review_status="gold", **common))
                else:
                    for tactic_id in _candidate_mentions(reason):
                        prototype_type = "unobservable" if any(marker in reason for marker in UNOBSERVABLE_MARKERS) else "targeted_negative"
                        episodes.append(_episode(tactic_id=tactic_id, source_tactic_name=label,
                                                 prototype_type=prototype_type, review_status="silver", **common))

    for dataset_id, dirname in MANIFEST_ONLY_DATASETS:
        records = _manifest_source_records(data_root, dataset_id, dirname, repo_root)
        source_rows.extend(records)
        for record in records:
            source_groups[record["source_group_id"]].append(record["clip_uid"])
            mapped_videos.add((repo_root / record["video_path"]).resolve())

    glossary_rows = _read_glossary(glossary)
    concepts = []
    for row in glossary_rows:
        source_id = row.get("id（英文标识）", "").strip()
        concepts.append({"schema_version": "tactical-concept-v1", "tactic_id": source_id or _tactic_id(row["中文名"]),
                         "source": "glossary", "card": row})
    known_ids = {concept["tactic_id"] for concept in concepts}
    for tactic_id, card in CONCEPT_CARDS.items():
        if tactic_id not in known_ids:
            concepts.append({"schema_version": "tactical-concept-v1", "tactic_id": tactic_id,
                             "source": "mvp_overlay", "card": card})

    source_rows.sort(key=lambda row: (row["dataset_id"], row["source_row_number"]))
    episodes.sort(key=lambda row: row["prototype_id"])
    concepts.sort(key=lambda row: row["tactic_id"])
    all_videos = {path.resolve() for _, dirname in ALL_DATASETS for path in (data_root / dirname).glob("*.mp4")}
    orphan_videos = sorted(path.relative_to(repo_root).as_posix() for path in all_videos - mapped_videos)
    positive = [row for row in episodes if row["prototype_type"] == "positive"]
    mvp_counts = Counter(row["tactic_id"] for row in positive if row["tactic_id"] in MVP_TACTICS)
    report = {
        "schema_version": "tactical-prototype-migration-report-v1",
        "source_row_count": len(source_rows),
        "positive_count": len(positive),
        "mvp_positive_count": sum(mvp_counts.values()),
        "mvp_positive_counts": dict(sorted(mvp_counts.items())),
        "derived_silver_count": sum(row["review_status"] == "silver" for row in episodes),
        "aliases": dict(sorted(aliases.items())),
        "unmapped_positive_labels": sorted(label for label, tactic in aliases.items() if tactic.startswith("source_label_")),
        "orphan_videos": orphan_videos,
        "normalizations": {"unicode_time_dash": True, "end_rounding_clamp_seconds": 1.0},
    }
    _write_jsonl(out / "concepts.jsonl", concepts)
    _write_jsonl(out / "source_rows.jsonl", source_rows)
    _write_jsonl(out / "episodes.jsonl", episodes)
    _write_evaluation_report(out, episodes)
    _write_json(out / "migration_report.json", report)
    _write_json(out / "source_groups.json", {key: sorted(set(values)) for key, values in sorted(source_groups.items())})
    _atomic_write_text(out / "prototype_rules.yaml", yaml.safe_dump(DEFAULT_RULES, allow_unicode=True, sort_keys=True))
    return report



def _write_evaluation_report(root: Path, episodes: list[dict[str, Any]]) -> None:
    groups = sorted({row["source"]["source_group_id"] for row in episodes})
    known_development = {"soccernet_game:7", "soccernet_challenge:opaque"}
    assignments = {group: ("test" if group not in known_development and int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "retrieval")
                   for group in groups}
    if groups and "test" not in assignments.values():
        fallback = next(group for group in reversed(groups) if group not in known_development)
        assignments[fallback] = "test"
    report = {
        "schema_version": "tactical-source-disjoint-evaluation-v1",
        "group_assignments": assignments,
        "retrieval_group_count": sum(split == "retrieval" for split in assignments.values()),
        "test_group_count": sum(split == "test" for split in assignments.values()),
        "cross_split_group_collisions": [],
        "known_development_groups": sorted(known_development & set(groups)),
        "gold_count": sum(row["review_status"] == "gold" for row in episodes),
        "silver_count": sum(row["review_status"] == "silver" for row in episodes),
    }
    _write_json(root / "evaluation_report.json", report)
def validate_store(root: Path | str) -> list[str]:
    root = Path(root)
    errors: list[str] = []
    required = ("concepts.jsonl", "source_rows.jsonl", "episodes.jsonl", "migration_report.json", "source_groups.json", "evaluation_report.json", "prototype_rules.yaml")
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors
    sources, episodes = load_jsonl(root / "source_rows.jsonl"), load_jsonl(root / "episodes.jsonl")
    ids = [row.get("prototype_id") for row in episodes]
    if len(ids) != len(set(ids)):
        errors.append("prototype_id is not globally unique")
    report = json.loads((root / "migration_report.json").read_text(encoding="utf-8"))
    expected_sources = report.get("source_row_count")
    if len(sources) != expected_sources:
        errors.append(f"expected {expected_sources} source rows, found {len(sources)}")
    if sum(row.get("prototype_type") == "positive" and row.get("review_status") == "gold" for row in episodes) != 187:
        errors.append("expected 187 positive episodes")
    valid_types = {"positive", "targeted_negative", "unobservable"}
    for row in episodes:
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"invalid schema: {row.get('prototype_id')}")
        if row.get("prototype_type") not in valid_types:
            errors.append(f"invalid prototype_type: {row.get('prototype_id')}")
    return errors


def _diverse(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen, used = [], set()
    for row in rows:
        group = row["source"]["source_group_id"]
        if group not in used:
            chosen.append(row)
            used.add(group)
            if len(chosen) == limit:
                return chosen
    for row in rows:
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == limit:
                break
    return chosen


def retrieve(root: Path | str, tactic: str, *, exclude_source_group: str | None = None,
             positive_limit: int = 3, negative_limit: int = 3, unobservable_limit: int = 2,
             phase: str | None = None) -> dict[str, list[dict[str, Any]]]:
    root = Path(root)
    test_groups: set[str] = set()
    report_path = root / "evaluation_report.json"
    if exclude_source_group and report_path.is_file():
        assignments = json.loads(report_path.read_text(encoding="utf-8")).get("group_assignments", {})
        if assignments.get(exclude_source_group) == "test":
            test_groups = {group for group, split in assignments.items() if split == "test"}
    rows = [row for row in load_jsonl(root / "episodes.jsonl")
            if row["tactic_id"] == tactic and row["review_status"] == "gold"
            and row["source"]["source_group_id"] != exclude_source_group
            and row["source"]["source_group_id"] not in test_groups]
    rows.sort(key=lambda row: (row["source"].get("phase") != phase if phase else False, row["prototype_id"]))
    return {
        "positive": _diverse([row for row in rows if row["prototype_type"] == "positive"], positive_limit),
        "targeted_negative": _diverse([row for row in rows if row["prototype_type"] == "targeted_negative"], negative_limit),
        "unobservable": _diverse([row for row in rows if row["prototype_type"] == "unobservable"], unobservable_limit),
    }


def _condition(name: str, status: str, *, value: Any = None, timestamp: float | None = None,
               source: str = "tactical_state") -> dict[str, Any]:
    return {"condition": name, "status": status, "value": value, "timestamp": timestamp, "source": source}


def _track(state: dict[str, Any], track_id: Any) -> dict[str, Any] | None:
    return state.get("measurements", {}).get("tracks", {}).get(str(track_id))


def _samples(track: dict[str, Any] | None, start: float, end: float) -> list[dict[str, Any]]:
    if not track:
        return []
    inside = [sample for sample in track.get("samples", []) if start <= float(sample.get("t", -1)) <= end]
    return inside or track.get("samples", [])


def _progress(x: float, direction: str) -> float:
    return x if direction == "left_to_right" else -x


def _finish(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {condition["status"] for condition in conditions}
    status = "unverifiable" if "unverifiable" in statuses else "failed" if "failed" in statuses else "passed"
    return {"status": status, "conditions": conditions}


def verify_candidate(candidate: dict[str, Any], state: dict[str, Any] | None,
                     rules: dict[str, Any] | None = None) -> dict[str, Any]:
    tactic = candidate.get("tactic_id")
    if tactic not in MVP_TACTICS:
        raise ValueError(f"unsupported MVP tactic: {tactic}")
    if not state or state.get("schema_version") != "tactical-state-v1":
        return _finish([_condition("tactical_state", "unverifiable", value="missing")])
    rules = rules or DEFAULT_RULES
    window = candidate.get("window", {})
    start, end = float(window.get("start_s", 0)), float(window.get("end_s", 0))
    team_id, actors = candidate.get("team_id"), candidate.get("actors", {})
    direction = state.get("teams", {}).get(str(team_id), {}).get("attack_direction")
    ball = [sample for sample in state.get("measurements", {}).get("ball", []) if start <= float(sample.get("t", -1)) <= end]
    conditions = [
        _condition("team_id", "passed" if team_id in state.get("teams", {}) else "unverifiable", value=team_id),
        _condition("attack_direction", "passed" if direction in {"left_to_right", "right_to_left"} else "unverifiable", value=direction),
        _condition("ball", "passed" if len(ball) >= 2 and state.get("quality", {}).get("ball_coverage", 0) >= rules["common"]["min_ball_coverage"] else "unverifiable",
                   value={"samples": len(ball), "coverage": state.get("quality", {}).get("ball_coverage")}),
        _condition("visibility", "passed" if state.get("quality", {}).get("pitch_coordinate_coverage", 0) >= rules["common"]["min_pitch_coverage"] else "unverifiable",
                   value=state.get("quality", {}).get("pitch_coordinate_coverage")),
    ]
    if any(condition["status"] == "unverifiable" for condition in conditions):
        return _finish(conditions)

    role_names = {
        "overlap_run": ("carrier_track_id", "runner_track_id"),
        "run_in_behind": ("runner_track_id",),
        "line_break": ("passer_track_id", "receiver_track_id"),
        "cutback": ("passer_track_id", "receiver_track_id"),
        "fast_break_pattern": ("carrier_track_id", "runner_track_id"),
    }[tactic]
    tracks: dict[str, list[dict[str, Any]]] = {}
    for role in role_names:
        track_id = actors.get(role)
        track = _track(state, track_id)
        samples = _samples(track, start, end)
        same_team = bool(track and track.get("team_id") == team_id)
        continuity = state.get("quality", {}).get("track_continuity", {}).get(str(track_id), 0)
        status = "passed" if len(samples) >= 2 and same_team and continuity >= rules["common"]["min_track_continuity"] else "unverifiable"
        conditions.append(_condition(role, status, value={"track_id": track_id, "samples": len(samples), "same_team": same_team, "continuity": continuity}))
        tracks[role] = samples
    if any(condition["status"] == "unverifiable" for condition in conditions):
        return _finish(conditions)

    if tactic == "overlap_run":
        carrier, runner = tracks["carrier_track_id"], tracks["runner_track_id"]
        behind_start = _progress(runner[0]["x"], direction) < _progress(carrier[0]["x"], direction)
        ahead_end = _progress(runner[-1]["x"], direction) > _progress(carrier[-1]["x"], direction)
        outside = abs(runner[-1]["y"]) > abs(carrier[-1]["y"]) + rules[tactic]["min_lateral_separation"]
        longitudinal_change = _progress(runner[-1]["x"], direction) - _progress(runner[0]["x"], direction)
        conditions.extend([
            _condition("starts_behind_carrier", "passed" if behind_start else "failed", value=behind_start, timestamp=runner[0]["t"]),
            _condition("ends_ahead_carrier", "passed" if ahead_end else "failed", value=ahead_end, timestamp=runner[-1]["t"]),
            _condition("outside_channel", "passed" if outside else "failed", value=outside, timestamp=runner[-1]["t"]),
            _condition("runner_progress", "passed" if longitudinal_change >= rules[tactic]["min_longitudinal_change"] else "failed", value=longitudinal_change),
        ])
    elif tactic == "run_in_behind":
        runner = tracks["runner_track_id"]
        opponents = [_samples(track, start, end) for track in state["measurements"]["tracks"].values()
                     if track.get("team_id") != team_id and track.get("role") == "player"]
        opponents = [samples for samples in opponents if len(samples) >= 2]
        if not opponents:
            conditions.append(_condition("visible_defensive_line", "unverifiable"))
        else:
            line0 = max(_progress(samples[0]["x"], direction) for samples in opponents)
            line1 = max(_progress(samples[-1]["x"], direction) for samples in opponents)
            crossed = _progress(runner[0]["x"], direction) <= line0 and _progress(runner[-1]["x"], direction) > line1
            moved = _progress(runner[-1]["x"], direction) > _progress(runner[0]["x"], direction) + rules[tactic]["min_line_crossing"]
            conditions.append(_condition("runner_crosses_line", "passed" if crossed and moved else "failed", value={"crossed": crossed, "self_moved": moved}))
    elif tactic in {"line_break", "cutback"}:
        passer, receiver = tracks["passer_track_id"], tracks["receiver_track_id"]
        ball_start, ball_end = ball[0], ball[-1]
        if tactic == "cutback":
            near_bottom = _progress(passer[0]["x"], direction) >= rules[tactic]["bottom_zone_x"]
            backward = _progress(ball_end["x"], direction) < _progress(ball_start["x"], direction) - rules[tactic]["min_backward_change"]
            conditions.extend([_condition("bottom_zone", "passed" if near_bottom else "failed", value=near_bottom),
                               _condition("backward_ball_path", "passed" if backward else "failed", value=backward)])
        else:
            opponents = [_samples(track, start, end) for track in state["measurements"]["tracks"].values()
                         if track.get("team_id") != team_id and track.get("role") == "player"]
            opponents = [samples for samples in opponents if len(samples) >= 2]
            if not opponents:
                conditions.append(_condition("visible_defensive_line", "unverifiable"))
            else:
                line = sum(_progress(samples[0]["x"], direction) for samples in opponents) / len(opponents)
                ball_change = _progress(ball_end["x"], direction) - _progress(ball_start["x"], direction)
                crossed = (_progress(ball_start["x"], direction) <= line < _progress(ball_end["x"], direction)
                           and ball_change >= rules[tactic]["min_line_crossing"])
                controlled = math.dist((ball_end["x"], ball_end["y"]), (receiver[-1]["x"], receiver[-1]["y"])) <= rules[tactic]["max_receive_distance"]
                conditions.extend([_condition("ball_crosses_line", "passed" if crossed else "failed", value=crossed),
                                   _condition("controlled_reception", "passed" if controlled else "failed", value=controlled)])
    else:
        carrier, runner = tracks["carrier_track_id"], tracks["runner_track_id"]
        forward = _progress(ball[-1]["x"], direction) - _progress(ball[0]["x"], direction)
        runner_forward = _progress(runner[-1]["x"], direction) > _progress(runner[0]["x"], direction)
        duration = end - start
        transitions = state.get("measurements", {}).get("possession_transitions", [])
        matching = [event for event in transitions if start <= float(event.get("t", -1)) <= end and event.get("to_team_id") == team_id]
        transition_status = "passed" if matching else "failed" if transitions else "unverifiable"
        conditions.extend([
            _condition("observable_possession_transition", transition_status, value=matching[0] if matching else None),
            _condition("immediate_forward_progress", "passed" if forward >= rules[tactic]["min_forward_change"] and duration <= rules[tactic]["max_transition_seconds"] else "failed", value=forward),
            _condition("supporting_run", "passed" if runner_forward else "failed", value=runner_forward),
        ])
    return _finish(conditions)


def _normalize_claimed_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate.get("claimed_evidence")
    raw = raw if isinstance(raw, dict) else {}
    allowed = {
        "restart_at_origin": RESTART_KINDS,
        "regain": CONTROLLED_REGAIN_KINDS,
        "delivery_kind": DELIVERY_KINDS,
        "corner_landing_zone": LANDING_ZONES,
    }
    normalized = {}
    for field, choices in allowed.items():
        value = raw.get(field)
        if value is not None and (not isinstance(value, str) or value not in choices):
            LOGGER.warning("invalid claimed_evidence %s=%r; using null", field, value)
            value = None
        normalized[field] = value
    chaos = raw.get("chaos")
    if chaos is not None and not isinstance(chaos, bool):
        LOGGER.warning("invalid claimed_evidence chaos=%r; using null", chaos)
        chaos = None
    normalized["chaos"] = chaos
    return normalized


def parse_claimed_evidence(clip_uid: str, candidate: dict[str, Any]) -> ClipEvidence:
    claim = _normalize_claimed_evidence(candidate)
    start = float(candidate["window"]["start_s"])
    events = []
    if claim["restart_at_origin"] is not None:
        events.append(PossessionEvent(start, "attacking", claim["restart_at_origin"]))
    if claim["regain"] is not None:
        events.append(PossessionEvent(start, "attacking", claim["regain"]))
    return ClipEvidence(
        clip_uid, candidate["tactic_id"], events, claim["delivery_kind"],
        claim["corner_landing_zone"], "vlm_claim",
    )


def should_call_sol(observations: list[dict[str, Any]], trajectory_tiers: list[str]) -> bool:
    return (
        len(observations) >= 2
        and "candidate_or_decision_disagreement" in skeptic_challenge(observations)["reasons"]
        and "insufficient" in trajectory_tiers
    )


def _claimed_checker_results(candidate: dict[str, Any], evidence: ClipEvidence) -> list[CheckResult]:
    claim = _normalize_claimed_evidence(candidate)
    kept = []
    for result in run_checkers(evidence, candidate["window"]):
        if result.checker == "restart_gate":
            if claim["restart_at_origin"] is None:
                continue
            if claim["regain"] is not None:
                result = CheckResult("restart_gate", "insufficient", "restart followed by claimed regain")
        elif result.checker == "clean_regain":
            if claim["regain"] is None:
                continue
            if claim["chaos"] is True:
                result = CheckResult("clean_regain", "insufficient", "chaos level not established")
        elif result.checker == "delivery_gate" and claim["delivery_kind"] is None:
            continue
        elif result.checker == "corner_landing" and claim["corner_landing_zone"] is None:
            continue
        kept.append(result)
    return kept


def evaluate_candidates(clip_uid: str, observations: list[dict[str, Any]],
                        prototype_contexts: dict[str, list[dict[str, Any]]],
                        trajectory_clip: Any, sol_call: Any = None) -> dict[str, Any]:
    records, trajectory_tiers = [], []
    for side, observation in enumerate(observations):
        for candidate in observation.get("candidates", []):
            candidate["claimed_evidence"] = _normalize_claimed_evidence(candidate)
            trajectory_evidence = (
                build_claim_evidence(trajectory_clip, clip_uid, candidate["tactic_id"], candidate["window"])
                if trajectory_clip is not None else None
            )
            trajectory_results = (
                run_checkers(trajectory_evidence, candidate["window"]) if trajectory_evidence else []
            )
            vlm_evidence = parse_claimed_evidence(clip_uid, candidate)
            vlm_results = _claimed_checker_results(candidate, vlm_evidence)
            trajectory_tier = decide(candidate["tactic_id"], trajectory_results, []).tier
            trajectory_tiers.append(trajectory_tier)
            records.append({
                "side": side, "candidate": candidate,
                "trajectory_evidence": trajectory_evidence,
                "trajectory_results": trajectory_results,
                "vlm_evidence": vlm_evidence, "vlm_results": vlm_results,
                "sol_unobservable": False,
            })

    arbitration = {"triggered": False}
    if should_call_sol(observations, trajectory_tiers):
        if sol_call is None:
            arbitration = {"triggered": True, "status": "pending_configuration"}
        else:
            payload = {
                "candidates_A": observations[0].get("candidates", []),
                "candidates_B": observations[1].get("candidates", []),
                "checker_results": [{
                    "side": record["side"], "tactic_id": record["candidate"]["tactic_id"],
                    "trajectory": [result.__dict__ for result in record["trajectory_results"]],
                    "vlm_claim": [result.__dict__ for result in record["vlm_results"]],
                } for record in records],
                "prototype_context": prototype_contexts,
            }
            prompt = (
                "Choose exactly one of support_A, support_B, neither, unobservable. "
                "Return JSON {choice, decisive_condition}. Do not add tactics or evidence.\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            answer = sol_call(prompt)
            choice = answer.get("choice") if isinstance(answer, dict) else None
            condition = answer.get("decisive_condition") if isinstance(answer, dict) else None
            if choice not in {"support_A", "support_B", "neither", "unobservable"} or not condition:
                raise ValueError("Sol returned invalid bounded arbitration")
            arbitration = {"triggered": True, "status": "completed", **answer}
            selected = {"support_A": 0, "support_B": 1}.get(choice)
            for record in records:
                if choice == "unobservable":
                    record["sol_unobservable"] = True
                elif record["side"] == selected and not any(
                        result.verdict == "veto" for result in record["vlm_results"]):
                    record["vlm_results"] = [
                        result for result in record["vlm_results"] if result.passed
                    ] + [CheckResult("sol", "pass", condition)]

    for record in records:
        candidate = record["candidate"]
        verdict = decide(
            candidate["tactic_id"], record["trajectory_results"], record["vlm_results"],
            candidate.get("decision") != "unobservable" and not record["sol_unobservable"],
        )
        candidate["evidence"] = {
            "trajectory": (record["trajectory_evidence"].to_dict()
                           if record["trajectory_evidence"] else None),
            "vlm_claim": record["vlm_evidence"].to_dict(),
        }
        candidate["checker_results"] = {
            "trajectory": [result.__dict__ for result in record["trajectory_results"]],
            "vlm_claim": [result.__dict__ for result in record["vlm_results"]],
        }
        candidate["verdict"] = verdict.__dict__
    return arbitration


def _observer_prompt(clip_uid: str) -> str:
    cards = [{"tactic_id": tactic, **CONCEPT_CARDS[tactic]} for tactic in sorted(SYSTEM_ARM_TACTICS)]
    schema = {"clip_uid": clip_uid, "observer": "provider", "candidates": [{
        "tactic_id": "one_of_the_cards", "window": {"start_s": 0.0, "end_s": 0.0},
        "decision": "positive|targeted_negative|unobservable", "positive_evidence": [],
        "exclusion_checks": [], "missing_evidence": [], "confidence": "low|medium|high",
        "claimed_evidence": {
            "restart_at_origin": sorted(RESTART_KINDS) + [None],
            "regain": sorted(CONTROLLED_REGAIN_KINDS) + [None], "chaos": "bool|null",
            "delivery_kind": sorted(DELIVERY_KINDS) + [None],
            "corner_landing_zone": sorted(LANDING_ZONES) + [None],
        }}]}
    return (
        "Independently inspect the timestamped football contact sheets. Do not infer intent. "
        "Return JSON only, nominate at most three candidates, and abstain when required evidence is not visible. Fill claimed_evidence only from direct observation; otherwise use null.\n"
        f"clip_uid={clip_uid}\nconcept_cards={json.dumps(cards, ensure_ascii=False)}\n"
        f"output_schema={json.dumps(schema, ensure_ascii=False)}"
    )


def _validate_observation(payload: dict[str, Any], clip_uid: str, observer: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError(f"{observer} returned invalid observation")
    if len(payload["candidates"]) > 3:
        raise ValueError(f"{observer} returned more than three candidates")
    payload["clip_uid"], payload["observer"] = clip_uid, observer
    kept = [candidate for candidate in payload["candidates"]
            if candidate.get("tactic_id") in SYSTEM_ARM_TACTICS]
    dropped = len(payload["candidates"]) - len(kept)
    if dropped:
        LOGGER.info("dropped %d out-of-scope tactic nominations for %s", dropped, clip_uid)
    payload["candidates"] = kept
    for candidate in kept:
        candidate["claimed_evidence"] = _normalize_claimed_evidence(candidate)
    return payload


def run_observers(clip_uid: str, image_paths: list[Path], providers: list[str], provider_call: Any,
                  allow_external_upload: bool) -> list[dict[str, Any]]:
    if not allow_external_upload:
        raise PermissionError("external upload requires --allow-external-upload")
    prompt = _observer_prompt(clip_uid)
    return [_validate_observation(provider_call(observer, prompt, image_paths), clip_uid, observer)
            for observer in providers]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    return json.loads(text)


def _image_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def live_provider_call(observer: str, prompt: str, image_paths: list[Path]) -> dict[str, Any]:
    if observer == "doubao":
        from pipeline.stage2b.generate import ark_chat
        usage: list[dict[str, Any]] = []
        payload = _extract_json(ark_chat(
            prompt, image_paths=[str(path) for path in image_paths], temperature=0.0,
            usage_callback=usage.append,
        ))
        payload["api_usage"] = usage
        return payload
    if observer != "gemini":
        raise ValueError(f"unknown provider: {observer}")
    import os
    from openai import OpenAI
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": _image_data_url(path)}} for path in image_paths)
    client = OpenAI(api_key=key, base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    response = client.chat.completions.create(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        messages=[{"role": "user", "content": content}], temperature=0)
    payload = _extract_json(response.choices[0].message.content or "")
    payload["api_usage"] = ([response.usage.model_dump(exclude_none=True)] if response.usage else [])
    return payload


def live_sol_call(prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ["SOL_MODEL"],
        messages=[{"role": "user", "content": prompt}], temperature=0,
    )
    return _extract_json(response.choices[0].message.content or "")


def make_contact_sheets(video_path: Path | str, out_dir: Path | str,
                        window: dict[str, float] | None = None,
                        profile: str = "640") -> list[Path]:
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if profile not in CONTACT_SHEET_PROFILES:
        raise ValueError(f"unknown contact-sheet profile: {profile}")
    tile_width, split = CONTACT_SHEET_PROFILES[profile]
    duration = _video_duration(video_path)
    if window:
        start = max(0.0, float(window["start_s"]))
        end = min(duration, float(window["end_s"]))
        count, grid, name = 16, "4x4", "candidate"
        prefix = f"trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS+{start:.3f}/TB,"
    else:
        start, end, count, grid, name, prefix = 0.0, duration, 12, "4x3", "overview", ""
    span = end - start
    if span <= 0:
        raise ValueError("empty contact-sheet window")
    if split:
        grid, frames = "2x2", count // 4
        sheet = out_dir / f"{name}-%02d.jpg"
        paths = [out_dir / f"{name}-{index:02d}.jpg" for index in range(1, frames + 1)]
        scale = "1920:1080"
    else:
        frames = 1
        sheet = out_dir / f"{name}.jpg"
        paths = [sheet]
        scale = f"{tile_width}:-2"
    fps = max(count / span, 0.01)
    font_size = round(22 * tile_width / 640)
    filters = (prefix + f"fps={fps:.6f},scale={scale}," +
               f"drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize={font_size}:fontcolor=white:box=1:boxcolor=black@0.6," +
               f"tile={grid}:padding=4:margin=4")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                    "-vf", filters, "-q:v", "3", "-frames:v", str(frames), "-y", str(sheet)], check=True)
    return paths



def should_call_reviewer(primary: dict[str, Any], clip_uid: str, sample_rate: float = 0.10) -> bool:
    candidates = primary.get("candidates", [])
    if candidates or primary.get("abstained"):
        return True
    threshold = int(sample_rate * 10_000)
    draw = int(hashlib.sha256(clip_uid.encode()).hexdigest()[:8], 16) % 10_000
    return draw < threshold


def run_selective_observers(clip_uid: str, image_paths: list[Path], provider_call: Any,
                            allow_external_upload: bool) -> list[dict[str, Any]]:
    primary = run_observers(clip_uid, image_paths, ["doubao"], provider_call, allow_external_upload)[0]
    results = [primary]
    if should_call_reviewer(primary, clip_uid):
        results.extend(run_observers(clip_uid, image_paths, ["gemini"], provider_call, allow_external_upload))
    return results


def skeptic_challenge(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(observations) < 2:
        return {"triggered": False, "reasons": []}
    signatures = [{(item["tactic_id"], item["decision"]) for item in row.get("candidates", [])}
                  for row in observations]
    reasons = []
    if signatures[0] != signatures[1]:
        reasons.append("candidate_or_decision_disagreement")
    windows = {item["tactic_id"]: item.get("window", {}) for item in observations[0].get("candidates", [])}
    for item in observations[1].get("candidates", []):
        other = windows.get(item["tactic_id"])
        if not other:
            continue
        overlap = max(0.0, min(float(other.get("end_s", 0)), float(item["window"].get("end_s", 0))) -
                      max(float(other.get("start_s", 0)), float(item["window"].get("start_s", 0))))
        union = (max(float(other.get("end_s", 0)), float(item["window"].get("end_s", 0))) -
                 min(float(other.get("start_s", 0)), float(item["window"].get("start_s", 0))))
        if union and overlap / union < 0.3:
            reasons.append(f"low_window_overlap:{item['tactic_id']}")
    return {"triggered": bool(reasons), "reasons": sorted(set(reasons)),
            "instruction": "Challenge only the relevant exclusion conditions once." if reasons else ""}


def prototype_context(root: Path | str, tactic_ids: list[str], exclude_source_group: str | None) -> tuple[list[dict[str, Any]], int]:
    context, excluded = [], 0
    episodes = load_jsonl(Path(root) / "episodes.jsonl")
    for tactic in sorted(set(tactic_ids)):
        stored_tactic = "corner_near_far_post" if tactic == "corner-near-far-post" else tactic
        excluded += sum(
            row["tactic_id"] == stored_tactic and row["review_status"] == "gold"
            and row["source"]["source_group_id"] == exclude_source_group
            for row in episodes
        )
        result = retrieve(root, stored_tactic, exclude_source_group=exclude_source_group)
        for kind, rows in result.items():
            for row in rows:
                context.append({"tactic_id": tactic, "prototype_type": kind, "phase": row["source"].get("phase"),
                                "evidence_summary": row["evidence_summary"], "failed_conditions": row["failed_conditions"],
                                "missing_evidence": row["missing_evidence"]})
    return context, excluded

def _run_rag_observer(root: Path, clip_uid: str, source_group: str, observer: str,
                      video_path: Path, out_dir: Path, image_paths: list[Path],
                      allow_external_upload: bool, contact_sheet_profile: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    initial = run_observers(clip_uid, image_paths, [observer], live_provider_call, allow_external_upload)[0]
    tactic_ids = [item["tactic_id"] for item in initial["candidates"]]
    context, excluded = prototype_context(root, tactic_ids, source_group)
    if not context:
        return initial, context, excluded
    starts = [float(item["window"]["start_s"]) for item in initial["candidates"]]
    ends = [float(item["window"]["end_s"]) for item in initial["candidates"]]
    dense = make_contact_sheets(
        video_path, out_dir / observer, {"start_s": min(starts), "end_s": max(ends)},
        contact_sheet_profile,
    )
    prompt = (_observer_prompt(clip_uid) + "\nYour own first-pass nominations=" +
              json.dumps(initial["candidates"], ensure_ascii=False) +
              "\nSource-disjoint prototype context=" + json.dumps(context, ensure_ascii=False) +
              "\nRe-evaluate only your nominations against these prototypes and the dense candidate-window sheet. Return the output schema JSON only.")
    refined = _validate_observation(live_provider_call(observer, prompt, image_paths + dense), clip_uid, observer)
    refined["api_usage"] = initial.get("api_usage", []) + refined.get("api_usage", [])
    return refined, context, excluded


def run_agents(root: Path | str, clip_uid: str, providers: list[str], allow_external_upload: bool,
               contact_sheet_profile: str = "640") -> dict[str, Any]:
    root = Path(root)
    source = next((row for row in load_jsonl(root / "source_rows.jsonl") if row["clip_uid"] == clip_uid), None)
    if not source or not source.get("video_path"):
        raise ValueError(f"unknown or unmapped clip_uid: {clip_uid}")
    if not providers:
        raise ValueError("at least one provider is required")
    repo_root = root.resolve().parents[1]
    observations, contexts, excluded_counts = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="tactical-sheets-") as temp_dir:
        video_path, sheet_dir = repo_root / source["video_path"], Path(temp_dir)
        sheets = make_contact_sheets(video_path, sheet_dir, profile=contact_sheet_profile)
        first = providers[0]
        primary, contexts[first], excluded_counts[first] = _run_rag_observer(
            root, clip_uid, source["source_group_id"], first, video_path, sheet_dir, sheets,
            allow_external_upload, contact_sheet_profile)
        observations.append(primary)
        for observer in providers[1:]:
            if observer != "gemini" or should_call_reviewer(primary, clip_uid):
                observation, contexts[observer], excluded_counts[observer] = _run_rag_observer(
                    root, clip_uid, source["source_group_id"], observer, video_path, sheet_dir, sheets,
                    allow_external_upload, contact_sheet_profile)
                observations.append(observation)
    predictions = repo_root / "outputs" / source["clip_id"] / "predictions.json"
    trajectory_clip = load_gsr(predictions) if predictions.is_file() else None
    sol_call = (live_sol_call
                if os.environ.get("OPENAI_API_KEY") and os.environ.get("SOL_MODEL") else None)
    sol_arbitration = evaluate_candidates(
        clip_uid, observations, contexts, trajectory_clip, sol_call,
    )
    result = {"schema_version": "tactical-agent-run-v1", "clip_uid": clip_uid,
              "contact_sheet_profile": contact_sheet_profile,
              "observations": observations, "prototype_context": contexts,
              "prototype_excluded_counts": excluded_counts,
              "sol_arbitration": sol_arbitration,
              "skeptic": skeptic_challenge(observations)}
    runs = root / "agent_runs"
    runs.mkdir(exist_ok=True)
    _write_json(runs / f"{clip_uid.replace(':', '__')}.json", result)
    return result


def export_sol_bundle(root: Path | str, out: Path | str) -> dict[str, Any]:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    disputes = []
    run_dir = root / "agent_runs"
    for path in sorted(run_dir.glob("*.json")) if run_dir.is_dir() else []:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("sol_arbitration", {}).get("triggered"):
            disputes.append(run)
    bundle = {"schema_version": "tactical-sol-review-v1", "disputes": disputes,
              "review_instructions": "Review exclusions once; never promote to gold without a human label."}
    _write_json(out / "disputes.json", bundle)
    return bundle


def import_sol_review(root: Path | str, review_path: Path | str) -> int:
    root, review_path = Path(root), Path(review_path)
    decisions = json.loads(review_path.read_text(encoding="utf-8")).get("decisions", [])
    ledger = root / "sol_reviews.jsonl"
    existing = load_jsonl(ledger) if ledger.is_file() else []
    merged = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in existing + decisions}
    _write_jsonl(ledger, [merged[key] for key in sorted(merged)])
    return len(decisions)

ACTIVE_FIELDS = ("query_id", "clip_uid", "tactic_id", "window", "selection_reason", "human_label", "failed_condition", "notes")


def select_active(root: Path | str, budget: int = 50, seed: int = 20260716) -> list[dict[str, str]]:
    root = Path(root)
    episodes = load_jsonl(root / "episodes.jsonl")
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in episodes:
        if row["review_status"] == "silver":
            rows_by_key.setdefault((row["clip_uid"], row["tactic_id"]), row)
    disagreements: set[tuple[str, str]] = set()
    run_dir = root / "agent_runs"
    for path in sorted(run_dir.glob("*.json")) if run_dir.is_dir() else []:
        run = json.loads(path.read_text(encoding="utf-8"))
        observations = run.get("observations", [])
        if len(observations) < 2 or not skeptic_challenge(observations)["triggered"]:
            continue
        for observation in observations:
            disagreements.update((run["clip_uid"], item["tactic_id"]) for item in observation.get("candidates", []))
    rule_conflicts: set[tuple[str, str]] = set()
    verification_path = root / "verification_runs.jsonl"
    if verification_path.is_file():
        for row in load_jsonl(verification_path):
            if row.get("model_decision") == "positive" and row.get("rule_status") in {"failed", "unverifiable"}:
                rule_conflicts.add((row["clip_uid"], row["tactic_id"]))
    gold_counts = Counter(row["tactic_id"] for row in episodes if row["review_status"] == "gold")
    priorities = {key: random.Random(f"{seed}:{key[0]}:{key[1]}").random() for key in rows_by_key}
    buckets = [
        ("agent_disagreement", 15, [key for key in rows_by_key if key in disagreements]),
        ("model_rule_conflict", 10, [key for key in rows_by_key if key in rule_conflicts]),
        ("hard_negative_candidate", 10, [key for key, row in rows_by_key.items() if row["prototype_type"] == "targeted_negative"]),
        ("rare_or_low_retrieval_coverage", 10, sorted(rows_by_key, key=lambda key: (gold_counts[key[1]], priorities[key]))),
        ("deterministic_random_audit", 5, sorted(rows_by_key, key=priorities.get)),
    ]
    chosen: list[tuple[str, str]] = []
    reasons: dict[tuple[str, str], str] = {}
    group_counts, tactic_counts = Counter(), Counter()

    def take(candidates: list[tuple[str, str]], limit: int, reason: str) -> None:
        for key in sorted(set(candidates), key=lambda item: (priorities[item], item)):
            if len([value for value in chosen if reasons[value] == reason]) >= limit or len(chosen) >= budget:
                break
            if key in reasons:
                continue
            row = rows_by_key[key]
            group, tactic = row["source"]["source_group_id"], row["tactic_id"]
            if group_counts[group] >= 5 or tactic_counts[tactic] >= 15:
                continue
            chosen.append(key)
            reasons[key] = reason
            group_counts[group] += 1
            tactic_counts[tactic] += 1

    for reason, quota, candidates in buckets:
        take(candidates, quota, reason)
    while len(chosen) < budget:
        eligible = [key for key, row in rows_by_key.items() if key not in reasons
                    and group_counts[row["source"]["source_group_id"]] < 5
                    and tactic_counts[row["tactic_id"]] < 15]
        if not eligible:
            break
        key = min(eligible, key=lambda item: (
            group_counts[rows_by_key[item]["source"]["source_group_id"]],
            tactic_counts[rows_by_key[item]["tactic_id"]], priorities[item], item))
        row = rows_by_key[key]
        chosen.append(key)
        reasons[key] = "quota_backfill"
        group_counts[row["source"]["source_group_id"]] += 1
        tactic_counts[row["tactic_id"]] += 1
    return [{
        "query_id": hashlib.sha256(row["prototype_id"].encode()).hexdigest()[:16],
        "clip_uid": row["clip_uid"], "tactic_id": row["tactic_id"], "window": row["window"]["raw"],
        "selection_reason": reasons[key], "human_label": "", "failed_condition": "", "notes": "",
    } for key in chosen for row in [rows_by_key[key]]]


def import_agent_runs(root: Path | str, dataset_id: str | None = None) -> dict[str, int]:
    root = Path(root)
    sources = {row["clip_uid"]: row for row in load_jsonl(root / "source_rows.jsonl")}
    episodes = load_jsonl(root / "episodes.jsonl")
    by_id = {row["prototype_id"]: row for row in episodes}
    counts = Counter(runs=0, candidates=0, created=0, updated=0, unchanged=0, skipped_gold=0)
    run_dir = root / "agent_runs"

    for path in sorted(run_dir.glob("*.json")) if run_dir.is_dir() else []:
        run = json.loads(path.read_text(encoding="utf-8"))
        clip_uid = run.get("clip_uid", "")
        if dataset_id and not clip_uid.startswith(f"{dataset_id}:"):
            continue
        source = sources.get(clip_uid)
        if not source:
            raise ValueError(f"unknown clip_uid in agent run: {clip_uid}")
        counts["runs"] += 1
        candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for observation in run.get("observations", []):
            observer = str(observation.get("observer", "unknown"))
            for candidate in observation.get("candidates", []):
                tactic_id = candidate.get("tactic_id")
                decision = candidate.get("decision")
                if tactic_id not in MVP_TACTICS or decision not in {"positive", "targeted_negative", "unobservable"}:
                    raise ValueError(f"invalid candidate in {path}: {tactic_id}/{decision}")
                candidates[tactic_id].append((observer, candidate))

        for tactic_id, items in sorted(candidates.items()):
            observers = list(dict.fromkeys(observer for observer, _ in items))
            decisions = [str(candidate["decision"]) for _, candidate in items]
            primary = items[0][1]
            starts = [float(candidate["window"]["start_s"]) for _, candidate in items]
            ends = [float(candidate["window"]["end_s"]) for _, candidate in items]
            start, end = min(starts), max(ends)
            if start < 0 or end <= start:
                raise ValueError(f"invalid candidate window in {path}: {tactic_id}")
            positive = list(dict.fromkeys(
                str(value) for _, candidate in items for value in candidate.get("positive_evidence", [])
            ))
            exclusions = list(dict.fromkeys(
                str(value) for _, candidate in items for value in candidate.get("exclusion_checks", [])
            ))
            missing = list(dict.fromkeys(
                str(value) for _, candidate in items for value in candidate.get("missing_evidence", [])
            ))
            evidence = positive + exclusions + missing
            prototype_id = f"{source['dataset_id']}/{source['clip_id']}/agent/{tactic_id}"
            video_path = Path(source["video_path"]) if source.get("video_path") else None
            row = {
                "schema_version": SCHEMA_VERSION,
                "prototype_id": prototype_id,
                "clip_uid": clip_uid,
                "source": {
                    "dataset_id": source["dataset_id"],
                    "source_group_id": source["source_group_id"],
                    "clip_id": source["clip_id"],
                    "event_id": source.get("event_id"),
                    "video_path": source.get("video_path"),
                    "source_csv": source.get("source_csv"),
                    "source_row_number": source["source_row_number"],
                    "phase": source.get("row", {}).get("event_type") or _source_phase(None, video_path),
                },
                "window": {"raw": f"{start:.3f}s–{end:.3f}s", "start_s": start, "end_s": end},
                "tactic_id": tactic_id,
                "source_tactic_name": CONCEPT_CARDS[tactic_id]["name_zh"],
                "prototype_type": primary["decision"],
                "review_status": "silver",
                "confidence": primary.get("confidence", "unknown"),
                "evidence_summary": "；".join(evidence) or f"{','.join(observers)} nominated {tactic_id}",
                "positive_evidence": positive,
                "failed_conditions": exclusions if primary["decision"] == "targeted_negative" else [],
                "missing_evidence": missing if primary["decision"] == "unobservable" else [],
                "team_id": None,
                "actor_bindings": {},
                "observability": {"video": "direct", "tracking": "unknown", "ball": "unknown", "calibration": "unknown"},
                "provenance": {
                    "migration_version": MIGRATION_VERSION,
                    "agent_run": {
                        "file": path.relative_to(root).as_posix(),
                        "observers": observers,
                        "decisions": decisions,
                        "disagreement": bool(run.get("skeptic", {}).get("triggered")) or len(set(decisions)) > 1,
                    },
                },
            }
            counts["candidates"] += 1
            existing = by_id.get(prototype_id)
            if existing and existing["review_status"] == "gold":
                counts["skipped_gold"] += 1
            elif existing == row:
                counts["unchanged"] += 1
            elif existing:
                episodes[episodes.index(existing)] = row
                by_id[prototype_id] = row
                counts["updated"] += 1
            else:
                episodes.append(row)
                by_id[prototype_id] = row
                counts["created"] += 1

    episodes.sort(key=lambda row: row["prototype_id"])
    _write_jsonl(root / "episodes.jsonl", episodes)
    _write_evaluation_report(root, episodes)
    report_path = root / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["post_review_status_counts"] = dict(sorted(Counter(row["review_status"] for row in episodes).items()))
    report["post_review_type_counts"] = dict(sorted(Counter(row["prototype_type"] for row in episodes).items()))
    _write_json(report_path, report)
    return dict(counts)


def import_reviews(root: Path | str, reviews: Path | str) -> dict[str, int]:
    root, reviews = Path(root), Path(reviews)
    allowed = {"positive", "targeted_negative", "unobservable", "outside_taxonomy", "needs_adjudication"}
    episodes = load_jsonl(root / "episodes.jsonl")
    by_query = {hashlib.sha256(row["prototype_id"].encode()).hexdigest()[:16]: row for row in episodes}
    counts = Counter()
    outside = []
    with reviews.open(encoding="utf-8-sig", newline="") as handle:
        for review in csv.DictReader(handle):
            label = review["human_label"].strip()
            if not label:
                continue
            if label not in allowed:
                raise ValueError(f"invalid human_label: {label}")
            row = by_query.get(review["query_id"])
            if not row:
                raise ValueError(f"unknown query_id: {review['query_id']}")
            if label in {"positive", "targeted_negative", "unobservable"}:
                row["prototype_type"], row["review_status"] = label, "gold"
                row["provenance"]["human_review"] = {"failed_condition": review["failed_condition"], "notes": review["notes"]}
            elif label == "outside_taxonomy":
                outside.append(review)
            counts[label] += 1
    episodes.sort(key=lambda row: row["prototype_id"])
    _write_jsonl(root / "episodes.jsonl", episodes)
    _write_evaluation_report(root, episodes)
    report_path = root / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["post_review_status_counts"] = dict(sorted(Counter(row["review_status"] for row in episodes).items()))
    report["post_review_type_counts"] = dict(sorted(Counter(row["prototype_type"] for row in episodes).items()))
    _write_json(report_path, report)
    if outside:
        ledger = root / "outside_taxonomy.jsonl"
        existing = load_jsonl(ledger) if ledger.is_file() else []
        merged = {row["query_id"]: row for row in existing + outside}
        _write_jsonl(ledger, [merged[key] for key in sorted(merged)])
    return dict(sorted(counts.items()))


def stats(root: Path | str) -> dict[str, Any]:
    rows = load_jsonl(Path(root) / "episodes.jsonl")
    return {
        "episodes": len(rows),
        "by_type": dict(sorted(Counter(row["prototype_type"] for row in rows).items())),
        "by_review_status": dict(sorted(Counter(row["review_status"] for row in rows).items())),
        "mvp": {tactic: dict(sorted(Counter(row["prototype_type"] for row in rows if row["tactic_id"] == tactic).items())) for tactic in sorted(MVP_TACTICS)},
    }
