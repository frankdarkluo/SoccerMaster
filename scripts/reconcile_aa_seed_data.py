#!/usr/bin/env python3
"""Canonicalize SoccerNet-AA seed rows and inventory unread clean clips."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from copy import copy
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.styles import Font


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.annotate_soccernet_aa_tactics import FPS, frame_range, sha256
from scripts.select_aa_clips import (
    SAMPLE_HZ,
    exact_five_second_alignment,
    read_cache,
)


DEFAULT_WORKBOOK = ROOT / "data/战术标签标注.xlsx"
DEFAULT_SOURCE_WORKBOOK = ROOT / "data/annotations/多模态模型战术识别.xlsx"
DEFAULT_CACHE = ROOT / "outputs/soccernet_aa_screening/clip_scan.jsonl"
DEFAULT_SELECTED = ROOT / "outputs/soccernet_aa_screening/aa_selected_clips.csv"
DATA_ROOT = ROOT / "data/SoccerNet-AA"
REVIEWS_ROOT = ROOT / "outputs/soccernet_aa_tactics/reviews"

SOURCE_SHEET = "AI建议seed data"
CANONICAL_SHEET = "AI建议seed canonical"
AUDIT_SHEET = "AA_seed映射审计"
COVERAGE_SHEET = "AA_clean读取覆盖"
QUEUE_SHEET = "AA_clean待读队列"
GENERATED_SHEETS = (CANONICAL_SHEET, AUDIT_SHEET, COVERAGE_SHEET, QUEUE_SHEET)

CLIP_FRAMES = 30 * FPS
ID_RE = re.compile(r"(train|valid)/AA-([1-9]\d*)")
INTERVAL_RE = re.compile(r"(\d{2}):(\d{2})")
TACTIC_IDS = {
    "back-line-possession-management": "back-line-possession-management",
    "后场倒脚控场": "back-line-possession-management",
    "counter-attack": "counter-attack",
    "long-ball": "long-ball",
    "line-breaking-pass": "line-breaking-pass",
    "run-in-behind": "run-in-behind",
    "tactical-foul": "tactical-foul",
    "one-two": "one-two",
    "cutback": "cutback",
}


def normalize_id(value: Any) -> str:
    clip_id = str(value).strip()
    clip_id = clip_id.replace("train-AA-", "train/AA-")
    clip_id = clip_id.replace("valid-AA-", "valid/AA-")
    if not ID_RE.fullmatch(clip_id):
        raise ValueError(f"invalid SoccerNet-AA id: {value!r}")
    return clip_id


def clip_sort_key(clip_id: str) -> tuple[int, int]:
    split, video_id = normalize_id(clip_id).split("/")
    return (0 if split == "train" else 1, int(video_id.split("-")[1]))


def normalize_tactic(value: Any) -> str:
    tactic = str(value).strip()
    if tactic not in TACTIC_IDS:
        raise ValueError(f"unsupported seed tactic: {tactic!r}")
    return TACTIC_IDS[tactic]


def parse_interval(value: Any) -> tuple[int, int]:
    parts = INTERVAL_RE.findall(str(value))
    if len(parts) != 2:
        raise ValueError(f"invalid interval: {value!r}")
    seconds = tuple(int(minutes) * 60 + int(secs) for minutes, secs in parts)
    if not 0 <= seconds[0] < seconds[1] <= 30:
        raise ValueError(f"interval outside 30-second clip: {value!r}")
    return seconds


def interval_iou(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> float:
    """IoU for inclusive frame intervals."""
    intersection = max(
        0, min(left_end, right_end) - max(left_start, right_start) + 1
    )
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union else 1.0


def choose_clean_mapping(
    source_id: str,
    start_frame: int,
    end_frame: int,
    clean_ids: set[str],
    alignments: list[dict[str, Any]],
) -> dict[str, Any]:
    usable = [item for item in alignments if item["clean_id"] in clean_ids]
    full = []
    for item in usable:
        shift_frames = round(item["shift_s"] * FPS)
        if 0 <= start_frame + shift_frames <= end_frame + shift_frames < CLIP_FRAMES:
            full.append(item)
    if len(full) > 1:
        raise ValueError(f"multiple clean clips fully contain {source_id} event")
    if full:
        selected = full[0]
        status = "already_clean" if selected["clean_id"] == source_id else "mapped_clean"
        return {
            "status": status,
            "target_id": selected["clean_id"],
            "shift_s": selected["shift_s"],
            "reason": "event_fully_contained",
        }

    event_start, event_end = start_frame, end_frame + 1
    covered = []
    for item in usable:
        shift_frames = round(item["shift_s"] * FPS)
        overlap_start = max(event_start, -shift_frames)
        overlap_end = min(event_end, CLIP_FRAMES - shift_frames)
        if overlap_start < overlap_end:
            covered.append((overlap_start, overlap_end, item["clean_id"]))
    if not covered:
        return {
            "status": "no_clean_overlap",
            "target_id": source_id,
            "shift_s": 0.0,
            "reason": "event_has_no_clean_content",
        }

    merged_start, merged_end, _ = min(covered)
    clean_count = 1
    for overlap_start, overlap_end, _ in sorted(covered)[1:]:
        if overlap_start <= merged_end:
            merged_end = max(merged_end, overlap_end)
            clean_count += 1
    if merged_start <= event_start and merged_end >= event_end and clean_count >= 2:
        status = "crosses_clean_boundary"
        reason = "event_spans_adjacent_clean_clips"
    else:
        status = "partial_clean_gap"
        reason = "clean_covers_only_part_of_event"
    return {
        "status": status,
        "target_id": source_id,
        "shift_s": 0.0,
        "reason": reason,
    }


def merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: min(row["source_rows"])):
        match = next(
            (
                row
                for row in merged
                if row["id"] == event["id"]
                and row["tactic"] == event["tactic"]
                and interval_iou(
                    row["start_frame"],
                    row["end_frame"],
                    event["start_frame"],
                    event["end_frame"],
                )
                >= 0.5
            ),
            None,
        )
        if match is None:
            merged.append(dict(event))
            continue
        if (match["tier"], match["evidence"]) != (event["tier"], event["evidence"]):
            raise ValueError("refusing to merge events with different provenance")
        match["start_frame"] = min(match["start_frame"], event["start_frame"])
        match["end_frame"] = max(match["end_frame"], event["end_frame"])
        match["source_rows"] = sorted(match["source_rows"] + event["source_rows"])
    return merged


def sheet_rows(worksheet) -> tuple[list[str], list[dict[str, Any]]]:
    headers = [cell.value for cell in worksheet[1]]
    if not headers or any(not isinstance(header, str) for header in headers):
        raise ValueError(f"invalid headers in sheet {worksheet.title!r}")
    rows = []
    for row_number, values in enumerate(
        worksheet.iter_rows(min_row=2, values_only=True), start=2
    ):
        if values[0] is None:
            continue
        rows.append({
            **dict(zip(headers, values)),
            "_source_row": row_number,
        })
    return headers, rows


def sheet_content_sha256(worksheet) -> str:
    values = [
        list(row)
        for row in worksheet.iter_rows(values_only=True)
        if any(value is not None for value in row)
    ]
    semantic_values = [
        [int(value) if isinstance(value, float) and value.is_integer() else value for value in row]
        for row in values
    ]
    raw = json.dumps(semantic_values, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def locate_video(data_root: Path, clip_id: str) -> Path:
    split, video_id = normalize_id(clip_id).split("/")
    candidates = [
        data_root / split / "clean" / f"{video_id}.mp4",
        data_root / split / "dup" / f"{video_id}.mp4",
        data_root / split / f"{video_id}.mp4",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"expected one current video path for {clip_id}, found {found}")
    return found[0]


def repository_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_reviews(reviews_root: Path) -> dict[str, dict[str, Any]]:
    reviews = {}
    for path in sorted(reviews_root.glob("*/*.json")):
        review = json.loads(path.read_text(encoding="utf-8"))
        if review.get("status") != "complete":
            continue
        clip_id = normalize_id(review["id"])
        if clip_id in reviews:
            raise ValueError(f"duplicate Codex review: {clip_id}")
        reviews[clip_id] = review
    return reviews


def model_ids(worksheet) -> set[str]:
    return {
        normalize_id(row[0])
        for row in worksheet.iter_rows(min_row=2, values_only=True)
        if row[0]
    }


def match_codex_event(seed: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    tactic_id = normalize_tactic(seed["tactic"])
    candidates = [
        tactic
        for tactic in review.get("tactics", [])
        if tactic.get("tactic_id") == tactic_id and tactic.get("verdict") == "present"
    ]
    if not candidates:
        raise ValueError(f"Codex review lacks seed tactic: {seed['id']} {tactic_id}")
    seed_start_s, seed_end_s = parse_interval(seed["interval"])
    seed_start, seed_end = seed_start_s * FPS, seed_end_s * FPS - 1
    ranked = sorted(
        candidates,
        key=lambda item: (
            interval_iou(
                seed_start,
                seed_end,
                int(item["start_frame"]),
                int(item["end_frame"]),
            ),
            -int(item["start_frame"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    if interval_iou(
        seed_start,
        seed_end,
        int(best["start_frame"]),
        int(best["end_frame"]),
    ) == 0:
        raise ValueError(f"Codex occurrence does not overlap seed: {seed['id']} {tactic_id}")
    return best


def clean_ids_from_csv(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            normalize_id(row["clip_id"])
            for row in csv.DictReader(handle)
            if row["split"] in {"train", "valid"}
        }


def actual_clean_ids(data_root: Path) -> set[str]:
    return {
        f"{split}/{path.stem}"
        for split in ("train", "valid")
        for path in (data_root / split / "clean").glob("AA-*.mp4")
    }


def cache_values(cache: Path) -> dict[str, list[int]]:
    rows = read_cache(cache)
    return {
        normalize_id(row["clip_id"]): [int(value, 16) for value in row["dhash"]]
        for row in rows
        if row.get("split") in {"train", "valid"}
    }


def build_clean_index(
    hashes: dict[str, list[int]], clean_ids: set[str]
) -> dict[int, set[str]]:
    index: dict[int, set[str]] = collections.defaultdict(set)
    for clip_id in clean_ids:
        for value in set(hashes[clip_id]):
            index[value].add(clip_id)
    return index


def clean_alignments(
    source_id: str,
    hashes: dict[str, list[int]],
    clean_index: dict[int, set[str]],
) -> list[dict[str, Any]]:
    split = source_id.split("/", 1)[0]
    candidates = {
        clean_id
        for value in set(hashes[source_id])
        for clean_id in clean_index.get(value, set())
        if clean_id.startswith(f"{split}/")
    }
    alignments = []
    for clean_id in sorted(candidates, key=clip_sort_key):
        result = exact_five_second_alignment(hashes[source_id], hashes[clean_id])
        if result is not None:
            alignments.append({"clean_id": clean_id, **result})
    return alignments


def copy_cell_style(source, target) -> None:
    if source.has_style:
        target._style = copy(source._style)
    target.number_format = source.number_format
    target.alignment = copy(source.alignment)
    target.protection = copy(source.protection)


def replace_sheet(workbook, name: str, *, index: int | None = None):
    if name in workbook.sheetnames:
        workbook.remove(workbook[name])
    return workbook.create_sheet(name, index)


def write_canonical_sheet(
    workbook,
    source,
    headers: list[str],
    rows: list[dict[str, Any]],
) -> None:
    sheet = replace_sheet(
        workbook, CANONICAL_SHEET, index=workbook.index(source) + 1
    )
    sheet.freeze_panes = source.freeze_panes
    for column_letter, dimension in source.column_dimensions.items():
        sheet.column_dimensions[column_letter].width = dimension.width
    for column, header in enumerate(headers, start=1):
        sheet.cell(1, column, header)
        copy_cell_style(source.cell(1, column), sheet.cell(1, column))
    for output_row, row in enumerate(rows, start=2):
        source_row = row["_source_row"]
        for column, header in enumerate(headers, start=1):
            sheet.cell(output_row, column, row.get(header))
            copy_cell_style(
                source.cell(source_row, column), sheet.cell(output_row, column)
            )


def write_plain_sheet(
    workbook,
    name: str,
    headers: list[str],
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    sheet = replace_sheet(workbook, name)
    row_number = 1
    if metadata:
        sheet.cell(row_number, 1, "metadata_key").font = Font(bold=True)
        sheet.cell(row_number, 2, "metadata_value").font = Font(bold=True)
        row_number += 1
        for key, value in metadata.items():
            sheet.cell(row_number, 1, key)
            sheet.cell(row_number, 2, value)
            row_number += 1
        row_number += 1
    header_row = row_number
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(header_row, column, header)
        cell.font = Font(bold=True)
    for row in rows:
        row_number += 1
        for column, header in enumerate(headers, start=1):
            sheet.cell(row_number, column, row.get(header))
    sheet.freeze_panes = f"A{header_row + 1}"
    for column in range(1, len(headers) + 1):
        values = [
            str(sheet.cell(row, column).value or "")
            for row in range(header_row, min(sheet.max_row, header_row + 200) + 1)
        ]
        sheet.column_dimensions[sheet.cell(header_row, column).column_letter].width = min(
            60, max(10, max(map(len, values), default=10) + 2)
        )


def save_atomic(workbook, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.xlsx")
    workbook.save(temporary)
    temporary.replace(path)


def reconcile(
    workbook_path: Path,
    source_workbook_path: Path,
    cache_path: Path,
    selected_path: Path,
    *,
    write: bool = False,
) -> dict[str, Any]:
    workbook = load_workbook(workbook_path)
    if SOURCE_SHEET not in workbook.sheetnames:
        raise ValueError(f"missing sheet: {SOURCE_SHEET}")
    seed_sheet = workbook[SOURCE_SHEET]
    seed_hash = sheet_content_sha256(seed_sheet)
    headers, seed_rows = sheet_rows(seed_sheet)
    gsr_rows = [row for row in seed_rows if row["source"] == "GSR-2025"]
    aa_rows = [row for row in seed_rows if row["source"] == "SoccerNet-AA"]
    if len(gsr_rows) != 45 or len(aa_rows) != 125:
        raise ValueError(
            f"unexpected seed composition: GSR={len(gsr_rows)} AA={len(aa_rows)}"
        )

    source_workbook = load_workbook(source_workbook_path, data_only=True)
    pro_ids = model_ids(source_workbook["5.6 Sol Pro AA"])
    codex_sheet_ids = model_ids(source_workbook["5.6 Sol Codex xhigh AA"])

    selected_clean = clean_ids_from_csv(selected_path)
    clean_ids = actual_clean_ids(DATA_ROOT)
    if selected_clean != clean_ids or len(clean_ids) != 828:
        raise ValueError("selected CSV and clean directories disagree")

    hashes = cache_values(cache_path)
    required_ids = clean_ids | {normalize_id(row["id"]) for row in aa_rows}
    if missing := required_ids - hashes.keys():
        raise ValueError(f"hash cache misses ids: {sorted(missing)[:10]}")
    clean_index = build_clean_index(hashes, clean_ids)
    reviews = load_reviews(REVIEWS_ROOT)
    file_hashes: dict[Path, str] = {}

    def current_sha(path: Path) -> str:
        if path not in file_hashes:
            file_hashes[path] = sha256(path)
        return file_hashes[path]

    events = []
    audit_rows = []
    mapping_counts: collections.Counter[str] = collections.Counter()
    for seed in aa_rows:
        source_id = normalize_id(seed["id"])
        source_path = locate_video(DATA_ROOT, source_id)
        review = reviews.get(source_id)
        if review is None:
            raise ValueError(f"missing Codex raw review for seed: {source_id}")
        if current_sha(source_path) != review.get("video_sha256"):
            raise ValueError(f"Codex review SHA mismatch: {source_id}")
        tactic = match_codex_event(seed, review)
        start_frame = int(tactic["start_frame"])
        end_frame = int(tactic["end_frame"])
        mapping = choose_clean_mapping(
            source_id,
            start_frame,
            end_frame,
            clean_ids,
            clean_alignments(source_id, hashes, clean_index),
        )
        mapping_counts[mapping["status"]] += 1
        target_id = mapping["target_id"]
        shift_frames = round(mapping["shift_s"] * FPS)
        new_start = start_frame + shift_frames
        new_end = end_frame + shift_frames
        target_path = locate_video(DATA_ROOT, target_id)
        event = {
            **{header: seed.get(header) for header in headers},
            "tactic": normalize_tactic(seed["tactic"]),
            "id": target_id,
            "video_path": repository_path(target_path),
            "interval": frame_range(new_start, new_end),
            "start_frame": new_start,
            "end_frame": new_end,
            "source_rows": [seed["_source_row"]],
            "_source_row": seed["_source_row"],
        }
        rejected = str(seed.get("人工核验结论") or "").strip().startswith("❌")
        audit = {
            "source_row": seed["_source_row"],
            "tier": seed["tier"],
            "tactic": normalize_tactic(seed["tactic"]),
            "old_id": source_id,
            "old_path": seed["video_path"],
            "source_actual_path": repository_path(source_path),
            "old_interval": seed["interval"],
            "codex_start_frame": start_frame,
            "codex_end_frame": end_frame,
            "codex_interval": frame_range(start_frame, end_frame),
            "new_id": target_id,
            "new_path": repository_path(target_path),
            "new_start_frame": new_start,
            "new_end_frame": new_end,
            "new_interval": frame_range(new_start, new_end),
            "shift_s": mapping["shift_s"],
            "mapping_status": mapping["status"],
            "mapping_reason": mapping["reason"],
            "timestamp_source": "Codex raw frames",
            "canonical_action": "excluded" if rejected else "",
            "merged_into": "",
            "exclusion_reason": "人工核验：❌ 无犯规" if rejected else "",
        }
        audit_rows.append(audit)
        if not rejected:
            events.append(event)

    merged_events = merge_events(events)
    audit_by_source = {row["source_row"]: row for row in audit_rows}
    for event in merged_events:
        event["interval"] = frame_range(event["start_frame"], event["end_frame"])
        key = (
            f"{event['id']}|{event['tactic']}|"
            f"{event['start_frame']}-{event['end_frame']}"
        )
        for position, source_row in enumerate(event["source_rows"]):
            audit = audit_by_source[source_row]
            audit["merged_into"] = key
            audit["canonical_action"] = (
                "kept"
                if len(event["source_rows"]) == 1
                else "merge_primary"
                if position == 0
                else "merged"
            )

    canonical_rows = sorted(
        [
            *gsr_rows,
            *[
                {**event, "_source_row": min(event["source_rows"])}
                for event in merged_events
            ],
        ],
        key=lambda row: row["_source_row"],
    )
    active_seed_counts = collections.Counter(
        event["id"] for event in merged_events if event["id"] in clean_ids
    )

    coverage_rows = []
    queue_rows = []
    for clip_id in sorted(clean_ids, key=clip_sort_key):
        path = locate_video(DATA_ROOT, clip_id)
        review = reviews.get(clip_id)
        raw_complete = review is not None
        raw_sha_match = (
            raw_complete and current_sha(path) == review.get("video_sha256")
        )
        pro_seen = clip_id in pro_ids
        codex_seen = clip_id in codex_sheet_ids
        directly_read = pro_seen or raw_sha_match
        coverage = {
            "id": clip_id,
            "split": clip_id.split("/", 1)[0],
            "video_path": repository_path(path),
            "pro_sheet_seen": pro_seen,
            "codex_sheet_seen": codex_seen,
            "codex_raw_complete": raw_complete,
            "codex_raw_sha_match": raw_sha_match,
            "directly_read": directly_read,
            "active_seed_events": active_seed_counts.get(clip_id, 0),
            "queue_status": "已读" if directly_read else "待读",
        }
        coverage_rows.append(coverage)
        if not directly_read:
            queue_rows.append({
                "queue_order": len(queue_rows) + 1,
                "id": clip_id,
                "split": coverage["split"],
                "video_path": coverage["video_path"],
                "reason": (
                    "Codex原始review SHA不匹配"
                    if raw_complete
                    else "无Pro记录且无SHA匹配Codex原始review"
                ),
            })

    tier_counts = collections.Counter(event["tier"] for event in merged_events)
    unread_by_split = collections.Counter(row["split"] for row in queue_rows)
    expected_mapping = {
        "already_clean": 14,
        "mapped_clean": 77,
        "crosses_clean_boundary": 28,
        "partial_clean_gap": 3,
        "no_clean_overlap": 3,
    }
    summary = {
        "seed_source": {
            "gsr_rows": len(gsr_rows),
            "aa_rows": len(aa_rows),
            "actual_rule": "59 dual-model agreement + 66 Codex high confidence",
        },
        "mapping": dict(sorted(mapping_counts.items())),
        "canonical": {
            "gsr_rows": len(gsr_rows),
            "aa_rows": len(merged_events),
            "total_rows": len(canonical_rows),
            "aa_tiers": dict(sorted(tier_counts.items())),
        },
        "clean_coverage": {
            "clean_total": len(coverage_rows),
            "directly_read": sum(row["directly_read"] for row in coverage_rows),
            "unread": len(queue_rows),
            "unread_by_split": dict(sorted(unread_by_split.items())),
        },
        "written": write,
    }
    if dict(mapping_counts) != expected_mapping:
        raise ValueError(f"mapping golden mismatch: {dict(mapping_counts)}")
    if (
        len(merged_events) != 122
        or tier_counts != {"A": 58, "B": 64}
        or len(canonical_rows) != 167
    ):
        raise ValueError(f"canonical golden mismatch: {summary['canonical']}")
    if (
        len(coverage_rows) != 828
        or len(queue_rows) != 676
        or unread_by_split != {"train": 633, "valid": 43}
    ):
        raise ValueError(f"coverage golden mismatch: {summary['clean_coverage']}")
    if any(
        row["mapping_status"] in {"already_clean", "mapped_clean"}
        and row["shift_s"] % 5
        for row in audit_rows
    ):
        raise ValueError("automatic mapping contains a non-5-second shift")

    for source_id, target_id, expected_shift in (
        ("valid/AA-1166", "valid/AA-271", 10.0),
        ("valid/AA-1024", "valid/AA-252", 5.0),
    ):
        alignment = exact_five_second_alignment(hashes[source_id], hashes[target_id])
        if alignment is None or alignment["shift_s"] != expected_shift:
            raise ValueError(f"known alignment changed: {source_id} -> {target_id}")
        reverse = exact_five_second_alignment(hashes[target_id], hashes[source_id])
        if reverse is None or reverse["shift_s"] != -expected_shift:
            raise ValueError(f"known reverse alignment changed: {target_id} -> {source_id}")

    if write:
        metadata = {
            "seed_sheet_sha256": seed_hash,
            "source_workbook_sha256": sha256(source_workbook_path),
            "clip_scan_sha256": sha256(cache_path),
            "selected_csv_sha256": sha256(selected_path),
            "timestamp_source": "Codex raw frames",
            "mapping_rule": "single clean clip fully contains event; 5s exact-dHash shift",
        }
        write_canonical_sheet(workbook, seed_sheet, headers, canonical_rows)
        audit_headers = [
            "source_row", "tier", "tactic", "old_id", "old_path",
            "source_actual_path", "old_interval", "codex_start_frame",
            "codex_end_frame", "codex_interval", "new_id", "new_path",
            "new_start_frame", "new_end_frame", "new_interval", "shift_s",
            "mapping_status", "mapping_reason", "timestamp_source",
            "canonical_action", "merged_into", "exclusion_reason",
        ]
        write_plain_sheet(
            workbook,
            AUDIT_SHEET,
            audit_headers,
            sorted(audit_rows, key=lambda row: row["source_row"]),
            metadata=metadata,
        )
        coverage_headers = [
            "id", "split", "video_path", "pro_sheet_seen",
            "codex_sheet_seen", "codex_raw_complete", "codex_raw_sha_match",
            "directly_read", "active_seed_events", "queue_status",
        ]
        write_plain_sheet(
            workbook, COVERAGE_SHEET, coverage_headers, coverage_rows
        )
        queue_headers = [
            "queue_order", "id", "split", "video_path", "reason",
        ]
        write_plain_sheet(workbook, QUEUE_SHEET, queue_headers, queue_rows)
        save_atomic(workbook, workbook_path)
        check = load_workbook(workbook_path, read_only=True)
        if sheet_content_sha256(check[SOURCE_SHEET]) != seed_hash:
            raise ValueError("original seed sheet changed during write")
        for sheet_name in GENERATED_SHEETS:
            if sheet_name not in check.sheetnames:
                raise ValueError(f"generated sheet missing after save: {sheet_name}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument(
        "--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace only the four generated sheets in the target workbook",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = reconcile(
        args.workbook,
        args.source_workbook,
        args.cache,
        args.selected,
        write=args.write,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
