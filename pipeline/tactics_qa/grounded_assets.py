"""Dependency-free import and validation for GroundedTactics candidate assets.

The source CSV/XLSX files are proposals, never gold.  This module preserves
their provenance and emits only explicitly provisional derived records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


SCHEMA_VERSION = "grounded-tactics-assets-v1"
CLAIM_SCHEMA_VERSION = "grounded-tactics-candidate-claim-v1"
QUESTION_SCHEMA_VERSION = "grounded-tactics-provisional-question-v1"
STRUCTURED_SCHEMA_VERSION = "grounded-tactics-structured-annotation-v1"
CRITERION_STATES = {"met", "violated", "not_observable"}
VERDICTS = {"true", "false", "insufficient", "none_of_ontology"}
TRACKS = {"T1_TEMPORAL", "T2_OBSERVATION", "T3_DIAGNOSIS", "T4_INTEGRATED"}
NONE_LABELS = {"无明确战术"}
CLIP_RE = re.compile(r"^(?:(2024|2025)-)?(SNGS-\d{3})$")
SPAN_RE = re.compile(r"^(\d{2}):(\d{2})\s*[–-]\s*(\d{2}):(\d{2})$")
ANSWER_RE = re.compile(r"^([ABCD])(?:\s|$|\()")
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def parse_span(raw: str, duration_s: float = 30.0) -> tuple[dict[str, float] | None, str]:
    """Parse a strict mm:ss interval without silently repairing source typos."""

    value = (raw or "").strip()
    if not value:
        return None, "missing"
    match = SPAN_RE.fullmatch(value)
    if not match:
        return None, "malformed"
    start_minute, start_second, end_minute, end_second = map(int, match.groups())
    if start_second >= 60 or end_second >= 60:
        return None, "malformed"
    start = float(start_minute * 60 + start_second)
    end = float(end_minute * 60 + end_second)
    span = {"start_s": start, "end_s": end}
    if start < 0 or start >= end or end > duration_s:
        return span, "out_of_bounds"
    return span, "valid"


def canonical_identity(clip_id: str) -> dict[str, Any]:
    match = CLIP_RE.fullmatch(clip_id)
    if not match:
        raise ValueError(f"invalid GSR clip id: {clip_id!r}")
    release = match.group(1) or "2024"
    source_clip_id = match.group(2)
    return {
        "clip_uid": f"gsr{release}:{source_clip_id}",
        "legacy_clip_uids": [f"soccernetgs:{source_clip_id}"],
        "content_group_id": f"soccernetgs-v1.3:{source_clip_id}",
    }


def _label_key(value: str) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def load_label_map(ontology_path: Path, alias_path: Path) -> dict[str, str]:
    """Build an exact alias map; no fuzzy matching is used."""

    aliases: dict[str, str] = {}

    def add(label: str, tactic_id: str) -> None:
        key = _label_key(label)
        if not key:
            return
        previous = aliases.get(key)
        if previous and previous != tactic_id:
            raise ValueError(f"ambiguous tactic alias {label!r}: {previous!r} vs {tactic_id!r}")
        aliases[key] = tactic_id

    with ontology_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id（英文标识）", "中文名", "英文名", "别名|同义词"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{ontology_path}: missing ontology columns {sorted(required)}")
        for row in reader:
            tactic_id = row["id（英文标识）"].strip()
            if not tactic_id or tactic_id.startswith("英文小写"):
                continue
            add(tactic_id, tactic_id)
            for field in ("中文名", "英文名"):
                label = row[field].strip()
                add(label, tactic_id)
                for part in re.split(r"\s+/\s+", label):
                    add(part, tactic_id)
            for label in re.split(r"[|\n]", row["别名|同义词"] or ""):
                add(label, tactic_id)

    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "grounded-tactics-label-aliases-v1":
        raise ValueError(f"{alias_path}: unsupported alias schema")
    for label, tactic_id in payload.get("aliases", {}).items():
        add(label, tactic_id)
    return aliases


def normalize_tactic(label: str, label_map: dict[str, str]) -> tuple[str, str | None]:
    value = (label or "").strip()
    if value in NONE_LABELS:
        return "no_explicit_tactic", None
    tactic_id = label_map.get(_label_key(value))
    return ("positive", tactic_id) if tactic_id else ("unmapped", None)


def _relative(path: Path, root: Path | None) -> str:
    if root:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _load_candidate_rows(
    raw_rows: Iterable[dict[str, str]],
    *,
    path: Path,
    source_id: str,
    source_type: str,
    label_map: dict[str, str],
    root: Path | None = None,
    duration_s: float = 30.0,
    release: str | None = None,
) -> list[dict[str, Any]]:
    if release not in {None, "2024", "2025"}:
        raise ValueError(f"unsupported GSR release: {release!r}")
    source_sha256 = sha256_file(path)
    source_rows = list(raw_rows)
    fieldnames = set(source_rows[0]) if source_rows else set()
    label_column = "战术tactics" if "战术tactics" in fieldnames else "tactics"
    required = {"id", label_column, "时间范围", "判断依据"}
    if not required.issubset(fieldnames):
        raise ValueError(f"{path}: missing candidate columns {sorted(required)}")

    rows: list[dict[str, Any]] = []
    for row_number, raw in enumerate(source_rows, 2):
        raw_clip_id = (raw.get("id") or "").strip()
        match = CLIP_RE.fullmatch(raw_clip_id)
        if not match:
            raise ValueError(f"invalid GSR clip id: {raw_clip_id!r}")
        source_release = match.group(1)
        if release and source_release and source_release != release:
            raise ValueError(
                f"clip release {source_release!r} conflicts with requested release {release!r}"
            )
        clip_id = f"{release}-{match.group(2)}" if release else raw_clip_id
        identity = canonical_identity(clip_id)
        label = (raw.get(label_column) or "").strip()
        label_kind, tactic_id = normalize_tactic(label, label_map)
        span, span_status = parse_span(raw.get("时间范围") or "", duration_s)
        rationale = (raw.get("判断依据") or "").strip()
        flags: list[str] = []
        if label_kind == "unmapped":
            flags.append("unmapped_tactic_label")
        if span_status != "valid":
            flags.append(f"span_{span_status}")
        if not rationale:
            flags.append("missing_rationale")
        if source_type == "model_weak_label":
            flags.append("model_weak_label")
        rows.append({
            "schema_version": CLAIM_SCHEMA_VERSION,
            "claim_id": f"candidate:{source_id}:{clip_id}:{row_number:04d}",
            "clip_id": clip_id,
            **identity,
            "claim_status": "candidate_not_gold",
            "label_kind": label_kind,
            "tactic_id": tactic_id,
            "span": span,
            "span_status": span_status,
            "rationale": rationale,
            "review_flags": flags,
            "source": {
                "source_id": source_id,
                "source_type": source_type,
                "path": _relative(path, root),
                "file_sha256": source_sha256,
                "row_number": row_number,
                "raw": {
                    "id": raw.get("id") or "",
                    "tactic_label": raw.get(label_column) or "",
                    "span": raw.get("时间范围") or "",
                    "rationale": raw.get("判断依据") or "",
                },
            },
        })
    return rows


def load_candidate_csv(
    path: Path,
    *,
    source_id: str,
    source_type: str,
    label_map: dict[str, str],
    root: Path | None = None,
    duration_s: float = 30.0,
    release: str | None = None,
) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return _load_candidate_rows(
            csv.DictReader(handle), path=path, source_id=source_id,
            source_type=source_type, label_map=label_map, root=root,
            duration_s=duration_s, release=release,
        )


def load_candidate_xlsx(
    path: Path,
    sheet_name: str,
    *,
    source_id: str,
    source_type: str,
    label_map: dict[str, str],
    root: Path | None = None,
    duration_s: float = 30.0,
    release: str | None = None,
) -> list[dict[str, Any]]:
    return _load_candidate_rows(
        read_xlsx_sheet(path, sheet_name), path=path, source_id=source_id,
        source_type=source_type, label_map=label_map, root=root,
        duration_s=duration_s, release=release,
    )


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path}; run the GSR-2024 migration first or omit --manifest"
        )
    index: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not CLIP_RE.fullmatch(clip_id):
            raise ValueError(f"{path}:{line_number}: invalid clip_id {clip_id!r}")
        if clip_id in index:
            raise ValueError(f"{path}:{line_number}: duplicate clip_id {clip_id}")
        expected = canonical_identity(clip_id)
        for field in ("clip_uid", "content_group_id"):
            if row.get(field) != expected[field]:
                raise ValueError(
                    f"{path}:{line_number}: {field} must be {expected[field]!r}, got {row.get(field)!r}"
                )
        aliases = set(row.get("legacy_clip_uids") or [])
        if expected["legacy_clip_uids"][0] not in aliases:
            raise ValueError(
                f"{path}:{line_number}: missing legacy alias {expected['legacy_clip_uids'][0]!r}"
            )
        index[clip_id] = row
    return index


def crosscheck_manifest(
    clip_ids: Iterable[str], manifest: dict[str, dict[str, Any]] | None
) -> dict[str, Any]:
    unique = sorted({CLIP_RE.fullmatch(clip_id).group(2) for clip_id in clip_ids})
    if manifest is None:
        return {"status": "not_checked", "clip_count": len(unique), "missing_clip_ids": []}
    missing = [clip_id for clip_id in unique if clip_id not in manifest]
    return {
        "status": "passed" if not missing else "failed",
        "clip_count": len(unique),
        "manifest_clip_count": len(manifest),
        "missing_clip_ids": missing,
    }


def build_candidate_queues(
    claims: list[dict[str, Any]], workbook_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    by_clip: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for claim in claims:
        by_clip[claim["clip_id"]][claim["source"]["source_id"]].append(claim)
    workbook_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in workbook_rows or []:
        workbook_by_clip[row["clip_id"]].append(row)

    entries = []
    counts: Counter[str] = Counter()
    for clip_id, sources in sorted(by_clip.items()):
        human = sources.get("human_annotator_1", [])
        sol = sources.get("gpt-5.6-sol-pro", [])
        clip_claims = [claim for source_claims in sources.values() for claim in source_claims]
        human_ids = {row["tactic_id"] for row in human if row["tactic_id"]}
        sol_ids = {row["tactic_id"] for row in sol if row["tactic_id"]}
        human_none = any(row["label_kind"] == "no_explicit_tactic" for row in human)
        sol_none = any(row["label_kind"] == "no_explicit_tactic" for row in sol)
        queue_ids: list[str] = []
        reasons: list[str] = []
        shared = sorted(human_ids & sol_ids)
        if shared or (human_none and sol_none):
            queue_ids.append("agreement")
            reasons.append("shared tactic claim" if shared else "both sources propose no explicit tactic")
        if human_ids != sol_ids or human_none != sol_none:
            queue_ids.append("tactic_conflict")
            reasons.append("normalized source label sets differ")
        if human_none and sol_ids:
            queue_ids.append("human_none_sol_positive")
            reasons.append("human seed proposes no tactic while Sol proposes positive tactics")
        if any(len({row["tactic_id"] for row in rows if row["tactic_id"]}) > 1 for rows in sources.values()):
            queue_ids.append("multi_tactic_cooccurrence")
            reasons.append("at least one source proposes multiple positive tactics")

        source_flags = sorted({
            flag
            for row in clip_claims
            for flag in row["review_flags"]
            if flag != "model_weak_label"
        })
        workbook_flags = sorted({
            flag
            for row in workbook_by_clip.get(clip_id, [])
            for flag in row.get("review_flags", [])
        })
        workbook_review_gate = any(
            row.get("benchmark_status") in {"HOLD", "REVIEW"}
            for row in workbook_by_clip.get(clip_id, [])
        )
        if source_flags or workbook_flags or workbook_review_gate:
            queue_ids.append("observability_review")
            reasons.append("source anomaly or provisional workbook review gate")
        counts.update(queue_ids)
        entries.append({
            "clip_id": clip_id,
            **canonical_identity(clip_id),
            "queue_ids": queue_ids,
            "human_tactic_ids": sorted(human_ids),
            "sol_tactic_ids": sorted(sol_ids),
            "shared_tactic_ids": shared,
            "human_no_explicit_tactic": human_none,
            "sol_no_explicit_tactic": sol_none,
            "review_flags": sorted(set(source_flags + workbook_flags)),
            "reasons": reasons,
        })
    return {
        "schema_version": "grounded-tactics-candidate-queues-v1",
        "queue_semantics": "non_exclusive",
        "definitions": {
            "agreement": "At least one shared normalized tactic, or both sources propose no explicit tactic.",
            "tactic_conflict": "The normalized source label sets differ.",
            "human_none_sol_positive": "Human proposes no explicit tactic while Sol proposes one or more tactics.",
            "multi_tactic_cooccurrence": "At least one source proposes more than one positive tactic.",
            "observability_review": "A source anomaly or provisional HOLD/review flag requires visual review.",
        },
        "counts": dict(sorted(counts.items())),
        "clips": entries,
    }


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    value = 0
    for char in letters.group():
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", NS))
    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read one simple tabular XLSX sheet using only the Python standard library."""

    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//main:t", NS))
                for item in root.findall("main:si", NS)
            ]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
        target = None
        for sheet in workbook.findall(".//main:sheet", NS):
            if sheet.attrib.get("name") == sheet_name:
                relation_id = sheet.attrib[f"{{{NS['rel']}}}id"]
                target = targets[relation_id]
                break
        if target is None:
            raise ValueError(f"{path}: missing XLSX sheet {sheet_name!r}")
        normalized_target = target.lstrip("/")
        sheet_path = normalized_target if normalized_target.startswith("xl/") else f"xl/{normalized_target}"
        sheet_root = ElementTree.fromstring(archive.read(sheet_path))

    matrix: list[list[str]] = []
    for row in sheet_root.findall(".//main:sheetData/main:row", NS):
        values: list[str] = []
        for cell in row.findall("main:c", NS):
            index = _column_index(cell.attrib["r"])
            if index >= len(values):
                values.extend([""] * (index + 1 - len(values)))
            values[index] = _xlsx_cell_value(cell, shared)
        matrix.append(values)
    if not matrix:
        return []
    headers = matrix[0]
    if not headers or len(set(headers)) != len(headers):
        raise ValueError(f"{path}:{sheet_name}: blank or duplicate headers")
    return [
        {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}
        for values in matrix[1:]
        if any(value.strip() for value in values)
    ]


def load_criterion_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "grounded-tactics-criterion-registry-v1":
        raise ValueError(f"{path}: unsupported criterion registry schema")
    if set(payload.get("allowed_states", [])) != CRITERION_STATES:
        raise ValueError(f"{path}: allowed_states must be exactly {sorted(CRITERION_STATES)}")
    entries = payload.get("criteria")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: criteria must be an array")
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        criterion_id = entry.get("criterion_id") if isinstance(entry, dict) else None
        if not isinstance(criterion_id, str) or not criterion_id:
            raise ValueError(f"{path}: every criterion needs a non-empty criterion_id")
        if criterion_id in index:
            raise ValueError(f"{path}: duplicate criterion_id {criterion_id!r}")
        index[criterion_id] = entry
    return {**payload, "index": index}


def parse_answer(raw: str) -> str | None:
    match = ANSWER_RE.match((raw or "").strip())
    return match.group(1) if match else None


def _issue(level: str, code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"level": level, "code": code, "message": message, **context}


def _parse_interval_options(raw: str) -> list[dict[str, Any]]:
    items = []
    for number, interval in re.findall(r"I([1-4])\s*=\s*([^;]+)", raw or ""):
        span, status = parse_span(interval.strip())
        items.append({"id": f"I{number}", "raw": interval.strip(), "span": span, "span_status": status})
    return items


def validate_question_bank(
    path: Path,
    registry: dict[str, Any],
    manifest: dict[str, dict[str, Any]] | None = None,
    *,
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_xlsx_sheet(path, "Question_Bank_3plus1")
    questions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    track_answers: dict[str, Counter[str]] = defaultdict(Counter)
    special_positions: dict[str, Counter[str]] = defaultdict(Counter)
    by_clip: dict[str, list[str]] = defaultdict(list)

    for row_number, raw in enumerate(raw_rows, 2):
        question_id = raw.get("question_id", "").strip()
        clip_id = raw.get("clip_id", "").strip()
        track = raw.get("track", "").strip()
        status = raw.get("benchmark_status", "").strip().upper()
        row_issues: list[dict[str, Any]] = []
        if question_id in seen_ids:
            row_issues.append(_issue("error", "duplicate_question_id", "question_id is duplicated"))
        seen_ids.add(question_id)
        if track not in TRACKS:
            row_issues.append(_issue("error", "unknown_track", f"unknown track {track!r}"))
        if status not in {"READY", "REVIEW", "HOLD"}:
            row_issues.append(_issue("error", "unknown_benchmark_status", f"unknown status {status!r}"))
        try:
            identity = canonical_identity(clip_id)
        except ValueError as exc:
            identity = {"clip_uid": None, "legacy_clip_uids": [], "content_group_id": None}
            row_issues.append(_issue("error", "invalid_clip_id", str(exc)))

        options = {letter: raw.get(f"option_{letter}", "").strip() for letter in "ABCD"}
        if any(not value for value in options.values()):
            row_issues.append(_issue("error", "missing_option", "all four options are required"))
        if len(set(options.values())) != 4:
            row_issues.append(_issue("error", "duplicate_option", "the four options must be distinct"))
        for letter, option in options.items():
            lowered = option.casefold()
            if any(token in lowered for token in ("not observable", "no clear", "insufficient")):
                special_positions[track][letter] += 1

        answer = parse_answer(raw.get("provisional_answer", ""))
        evidence_option = parse_answer(raw.get("provisional_evidence_option", ""))
        if answer:
            track_answers[track][answer] += 1
        elif status == "READY":
            row_issues.append(_issue("error", "ready_without_answer", "READY question needs one A-D answer"))
        else:
            row_issues.append(_issue("warning", "answer_pending", "provisional answer is unresolved"))

        provisional_span, span_status = parse_span(raw.get("gold_interval", ""))
        if span_status != "valid":
            level = "error" if status == "READY" else "warning"
            row_issues.append(_issue(level, f"provisional_span_{span_status}", "provisional interval is not valid"))

        interval_options: list[dict[str, Any]] = []
        if track == "T1_TEMPORAL":
            interval_options = [
                {"id": letter, "raw": option, "span": parse_span(option)[0], "span_status": parse_span(option)[1]}
                for letter, option in options.items()
            ]
        elif track == "T4_INTEGRATED":
            interval_options = _parse_interval_options(raw.get("interval_options_for_T4", ""))
        if track in {"T1_TEMPORAL", "T4_INTEGRATED"}:
            if len(interval_options) != 4 or any(item["span_status"] != "valid" for item in interval_options):
                row_issues.append(_issue("error", "invalid_interval_options", "temporal tracks require four valid intervals"))
        if track in {"T1_TEMPORAL", "T4_INTEGRATED"} and not evidence_option:
            level = "error" if status == "READY" else "warning"
            row_issues.append(_issue(level, "evidence_option_pending", "temporal question lacks an A-D evidence option"))

        output_raw = raw.get("output_schema", "").strip()
        output_contract: dict[str, Any] | None = None
        if output_raw:
            try:
                parsed = json.loads(output_raw)
                if not isinstance(parsed, dict):
                    raise ValueError("not an object")
                output_contract = parsed
            except (json.JSONDecodeError, ValueError) as exc:
                row_issues.append(_issue("error", "invalid_output_schema", f"output_schema is invalid JSON: {exc}"))
        else:
            row_issues.append(_issue("error", "missing_output_schema", "output_schema is required"))

        criterion_states = (output_contract or {}).get("criterion_states", {})
        if criterion_states and not isinstance(criterion_states, dict):
            row_issues.append(_issue("error", "invalid_criterion_states_shape", "criterion_states must be an object"))
            criterion_states = {}
        for criterion_id, state in criterion_states.items():
            if criterion_id not in registry["index"]:
                row_issues.append(_issue("warning", "unresolved_criterion_id", f"criterion {criterion_id!r} is not registered"))
            if state not in CRITERION_STATES:
                row_issues.append(_issue("warning", "invalid_provisional_criterion_state", f"{criterion_id}={state!r} is provisional, not a legal released state"))

        if status != "READY":
            row_issues.append(_issue("warning", "not_release_ready", f"question status is {status}"))

        question = {
            "schema_version": QUESTION_SCHEMA_VERSION,
            "question_id": question_id,
            "clip_id": clip_id,
            **identity,
            "track": track,
            "benchmark_status": status,
            "gold_status": "provisional_not_gold",
            "target_team": raw.get("target_team", "").strip(),
            "constraint": raw.get("constraint", "").strip(),
            "video_scope": raw.get("video_scope", "").strip(),
            "question": raw.get("question_en", "").strip(),
            "options": options,
            "interval_options": interval_options,
            "provisional_answer": answer,
            "provisional_answer_raw": raw.get("provisional_answer", ""),
            "provisional_evidence_option": evidence_option,
            "provisional_interval": provisional_span,
            "provisional_interval_raw": raw.get("gold_interval", ""),
            "provisional_tactic": raw.get("provisional_tactic", "").strip(),
            "provisional_verdict": raw.get("expected_verdict", "").strip(),
            "provisional_evidence_zh": raw.get("gold_evidence_zh", "").strip(),
            "provisional_output_contract": output_contract,
            "review_flags": sorted({item["code"] for item in row_issues}),
            "source": {
                "path": _relative(path, root),
                "sheet": "Question_Bank_3plus1",
                "row_number": row_number,
                "renamed_provisional_columns": {
                    "gold_interval": "provisional_interval",
                    "gold_evidence_zh": "provisional_evidence_zh",
                },
            },
        }
        questions.append(question)
        issues.extend({**item, "question_id": question_id, "row_number": row_number} for item in row_issues)
        if CLIP_RE.fullmatch(clip_id):
            by_clip[clip_id].append(track)

    for clip_id, tracks in sorted(by_clip.items()):
        if len(tracks) != 4 or set(tracks) != TRACKS:
            issues.append(_issue("error", "incomplete_3plus1", "clip must have exactly one question per track", clip_id=clip_id, tracks=tracks))

    answer_balance: dict[str, Any] = {}
    for track in sorted(TRACKS):
        counts = {letter: track_answers[track][letter] for letter in "ABCD"}
        values = list(counts.values())
        balanced = max(values) - min(values) <= 1
        answer_balance[track] = {"counts": counts, "balanced": balanced}
        if sum(values) and not balanced:
            issues.append(_issue("warning", "answer_position_imbalance", f"{track} provisional answers are imbalanced", track=track, counts=counts))
        specials = special_positions[track]
        if sum(specials.values()) >= 2 and len([value for value in specials.values() if value]) == 1:
            issues.append(_issue("warning", "fixed_special_option_position", f"{track} special options use one fixed position", track=track, counts=dict(specials)))

    manifest_result = crosscheck_manifest((row["clip_id"] for row in questions if CLIP_RE.fullmatch(row["clip_id"])), manifest)
    if manifest_result["status"] == "failed":
        issues.append(_issue("error", "manifest_missing_clips", "question clips are absent from the manifest", missing_clip_ids=manifest_result["missing_clip_ids"]))
    report = {
        "source_sha256": sha256_file(path),
        "question_count": len(questions),
        "clip_count": len(by_clip),
        "status_counts": dict(Counter(row["benchmark_status"] for row in questions)),
        "track_counts": dict(Counter(row["track"] for row in questions)),
        "parsed_answer_count": sum(row["provisional_answer"] is not None for row in questions),
        "pending_answer_count": sum(row["provisional_answer"] is None for row in questions),
        "answer_balance": answer_balance,
        "manifest": manifest_result,
        "issues": issues,
    }
    return questions, report


def validate_structured_annotations(
    path: Path,
    registry: dict[str, Any],
    manifest: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    rows = read_jsonl(path)
    clip_ids: list[str] = []
    required = {
        "schema_version", "clip_uid", "target_team", "phase", "tactic_ids",
        "minimum_sufficient_spans", "criterion_states", "verdict", "observability",
        "confounder", "free_text_evidence", "annotator_id_pseudonymous", "ontology_version",
    }
    for row_number, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            issues.append(_issue("error", "missing_structured_fields", "structured annotation is incomplete", row_number=row_number, fields=missing))
            continue
        if row["schema_version"] != STRUCTURED_SCHEMA_VERSION:
            issues.append(_issue("error", "unknown_structured_schema", "unsupported schema_version", row_number=row_number))
        clip_uid = row.get("clip_uid", "")
        uid_match = re.fullmatch(r"gsr(2024|2025):(SNGS-\d{3})", clip_uid) if isinstance(clip_uid, str) else None
        clip_id = uid_match.group(2) if uid_match else ""
        clip_ref = clip_id if uid_match and uid_match.group(1) == "2024" else f"2025-{clip_id}"
        if not uid_match or clip_uid != canonical_identity(clip_ref)["clip_uid"]:
            issues.append(_issue("error", "invalid_structured_clip_uid", "clip_uid must use gsr2024:SNGS-### or gsr2025:SNGS-###", row_number=row_number))
        else:
            clip_ids.append(clip_id)
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            issues.append(_issue("error", "invalid_verdict", f"invalid verdict {verdict!r}", row_number=row_number))
        states = row.get("criterion_states")
        if not isinstance(states, list):
            issues.append(_issue("error", "invalid_criterion_states_shape", "criterion_states must be an array", row_number=row_number))
            continue
        state_values = []
        seen_criteria: set[str] = set()
        for item in states:
            criterion_id = item.get("criterion_id") if isinstance(item, dict) else None
            state = item.get("state") if isinstance(item, dict) else None
            if criterion_id in seen_criteria:
                issues.append(_issue("error", "duplicate_criterion_state", f"duplicate criterion {criterion_id!r}", row_number=row_number))
            seen_criteria.add(criterion_id)
            if criterion_id not in registry["index"]:
                issues.append(_issue("error", "unresolved_criterion_id", f"unknown criterion {criterion_id!r}", row_number=row_number))
            if state not in CRITERION_STATES:
                issues.append(_issue("error", "invalid_criterion_state", f"invalid state {state!r}", row_number=row_number, criterion_id=criterion_id))
            state_values.append(state)
        if verdict == "false" and "violated" not in state_values:
            issues.append(_issue("error", "false_without_counterevidence", "false requires at least one violated criterion", row_number=row_number))
        if verdict == "insufficient" and "not_observable" not in state_values:
            issues.append(_issue("error", "insufficient_without_unobservable", "insufficient requires at least one not_observable criterion", row_number=row_number))
        if verdict == "none_of_ontology" and row.get("tactic_ids"):
            issues.append(_issue("error", "none_with_tactic_ids", "none_of_ontology requires an empty tactic_ids list", row_number=row_number))
        duration_s = 30.0
        if manifest and clip_id in manifest and manifest[clip_id].get("duration_s") is not None:
            manifest_duration = manifest[clip_id]["duration_s"]
            if not isinstance(manifest_duration, (int, float)) or manifest_duration <= 0:
                issues.append(_issue("error", "invalid_manifest_duration", "manifest duration_s must be positive", row_number=row_number))
            else:
                duration_s = float(manifest_duration)
        for span in row.get("minimum_sufficient_spans", []):
            if not isinstance(span, dict) or set(span) != {"start_s", "end_s"}:
                issues.append(_issue("error", "invalid_structured_span", "span must contain start_s and end_s", row_number=row_number))
                continue
            if not isinstance(span["start_s"], (int, float)) or not isinstance(span["end_s"], (int, float)) or not 0 <= span["start_s"] < span["end_s"]:
                issues.append(_issue("error", "invalid_structured_span", "span bounds are invalid", row_number=row_number))
            elif span["end_s"] > duration_s:
                issues.append(_issue("error", "structured_span_out_of_bounds", f"span exceeds clip duration {duration_s:g}s", row_number=row_number))

    manifest_result = crosscheck_manifest(clip_ids, manifest)
    if manifest_result["status"] == "failed":
        issues.append(_issue("error", "manifest_missing_clips", "structured annotation clips are absent from the manifest", missing_clip_ids=manifest_result["missing_clip_ids"]))
    return {
        "path": str(path),
        "row_count": len(rows),
        "manifest": manifest_result,
        "issues": issues,
    }


def issue_counts(issues: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(issue["level"] for issue in issues)
    return {"errors": counts["error"], "warnings": counts["warning"]}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
