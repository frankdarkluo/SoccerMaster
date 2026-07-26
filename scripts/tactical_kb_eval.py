"""Deterministic development-only Tactical KB feasibility audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.atomic import atomic_write_json
from pipeline.stage2b.kb import load_catalog

LOCKED_HELDOUT_CLIP_IDS = frozenset({
    *(f"SNGS-{value:03d}" for value in range(132, 151)),
    *(f"SNGS-{value:03d}" for value in range(187, 201)),
})
QUALITY_VALUES = frozenset({"usable", "degraded", "unsupported", "not_required"})
OBSERVABILITY_VALUES = frozenset({"direct", "degraded", "unsupported"})


def _sha256_json(value: dict) -> str:
    data = json.dumps(value, ensure_ascii=False, indent=1)
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def load_annotations(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number}: malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: annotation must be an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"line {number}: invalid sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def load_dataset_index(path: Path) -> dict[tuple[str, str], dict]:
    manifest = json.loads(Path(path).read_text())
    root = Path(manifest.get("root", ""))
    index = {}
    for row in manifest.get("sequences", []):
        key = (row.get("split"), row.get("sequence"))
        if key in index:
            raise ValueError(f"duplicate dataset sequence: {key}")
        if key[0] in {"train", "valid"} and isinstance(key[1], str):
            record = dict(row)
            record["_path"] = root / key[0] / key[1]
            index[key] = record
    return index


def validate_development_clip_manifest(
    manifest: dict,
    *,
    dataset_index: dict[tuple[str, str], dict],
) -> list[str]:
    errors = []
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "tactical-development-clips-v1":
        return ["invalid development clip manifest schema"]
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return ["clips must be an array"]
    seen = set()
    for index, row in enumerate(clips):
        label = f"clips[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        key = (row.get("dataset_split"), row.get("clip_id"))
        if key in seen:
            errors.append(f"{label} duplicates {key}")
        seen.add(key)
        trusted = dataset_index.get(key)
        if key[1] in LOCKED_HELDOUT_CLIP_IDS:
            errors.append(f"{label} is held out")
        if trusted is None:
            errors.append(f"{label} is not in trusted train/valid index")
            continue
        if str(row.get("match_id")) != str(trusted.get("game_id")):
            errors.append(f"{label} match_id differs from trusted metadata")
        if str(trusted.get("game_id")) in {"7", "8", "11"}:
            errors.append(f"{label} uses excluded game")
        path = trusted.get("_path")
        if path is not None and (not (path / "Labels-GameState.json").is_file() or not (path / "img1").is_dir()):
            errors.append(f"{label} trusted dataset path is missing")
    return errors


def _concept_map(catalog: dict) -> dict[str, dict]:
    return {concept["id"]: concept for concept in catalog["concepts"]}


def validate_annotation(
    row: dict,
    *,
    catalog: dict,
    dataset_index: dict[tuple[str, str], dict],
    reviewed_clips: set[tuple[str, str]],
    purpose: str,
) -> list[str]:
    errors = []
    if not isinstance(row, dict):
        return ["annotation must be an object"]
    required = {
        "schema_version", "sample_id", "dataset_split", "match_id", "clip_id",
        "window", "context_window", "concept_id", "label", "team_id",
        "actor_bindings", "actor_correspondence", "claim_labels",
        "hard_negative", "state_observability", "quality_by_state_source",
        "kb_version",
    }
    missing = sorted(required - set(row))
    if missing:
        errors.append(f"missing fields: {missing}")
    if row.get("schema_version") != "tactical-kb-dev-v1" or purpose != "development":
        errors.append("invalid development schema or purpose")
    key = (row.get("dataset_split"), row.get("clip_id"))
    if key[1] in LOCKED_HELDOUT_CLIP_IDS:
        errors.append("held-out clip is forbidden")
    trusted = dataset_index.get(key)
    if trusted is None:
        errors.append("clip is not in trusted train/valid index")
    else:
        if str(row.get("match_id")) != str(trusted.get("game_id")):
            errors.append("match_id differs from trusted metadata")
    if key not in reviewed_clips:
        errors.append("clip is not in frozen development manifest")
    for field in ("window", "context_window"):
        value = row.get(field)
        if not isinstance(value, dict) or set(value) != {"start_s", "end_s"}:
            errors.append(f"{field} must contain start_s and end_s")
        elif not all(type(value[x]) in (int, float) and math.isfinite(value[x]) for x in value) or value["start_s"] < 0 or value["start_s"] >= value["end_s"]:
            errors.append(f"{field} is invalid")
    concepts = _concept_map(catalog)
    concept = concepts.get(row.get("concept_id"))
    if concept is None:
        errors.append("unknown concept_id")
    if row.get("team_id") not in {"team_0", "team_1"}:
        errors.append("team_id must be canonical")
    label = row.get("label")
    if label not in {"positive", "hard_negative", "ambiguous_exclude"}:
        errors.append("invalid label")
    hard_negative = row.get("hard_negative")
    if label == "hard_negative":
        if not isinstance(hard_negative, dict) or not isinstance(hard_negative.get("type"), str) or hard_negative.get("target_concept_id") != row.get("concept_id"):
            errors.append("hard negative requires type and target concept")
    elif hard_negative is not None:
        errors.append("hard_negative must be null unless label is hard_negative")
    bindings = row.get("actor_bindings")
    expected_roles = set(concept.get("required_actor_roles", [])) if concept else set()
    if not isinstance(bindings, dict) or set(bindings) != {"ground_truth", "gsr_prediction"}:
        errors.append("actor_bindings must contain both state sources")
    else:
        gt, gsr = bindings["ground_truth"], bindings["gsr_prediction"]
        if not isinstance(gt, dict) or set(gt) != expected_roles:
            errors.append("ground_truth actor roles do not match concept")
        if not isinstance(gsr, dict) or set(gsr) != expected_roles:
            errors.append("gsr_prediction actor roles do not match concept")
    correspondence = row.get("actor_correspondence")
    expected_correspondence = {role.removesuffix("_track_id") for role in expected_roles}
    if not isinstance(correspondence, dict) or set(correspondence) != expected_correspondence:
        errors.append("actor_correspondence must match concept roles")
        correspondence = {}
    else:
        for name, record in correspondence.items():
            role = f"{name}_track_id"
            if not isinstance(record, dict) or record.get("status") not in {"reviewed_match", "unresolved"}:
                errors.append(f"invalid actor correspondence for {name}")
                continue
            gt_id = bindings.get("ground_truth", {}).get(role) if isinstance(bindings, dict) else None
            gsr_id = bindings.get("gsr_prediction", {}).get(role) if isinstance(bindings, dict) else None
            if record.get("ground_truth_track_id") != gt_id or record.get("gsr_prediction_track_id") != gsr_id:
                errors.append(f"actor correspondence IDs disagree for {name}")
            if record.get("status") == "unresolved" and gsr_id is not None:
                errors.append(f"unresolved GSR actor {name} must be null")
    observability = row.get("state_observability")
    if not isinstance(observability, dict) or set(observability) != {"ground_truth", "gsr_prediction"} or any(value not in OBSERVABILITY_VALUES for value in observability.values()):
        errors.append("invalid state_observability")
    elif any(record.get("status") == "unresolved" for record in correspondence.values()) and observability["gsr_prediction"] != "unsupported":
        errors.append("unresolved GSR actor requires unsupported observability")
    elif observability["gsr_prediction"] == "unsupported" and isinstance(bindings, dict):
        if any(value is not None for value in bindings.get("gsr_prediction", {}).values()):
            errors.append("unsupported GSR actors must be null")
    quality = row.get("quality_by_state_source")
    if not isinstance(quality, dict) or set(quality) != {"ground_truth", "gsr_prediction"}:
        errors.append("invalid quality_by_state_source")
    else:
        for source, values in quality.items():
            if not isinstance(values, dict) or set(values) != {"tracking", "ball", "calibration", "structure"} or any(value not in QUALITY_VALUES for value in values.values()):
                errors.append(f"invalid {source} quality")
    if row.get("claim_labels") != {"observation": "supported", "effect": "not_annotated", "intention": "not_annotated"}:
        errors.append("claim_labels must be observation-only")
    if row.get("kb_version") != "tactical-kb-v2" or "recipe_version" in row:
        errors.append("invalid KB version profile")
    return errors


def audit_feasibility(
    catalog: dict,
    rows: list[dict],
    *,
    dataset_index: dict[tuple[str, str], dict],
    reviewed_clips: set[tuple[str, str]],
) -> dict:
    candidates = list(catalog["feasibility_candidates"])
    metrics = {}
    for concept_id in candidates:
        eligible = []
        for row in rows:
            if row.get("concept_id") != concept_id:
                continue
            errors = validate_annotation(row, catalog=catalog, dataset_index=dataset_index, reviewed_clips=reviewed_clips, purpose="development")
            if errors:
                raise ValueError(f"{row.get('sample_id')}: {'; '.join(errors)}")
            trusted = dataset_index[(row["dataset_split"], row["clip_id"])]
            if str(trusted["game_id"]) != "7" and row["label"] != "ambiguous_exclude":
                eligible.append(row)
        positives = [row for row in eligible if row["label"] == "positive"]
        negatives = [row for row in eligible if row["label"] == "hard_negative"]
        matches = sorted({str(row["match_id"]) for row in positives + negatives})
        reasons = []
        if len(positives) < 6:
            reasons.append("fewer_than_6_positives")
        if len(negatives) < 6:
            reasons.append("fewer_than_6_hard_negatives")
        if len(matches) < 2:
            reasons.append("fewer_than_2_matches")
        source_counts = {
            source: {
                value: sum(row["state_observability"][source] == value for row in eligible)
                for value in sorted(OBSERVABILITY_VALUES)
            }
            for source in ("ground_truth", "gsr_prediction")
        }
        hard_negative_types = {}
        for negative in negatives:
            kind = negative["hard_negative"]["type"]
            hard_negative_types[kind] = hard_negative_types.get(kind, 0) + 1
        quality_counts = {
            source: {
                field: {
                    value: sum(row["quality_by_state_source"][source][field] == value for row in eligible)
                    for value in sorted(QUALITY_VALUES)
                }
                for field in ("tracking", "ball", "calibration", "structure")
            }
            for source in ("ground_truth", "gsr_prediction")
        }
        metrics[concept_id] = {
            "positive_count": len(positives),
            "hard_negative_count": len(negatives),
            "hard_negative_types": hard_negative_types,
            "quality_by_state_source": quality_counts,
            "eligible_match_count": len(matches),
            "eligible_matches": matches,
            "observability": source_counts,
            "gt_to_gsr_unsupported_increase": source_counts["gsr_prediction"]["unsupported"] - source_counts["ground_truth"]["unsupported"],
            "feasibility_passed": not reasons,
            "failure_reasons": reasons,
        }
    passed = [concept_id for concept_id in candidates if metrics[concept_id]["feasibility_passed"]]
    return {
        "schema_version": "tactical-feasibility-v1",
        "kb_version": catalog["catalog_version"],
        "concepts": metrics,
        "feasibility_passed": passed,
        "feasibility_failed": [concept_id for concept_id in candidates if concept_id not in passed],
    }


def validate_recipe_shortlist(shortlist: dict, *, feasibility_report: dict) -> list[str]:
    errors = []
    if not isinstance(shortlist, dict) or shortlist.get("schema_version") != "tactical-recipe-shortlist-v1":
        return ["invalid shortlist schema"]
    ids = shortlist.get("concept_ids")
    if not isinstance(ids, list) or len(ids) > 3 or len(ids) != len(set(ids)):
        errors.append("concept_ids must be a unique zero-to-three array")
        ids = []
    if not set(ids).issubset(feasibility_report.get("feasibility_passed", [])):
        errors.append("shortlist must be a subset of feasibility_passed")
    if shortlist.get("feasibility_report_hash") != _sha256_json(feasibility_report):
        errors.append("feasibility report hash mismatch")
    decisions = shortlist.get("decisions")
    passed = feasibility_report.get("feasibility_passed", [])
    if not isinstance(decisions, list):
        errors.append("decisions must be an array")
    else:
        decision_ids = [item.get("concept_id") for item in decisions if isinstance(item, dict)]
        if len(decision_ids) != len(decisions) or len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(passed):
            errors.append("decisions must cover every feasibility-passed concept exactly once")
        for item in decisions:
            if not isinstance(item, dict):
                continue
            if item.get("selected") is not (item.get("concept_id") in ids):
                errors.append("decision selected flag disagrees with concept_ids")
            for field in ("rationale", "reviewer", "reviewed_at"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    errors.append(f"decision {field} must be non-empty")
    return errors


def _load_reviewed(path: Path, index: dict) -> set[tuple[str, str]]:
    manifest = json.loads(path.read_text())
    errors = validate_development_clip_manifest(manifest, dataset_index=index)
    if errors:
        raise ValueError("; ".join(errors))
    return {(row["dataset_split"], row["clip_id"]) for row in manifest["clips"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-development-clips", "validate-annotations", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--dataset-manifest", type=Path, required=True)
        command.add_argument("--clips", type=Path, required=True)
        if name != "validate-development-clips":
            command.add_argument("--catalog", type=Path, required=True)
            command.add_argument("--annotations", type=Path, required=True)
        if name == "validate-annotations":
            command.add_argument("--purpose", choices=["development"], required=True)
        if name == "audit":
            command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("validate-shortlist")
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--shortlist", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-shortlist":
        report = json.loads(args.report.read_text())
        errors = validate_recipe_shortlist(json.loads(args.shortlist.read_text()), feasibility_report=report)
    else:
        index = load_dataset_index(args.dataset_manifest)
        reviewed = _load_reviewed(args.clips, index)
        errors = []
        if args.command != "validate-development-clips":
            catalog = load_catalog(args.catalog)
            rows = load_annotations(args.annotations)
            if args.command == "validate-annotations":
                for row in rows:
                    errors.extend(validate_annotation(row, catalog=catalog, dataset_index=index, reviewed_clips=reviewed, purpose=args.purpose))
            else:
                report = audit_feasibility(catalog, rows, dataset_index=index, reviewed_clips=reviewed)
                atomic_write_json(args.out, report)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
