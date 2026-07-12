from scripts.tactical_kb_eval import (
    audit_feasibility,
    load_annotations,
    load_dataset_index,
    validate_annotation,
    validate_development_clip_manifest,
    validate_recipe_shortlist,
)


import json

import pytest

from scripts.tactical_kb_eval import _sha256_json


def _catalog():
    return {
        "catalog_version": "tactical-kb-v2",
        "feasibility_candidates": ["overlap_run", "run_in_behind", "switch_of_play", "local_numerical_superiority"],
        "concepts": [
            {"id": "overlap_run", "required_actor_roles": ["carrier_track_id", "runner_track_id"]},
            {"id": "run_in_behind", "required_actor_roles": ["carrier_track_id", "runner_track_id"]},
            {"id": "switch_of_play", "required_actor_roles": ["passer_track_id", "receiver_track_id"]},
            {"id": "local_numerical_superiority", "required_actor_roles": ["anchor_track_id"]},
        ],
    }


def _index():
    return {
        ("train", "SNGS-001"): {"split": "train", "sequence": "SNGS-001", "game_id": "1"},
        ("valid", "SNGS-002"): {"split": "valid", "sequence": "SNGS-002", "game_id": "2"},
        ("train", "SNGS-116"): {"split": "train", "sequence": "SNGS-116", "game_id": "7"},
        ("valid", "SNGS-132"): {"split": "valid", "sequence": "SNGS-132", "game_id": "8"},
        ("valid", "SNGS-187"): {"split": "valid", "sequence": "SNGS-187", "game_id": "11"},
    }


def _row(sample_id="p1", *, clip_id="SNGS-001", split="train", match_id="1", label="positive", concept_id="overlap_run"):
    roles = {
        "overlap_run": ("carrier_track_id", "runner_track_id"),
        "run_in_behind": ("carrier_track_id", "runner_track_id"),
        "switch_of_play": ("passer_track_id", "receiver_track_id"),
        "local_numerical_superiority": ("anchor_track_id",),
    }[concept_id]
    gt = {role: index + 1 for index, role in enumerate(roles)}
    gsr = {role: index + 101 for index, role in enumerate(roles)}
    correspondence = {
        role.removesuffix("_track_id"): {
            "status": "reviewed_match",
            "ground_truth_track_id": gt[role],
            "gsr_prediction_track_id": gsr[role],
        }
        for role in roles
    }
    return {
        "schema_version": "tactical-kb-dev-v1",
        "sample_id": sample_id,
        "dataset_split": split,
        "match_id": match_id,
        "clip_id": clip_id,
        "window": {"start_s": 1.0, "end_s": 2.0},
        "context_window": {"start_s": 0.0, "end_s": 3.0},
        "concept_id": concept_id,
        "label": label,
        "team_id": "team_0",
        "actor_bindings": {"ground_truth": gt, "gsr_prediction": gsr},
        "actor_correspondence": correspondence,
        "claim_labels": {"observation": "supported", "effect": "not_annotated", "intention": "not_annotated"},
        "hard_negative": (
            {"type": "parallel_support", "target_concept_id": concept_id}
            if label == "hard_negative" else None
        ),
        "state_observability": {"ground_truth": "direct", "gsr_prediction": "direct"},
        "quality_by_state_source": {
            source: {"tracking": "usable", "ball": "usable", "calibration": "usable", "structure": "not_required"}
            for source in ("ground_truth", "gsr_prediction")
        },
        "kb_version": "tactical-kb-v2",
    }


def _errors(row, reviewed=None):
    return validate_annotation(
        row,
        catalog=_catalog(),
        dataset_index=_index(),
        reviewed_clips=reviewed or {("train", "SNGS-001"), ("valid", "SNGS-002")},
        purpose="development",
    )


def test_annotation_requires_match_clip_window_concept_and_actor_roles():
    row = _row()
    del row["window"]
    row["actor_bindings"]["ground_truth"].pop("runner_track_id")
    errors = _errors(row)
    assert any("window" in error for error in errors)
    assert any("actor roles" in error for error in errors)


def test_hard_negative_requires_type_and_target_concept():
    row = _row(label="hard_negative")
    row["hard_negative"] = {"type": "parallel_support", "target_concept_id": "run_in_behind"}
    assert any("target concept" in error for error in _errors(row))


def test_annotation_match_id_must_equal_trusted_clip_game_mapping():
    assert any("trusted metadata" in error for error in _errors(_row(match_id="999")))


def test_annotation_clip_must_belong_to_frozen_development_manifest():
    assert any("frozen development" in error for error in _errors(_row(), reviewed={("valid", "SNGS-002")}))


@pytest.mark.parametrize("clip_id,split", [("SNGS-132", "valid"), ("SNGS-187", "valid")])
def test_games8_and11_clips_are_rejected_even_with_falsified_match_id(clip_id, split):
    errors = _errors(_row(clip_id=clip_id, split=split, match_id="1"), reviewed={(split, clip_id)})
    assert any("held-out" in error for error in errors)


def test_game7_rows_are_diagnostic_and_do_not_count_toward_feasibility():
    rows = [_row(str(i), clip_id="SNGS-116", match_id="7") for i in range(12)]
    report = audit_feasibility(_catalog(), rows, dataset_index=_index(), reviewed_clips={("train", "SNGS-116")})
    assert report["concepts"]["overlap_run"]["positive_count"] == 0


def test_feasibility_requires_6_positive_6_hard_negative_and_2_matches():
    rows = []
    for i in range(6):
        rows.append(_row(f"p{i}"))
        rows.append(_row(f"n{i}", label="hard_negative"))
    report = audit_feasibility(_catalog(), rows, dataset_index=_index(), reviewed_clips={("train", "SNGS-001")})
    assert "overlap_run" not in report["feasibility_passed"]
    rows.extend([
        _row("p-other", clip_id="SNGS-002", split="valid", match_id="2"),
        _row("n-other", clip_id="SNGS-002", split="valid", match_id="2", label="hard_negative"),
    ])
    report = audit_feasibility(_catalog(), rows, dataset_index=_index(), reviewed_clips={("train", "SNGS-001"), ("valid", "SNGS-002")})
    assert "overlap_run" in report["feasibility_passed"]


def test_gt_and_gsr_observability_are_reported_separately():
    row = _row()
    row["state_observability"]["gsr_prediction"] = "degraded"
    report = audit_feasibility(_catalog(), [row], dataset_index=_index(), reviewed_clips={("train", "SNGS-001")})
    metric = report["concepts"]["overlap_run"]["observability"]
    assert metric["ground_truth"]["direct"] == 1
    assert metric["gsr_prediction"]["degraded"] == 1
    quality = report["concepts"]["overlap_run"]["quality_by_state_source"]
    assert quality["ground_truth"]["tracking"]["usable"] == 1


def test_gt_and_gsr_actor_bindings_can_use_different_track_ids():
    assert _errors(_row()) == []


def test_unresolved_gsr_actor_requires_unsupported_observability():
    row = _row()
    row["actor_correspondence"]["runner"]["status"] = "unresolved"
    row["actor_correspondence"]["runner"]["gsr_prediction_track_id"] = None
    row["actor_bindings"]["gsr_prediction"]["runner_track_id"] = None
    assert any("unsupported" in error for error in _errors(row))


def test_audit_does_not_cap_feasibility_passed_at_three():
    rows = []
    for concept in _catalog()["feasibility_candidates"]:
        for match, clip, split in (("1", "SNGS-001", "train"), ("2", "SNGS-002", "valid")):
            for i in range(3):
                rows.append(_row(f"{concept}-{match}-p{i}", concept_id=concept, match_id=match, clip_id=clip, split=split))
                rows.append(_row(f"{concept}-{match}-n{i}", concept_id=concept, match_id=match, clip_id=clip, split=split, label="hard_negative"))
    report = audit_feasibility(_catalog(), rows, dataset_index=_index(), reviewed_clips={("train", "SNGS-001"), ("valid", "SNGS-002")})
    assert len(report["feasibility_passed"]) == 4


def test_recipe_shortlist_is_at_most_three_and_subset_of_passed_report():
    report = {"feasibility_passed": ["overlap_run"]}
    shortlist = {
        "schema_version": "tactical-recipe-shortlist-v1",
        "feasibility_report_hash": _sha256_json(report),
        "concept_ids": ["run_in_behind"],
        "decisions": [],
    }
    assert validate_recipe_shortlist(shortlist, feasibility_report=report)


def test_recipe_shortlist_requires_exact_feasibility_report_hash():
    report = {"feasibility_passed": []}
    shortlist = {
        "schema_version": "tactical-recipe-shortlist-v1",
        "feasibility_report_hash": "sha256:" + "0" * 64,
        "concept_ids": [],
        "decisions": [],
    }
    assert any("hash" in error for error in validate_recipe_shortlist(shortlist, feasibility_report=report))


def test_load_annotations_and_dataset_index_are_strict(tmp_path):
    annotations = tmp_path / "rows.jsonl"
    annotations.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_annotations(annotations)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequences": list(_index().values())}))
    assert ("train", "SNGS-001") in load_dataset_index(manifest)


def test_development_manifest_rejects_excluded_trusted_games():
    manifest = {
        "schema_version": "tactical-development-clips-v1",
        "clips": [{"dataset_split": "train", "clip_id": "SNGS-116", "match_id": "7"}],
    }
    assert validate_development_clip_manifest(manifest, dataset_index=_index())
