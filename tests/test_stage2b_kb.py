from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from pipeline.stage2b.kb import (
    CatalogError,
    get_concept,
    load_catalog,
    project_public_catalog,
)


CANDIDATES = [
    "overlap_run",
    "run_in_behind",
    "switch_of_play",
    "local_numerical_superiority",
    "compact_block",
    "line_break",
]
PRIVATE_KEYS = {
    "sources",
    "source_registry",
    "recipe",
    "parameters",
    "thresholds",
    "maturity",
    "production_enabled",
    "production_gate",
}


def _source(source_id: str = "definitions") -> dict:
    return {
        "source_id": source_id,
        "title": "Definitions",
        "authors": ["SoccerMaster project"],
        "version": "1",
        "url_or_doi": "https://example.com/definitions",
        "license": "project-internal",
        "accessed_at": "2026-07-11",
        "content_hash": "sha256:" + "a" * 64,
        "used_for": ["name"],
    }


def _concept(concept_id: str = "overlap_run") -> dict:
    return {
        "id": concept_id,
        "name_zh": "套边",
        "name_en": "overlap run",
        "aliases": {"zh": ["套上"], "en": ["overlap"]},
        "observation_definition": "同队跑动者从持球者身后沿外侧移动到身前。",
        "required_actor_roles": ["carrier_track_id", "runner_track_id"],
        "actor_team_constraints": {
            "carrier_track_id": "same",
            "runner_track_id": "same",
        },
        "sources": {
            "terminology": ["definitions"],
            "formalization": [],
            "calibration": [],
        },
        "allowed_claim_levels": ["observation"],
        "maturity": "planned",
        "production_enabled": False,
    }


def _catalog() -> dict:
    return {
        "catalog_version": "tactical-kb-v2",
        "public_projection_version": "tactical-public-v1",
        "source_registry": [_source()],
        "feasibility_candidates": ["overlap_run"],
        "concepts": [_concept()],
    }


def _write(tmp_path: Path, catalog: dict) -> Path:
    path = tmp_path / "concepts.yaml"
    path.write_text(yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False))
    return path


def _assert_no_private_keys(value) -> None:
    if isinstance(value, dict):
        assert PRIVATE_KEYS.isdisjoint(value)
        for nested in value.values():
            _assert_no_private_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_private_keys(nested)


def test_repository_catalog_has_14_disabled_concepts():
    catalog = load_catalog()

    assert len(catalog["concepts"]) == 14
    assert all(not concept["production_enabled"] for concept in catalog["concepts"])


def test_candidate_pool_is_the_approved_six():
    catalog = load_catalog()

    assert catalog["feasibility_candidates"] == CANDIDATES


def test_legacy_aliases_resolve_to_canonical_ids(tmp_path):
    catalog_data = _catalog()
    concept = catalog_data["concepts"][0]
    concept["id"] = "run_in_behind"
    concept["aliases"]["en"].append("depth_run")
    catalog_data["feasibility_candidates"] = ["run_in_behind"]
    catalog = load_catalog(_write(tmp_path, catalog_data))

    assert get_concept(catalog, "depth_run")["id"] == "run_in_behind"


@pytest.mark.parametrize("duplicate", ["concept", "source", "alias"])
def test_duplicate_concept_source_and_aliases_fail(tmp_path, duplicate):
    catalog = _catalog()
    if duplicate == "concept":
        catalog["concepts"].append(deepcopy(catalog["concepts"][0]))
    elif duplicate == "source":
        catalog["source_registry"].append(deepcopy(catalog["source_registry"][0]))
    else:
        second = _concept("underlap_run")
        second["aliases"]["en"] = ["overlap"]
        catalog["concepts"].append(second)

    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(_write(tmp_path, catalog))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.pop("license"),
        lambda source: source.update(content_hash="sha256:ABC"),
        lambda source: source.update(authors=[]),
        lambda source: source.update(used_for=[]),
    ],
)
def test_source_records_require_complete_metadata_and_sha256(tmp_path, mutation):
    catalog = _catalog()
    mutation(catalog["source_registry"][0])

    with pytest.raises(CatalogError):
        load_catalog(_write(tmp_path, catalog))


def test_planned_concept_may_omit_recipe(tmp_path):
    catalog = load_catalog(_write(tmp_path, _catalog()))

    assert "recipe" not in catalog["concepts"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda concept: concept.pop("recipe"),
        lambda concept: concept["recipe"].pop("checker"),
        lambda concept: concept.pop("production_gate"),
        lambda concept: concept["production_gate"].update(passed=False),
    ],
)
def test_production_concept_requires_recipe_checker_and_gate(tmp_path, mutation):
    catalog = _catalog()
    concept = catalog["concepts"][0]
    concept.update(
        maturity="production",
        production_enabled=True,
        recipe={
            "id": "overlap_run",
            "version": 1,
            "checker": "overlap_run",
            "parameters": {},
        },
        production_gate={"passed": True},
    )
    mutation(concept)

    with pytest.raises(CatalogError):
        load_catalog(
            _write(tmp_path, catalog),
            checker_names=frozenset({"overlap_run"}),
        )


def test_public_projection_is_an_exact_whitelist(tmp_path):
    catalog = _catalog()
    catalog["concepts"][0]["recipe"] = {
        "id": "overlap_run",
        "version": 1,
        "checker": "overlap_run",
        "parameters": {"distance": 1.0},
    }
    loaded = load_catalog(_write(tmp_path, catalog))

    public = project_public_catalog(loaded, concept_ids={"overlap_run"})

    assert set(public) == {
        "catalog_version",
        "public_projection_version",
        "concepts",
    }
    assert set(public["concepts"][0]) == {
        "id",
        "name_zh",
        "name_en",
        "aliases",
        "observation_definition",
        "required_actor_roles",
    }
    _assert_no_private_keys(public)
