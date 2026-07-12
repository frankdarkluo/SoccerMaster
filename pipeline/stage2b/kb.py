"""Validated private Tactical KB catalog and strict public projection."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import re

import yaml


DEFAULT_CATALOG_PATH = Path(__file__).with_name("concepts.yaml")
PUBLIC_FIELDS = (
    "id",
    "name_zh",
    "name_en",
    "aliases",
    "observation_definition",
    "required_actor_roles",
)
MATURITIES = frozenset({"planned", "audited", "production"})
SOURCE_ROLES = ("terminology", "formalization", "calibration")
TEAM_RELATIONS = frozenset({"same", "opponent"})
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CatalogError(ValueError):
    pass


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{field} must be a non-empty string")
    return value


def _normalize_alias(value: str) -> str:
    return "_".join(value.casefold().replace("-", " ").split())


def _validate_source(source: object, *, index: int) -> str:
    if not isinstance(source, dict):
        raise CatalogError(f"source_registry[{index}] must be an object")
    required = (
        "source_id",
        "title",
        "authors",
        "version",
        "url_or_doi",
        "license",
        "accessed_at",
        "content_hash",
        "used_for",
    )
    missing = [field for field in required if field not in source]
    if missing:
        raise CatalogError(f"source_registry[{index}] missing {missing}")

    source_id = _nonempty_string(source["source_id"], "source_id")
    for field in ("title", "version", "url_or_doi", "license"):
        _nonempty_string(source[field], f"{source_id}.{field}")
    accessed_at = source["accessed_at"]
    if not isinstance(accessed_at, date):
        _nonempty_string(accessed_at, f"{source_id}.accessed_at")
    authors = source["authors"]
    if (
        not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
    ):
        raise CatalogError(f"{source_id}.authors must be non-empty strings")
    used_for = source["used_for"]
    if (
        not isinstance(used_for, list)
        or not used_for
        or any(not isinstance(item, str) or not item.strip() for item in used_for)
    ):
        raise CatalogError(f"{source_id}.used_for must be non-empty strings")
    if not isinstance(source["content_hash"], str) or not SHA256_RE.fullmatch(
        source["content_hash"]
    ):
        raise CatalogError(f"{source_id}.content_hash must be sha256:<lowercase hex>")
    return source_id


def _validate_aliases(concept_id: str, aliases: object) -> list[str]:
    if not isinstance(aliases, dict) or not aliases:
        raise CatalogError(f"{concept_id}.aliases must be a non-empty object")
    flattened: list[str] = []
    for language, values in aliases.items():
        _nonempty_string(language, f"{concept_id}.aliases language")
        if not isinstance(values, list):
            raise CatalogError(f"{concept_id}.aliases.{language} must be a list")
        for value in values:
            flattened.append(
                _nonempty_string(value, f"{concept_id}.aliases.{language}")
            )
    normalized = [_normalize_alias(value) for value in flattened]
    if len(normalized) != len(set(normalized)):
        raise CatalogError(f"duplicate alias within {concept_id}")
    return normalized


def _validate_recipe(concept_id: str, recipe: object) -> str:
    if not isinstance(recipe, dict):
        raise CatalogError(f"{concept_id}.recipe must be an object")
    required = ("id", "version", "checker", "parameters")
    missing = [field for field in required if field not in recipe]
    if missing:
        raise CatalogError(f"{concept_id}.recipe missing {missing}")
    if recipe["id"] != concept_id:
        raise CatalogError(f"{concept_id}.recipe.id must equal concept id")
    if (
        not isinstance(recipe["version"], (str, int))
        or isinstance(recipe["version"], bool)
        or str(recipe["version"]).strip() in {"", "0"}
    ):
        raise CatalogError(f"{concept_id}.recipe.version must be non-zero")
    checker = _nonempty_string(recipe["checker"], f"{concept_id}.recipe.checker")
    if not isinstance(recipe["parameters"], dict):
        raise CatalogError(f"{concept_id}.recipe.parameters must be an object")
    return checker


def _validate_concept(
    concept: object,
    *,
    index: int,
    source_ids: set[str],
    checker_names: frozenset[str] | None,
) -> tuple[str, list[str]]:
    if not isinstance(concept, dict):
        raise CatalogError(f"concepts[{index}] must be an object")
    required = (
        "id",
        "name_zh",
        "name_en",
        "aliases",
        "observation_definition",
        "required_actor_roles",
        "actor_team_constraints",
        "sources",
        "allowed_claim_levels",
        "maturity",
        "production_enabled",
    )
    missing = [field for field in required if field not in concept]
    if missing:
        raise CatalogError(f"concepts[{index}] missing {missing}")

    concept_id = _nonempty_string(concept["id"], f"concepts[{index}].id")
    for field in ("name_zh", "name_en", "observation_definition"):
        _nonempty_string(concept[field], f"{concept_id}.{field}")
    aliases = _validate_aliases(concept_id, concept["aliases"])

    roles = concept["required_actor_roles"]
    if (
        not isinstance(roles, list)
        or any(not isinstance(role, str) or not role.strip() for role in roles)
        or len(roles) != len(set(roles))
    ):
        raise CatalogError(f"{concept_id}.required_actor_roles must be unique strings")
    constraints = concept["actor_team_constraints"]
    if not isinstance(constraints, dict) or set(constraints) != set(roles):
        raise CatalogError(
            f"{concept_id}.actor_team_constraints must match required roles"
        )
    if any(value not in TEAM_RELATIONS for value in constraints.values()):
        raise CatalogError(f"{concept_id}.actor_team_constraints has invalid relation")

    sources = concept["sources"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_ROLES):
        raise CatalogError(f"{concept_id}.sources must contain {SOURCE_ROLES}")
    for source_role in SOURCE_ROLES:
        refs = sources[source_role]
        if (
            not isinstance(refs, list)
            or len(refs) != len(set(refs))
            or any(ref not in source_ids for ref in refs)
        ):
            raise CatalogError(f"{concept_id}.sources.{source_role} is invalid")

    if concept["allowed_claim_levels"] != ["observation"]:
        raise CatalogError(f"{concept_id} must allow observation claims only")
    maturity = concept["maturity"]
    if maturity not in MATURITIES:
        raise CatalogError(f"{concept_id}.maturity is invalid")
    if not isinstance(concept["production_enabled"], bool):
        raise CatalogError(f"{concept_id}.production_enabled must be boolean")
    if concept["production_enabled"] and maturity != "production":
        raise CatalogError(f"{concept_id} enabled without production maturity")

    checker = None
    if "recipe" in concept:
        checker = _validate_recipe(concept_id, concept["recipe"])
    if maturity == "production":
        if checker is None:
            raise CatalogError(f"{concept_id} production concept requires recipe")
        gate = concept.get("production_gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise CatalogError(f"{concept_id} production concept requires passed gate")
        if checker_names is not None and checker not in checker_names:
            raise CatalogError(f"{concept_id} checker is not registered")
    return concept_id, aliases


def load_catalog(
    path: Path = DEFAULT_CATALOG_PATH,
    *,
    checker_names: frozenset[str] | None = None,
) -> dict:
    """Load once with yaml.safe_load and validate all invariants."""
    try:
        catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CatalogError(f"cannot load catalog {path}: {error}") from error
    if not isinstance(catalog, dict):
        raise CatalogError("catalog must be an object")
    for field in ("catalog_version", "public_projection_version"):
        _nonempty_string(catalog.get(field), field)

    source_registry = catalog.get("source_registry")
    concepts = catalog.get("concepts")
    candidates = catalog.get("feasibility_candidates")
    if not isinstance(source_registry, list):
        raise CatalogError("source_registry must be a list")
    if not isinstance(concepts, list):
        raise CatalogError("concepts must be a list")
    if not isinstance(candidates, list):
        raise CatalogError("feasibility_candidates must be a list")

    source_ids: set[str] = set()
    for index, source in enumerate(source_registry):
        source_id = _validate_source(source, index=index)
        if source_id in source_ids:
            raise CatalogError(f"duplicate source id: {source_id}")
        source_ids.add(source_id)

    concept_ids: set[str] = set()
    aliases: dict[str, str] = {}
    for index, concept in enumerate(concepts):
        concept_id, normalized_aliases = _validate_concept(
            concept,
            index=index,
            source_ids=source_ids,
            checker_names=checker_names,
        )
        if concept_id in concept_ids:
            raise CatalogError(f"duplicate concept id: {concept_id}")
        concept_ids.add(concept_id)
        normalized_id = _normalize_alias(concept_id)
        if normalized_id in aliases:
            raise CatalogError(f"duplicate concept id or alias: {concept_id}")
        aliases[normalized_id] = concept_id
        for alias in normalized_aliases:
            if alias in aliases:
                raise CatalogError(f"duplicate alias: {alias}")
            aliases[alias] = concept_id

    if (
        any(not isinstance(candidate, str) for candidate in candidates)
        or len(candidates) != len(set(candidates))
        or any(candidate not in concept_ids for candidate in candidates)
    ):
        raise CatalogError("feasibility_candidates must be unique catalog ids")
    return catalog


def get_concept(catalog: dict, concept_id: str) -> dict:
    """Return the exact private entry by canonical ID or alias."""
    normalized = _normalize_alias(
        _nonempty_string(concept_id, "concept_id")
    )
    for concept in catalog.get("concepts", []):
        if _normalize_alias(concept["id"]) == normalized:
            return concept
        for values in concept["aliases"].values():
            if any(_normalize_alias(value) == normalized for value in values):
                return concept
    raise CatalogError(f"unknown concept id or alias: {concept_id}")


def project_public_catalog(catalog: dict, *, concept_ids: set[str]) -> dict:
    """Build a new object from public fields only."""
    known_ids = {concept["id"] for concept in catalog["concepts"]}
    unknown = concept_ids - known_ids
    if unknown:
        raise CatalogError(f"unknown public concept ids: {sorted(unknown)}")
    return {
        "catalog_version": catalog["catalog_version"],
        "public_projection_version": catalog["public_projection_version"],
        "concepts": [
            {
                field: deepcopy(concept[field])
                for field in PUBLIC_FIELDS
            }
            for concept in catalog["concepts"]
            if concept["id"] in concept_ids
        ],
    }
