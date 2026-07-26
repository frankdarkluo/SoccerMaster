#!/usr/bin/env python3
"""Command line entry point for the offline tactical prototype MVP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactical_prototype_mvp import (  # noqa: E402
    ACTIVE_FIELDS,
    export_sol_bundle,
    import_agent_runs,
    import_reviews,
    import_sol_review,
    migrate,
    register_manifest_source,
    retrieve,
    run_agents,
    select_active,
    stats,
    validate_store,
    verify_candidate,
)


def _json(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("migrate")
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--glossary", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    for name in ("validate", "stats"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)

    command = sub.add_parser("verify")
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--state", type=Path, required=True)
    command.add_argument("--rules", type=Path)

    command = sub.add_parser("retrieve")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--tactic", required=True)
    command.add_argument("--exclude-source-group")
    command.add_argument("--phase")

    command = sub.add_parser("select-active")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--budget", type=int, default=50)
    command.add_argument("--seed", type=int, default=20260716)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("register-source")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--dataset-id", required=True)
    command.add_argument("--directory", required=True)

    command = sub.add_parser("import-agent-runs")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--dataset-id")

    command = sub.add_parser("import-review")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--reviews", type=Path, required=True)

    command = sub.add_parser("run-agents")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--clip-uid", required=True)
    command.add_argument("--providers", default="doubao,gemini")
    command.add_argument("--allow-external-upload", action="store_true")

    command = sub.add_parser("export-sol")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("import-sol")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--reviews", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "migrate":
        _json(migrate(args.data_root, args.glossary, args.out))
    elif args.command == "validate":
        errors = validate_store(args.root)
        _json({"valid": not errors, "errors": errors})
        return int(bool(errors))
    elif args.command == "stats":
        _json(stats(args.root))
    elif args.command == "verify":
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        state = json.loads(args.state.read_text(encoding="utf-8"))
        rules = None
        if args.rules:
            import yaml
            rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
        _json(verify_candidate(candidate, state, rules))
    elif args.command == "retrieve":
        _json(retrieve(args.root, args.tactic, exclude_source_group=args.exclude_source_group, phase=args.phase))
    elif args.command == "select-active":
        rows = select_active(args.root, args.budget, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ACTIVE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        _json({"requested": args.budget, "selected": len(rows), "underfilled_by": args.budget - len(rows), "out": args.out.as_posix()})
    elif args.command == "register-source":
        _json(register_manifest_source(
            args.root, args.data_root, args.dataset_id, args.directory, repo_root=ROOT,
        ))
    elif args.command == "import-agent-runs":
        _json(import_agent_runs(args.root, args.dataset_id))
    elif args.command == "import-review":
        _json(import_reviews(args.root, args.reviews))
    elif args.command == "run-agents":
        _json(run_agents(args.root, args.clip_uid, args.providers.split(","), args.allow_external_upload))
    elif args.command == "export-sol":
        _json(export_sol_bundle(args.root, args.out))
    elif args.command == "import-sol":
        _json({"imported": import_sol_review(args.root, args.reviews)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
