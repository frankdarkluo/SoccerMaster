#!/usr/bin/env python3
"""Offline k=1/k=3 and per-tactic analysis for open tactical-tag runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa.opentac import claim_match, load_jsonl


OLD_TO_CURRENT = {
    "fast_break_pattern": "counter-attack",
    "line_break": "line-breaking-pass",
    "run_in_behind": "run-in-behind",
    "corner-near-far-post": "corner-near-far-post",
    "cutback": "cutback",
    "halfspace-penetration": "halfspace-penetration",
    "between-the-lines": "between-the-lines",
    "gk-in-buildup": "gk-in-buildup",
    "dummy-run": "dummy-run",
    "switch-of-play": "switch-of-play",
    "one_two": "one-two",
}


def load_runs(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempted = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]
    return attempted, {item["clip_uid"]: item for item in attempted if item.get("status") == "success"}


def prediction(run: dict[str, Any], claim: dict[str, Any], mode: str) -> str | None:
    samples = run.get("assessment_samples") or []
    candidates = set(run.get("screen", {}).get("candidate_tactic_ids", []))
    if mode == "k1":
        if len(samples) < 1:
            return None
        assessments = samples[0]
    elif mode == "k3":
        if len(samples) < 3:
            return None
        assessments = run.get("assessments", [])
    else:
        raise ValueError(f"unknown mode: {mode}")
    if claim["tactic_id"] not in candidates:
        return None
    return "present" if claim_match({"assessments": assessments}, claim, 0) else "absent"


def summarize(rows: list[tuple[dict[str, Any], str | None]]) -> dict[str, Any]:
    covered = [(claim, pred) for claim, pred in rows if pred is not None]
    tp = sum(pred == "present" and claim["gt_verdict"] == "present" for claim, pred in covered)
    fp = sum(pred == "present" and claim["gt_verdict"] == "absent" for claim, pred in covered)
    fn = sum(pred == "absent" and claim["gt_verdict"] == "present" for claim, pred in covered)
    tn = sum(pred == "absent" and claim["gt_verdict"] == "absent" for claim, pred in covered)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "expected_claims": len(rows),
        "covered_claims": len(covered),
        "abstained_claims": len(rows) - len(covered),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": None if precision is None or recall is None else 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "covered_accuracy": (tp + tn) / len(covered) if covered else None,
    }


def claims_for_mode(claims: list[dict[str, Any]], runs: dict[str, dict[str, Any]], mode: str) -> list[tuple[dict[str, Any], str | None]]:
    return [
        (claim, prediction(runs[claim["clip_uid"]], claim, mode))
        for claim in claims
        if claim["clip_uid"] in runs
    ]


def per_tactic(rows: list[tuple[dict[str, Any], str | None]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], str | None]]] = defaultdict(list)
    for claim, pred in rows:
        grouped[claim["tactic_id"]].append((claim, pred))
    return {tactic_id: summarize(items) for tactic_id, items in sorted(grouped.items())}


def consistency(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_cards = unanimous = pair_matches = pair_total = vote_changes = 0
    clip_rows = []
    for clip_uid, run in sorted(runs.items()):
        samples = run.get("assessment_samples") or []
        if len(samples) < 3:
            continue
        by_tactic = defaultdict(list)
        for sample in samples:
            for item in sample:
                by_tactic[item["tactic_id"]].append(item["verdict"])
        clip_changes = 0
        for tactic_id, verdicts in sorted(by_tactic.items()):
            if len(verdicts) != 3:
                continue
            total_cards += 1
            unanimous += len(set(verdicts)) == 1
            pair_matches += sum(verdicts[i] == verdicts[j] for i in range(3) for j in range(i + 1, 3))
            pair_total += 3
            final = next(item for item in run["assessments"] if item["tactic_id"] == tactic_id)["verdict"]
            if verdicts[0] != final:
                vote_changes += 1
                clip_changes += 1
        clip_rows.append({"clip_uid": clip_uid, "cards": len(by_tactic), "sample0_to_vote_changes": clip_changes})
    return {
        "three_sample_clips": len(clip_rows),
        "three_sample_card_decisions": total_cards,
        "unanimous_card_decisions": unanimous,
        "unanimous_rate": unanimous / total_cards if total_cards else None,
        "pairwise_verdict_agreement": pair_matches / pair_total if pair_total else None,
        "k1_to_k3_verdict_changes": vote_changes,
        "clips": clip_rows,
    }


def review_ledger(rows: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tactic_id = OLD_TO_CURRENT.get(row["tactic_id"])
        key = tactic_id or f"unmapped:{row['tactic_id']}"
        grouped[key].append(row)
        if tactic_id and row["clip_uid"] in runs:
            pred = prediction(runs[row["clip_uid"]], {"tactic_id": tactic_id, "window": row["window"]}, "k3")
            comparisons[tactic_id].append({
                "clip_uid": row["clip_uid"],
                "old_verdict": row["verdict"],
                "current_prediction": pred or "abstain",
                "old_error_fixed": row["verdict"] == "wrong" and pred == "absent",
                "old_correct_preserved": row["verdict"] == "correct" and pred == "present",
            })

    ledger = {}
    for tactic_id, items in sorted(grouped.items()):
        counts = Counter(item["verdict"] for item in items)
        ledger[tactic_id] = {"rows": len(items), "old_correct": counts["correct"], "old_wrong": counts["wrong"]}
    comparison = {}
    for tactic_id, items in sorted(comparisons.items()):
        counts = Counter(item["current_prediction"] for item in items)
        comparison[tactic_id] = {
            "review_rows_seen": len(items),
            "current_present": counts["present"],
            "current_absent": counts["absent"],
            "current_abstain": counts["abstain"],
            "old_error_fixed": sum(item["old_error_fixed"] for item in items),
            "old_correct_preserved": sum(item["old_correct_preserved"] for item in items),
        }
    return ledger, comparison


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    attempted, runs = load_runs(args.runs)
    claims = [row for row in load_jsonl(args.ground_truth) if row.get("score_set") == "primary"]
    review = load_jsonl(args.review)
    k1_rows = claims_for_mode(claims, runs, "k1")
    k3_rows = claims_for_mode(claims, runs, "k3")
    changes = [
        {"clip_uid": claim["clip_uid"], "tactic_id": claim["tactic_id"], "gt_verdict": claim["gt_verdict"], "k1": k1, "k3": k3}
        for (claim, k1), (_, k3) in zip(k1_rows, k3_rows)
        if k1 is not None and k3 is not None and k1 != k3
    ]
    ledger, comparison = review_ledger(review, runs)
    return {
        "input": str(args.runs),
        "execution": {
            "attempted_files": len(attempted),
            "successful_files": len(runs),
            "screened_files": sum("screen" in run for run in runs.values()),
            "screened_with_candidates": sum(bool(run.get("screen", {}).get("candidate_tactic_ids")) for run in runs.values()),
            "scored_files": sum(bool(run.get("assessment_samples")) for run in runs.values()),
            "scored_with_3_samples": sum(len(run.get("assessment_samples") or []) >= 3 for run in runs.values()),
        },
        "k1_vs_k3": {
            "consistency": consistency(runs),
            "claim_k1": summarize(k1_rows),
            "claim_k3": summarize(k3_rows),
            "verdict_changes": changes,
        },
        "primary_claims_by_tactic": {"k1": per_tactic(k1_rows), "k3": per_tactic(k3_rows)},
        "recognition_review_ledger_old_ids": ledger,
        "recognition_review_vs_current_k3": comparison,
        "caveat": "recognition_review is the old error ledger; no-matching/empty candidates are abstain, not clean negative evidence.",
    }


def markdown(report: dict[str, Any]) -> str:
    execution = report["execution"]
    comparison = report["k1_vs_k3"]
    lines = [
        "# Offline tactical assessment analysis",
        "",
        f"Input: `{report['input']}`",
        "",
        "## Coverage",
        "",
        "| attempted | success | screened | candidates | scored | score k=3 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {execution['attempted_files']} | {execution['successful_files']} | {execution['screened_files']} | {execution['screened_with_candidates']} | {execution['scored_files']} | {execution['scored_with_3_samples']} |",
        "",
        "Empty candidates are abstentions; they are not counted as model-verified absences.",
        "",
        "## k=1 vs k=3",
        "",
        f"3-sample clips: {comparison['consistency']['three_sample_clips']}; card-level unanimous rate: {comparison['consistency']['unanimous_rate']}; pairwise verdict agreement: {comparison['consistency']['pairwise_verdict_agreement']}; sample-0 → vote changes: {comparison['consistency']['k1_to_k3_verdict_changes']}.",
        "",
        "| mode | expected claims | covered | abstain | precision | recall | f1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("claim_k1", "claim_k3"):
        item = comparison[mode]
        lines.append(f"| {mode[6:]} | {item['expected_claims']} | {item['covered_claims']} | {item['abstained_claims']} | {item['precision']} | {item['recall']} | {item['f1']} |")
    lines += ["", "## Primary claims by tactic (k=3)", "", "| tactic | n | covered | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|"]
    for tactic_id, item in report["primary_claims_by_tactic"]["k3"].items():
        lines.append(f"| {tactic_id} | {item['expected_claims']} | {item['covered_claims']} | {item['tp']} | {item['fp']} | {item['fn']} | {item['tn']} |")
    lines += ["", "## Old error ledger", "", "Grouped by current tactic ID; this is not a clean negative set.", "", "| tactic | old correct | old wrong | current rows seen | fixed | preserved |", "|---|---:|---:|---:|---:|---:|"]
    for tactic_id, item in report["recognition_review_ledger_old_ids"].items():
        current = report["recognition_review_vs_current_k3"].get(tactic_id, {})
        lines.append(f"| {tactic_id} | {item['old_correct']} | {item['old_wrong']} | {current.get('review_rows_seen', 0)} | {current.get('old_error_fixed', 0)} | {current.get('old_correct_preserved', 0)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/tactical_claim_benchmark/opentac/phase2_observation_first/gemini")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl")
    parser.add_argument("--review", type=Path, default=ROOT / "benchmark/tactical_prototypes/recognition_review.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/tactical_claim_benchmark/opentac/evaluation/offline_analysis.json")
    args = parser.parse_args()
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["execution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
