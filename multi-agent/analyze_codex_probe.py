#!/usr/bin/env python3
"""Compare the paired 4/8 fps Codex probe runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs/codex_probe"
ARMS = (4, 8)
ZERO_HIT_CLASSES = {
    "one-two",
    "tactical-foul",
    "cutback",
    "run-in-behind",
    "back-line-possession-management",
}


def load_results(path: Path) -> dict[str, dict]:
    results = {}
    if not path.is_dir():
        return results
    for file in sorted(path.glob("*.json")):
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("clip"):
            results[value["clip"]] = value
    return results


def _fact(a: dict, name: str) -> bool:
    return a.get("key_spatial_facts", {}).get(name) == "yes"


def _event_after(events: list[dict], index: int, actions: set[str]) -> int | None:
    return next((i for i in range(index + 1, len(events)) if events[i]["action"] in actions), None)


def _one_two(events: list[dict]) -> bool:
    return_actions = pilot.PASSES | {"one_touch_return_pass"}
    for first_i, first in enumerate(events):
        actor = first.get("actor")
        if first.get("action") not in pilot.PASSES or actor in {None, "unknown"}:
            continue
        for return_i in range(first_i + 1, len(events)):
            returned = events[return_i]
            if (
                returned.get("action") not in return_actions
                or returned.get("actor") in {actor, "unknown", None}
                or returned["time"] - first["time"] > 2.5
            ):
                continue
            if any(
                event.get("actor") == actor and event.get("action") in pilot.RECEIVE_ACTIONS
                for event in events[return_i + 1:]
            ):
                return True
    return False


def class_evidence(tactic: str, a: dict, events: list[dict], interval: dict) -> bool:
    lo, hi = interval["start_s"], interval["end_s"]
    window = [event for event in events if lo <= event["time"] <= hi]
    if tactic == "one-two":
        return _one_two(window)
    if tactic == "tactical-foul":
        change = a.get("possession_change")
        if change is None or not lo <= change <= hi:
            return False
        for foul_i, event in enumerate(window):
            if event["action"] not in {"foul", "pull_or_block"} or not 0 <= event["time"] - change <= 6:
                continue
            return any(
                prior["action"] in pilot.FORWARD_ACTIONS and change <= prior["time"] <= event["time"]
                for prior in window[:foul_i]
            )
        return False
    if tactic == "cutback":
        if not (_fact(a, "ball_carrier_reaches_left_byline")
                or _fact(a, "ball_carrier_reaches_right_byline")):
            return False
        pass_i = next((i for i, event in enumerate(window)
                       if event["action"] == "backward_ground_pass"), None)
        return pass_i is not None and _event_after(window, pass_i, {"reception", "shot"}) is not None
    if tactic == "run-in-behind":
        if not _fact(a, "runner_moves_from_last_line_front_to_behind"):
            return False
        pass_i = next((i for i, event in enumerate(window)
                       if event["action"] in pilot.FORWARD_PASSES), None)
        return pass_i is not None and _event_after(window, pass_i, {"reception"}) is not None
    if tactic == "line-breaking-pass":
        if not (_fact(a, "pass_path_crosses_defensive_line")
                and _fact(a, "receiver_controls_between_or_behind_lines")):
            return False
        pass_i = next((i for i, event in enumerate(window)
                       if event["action"] in pilot.FORWARD_PASSES), None)
        return pass_i is not None and _event_after(window, pass_i, {"reception"}) is not None
    if tactic == "back-line-possession-management":
        passes = [event for event in window if event["action"] in pilot.BACK_LINE_PASSES]
        return (
            _fact(a, "ball_circulates_across_back_line")
            and _fact(a, "ball_remains_in_own_half")
            and len(passes) >= 3
            and not any(event["action"] in pilot.FORWARD_ACTIONS for event in window)
        )
    return False


def evidence(result: dict, *, a_override: dict | None = None) -> dict:
    interval = result["gt"]["interval"]
    a = a_override or result.get("agents", {}).get("A", {}).get("output", {})
    events = result.get("agents", {}).get("B", {}).get("output", {}).get("events", [])
    lo, hi = interval["start_s"], interval["end_s"]
    window = [event for event in events if lo <= event["time"] <= hi]
    actors = sorted({
        event["actor"] for event in window
        if event.get("actor") not in {None, "unknown"}
    })
    return {
        "window_hit": bool(window),
        "window_event_count": len(window),
        "window_actors": actors,
        "window_actor_count": len(actors),
        "class_evidence": class_evidence(result["gt"]["tactic"], a, events, interval),
    }


def result_row(result: dict) -> dict:
    gt = result["gt"]["tactic"]
    rule = (result.get("rule") or {}).get("label")
    final = (result.get("final") or {}).get("label")
    status = result.get("status")
    rule_correct = rule == gt
    final_correct = status == "success" and final == gt
    return {
        "clip": result["clip"],
        "gt": gt,
        "status": status,
        "rule": rule,
        "final": final,
        "rule_correct": rule_correct,
        "final_correct": final_correct,
        "c_transition": (
            f"{'correct' if rule_correct else 'wrong'}_to_"
            f"{'correct' if final_correct else 'wrong'}"
        ) if status == "success" else None,
        "elapsed_s": result.get("elapsed_s"),
        **evidence(result),
    }


def summarize_arm(results: dict[str, dict]) -> dict:
    rows = [result_row(result) for result in results.values()]
    complete = [row for row in rows if row["status"] == "success"]
    status_counts = Counter(row["status"] for row in rows)
    per_class = {}
    for tactic in sorted({row["gt"] for row in rows}):
        class_rows = [row for row in rows if row["gt"] == tactic]
        per_class[tactic] = {
            "support": len(class_rows),
            "complete": sum(row["status"] == "success" for row in class_rows),
            "final_correct": sum(row["final_correct"] for row in class_rows),
            "rule_correct": sum(row["rule_correct"] for row in class_rows),
        }
    elapsed = [row["elapsed_s"] for row in complete if isinstance(row["elapsed_s"], (int, float))]
    return {
        "n_results": len(rows),
        "n_complete": len(complete),
        "n_failed": status_counts["failed"],
        "n_partial": status_counts["partial"],
        "failure_rate": round((len(rows) - len(complete)) / len(rows), 3) if rows else None,
        "status_counts": dict(sorted(status_counts.items())),
        "final_correct_n": sum(row["final_correct"] for row in complete),
        "rule_correct_n": sum(row["rule_correct"] for row in complete),
        "window_hit_n": sum(row["window_hit"] for row in complete),
        "class_evidence_n": sum(row["class_evidence"] for row in complete),
        "mean_window_events": round(
            sum(row["window_event_count"] for row in complete) / len(complete), 3
        ) if complete else None,
        "mean_window_actors": round(
            sum(row["window_actor_count"] for row in complete) / len(complete), 3
        ) if complete else None,
        "mean_elapsed_s": round(sum(elapsed) / len(elapsed), 1) if elapsed else None,
        "c_transitions": dict(sorted(Counter(
            row["c_transition"] for row in complete if row["c_transition"]
        ).items())),
        "per_class": per_class,
        "rows": sorted(rows, key=lambda row: row["clip"]),
    }


def repeat_state(root: Path, fps: int, result: dict) -> dict:
    outcomes = []
    statuses = []
    for run in (1, 2, 3):
        path = root / f"fps{fps}" / f"run{run}" / f"{result['clip']}.json"
        if not path.is_file():
            statuses.append("missing")
            outcomes.append(None)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(value.get("status"))
        outcomes.append(
            value.get("final", {}).get("label") == result["gt"]["tactic"]
            if value.get("status") == "success" else None
        )
    complete = all(status == "success" for status in statuses)
    return {
        "clip": result["clip"],
        "fps": fps,
        "gt": result["gt"]["tactic"],
        "statuses": statuses,
        "correct": outcomes,
        "stable_correct": sum(value is True for value in outcomes) >= 2 if complete else None,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {report['material_passport']['origin_date']}",
        "- Verification Status: ANALYZED",
        "- Version Label: codex_probe_comparison_v1",
        "",
        "## Codex 4/8 fps Comparison",
        "",
        "| fps | complete | failed | partial | failure rate | final correct | rule correct | window hit | class evidence | mean elapsed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fps in ARMS:
        arm = report["arms"][f"fps{fps}"]
        lines.append(
            f"| {fps} | {arm['n_complete']}/{arm['n_results']} | {arm['n_failed']} | "
            f"{arm['n_partial']} | {arm['failure_rate']} | {arm['final_correct_n']} | "
            f"{arm['rule_correct_n']} | {arm['window_hit_n']} | {arm['class_evidence_n']} | "
            f"{arm['mean_elapsed_s']} |"
        )
    paired = report["paired"]
    lines += [
        "",
        "### Paired Result",
        "",
        f"- Complete pairs: {paired['n_complete_pairs']}",
        f"- Both correct: {paired['both_correct']}",
        f"- 4 fps only: {paired['fps4_only']}",
        f"- 8 fps only: {paired['fps8_only']}",
        f"- Neither correct: {paired['neither_correct']}",
        "",
        "### Follow-up",
        "",
        f"- Repeat candidates: {len(report['repeat_candidates'])}",
        f"- Stable correct: {len(report['stable_correct'])}",
        f"- New zero-hit-class breakthroughs: {len(report['new_class_breakthroughs'])}",
        f"- Window-probe candidates: {len(report['window_probe_candidates'])}",
        "",
        "### Limits",
        "",
        "- This is an internal, descriptive diagnostic on 16 selected clips.",
        "- Tool traces are not audited. The result cannot establish visual-only inference.",
        "- Historical 1 fps output is not a matched control.",
        "",
    ]
    return "\n".join(lines)


def analyze(root: Path) -> dict:
    runs = {fps: load_results(root / f"fps{fps}" / "run1") for fps in ARMS}
    arms = {f"fps{fps}": summarize_arm(runs[fps]) for fps in ARMS}
    common = sorted(set(runs[4]) & set(runs[8]))
    paired_rows = []
    for clip in common:
        row4, row8 = result_row(runs[4][clip]), result_row(runs[8][clip])
        if row4["status"] != "success" or row8["status"] != "success":
            continue
        paired_rows.append({
            "clip": clip,
            "gt": row4["gt"],
            "fps4": row4,
            "fps8": row8,
        })
    paired = {
        "n_complete_pairs": len(paired_rows),
        "both_correct": sum(row["fps4"]["final_correct"] and row["fps8"]["final_correct"]
                            for row in paired_rows),
        "fps4_only": sum(row["fps4"]["final_correct"] and not row["fps8"]["final_correct"]
                         for row in paired_rows),
        "fps8_only": sum(not row["fps4"]["final_correct"] and row["fps8"]["final_correct"]
                         for row in paired_rows),
        "neither_correct": sum(not row["fps4"]["final_correct"] and not row["fps8"]["final_correct"]
                               for row in paired_rows),
        "rows": paired_rows,
    }

    repeats = []
    for fps in ARMS:
        for result in runs[fps].values():
            if result.get("status") == "success" and result.get("final", {}).get("label") == result["gt"]["tactic"]:
                repeats.append(repeat_state(root, fps, result))
    repeat_candidates = [
        {"clip": item["clip"], "fps": item["fps"], "gt": item["gt"]}
        for item in repeats
        if item["stable_correct"] is None
        and "missing" in item["statuses"]
        and set(item["statuses"]) <= {"success", "missing"}
    ]
    stable = [item for item in repeats if item["stable_correct"] is True]

    window_candidates = [
        row["clip"] for row in paired_rows
        if not row["fps4"]["final_correct"]
        and not row["fps8"]["final_correct"]
        and not row["fps4"]["class_evidence"]
        and not row["fps8"]["class_evidence"]
    ]
    window_results = load_results(root / "window_probe" / "fps8")
    focused = {}
    for clip in window_candidates:
        value = window_results.get(clip)
        if not value or value.get("status") != "success":
            focused[clip] = {"status": (value or {}).get("status", "missing")}
            continue
        pseudo = {
            **runs[8][clip],
            "agents": {
                **runs[8][clip]["agents"],
                "B": value["agents"]["B"],
            },
        }
        focused[clip] = {"status": "success", **evidence(pseudo)}

    created = datetime.now(timezone.utc).isoformat()
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": created,
            "verification_status": "ANALYZED",
            "version_label": "codex_probe_comparison_v1",
        },
        "arms": arms,
        "paired": paired,
        "repeat_candidates": repeat_candidates,
        "repeat_results": repeats,
        "stable_correct": stable,
        "new_class_breakthroughs": [
            item for item in stable if item["gt"] in ZERO_HIT_CLASSES
        ],
        "window_probe_candidates": window_candidates,
        "window_probe_results": focused,
        "interpretation_limits": [
            "internal descriptive diagnostic only",
            "tool traces are not audited",
            "historical 1 fps output is not a matched control",
        ],
    }
    pilot.write_json(root / "comparison.json", report)
    temporary = root / "comparison.md.tmp"
    temporary.write_text(markdown_report(report), encoding="utf-8")
    temporary.replace(root / "comparison.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    report = analyze(args.root)
    print(json.dumps({
        "fps4_correct": report["arms"]["fps4"]["final_correct_n"],
        "fps8_correct": report["arms"]["fps8"]["final_correct_n"],
        "complete_pairs": report["paired"]["n_complete_pairs"],
        "repeat_candidates": report["repeat_candidates"],
        "window_probe_candidates": report["window_probe_candidates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
