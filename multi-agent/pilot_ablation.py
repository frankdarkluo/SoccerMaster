#!/usr/bin/env python3
"""Run attributable Doubao B/C ablations against the frozen r3 pilot outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pilot


ROOT = pilot.ROOT
SOURCE_RUN = pilot.OUTPUT_ROOT / "r3"
OUTPUT_ROOT = pilot.OUTPUT_ROOT / "ablations"
C_SMOKE = ("SNGS-045", "SNGS-078", "SNGS-049", "SNGS-058", "SNGS-111")
B_POSITIVES = ("SNGS-022", "SNGS-031", "SNGS-039", "SNGS-137", "SNGS-184", "SNGS-198")
B_NEGATIVES = ("SNGS-141", "SNGS-049", "SNGS-160", "SNGS-058", "SNGS-155", "SNGS-173")
B_CLIPS = B_POSITIVES + B_NEGATIVES
B_ARMS = {
    "baseline6": (6, False),
    "two-leg6": (6, True),
    "baseline10": (10, False),
    "two-leg10": (10, True),
}
PASS_ACTIONS = pilot.PASSES | {"one_touch_return_pass"}
MAX_ATTEMPTS = 2
PAIR_WINDOW_S = 2.5


def json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_matches(result: dict[str, Any], config: dict[str, Any]) -> bool:
    return result.get("config_signature") == json_hash(config)


def run_attempts(
    target: Path,
    base: dict[str, Any],
    config: dict[str, Any],
    invoke: Callable[[], dict[str, Any]],
    *,
    resume: bool,
) -> dict[str, Any]:
    if target.is_file():
        result = read_json(target)
        if not result_matches(result, config):
            raise RuntimeError(f"configuration mismatch for {target}")
        if not resume:
            raise FileExistsError(f"result exists; pass --resume: {target}")
        if result.get("status") == "success" or len(result.get("attempt_history", [])) >= MAX_ATTEMPTS:
            return result
    else:
        result = {
            **base,
            "status": "pending",
            "config": config,
            "config_signature": json_hash(config),
            "attempt_history": [],
        }

    while result["status"] != "success" and len(result["attempt_history"]) < MAX_ATTEMPTS:
        number = len(result["attempt_history"]) + 1
        started = time.monotonic()
        try:
            agent = invoke()
            attempt = {
                "attempt": number,
                "status": "success",
                "elapsed_s": round(time.monotonic() - started, 3),
                "agent": agent,
            }
            result.update(status="success", agent=agent)
        except Exception as exc:
            attempt = {
                "attempt": number,
                "status": "failed",
                "elapsed_s": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        result["attempt_history"].append(attempt)
        result["first_attempt_failed"] = result["attempt_history"][0]["status"] == "failed"
        result["recovered"] = result["first_attempt_failed"] and result["status"] == "success"
        pilot.write_json(target, result)
    return result


def call_video_agent(
    video_path: Path,
    prompt: str,
    schema: dict[str, Any],
    fps: float,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    generator: pilot.Generator = pilot.generate_json,
) -> dict[str, Any]:
    started = time.monotonic()
    with pilot.temporary_compact_clip(video_path, **pilot.VIDEO_PREPROCESS) as upload:
        payload, usage, transport_attempts = generator(
            pilot.PROVIDER,
            prompt,
            schema,
            video_path=upload,
            fps=fps,
            temperature=pilot.TEMPERATURE,
            retries=0,
        )
    validator(payload)
    return {
        "output": payload,
        "usage": usage,
        "transport_attempts": transport_attempts,
        "elapsed_s": round(time.monotonic() - started, 3),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": json_hash(schema),
        "fps": fps,
    }


def gt_by_clip() -> dict[str, dict[str, Any]]:
    return {row["clip"]: row for row in pilot.load_gt()}


def source_results() -> dict[str, dict[str, Any]]:
    return {
        path.stem: read_json(path)
        for path in sorted((SOURCE_RUN / "clips").glob("*.json"))
    }


def source_manifest() -> dict[str, Any]:
    manifest = read_json(SOURCE_RUN / "manifest.json")
    if not isinstance(manifest.get("criteria_snapshot"), dict):
        raise ValueError("r3 manifest has no frozen criteria_snapshot")
    return manifest


def video_path(result: dict[str, Any]) -> Path:
    path = Path(result["video_path"])
    return path if path.is_absolute() else ROOT / path


def c_transition(gt: str, rule: str | None, final: str | None) -> str:
    return f"{'correct' if rule == gt else 'wrong'}_to_{'correct' if final == gt else 'wrong'}"


def run_c_replay(
    phase: str,
    replicate: int,
    *,
    resume: bool,
    generator: pilot.Generator = pilot.generate_json,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, int]:
    sources = source_results()
    manifest = source_manifest()
    criteria = manifest["criteria_snapshot"]
    if phase == "smoke":
        if replicate != 1:
            raise ValueError("C smoke only supports replicate 1")
        clips = list(C_SMOKE)
    elif replicate == 1:
        clips = sorted(clip for clip, result in sources.items() if result.get("status") == "success")
    else:
        run1 = output_root / "c_prompt" / "run1"
        clips = sorted(
            path.stem for path in run1.glob("*.json")
            if (value := read_json(path)).get("final", {}).get("label") != value.get("rule", {}).get("label")
        )
    counts = Counter(selected=len(clips), success=0, failed=0, skipped=0)
    for clip in clips:
        source = sources.get(clip)
        if not source or source.get("status") != "success":
            raise ValueError(f"C replay requires a successful r3 source: {clip}")
        a = source["agents"]["A"]["output"]
        b = source["agents"]["B"]["output"]
        rule = source["rule"]
        candidates = pilot.c_candidates(rule)
        prompt = pilot.build_c_prompt(a, b, rule, criteria)
        schema = pilot.C_SCHEMA
        path = video_path(source)
        config = {
            "mode": "c-replay",
            "replicate": replicate,
            "provider": pilot.PROVIDER,
            "model": pilot.model_name(pilot.PROVIDER),
            "temperature": pilot.TEMPERATURE,
            "fps": pilot.MODEL_FPS["C"],
            "video_preprocess": pilot.VIDEO_PREPROCESS,
            "video_sha256": source["video_sha256"],
            "source_manifest_signature": manifest["signature"],
            "source_a_sha256": json_hash(a),
            "source_b_sha256": json_hash(b),
            "source_rule_sha256": json_hash(rule),
            "criteria_snapshot_sha256": manifest["criteria_snapshot_sha256"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": json_hash(schema),
        }
        base = {
            "mode": "c-replay",
            "replicate": replicate,
            "clip": clip,
            "video_path": source["video_path"],
            "video_sha256": source["video_sha256"],
            "gt": source["gt"],
            "source_run": "r3",
            "source_manifest_signature": manifest["signature"],
            "agents": {"A": source["agents"]["A"], "B": source["agents"]["B"]},
            "rule": rule,
            "old_final": source.get("final"),
        }
        target = output_root / "c_prompt" / f"run{replicate}" / f"{clip}.json"
        existed = target.is_file()
        result = run_attempts(
            target,
            base,
            config,
            lambda: call_video_agent(
                path, prompt, schema, pilot.MODEL_FPS["C"],
                lambda payload: pilot.validate_c(payload, candidates), generator,
            ),
            resume=resume,
        )
        verdict = result.get("agent", {}).get("output") if result.get("status") == "success" else None
        result["agents"]["C"] = result.get("agent", {"error_type": result.get("error_type"), "error": result.get("error")})
        result["final"] = {
            "label": verdict["label"] if verdict else rule.get("label"),
            "source": "arbiter" if verdict else "rule",
            "candidates": candidates,
        }
        result["transition"] = c_transition(source["gt"]["tactic"], rule.get("label"), result["final"]["label"])
        pilot.write_json(target, result)
        if existed and resume and len(result.get("attempt_history", [])) != 1:
            counts["skipped" if result["status"] == "success" else "failed"] += 1
        else:
            counts[result["status"]] += 1
    return dict(counts)


def matching_prior_a(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    run_dir = pilot.OUTPUT_ROOT / "r2_pro"
    manifest = read_json(run_dir / "manifest.json")
    source = read_json(run_dir / "clips" / "SNGS-039.json")
    expected = {
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "temperature": manifest.get("temperature"),
        "fps": manifest.get("fps", {}).get("A"),
        "video_preprocess": manifest.get("video_preprocess"),
        "video_sha256": source.get("video_sha256"),
        "prompt_sha256": manifest.get("prompt_sha256", {}).get("A"),
    }
    agent = source.get("agents", {}).get("A")
    if source.get("status") != "success" or expected != {key: config.get(key) for key in expected} or not agent:
        return None
    pilot.validate_a(agent["output"])
    provenance = {
        "run": "r2_pro",
        "manifest_signature": manifest["signature"],
        "clip_sha256": json_hash(source),
        "agent_sha256": json_hash(agent),
        "matched_config": expected,
    }
    return agent, provenance


def ensure_fixed_a(
    *,
    resume: bool,
    generator: pilot.Generator,
    output_root: Path,
) -> dict[str, Any]:
    gt = gt_by_clip()["SNGS-039"]
    prompt, schema = pilot.A_PROMPT, pilot.A_SCHEMA
    config = {
        "mode": "fixed-a",
        "clip": "SNGS-039",
        "provider": pilot.PROVIDER,
        "model": pilot.model_name(pilot.PROVIDER),
        "temperature": pilot.TEMPERATURE,
        "fps": pilot.MODEL_FPS["A"],
        "video_preprocess": pilot.VIDEO_PREPROCESS,
        "video_sha256": pilot.sha256_file(gt["video_path"]),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": json_hash(schema),
    }
    base = {
        "mode": "fixed-a",
        "clip": "SNGS-039",
        "video_path": gt["video_path"].relative_to(ROOT).as_posix(),
        "video_sha256": config["video_sha256"],
        "gt": {"tactic": gt["tactic"], "interval": gt["interval"]},
    }
    target = output_root / "b_events" / "fixed_a" / "SNGS-039.json"
    result = run_attempts(
        target,
        base,
        config,
        lambda: call_video_agent(
            gt["video_path"], prompt, schema, pilot.MODEL_FPS["A"], pilot.validate_a, generator,
        ),
        resume=resume,
    )
    if result.get("status") != "success" and (prior := matching_prior_a(config)):
        result.update(
            status="success", agent=prior[0], fixed_a_resolution="reused_matching_prior",
            reuse_source=prior[1],
        )
        pilot.write_json(target, result)
    return result


def run_b_probe(
    arm: str,
    replicate: int,
    *,
    resume: bool,
    generator: pilot.Generator = pilot.generate_json,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, int]:
    max_items, two_leg = B_ARMS[arm]
    prompt = pilot.build_b_prompt(max_items, two_leg)
    schema = pilot.build_b_schema(max_items)
    sources = source_results()
    rows = gt_by_clip()
    fixed_a = ensure_fixed_a(resume=True, generator=generator, output_root=output_root)
    if fixed_a.get("status") != "success":
        raise RuntimeError("SNGS-039 fixed A failed after two attempts")
    counts = Counter(selected=len(B_CLIPS), success=0, failed=0, skipped=0)
    for clip in B_CLIPS:
        gt = rows[clip]
        source = sources.get(clip)
        if clip == "SNGS-039":
            a_agent = fixed_a["agent"]
        elif source and source.get("status") == "success":
            a_agent = source["agents"]["A"]
        else:
            raise ValueError(f"B probe has no fixed A for {clip}")
        a = a_agent["output"]
        path = gt["video_path"]
        video_sha = pilot.sha256_file(path)
        config = {
            "mode": "b-probe",
            "arm": arm,
            "replicate": replicate,
            "provider": pilot.PROVIDER,
            "model": pilot.model_name(pilot.PROVIDER),
            "temperature": pilot.TEMPERATURE,
            "fps": pilot.MODEL_FPS["B"],
            "video_preprocess": pilot.VIDEO_PREPROCESS,
            "b_max_items": max_items,
            "b_prompt_variant": "two-leg" if two_leg else "baseline",
            "video_sha256": video_sha,
            "source_a_sha256": json_hash(a),
            "rule_sha256": pilot.sha256_file(Path(pilot.__file__)),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": json_hash(schema),
        }
        base = {
            "mode": "b-probe",
            "arm": arm,
            "replicate": replicate,
            "clip": clip,
            "video_path": path.relative_to(ROOT).as_posix(),
            "video_sha256": video_sha,
            "gt": {"tactic": gt["tactic"], "interval": gt["interval"]},
            "agents": {"A": a_agent},
        }
        target = output_root / "b_events" / arm / f"run{replicate}" / f"{clip}.json"
        existed = target.is_file()
        result = run_attempts(
            target,
            base,
            config,
            lambda: call_video_agent(
                path, prompt, schema, pilot.MODEL_FPS["B"],
                lambda payload: pilot.validate_b(payload, max_items), generator,
            ),
            resume=resume,
        )
        result["agents"]["B"] = result.get("agent", {"error_type": result.get("error_type"), "error": result.get("error")})
        if result.get("status") == "success":
            result["rule"] = pilot.classify(a, result["agent"]["output"], max_items)
        pilot.write_json(target, result)
        if existed and resume and len(result.get("attempt_history", [])) != 1:
            counts["skipped" if result["status"] == "success" else "failed"] += 1
        else:
            counts[result["status"]] += 1
    return dict(counts)


def events_in_window(events: Iterable[dict[str, Any]], interval: dict[str, float], padding: float = 0.0) -> list[dict[str, Any]]:
    lo = max(0.0, interval["start_s"] - padding)
    hi = min(30.0, interval["end_s"] + padding)
    return [event for event in events if lo <= event["time"] <= hi]


def pair_evidence(events: Iterable[dict[str, Any]], interval: dict[str, float], padding: float = 0.0) -> bool:
    passes = [event for event in events_in_window(events, interval, padding) if event["action"] in PASS_ACTIONS]
    return any(0 < later["time"] - first["time"] <= PAIR_WINDOW_S
               for index, first in enumerate(passes) for later in passes[index + 1:])


def confirmed_xyx(events: Iterable[dict[str, Any]], interval: dict[str, float]) -> bool:
    window = events_in_window(events, interval)
    for first_i, first in enumerate(window):
        if first["action"] not in pilot.PASSES or first["actor"] == "unknown":
            continue
        for return_i in range(first_i + 1, len(window)):
            returned = window[return_i]
            if returned["action"] not in PASS_ACTIONS or not 0 < returned["time"] - first["time"] <= PAIR_WINDOW_S:
                continue
            if returned["actor"] in {"unknown", first["actor"]}:
                continue
            if any(event["actor"] == first["actor"] and event["action"] in pilot.RECEIVE_ACTIONS
                   for event in window[return_i + 1:]):
                return True
    return False


def load_ablation_run(path: Path) -> dict[str, dict[str, Any]]:
    return {file.stem: read_json(file) for file in sorted(path.glob("*.json"))}


def run_stats(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    first_failures = sum(value.get("first_attempt_failed") is True for value in values)
    recovered = sum(value.get("recovered") is True for value in values)
    final_failures = sum(value.get("status") != "success" for value in values)
    elapsed = [sum(float(attempt.get("elapsed_s", 0.0)) for attempt in value.get("attempt_history", [])) for value in values]
    total = len(values)
    return {
        "units": total,
        "success": total - final_failures,
        "first_attempt_failures": first_failures,
        "first_attempt_failure_rate": first_failures / total if total else 0.0,
        "recovered": recovered,
        "recovery_rate": recovered / first_failures if first_failures else 0.0,
        "final_failures": final_failures,
        "final_failure_rate": final_failures / total if total else 0.0,
        "elapsed_total_s": round(sum(elapsed), 3),
        "elapsed_median_s": round(statistics.median(elapsed), 3) if elapsed else 0.0,
    }


def b_clip_metrics(result: dict[str, Any]) -> dict[str, Any]:
    output = result.get("agents", {}).get("B", {}).get("output", {})
    events = output.get("events", [])
    interval = result["gt"]["interval"]
    window = events_in_window(events, interval)
    return {
        "strict_pair": pair_evidence(events, interval),
        "padded_pair": pair_evidence(events, interval, 1.0),
        "explicit_return": any(event["action"] == "one_touch_return_pass" for event in window),
        "actor_count": len({event["actor"] for event in window if event["actor"] != "unknown"}),
        "confirmed_xyx": confirmed_xyx(events, interval),
        "one_two_top1": result.get("rule", {}).get("label") == "one-two",
    }


def majority(values: list[bool]) -> bool:
    return len(values) == 3 and sum(values) >= 2


def summarize_b_arm(output_root: Path, arm: str) -> dict[str, Any]:
    runs = [load_ablation_run(output_root / "b_events" / arm / f"run{rep}") for rep in (1, 2, 3)]
    run1 = runs[0]
    initial_success = all(run1.get(clip, {}).get("status") == "success" for clip in B_CLIPS)
    initial = {
        "run_stats": run_stats(run1.values()),
        "positive_hits": sum(b_clip_metrics(run1[clip])["strict_pair"] for clip in B_POSITIVES if run1.get(clip, {}).get("status") == "success"),
        "negative_false_positives": sum(b_clip_metrics(run1[clip])["one_two_top1"] for clip in B_NEGATIVES if run1.get(clip, {}).get("status") == "success"),
    }
    initial["qualifies_for_repeats"] = initial_success and initial["positive_hits"] >= 4 and initial["negative_false_positives"] <= 1
    stable_ready = all(all(clip in run for run in runs) for clip in B_CLIPS)
    stable_positive = sum(
        majority([run[clip].get("status") == "success" and b_clip_metrics(run[clip])["strict_pair"] for run in runs])
        for clip in B_POSITIVES
    ) if stable_ready else None
    stable_negative = sum(
        majority([run[clip].get("status") == "success" and b_clip_metrics(run[clip])["one_two_top1"] for run in runs])
        for clip in B_NEGATIVES
    ) if stable_ready else None
    return {
        "arm": arm,
        "initial": initial,
        "stable_ready": stable_ready,
        "stable_positive_hits": stable_positive,
        "stable_negative_false_positives": stable_negative,
        "rows": {
            clip: [b_clip_metrics(run[clip]) if run.get(clip, {}).get("status") == "success" else None for run in runs]
            for clip in B_CLIPS
        },
    }


def stable_c_label(results: list[dict[str, Any]]) -> str | None:
    labels = [result.get("final", {}).get("label") for result in results]
    if len(results) == 1 and labels[0] == results[0].get("rule", {}).get("label"):
        return labels[0]
    if len(results) != 3:
        return None
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label if count >= 2 else None


def summarize_c(output_root: Path) -> dict[str, Any]:
    runs = [load_ablation_run(output_root / "c_prompt" / f"run{rep}") for rep in (1, 2, 3)]
    run1 = runs[0]
    smoke_complete = all(run1.get(clip, {}).get("status") == "success" for clip in C_SMOKE)
    smoke_safe = smoke_complete and all(
        run1[clip].get("final", {}).get("label") == run1[clip].get("rule", {}).get("label")
        for clip in ("SNGS-045", "SNGS-078")
    )
    full_complete = len(run1) == 35 and all(value.get("status") == "success" for value in run1.values())
    changed = [clip for clip, value in run1.items() if value.get("status") == "success" and value.get("final", {}).get("label") != value.get("rule", {}).get("label")]
    stable = Counter()
    pending = []
    for clip, value in run1.items():
        available = [run[clip] for run in runs if clip in run and run[clip].get("status") == "success"]
        label = stable_c_label(available)
        if label is None and clip in changed:
            pending.append(clip)
            continue
        stable[c_transition(value["gt"]["tactic"], value["rule"].get("label"), label)] += 1
    first_failures = sum(value.get("first_attempt_failed") is True for value in run1.values())
    accepted = (
        full_complete and not pending and first_failures <= 1
        and stable["correct_to_wrong"] == 0 and stable["wrong_to_correct"] >= 1
    )
    return {
        "smoke_complete": smoke_complete,
        "smoke_pass": smoke_safe,
        "run1_stats": run_stats(run1.values()),
        "changed_run1": changed,
        "pending_repeats": pending,
        "stable_transitions": dict(stable),
        "accepted_for_r4": accepted,
    }


def analyze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    b = {arm: summarize_b_arm(output_root, arm) for arm in B_ARMS}
    baseline = b["baseline6"]
    eligible = []
    if baseline["stable_ready"]:
        for arm in ("two-leg6", "baseline10", "two-leg10"):
            item = b[arm]
            item["passes_absolute_and_relative"] = bool(
                item["stable_ready"]
                and item["stable_positive_hits"] >= 4
                and item["stable_negative_false_positives"] <= 1
                and item["stable_positive_hits"] >= baseline["stable_positive_hits"] + 2
                and item["stable_negative_false_positives"] <= baseline["stable_negative_false_positives"]
            )
            if item["passes_absolute_and_relative"]:
                eligible.append(item)
    selected = None
    if eligible:
        preference = {"two-leg6": 0, "baseline10": 1, "two-leg10": 2}
        selected = sorted(
            eligible,
            key=lambda item: (-item["stable_positive_hits"], item["stable_negative_false_positives"], preference[item["arm"]]),
        )[0]["arm"]
    c = summarize_c(output_root)
    initial_arms_complete = all(b[arm]["initial"]["run_stats"]["units"] == len(B_CLIPS) for arm in ("baseline6", "two-leg6", "baseline10"))
    verification_status = "ANALYZED" if initial_arms_complete else "PARTIAL"
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": verification_status,
            "version_label": "doubao_bc_ablation_v1",
        },
        "c": c,
        "b": b,
        "decision": {
            "b_selected": selected,
            "b_initial_arms_complete": initial_arms_complete,
            "c_stopped_after_smoke": not c["smoke_pass"],
            "c_accepted": c["accepted_for_r4"],
            "ready_for_r4": selected is not None or c["accepted_for_r4"],
            "run_two_leg10_next": (
                selected is None
                and b["two-leg6"]["stable_ready"]
                and b["baseline10"]["stable_ready"]
                and not b["two-leg10"]["initial"]["qualifies_for_repeats"]
            ),
        },
    }
    pilot.write_json(output_root / "comparison.json", report)
    c_stats = c["run1_stats"]
    lines = [
        "## Material Passport", "", "- Origin Skill: experiment-agent", "- Origin Mode: run",
        f"- Origin Date: {report['material_passport']['origin_date']}", f"- Verification Status: {verification_status}",
        "- Version Label: doubao_bc_ablation_v1", "", "# Doubao B/C Ablation", "",
        f"- C smoke pass: {c['smoke_pass']}",
        f"- C run1: {c_stats['success']}/{c_stats['units']} success; first failures {c_stats['first_attempt_failures']}; "
        f"final failures {c_stats['final_failures']}; elapsed {c_stats['elapsed_total_s'] / 60:.1f} min",
        f"- C accepted for r4: {c['accepted_for_r4']}",
        f"- B initial arms complete: {initial_arms_complete}",
        f"- B selected: {selected}", f"- Ready for r4: {report['decision']['ready_for_r4']}", "",
        "| B arm | success | first fail | final fail | elapsed min | positive | negative FP | stable + | stable FP | passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, item in b.items():
        stats = item["initial"]["run_stats"]
        lines.append(
            f"| {arm} | {stats['success']}/{stats['units']} | {stats['first_attempt_failures']} | "
            f"{stats['final_failures']} | {stats['elapsed_total_s'] / 60:.1f} | "
            f"{item['initial']['positive_hits']} | {item['initial']['negative_false_positives']} | "
            f"{item['stable_positive_hits']} | {item['stable_negative_false_positives']} | "
            f"{item.get('passes_absolute_and_relative', False)} |"
        )
    (output_root / "comparison.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    c = commands.add_parser("c-replay")
    c.add_argument("--phase", required=True, choices=("smoke", "all"))
    c.add_argument("--replicate", type=int, default=1, choices=(1, 2, 3))
    c.add_argument("--resume", action="store_true")
    c.add_argument("--allow-external-upload", action="store_true")
    b = commands.add_parser("b-probe")
    b.add_argument("--arm", required=True, choices=tuple(B_ARMS))
    b.add_argument("--replicate", type=int, default=1, choices=(1, 2, 3))
    b.add_argument("--resume", action="store_true")
    b.add_argument("--allow-external-upload", action="store_true")
    commands.add_parser("analyze")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command in {"c-replay", "b-probe"} and not args.allow_external_upload:
        raise SystemExit("Refusing external video upload without --allow-external-upload")
    if args.command in {"c-replay", "b-probe"} and not os.environ.get("ARK_API_KEY"):
        raise SystemExit("ARK_API_KEY is not set; load the project environment before running")
    if args.command == "c-replay":
        value = run_c_replay(args.phase, args.replicate, resume=args.resume)
    elif args.command == "b-probe":
        value = run_b_probe(args.arm, args.replicate, resume=args.resume)
    else:
        value = analyze()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
