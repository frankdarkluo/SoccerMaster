#!/usr/bin/env python3
"""Run the r4 Codex and direct C-only model comparisons on the frozen 37 clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_probe
import pilot
import pilot_ablation


MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "xhigh"
HEIGHT = 720
B_MAX_ITEMS = 15
FPS = {"A": 2.0, "B": 4.0, "C": 2.0}
MAX_ATTEMPTS = 2
OUTPUT_ROOT = pilot.OUTPUT_ROOT
R4_CODEX_ROOT = OUTPUT_ROOT / "r4_codex"
C_ONLY_ROOT = OUTPUT_ROOT / "c_only"
C_ONLY_DIRS = {
    "codex": C_ONLY_ROOT / "codex_gpt_5_6_sol_xhigh",
    "doubao": C_ONLY_ROOT / "doubao_seed_2_1_pro",
}
VIDEO_FIELDS = ("视频侧-参与者", "视频侧-空间约束", "视频侧-时间窗", "视频侧-事件序列")

C_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(pilot.TACTICS)},
        "evidence_interval": {
            "type": "object",
            "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}},
            "required": ["start_s", "end_s"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": ["label", "evidence_interval", "rationale"],
    "additionalProperties": False,
}


def json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stored_path(path: Path) -> str:
    return str(path.relative_to(pilot.ROOT)) if path.is_relative_to(pilot.ROOT) else str(path)


def build_c_only_prompt(criteria: dict[str, dict[str, str]]) -> str:
    sections = []
    for tactic in pilot.TACTICS:
        row = criteria[tactic]
        lines = [f"战术：{tactic}"]
        lines.extend(f"- {field}: {row[field]}" for field in VIDEO_FIELDS)
        sections.append("\n".join(lines))
    return f"""你是足球视频战术分类器。观看完整30秒片段，从下面8个战术中选出整体最符合画面的一个。

判断原则：
- 四项视频侧描述是典型识别依据，不是必须逐项满足的合取清单。
- 比较八个战术与完整画面的整体吻合度，不要仅凭一个孤立动作下结论。
- 必须八选一，不允许null，也不要创造新标签。
- evidence_interval写最能支持该判断的连续时间段，起止均精确到0.5秒，且0<=start_s<end_s<=30。
- rationale用一到两句话说明画面里的直接证据。
- 系统按2 fps读取画面。若输入呈现为静止图片，文件名就是时间戳，例如t03.500s.jpg表示第3.5秒；两帧间看不到的动作不要猜。

八个战术及视频侧定义：
{chr(10).join(sections)}

只输出严格JSON：label、evidence_interval、rationale。"""


def validate_c_only(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"label", "evidence_interval", "rationale"}:
        raise ValueError("C-only fields are not strict")
    if payload["label"] not in pilot.TACTICS:
        raise ValueError("C-only label must be one of the eight tactics")
    interval = payload["evidence_interval"]
    if not isinstance(interval, dict) or set(interval) != {"start_s", "end_s"}:
        raise ValueError("C-only evidence_interval fields are not strict")
    start, end = interval["start_s"], interval["end_s"]
    if not pilot._half_second(start) or not pilot._half_second(end) or not float(start) < float(end):
        raise ValueError("C-only evidence interval must be ordered on the 0.5-second grid")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("C-only rationale must be non-empty")
    return payload


def frame_note(n: int, fps: float) -> str:
    return codex_probe.FRAME_NOTE.format(n=n, step=1 / fps)


def agent_record(
    payload: dict[str, Any], elapsed: float, prompt: str, schema: dict[str, Any], fps: float,
    n_frames: int, *, provider: str, model: str, transport_attempts: int | None = None,
    usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "output": payload,
        "elapsed_s": elapsed,
        "prompt_sha256": text_hash(prompt),
        "schema_sha256": json_hash(schema),
        "fps": fps,
        "height": HEIGHT,
        "n_frames": n_frames,
        "provider": provider,
        "model": model,
        "reasoning_effort": REASONING_EFFORT if provider == "codex" else None,
        "usage": usage,
        "transport_attempts": transport_attempts,
    }


def codex_agent(
    prompt: str, schema: dict[str, Any], frames: list[Path], workdir: Path, role: str,
    fps: float, validator: Callable[[dict[str, Any]], dict[str, Any]], timeout: float,
) -> dict[str, Any]:
    payload, elapsed = codex_probe.call_codex(
        prompt, schema, frames, workdir, role, MODEL, REASONING_EFFORT, timeout,
    )
    payload = codex_probe.normalize_agent_times(role, payload)
    validator(payload)
    return agent_record(
        payload, elapsed, prompt, schema, fps, len(frames), provider="codex", model=MODEL,
    )


def final_from_c(rule: dict[str, Any], c: dict[str, Any] | None) -> dict[str, Any]:
    label = c["label"] if c else rule.get("label")
    return {
        "label": label,
        "source": "arbiter" if c else "rule",
        "candidates": pilot.c_candidates(rule),
        "interval": next(
            (item["interval"] for item in rule["ranked"] if c and item["tactic"] == label),
            rule.get("interval"),
        ),
    }


def run_r4_codex_attempt(row: dict[str, Any], criteria: dict[str, dict[str, str]], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"r4-codex-{row['clip']}-") as directory:
        workdir = Path(directory)
        frames_b = codex_probe.extract_frames(row["video_path"], workdir / "frames", FPS["B"], HEIGHT)
        if len(frames_b) != 120 or len(set(frames_b)) != 120:
            raise RuntimeError(f"expected 120 unique B frames, got {len(frames_b)}")
        frames_ac = frames_b[::2]
        if len(frames_ac) != 60:
            raise RuntimeError(f"expected 60 A/C frames, got {len(frames_ac)}")

        a_prompt = pilot.A_PROMPT + frame_note(60, FPS["A"])
        b_prompt = pilot.build_b_prompt(B_MAX_ITEMS) + frame_note(120, FPS["B"])
        a = codex_agent(a_prompt, pilot.A_SCHEMA, frames_ac, workdir, "A", FPS["A"], pilot.validate_a, timeout)
        b_schema = pilot.build_b_schema(B_MAX_ITEMS)
        b = codex_agent(
            b_prompt, b_schema, frames_b, workdir, "B", FPS["B"],
            lambda payload: pilot.validate_b(payload, B_MAX_ITEMS), timeout,
        )
        rule = pilot.classify(a["output"], b["output"], B_MAX_ITEMS)
        candidates = pilot.c_candidates(rule)
        c_prompt = pilot.build_c_prompt(a["output"], b["output"], rule, criteria) + frame_note(60, FPS["C"])
        try:
            c = codex_agent(
                c_prompt, pilot.C_SCHEMA, frames_ac, workdir, "C", FPS["C"],
                lambda payload: pilot.validate_c(payload, candidates), timeout,
            )
            status = "success"
        except Exception as exc:
            c = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            status = "partial"
        verdict = c.get("output")
        return {
            "status": status,
            "agents": {"A": a, "B": b, "C": c},
            "rule": rule,
            "final": final_from_c(rule, verdict),
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def run_c_only_codex_attempt(row: dict[str, Any], prompt: str, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"c-only-codex-{row['clip']}-") as directory:
        workdir = Path(directory)
        frames = codex_probe.extract_frames(row["video_path"], workdir / "frames", FPS["C"], HEIGHT)
        if len(frames) != 60 or len(set(frames)) != 60:
            raise RuntimeError(f"expected 60 unique C-only frames, got {len(frames)}")
        agent = codex_agent(
            prompt, C_ONLY_SCHEMA, frames, workdir, "C_only", FPS["C"], validate_c_only, timeout,
        )
    output = agent["output"]
    return {
        "status": "success",
        "agents": {"C": agent},
        "final": {"label": output["label"], "source": "direct_c", "interval": output["evidence_interval"]},
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def run_c_only_doubao_attempt(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    started = time.monotonic()
    with pilot.temporary_compact_clip(row["video_path"], **pilot.VIDEO_PREPROCESS) as upload:
        call_started = time.monotonic()
        payload, usage, transport_attempts = pilot.generate_json(
            "doubao", prompt, C_ONLY_SCHEMA, video_path=upload, fps=FPS["C"],
            temperature=pilot.TEMPERATURE, retries=0,
        )
        call_elapsed = round(time.monotonic() - call_started, 3)
    validate_c_only(payload)
    agent = agent_record(
        payload, call_elapsed, prompt, C_ONLY_SCHEMA, FPS["C"], 60,
        provider="doubao", model=pilot.model_name("doubao"),
        transport_attempts=transport_attempts, usage=usage,
    )
    return {
        "status": "success",
        "agents": {"C": agent},
        "final": {"label": payload["label"], "source": "direct_c", "interval": payload["evidence_interval"]},
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def ensure_manifest(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    signature = json_hash(value)
    path = root / "manifest.json"
    if path.is_file():
        existing = read_json(path)
        if existing.get("signature") != signature:
            raise RuntimeError(f"manifest mismatch at {path}; use a new output directory")
        return existing
    manifest = {**value, "signature": signature, "created_at": datetime.now(timezone.utc).isoformat()}
    pilot.write_json(path, manifest)
    return manifest


def run_unit(
    root: Path, row: dict[str, Any], config: dict[str, Any], invoke: Callable[[], dict[str, Any]],
    *, resume: bool, retry_failed: bool,
) -> str:
    target = root / "clips" / f"{row['clip']}.json"
    existing = read_json(target) if target.is_file() else None
    signature = json_hash(config)
    if existing:
        if existing.get("config_signature") != signature:
            raise RuntimeError(f"configuration mismatch for {target}")
        if not resume:
            raise FileExistsError(f"result exists; pass --resume: {target}")
        history = existing.get("attempt_history", [])
        if existing.get("status") == "success" or not retry_failed or len(history) >= MAX_ATTEMPTS:
            return "skipped"
    else:
        if retry_failed:
            return "skipped"
        history = []

    attempt_number = len(history) + 1
    base = {
        "clip": row["clip"],
        "video_path": stored_path(row["video_path"]),
        "video_sha256": pilot.sha256_file(row["video_path"]),
        "gt": {"tactic": row["tactic"], "interval": row["interval"]},
    }
    started = time.monotonic()
    try:
        value = invoke()
    except Exception as exc:
        value = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
    elapsed = round(time.monotonic() - started, 3)
    attempt = {
        **base,
        **value,
        "attempt": attempt_number,
        "config": config,
        "config_signature": signature,
        "attempt_elapsed_s": elapsed,
    }
    attempt_path = root / "attempts" / row["clip"] / f"attempt{attempt_number}.json"
    pilot.write_json(attempt_path, attempt)
    history = [*history, {
        "attempt": attempt_number,
        "status": value["status"],
        "elapsed_s": elapsed,
        "path": stored_path(attempt_path),
        **({"error_type": value.get("error_type"), "error": value.get("error")} if value["status"] == "failed" else {}),
    }]
    result = {
        **base,
        **value,
        "config": config,
        "config_signature": signature,
        "attempt_history": history,
        "first_attempt_failed": history[0]["status"] != "success",
        "recovered": history[0]["status"] != "success" and value["status"] == "success",
    }
    pilot.write_json(target, result)
    return value["status"]


def selected_rows(clip_ids: list[str] | None) -> list[dict[str, Any]]:
    rows = pilot.load_gt()
    if not clip_ids:
        return rows
    by_clip = {row["clip"]: row for row in rows}
    unknown = sorted(set(clip_ids) - set(by_clip))
    if unknown:
        raise ValueError(f"unknown clips: {', '.join(unknown)}")
    return [by_clip[clip] for clip in dict.fromkeys(clip_ids)]


def base_manifest(mode: str, provider: str, criteria: dict[str, dict[str, str]], cli_version: str | None) -> dict[str, Any]:
    prompt = build_c_only_prompt(criteria) if mode == "c-only" else None
    return {
        "mode": mode,
        "expected_clips": 37,
        "provider": provider,
        "model": MODEL if provider == "codex" else pilot.model_name("doubao"),
        "reasoning_effort": REASONING_EFFORT if provider == "codex" else None,
        "codex_cli_version": cli_version,
        "temperature": None if provider == "codex" else pilot.TEMPERATURE,
        "fps": FPS if mode == "r4-codex" else FPS["C"],
        "height": HEIGHT,
        "b_max_items": B_MAX_ITEMS if mode == "r4-codex" else None,
        "search_enabled": provider == "codex",
        "tool_trace_audited": False if provider == "codex" else None,
        "sandbox": "danger-full-access" if provider == "codex" else None,
        "max_experiment_attempts": MAX_ATTEMPTS,
        "gt_sha256": pilot.sha256_file(pilot.GT_PATH),
        "glossary_sha256": pilot.sha256_file(pilot.GLOSSARY_PATH),
        "criteria_snapshot_sha256": json_hash(criteria),
        "criteria_snapshot": criteria,
        "prompt_sha256": text_hash(prompt) if prompt else {
            "A": text_hash(pilot.A_PROMPT + frame_note(60, FPS["A"])),
            "B": text_hash(pilot.build_b_prompt(B_MAX_ITEMS) + frame_note(120, FPS["B"])),
        },
        "schema_sha256": json_hash(C_ONLY_SCHEMA) if prompt else {
            "A": json_hash(pilot.A_SCHEMA), "B": json_hash(pilot.build_b_schema(B_MAX_ITEMS)),
            "C": json_hash(pilot.C_SCHEMA),
        },
        "pilot_sha256": pilot.sha256_file(Path(pilot.__file__)),
    }


def codex_version() -> str:
    if not shutil.which("codex"):
        raise RuntimeError("codex CLI not found on PATH")
    return subprocess.run(["codex", "--version"], check=True, capture_output=True, text=True).stdout.strip()


def run_r4_codex(args: argparse.Namespace) -> None:
    criteria, cli_version = pilot.load_criteria(), codex_version()
    manifest = ensure_manifest(R4_CODEX_ROOT, base_manifest("r4-codex", "codex", criteria, cli_version))
    counts = Counter()
    for row in selected_rows(args.clip_ids):
        config = {
            "manifest_signature": manifest["signature"],
            "video_sha256": pilot.sha256_file(row["video_path"]),
        }
        status = run_unit(
            R4_CODEX_ROOT, row, config,
            lambda row=row: run_r4_codex_attempt(row, criteria, args.timeout),
            resume=args.resume, retry_failed=args.retry_failed,
        )
        counts[status] += 1
        print(f"{row['clip']} {status}", flush=True)
    print(dict(counts), flush=True)


def run_c_only(args: argparse.Namespace) -> None:
    if args.provider == "doubao" and not args.allow_external_upload:
        raise SystemExit("Doubao uploads video externally; pass --allow-external-upload")
    criteria = pilot.load_criteria()
    prompt = build_c_only_prompt(criteria)
    cli_version = codex_version() if args.provider == "codex" else None
    root = C_ONLY_DIRS[args.provider]
    manifest = ensure_manifest(root, base_manifest("c-only", args.provider, criteria, cli_version))
    counts = Counter()
    for row in selected_rows(args.clip_ids):
        config = {
            "manifest_signature": manifest["signature"],
            "video_sha256": pilot.sha256_file(row["video_path"]),
            "prompt_sha256": text_hash(prompt),
            "schema_sha256": json_hash(C_ONLY_SCHEMA),
        }
        invoke = (
            (lambda row=row: run_c_only_codex_attempt(row, prompt, args.timeout))
            if args.provider == "codex"
            else (lambda row=row: run_c_only_doubao_attempt(row, prompt))
        )
        status = run_unit(
            root, row, config, invoke, resume=args.resume, retry_failed=args.retry_failed,
        )
        counts[status] += 1
        print(f"{row['clip']} {status}", flush=True)
    print(dict(counts), flush=True)


def load_results(root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((root / "clips").glob("*.json"))]


def complete_run(results: list[dict[str, Any]]) -> bool:
    return len({result.get("clip") for result in results}) == 37


def prediction_metrics(results: list[dict[str, Any]], *, include_partial: bool = False, key: str = "final") -> dict[str, Any]:
    by_clip = {result["clip"]: result for result in results}
    pairs = []
    confusion = {tactic: Counter() for tactic in pilot.TACTICS}
    temporal = []
    statuses = Counter(result.get("status", "missing") for result in results)
    statuses["missing"] += 37 - len(results)
    allowed = {"success", "partial"} if include_partial else {"success"}
    for row in pilot.load_gt():
        result = by_clip.get(row["clip"])
        chosen = result.get(key, {}) if result and result.get("status") in allowed else {}
        predicted = chosen.get("label")
        pairs.append((row["tactic"], predicted))
        confusion[row["tactic"]][predicted or ("missing" if result is None else result.get("status", "failed"))] += 1
        if row["tactic"] != "back-line-possession-management":
            temporal.append(pilot.temporal_iou(chosen.get("interval") if predicted == row["tactic"] else None, row["interval"]))
    per_class = {}
    for tactic in pilot.TACTICS:
        tp = sum(truth == prediction == tactic for truth, prediction in pairs)
        fp = sum(truth != tactic and prediction == tactic for truth, prediction in pairs)
        fn = sum(truth == tactic and prediction != tactic for truth, prediction in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[tactic] = {
            "support": tp + fn, "correct_n": tp, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    correct = sum(truth == prediction for truth, prediction in pairs)
    first_failures = sum(
        bool(result.get("attempt_history")) and result["attempt_history"][0].get("status") != "success"
        for result in results
    )
    recovered = sum(result.get("recovered") is True for result in results)
    return {
        "n_expected": 37,
        "n_present": len(results),
        "status_counts": dict(statuses),
        "attempts": {
            "first_failures": first_failures,
            "recovered": recovered,
            "final_failures": sum(result.get("status") != "success" for result in results) + 37 - len(results),
            "total": sum(len(result.get("attempt_history", [])) for result in results),
            "elapsed_total_s": round(sum(
                float(attempt.get("elapsed_s", 0.0))
                for result in results for attempt in result.get("attempt_history", [])
            ), 3),
        },
        "correct_n": correct,
        "accuracy": correct / 37,
        "macro_f1": statistics.fmean(item["f1"] for item in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": {tactic: dict(values) for tactic, values in confusion.items()},
        "temporal": {
            "eligible_n": len(temporal),
            "mean_iou": statistics.fmean(temporal) if temporal else 0.0,
            "tiou_at_0_5": sum(value >= 0.5 for value in temporal) / len(temporal) if temporal else 0.0,
        },
    }


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by = {item["clip"]: item for item in left}
    right_by = {item["clip"]: item for item in right}
    counts = Counter()
    changes = []
    for row in pilot.load_gt():
        left_result = left_by.get(row["clip"], {})
        right_result = right_by.get(row["clip"], {})
        l = left_result.get("final", {}).get("label") if left_result.get("status") == "success" else None
        r = right_result.get("final", {}).get("label") if right_result.get("status") == "success" else None
        lc, rc = l == row["tactic"], r == row["tactic"]
        bucket = "both_correct" if lc and rc else "left_only" if lc else "right_only" if rc else "neither"
        counts[bucket] += 1
        counts["label_agreement"] += l is not None and l == r
        if l != r:
            changes.append({"clip": row["clip"], "gt": row["tactic"], "left": l, "right": r})
    return {**dict(counts), "changes": changes}


def r4_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter()
    strict = xyx = 0
    one_two_false_positives = 0
    for result in results:
        gt = result["gt"]["tactic"]
        rule_label = result.get("rule", {}).get("label")
        final_label = result.get("final", {}).get("label")
        if result.get("status") == "success":
            transitions[f"{'correct' if rule_label == gt else 'wrong'}_to_{'correct' if final_label == gt else 'wrong'}"] += 1
        events = result.get("agents", {}).get("B", {}).get("output", {}).get("events", [])
        if gt == "one-two":
            strict += pilot_ablation.pair_evidence(events, result["gt"]["interval"])
            xyx += pilot_ablation.confirmed_xyx(events, result["gt"]["interval"])
        elif final_label == "one-two":
            one_two_false_positives += 1
    return {
        "rule_to_c": dict(transitions),
        "one_two": {"strict_two_pass": strict, "confirmed_xyx": xyx, "final_false_positives": one_two_false_positives},
    }


def write_review(root: Path, results: list[dict[str, Any]], title: str) -> None:
    lines = [f"# {title}", "", "| clip | GT | prediction | status | correct | evidence |", "|---|---|---|---|---:|---|"]
    for result in sorted(results, key=lambda item: item["clip"]):
        final = result.get("final", {})
        interval = final.get("interval")
        evidence = f"{interval['start_s']:.1f}–{interval['end_s']:.1f}" if interval else "—"
        lines.append(
            f"| {result['clip']} | {result['gt']['tactic']} | {final.get('label') or '—'} | "
            f"{result.get('status')} | {'yes' if final.get('label') == result['gt']['tactic'] else 'no'} | {evidence} |"
        )
    (root / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    r4_codex = load_results(R4_CODEX_ROOT)
    r4_seed = load_results(OUTPUT_ROOT / "r4")
    c_codex = load_results(C_ONLY_DIRS["codex"])
    c_doubao = load_results(C_ONLY_DIRS["doubao"])

    report: dict[str, Any] = {}
    if r4_codex:
        r4_seed_metrics = prediction_metrics(r4_seed)
        strict = prediction_metrics(r4_codex)
        operational = prediction_metrics(r4_codex, include_partial=True)
        rule = prediction_metrics(r4_codex, include_partial=True, key="rule")
        metrics = {
            **strict,
            "operational_with_rule_fallback": operational,
            "rule_only": rule,
            "diagnostics": r4_diagnostics(r4_codex),
        }
        pilot.write_json(R4_CODEX_ROOT / "metrics.json", metrics)
        write_review(R4_CODEX_ROOT, r4_codex, "r4 Codex Review")
        comparison = {
            "r4_doubao_current": r4_seed_metrics,
            "r4_codex": strict,
            "r4_doubao_left_vs_r4_codex_right": paired(r4_seed, r4_codex),
        }
        pilot.write_json(R4_CODEX_ROOT / "comparison.json", comparison)
        (R4_CODEX_ROOT / "comparison.md").write_text(
            f"# r4 Doubao vs r4 Codex\n\nDoubao current: {r4_seed_metrics['correct_n']}/37. "
            f"Codex strict: {strict['correct_n']}/37. Codex with Rule fallback: {operational['correct_n']}/37.\n",
            encoding="utf-8",
        )
        report["r4_codex"] = metrics

    for provider, results in (("codex", c_codex), ("doubao", c_doubao)):
        if results:
            metrics = prediction_metrics(results)
            root = C_ONLY_DIRS[provider]
            pilot.write_json(root / "metrics.json", metrics)
            write_review(root, results, f"C-only {provider} Review")
            report[f"c_only_{provider}"] = metrics

    comparisons = {
        "availability": {
            "r4_codex": {"n_present": len(r4_codex), "complete": complete_run(r4_codex)},
            "c_only_codex": {"n_present": len(c_codex), "complete": complete_run(c_codex)},
            "c_only_doubao": {"n_present": len(c_doubao), "complete": complete_run(c_doubao)},
        },
        "c_only_doubao_left_vs_codex_right": paired(c_doubao, c_codex) if complete_run(c_doubao) and complete_run(c_codex) else None,
        "r4_doubao_left_vs_c_only_doubao_right": paired(r4_seed, c_doubao) if complete_run(c_doubao) else None,
        "r4_codex_left_vs_c_only_codex_right": paired(r4_codex, c_codex) if complete_run(r4_codex) and complete_run(c_codex) else None,
    }
    pilot.write_json(C_ONLY_ROOT / "comparison.json", comparisons)
    lines = ["# C-only Model Comparison", ""]
    for name, results in (("Codex", c_codex), ("Doubao", c_doubao)):
        if results:
            metrics = prediction_metrics(results)
            state = "complete" if complete_run(results) else "pending"
            lines.append(f"- {name}: {state}, {metrics['correct_n']}/37 ({metrics['accuracy']:.1%}), present={metrics['n_present']}")
        else:
            lines.append(f"- {name}: pending")
    lines.extend(["", "Internal diagnostic only; no significance or population-level claim.", ""])
    C_ONLY_ROOT.mkdir(parents=True, exist_ok=True)
    (C_ONLY_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="run only existing failed/partial units with fewer than two attempts")
    parser.add_argument("--timeout", type=float, default=900.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r4 = sub.add_parser("r4-codex")
    add_run_flags(r4)
    direct = sub.add_parser("c-only")
    add_run_flags(direct)
    direct.add_argument("--provider", required=True, choices=("codex", "doubao"))
    direct.add_argument("--allow-external-upload", action="store_true")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if getattr(args, "timeout", 1) <= 0:
        raise SystemExit("timeout must be positive")
    if getattr(args, "retry_failed", False) and not args.resume:
        raise SystemExit("--retry-failed requires --resume")
    if args.command == "r4-codex":
        run_r4_codex(args)
    elif args.command == "c-only":
        run_c_only(args)
    else:
        analyze()


if __name__ == "__main__":
    main()
