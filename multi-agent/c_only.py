#!/usr/bin/env python3
"""C-only arm: one video call picks the tactic directly, with no A, B or Rule.

The control for pilot.py. Same clips, same ground truth, same metrics — the only
difference is that nothing structured reaches the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot

OUTPUT_ROOT = pilot.OUTPUT_ROOT / "c_only"
SCHEMA_VERSION = "gsr2025-c-only-v1"
PROMPT_VERSION = "c-only-direct-classification-v1"
FPS = pilot.MODEL_FPS["C"]
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


def build_prompt(criteria: dict[str, dict[str, str]]) -> str:
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

八个战术及视频侧定义：
{chr(10).join(sections)}

只输出严格JSON：label、evidence_interval、rationale。"""


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    pilot._schema_keys(payload, {"label", "evidence_interval", "rationale"}, "C-only")
    if payload["label"] not in pilot.TACTICS:
        raise ValueError("C-only label must be one of the eight tactics")
    interval = payload["evidence_interval"]
    pilot._schema_keys(interval, {"start_s", "end_s"}, "C-only evidence_interval")
    start, end = interval["start_s"], interval["end_s"]
    if not pilot._half_second(start) or not pilot._half_second(end) or not float(start) < float(end):
        raise ValueError("C-only evidence interval must be ordered on the 0.5-second grid")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("C-only rationale must be non-empty")
    return payload


def run_clip(
    video_path: Path,
    prompt: str,
    generator: pilot.Generator = pilot.generate_json,
    *,
    provider: str = "doubao",
) -> dict[str, Any]:
    started = time.monotonic()
    tuning = {"fps": FPS, "temperature": pilot.TEMPERATURE} if provider == "doubao" else {}
    with pilot.temporary_compact_clip(video_path, **pilot.VIDEO_PREPROCESS) as upload:
        payload, usage, attempts = generator(
            provider, prompt, C_ONLY_SCHEMA, video_path=upload, **tuning,
        )
    validate(payload)
    return {
        "agents": {"C": {
            "output": payload,
            "usage": usage,
            "attempts": attempts,
            "elapsed_s": round(time.monotonic() - started, 3),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "fps": tuning.get("fps"),
        }},
        "final": {
            "label": payload["label"],
            "source": "direct_c",
            "interval": payload["evidence_interval"],
        },
    }


def ensure_manifest(run_dir: Path, prompt: str, provider: str) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": pilot.model_name(provider),
        "provider": provider,
        "fps": FPS,
        "temperature": pilot.TEMPERATURE,
        "video_preprocess": pilot.VIDEO_PREPROCESS,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": pilot._json_hash(C_ONLY_SCHEMA),
        "gt_sha256": pilot.sha256_file(pilot.GT_PATH),
        "glossary_sha256": pilot.sha256_file(pilot.GLOSSARY_PATH),
    }
    current = {**value, "signature": pilot._json_hash(value)}
    path = run_dir / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != current["signature"]:
            raise RuntimeError(f"{run_dir} manifest does not match current prompt/data; use a new run-id")
        return existing
    manifest = {
        **current,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_clips": pilot.EXPECTED_CLIPS,
    }
    pilot.write_json(path, manifest)
    return manifest


def run(
    run_id: str,
    *,
    provider: str = "doubao",
    clip_ids: Iterable[str] | None = None,
    retry_failed: bool = False,
    generator: pilot.Generator = pilot.generate_json,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, int]:
    rows = pilot.load_gt()
    prompt = build_prompt(pilot.load_criteria())
    run_dir = output_root / run_id
    ensure_manifest(run_dir, prompt, provider)
    selected = set(clip_ids or (row["clip"] for row in rows))
    if unknown := selected - {row["clip"] for row in rows}:
        raise ValueError(f"unknown clip ids: {sorted(unknown)}")
    counts = Counter(selected=len(selected), success=0, failed=0, skipped=0)
    for gt in rows:
        if gt["clip"] not in selected:
            continue
        target = run_dir / "clips" / f"{gt['clip']}.json"
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("status") == "success" or not retry_failed:
                counts["skipped"] += 1
                continue
        started = time.monotonic()
        try:
            result = {"status": "success", **run_clip(gt["video_path"], prompt, generator, provider=provider)}
        except Exception as exc:
            result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        result.update({
            "schema_version": SCHEMA_VERSION,
            "clip": gt["clip"],
            "video_path": gt["video_path"].relative_to(pilot.ROOT).as_posix() if gt["video_path"].is_relative_to(pilot.ROOT) else gt["video_path"].as_posix(),
            "video_sha256": pilot.sha256_file(gt["video_path"]),
            "gt": {"tactic": gt["tactic"], "interval": gt["interval"]},
            "elapsed_s": round(time.monotonic() - started, 3),
        })
        pilot.write_json(target, result)
        counts[result["status"]] += 1
    return dict(counts)


def _render_review(results: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    lines = [
        "# GSR-2025 C-only Review",
        "",
        f"macro-F1={metrics['macro_f1']:.3f} · accuracy={metrics['correct_n']}/{metrics['n_expected']} "
        f"({metrics['accuracy']:.1%}) · mean tIoU={metrics['temporal']['mean_iou']:.3f}",
        "",
        "| clip | GT | prediction | correct | evidence | rationale |",
        "|---|---|---|---|---|---|",
    ]
    for result in sorted(results, key=lambda item: item["clip"]):
        final = result.get("final", {})
        interval = final.get("interval")
        evidence = f"{interval['start_s']:.1f}–{interval['end_s']:.1f}" if interval else "—"
        rationale = result.get("agents", {}).get("C", {}).get("output", {}).get("rationale", "")
        lines.append(
            f"| {result['clip']} | {result['gt']['tactic']} | {final.get('label') or result.get('status')} | "
            f"{'yes' if final.get('label') == result['gt']['tactic'] else 'no'} | {evidence} | {rationale} |"
        )
    return "\n".join(lines) + "\n"


def report(run_id: str, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    run_dir = output_root / run_id
    results = pilot.load_results(run_dir)
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8",
    )
    metrics = pilot.classification_metrics(results)
    pilot.write_json(run_dir / "metrics.json", metrics)
    (run_dir / "review.md").write_text(_render_review(results, metrics), encoding="utf-8")
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--provider", required=True, choices=pilot.PROVIDERS)
    run_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--allow-external-upload", action="store_true")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run-id", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        if not args.allow_external_upload:
            raise SystemExit("Refusing external video upload without --allow-external-upload")
        value = run(
            args.run_id, provider=args.provider,
            clip_ids=args.clip_ids, retry_failed=args.retry_failed,
        )
    else:
        value = report(args.run_id)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
