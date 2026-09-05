#!/usr/bin/env python3
"""Score every MP4 in a directory against the 12 tactic evidence rubrics."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa import criteria  # noqa: E402
from pipeline.video_models import (  # noqa: E402
    generate_json,
    model_name,
    temporary_compact_clip,
)

PROMPT_SPEC = ROOT / "docs/specs/2026-08-29-ai-tactic-scoring-prompt-v3.md"
VALIDATION_RETRIES = 2
ENV_KEYS = {
    "ARK_API_KEY", "ARK_BASE_URL", "ARK_RESPONSES_MODEL", "ARK_TIMEOUT_S",
    "GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3",
    "GEMINI_KEY_ROTATION_OFFSET", "GEMINI_INTERACTIONS_URL",
}


def build_prompt() -> str:
    text = PROMPT_SPEC.read_text(encoding="utf-8")
    try:
        fixed = text.split("## 输入\n", 1)[1].split("## 12 类证据清单\n", 1)[0]
    except IndexError as exc:
        raise ValueError(f"prompt sections missing from {PROMPT_SPEC}") from exc
    return "## 输入\n" + fixed + "## 12 类证据清单\n\n" + criteria.prompt_block()


def load_repo_env() -> None:
    """Load the provider settings used by this script without adding a dependency."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key = key.strip()
        if key in ENV_KEYS:
            os.environ.setdefault(key, value.strip().strip("'\""))


def check_provider_config(provider: str) -> None:
    if provider != "doubao":
        return
    expected = model_name(provider)
    configured = os.environ.get("ARK_RESPONSES_MODEL", "").strip() or expected
    if configured != expected:
        raise ValueError(f"ARK_RESPONSES_MODEL must be {expected}, got {configured}")
    if not os.environ.get("ARK_API_KEY", "").strip():
        raise RuntimeError("ARK_API_KEY is not set")


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        clip_id = row.get("clip_id") if isinstance(row, dict) else None
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"{path}:{lineno}: missing clip_id")
        completed.add(clip_id)
    return completed


def score_clip(
    video: Path,
    provider: str,
    fps: float,
    prompt: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    usage: list[dict[str, Any]] = []
    try:
        duration_s = duration(video)
        with temporary_compact_clip(video) as compact:
            validation_error: ValueError | None = None
            for _ in range(VALIDATION_RETRIES + 1):
                payload, call_usage, _ = generate_json(
                    provider,
                    prompt,
                    schema,
                    video_path=compact,
                    fps=fps,
                )
                usage.extend(call_usage)
                try:
                    criteria.validate(payload, duration_s=duration_s)
                except ValueError as exc:
                    validation_error = exc
                    continue
                return {**payload, "clip_id": video.stem}, usage, None
        return None, usage, f"validation failed after {VALIDATION_RETRIES + 1} responses: {validation_error}"
    except Exception as exc:  # report the clip and continue the batch
        return None, usage, f"{type(exc).__name__}: {exc}"


def run(
    input_dir: Path,
    output: Path,
    *,
    provider: str = "doubao",
    fps: float = 5.0,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if fps <= 0 or provider == "doubao" and not 0.2 <= fps <= 5.0:
        raise ValueError("fps must be in [0.2, 5.0] for doubao and positive otherwise")

    selected = sorted(input_dir.glob("*.mp4"))
    if limit is not None:
        selected = selected[:limit]
    completed = completed_ids(output)
    pending = [video for video in selected if video.stem not in completed]
    skipped = len(selected) - len(pending)

    output.parent.mkdir(parents=True, exist_ok=True)
    failed_path = output.with_suffix(output.suffix + ".failed.txt")
    totals: Counter[str] = Counter()
    succeeded = failed = 0
    prompt, schema = build_prompt(), criteria.response_schema()
    print(f"selected={len(selected)} skipped={skipped} pending={len(pending)}", flush=True)

    with output.open("a", encoding="utf-8") as destination, failed_path.open("w", encoding="utf-8") as failures:
        if pending:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                futures = {
                    pool.submit(score_clip, video, provider, fps, prompt, schema): video
                    for video in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    video = futures[future]
                    row, usage, error = future.result()
                    for record in usage:
                        totals.update({
                            key: value for key, value in record.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                        })
                    if row is not None:
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                        destination.flush()
                        succeeded += 1
                        status = "OK"
                    else:
                        reason = (error or "unknown error").replace("\t", " ").replace("\n", " ")
                        failures.write(f"{video.stem}\t{reason}\n")
                        failures.flush()
                        failed += 1
                        status = f"FAILED {reason}"
                    print(f"[{index}/{len(pending)}] {video.stem} {status}", flush=True)

    summary = {
        "selected": len(selected), "skipped": skipped, "succeeded": succeeded,
        "failed": failed, "usage": dict(totals),
    }
    print("summary " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider", choices=("doubao", "gemini"), default="doubao")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    load_repo_env()
    check_provider_config(args.provider)
    if run(
        args.input_dir,
        args.output,
        provider=args.provider,
        fps=args.fps,
        limit=args.limit,
        workers=args.workers,
    )["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
