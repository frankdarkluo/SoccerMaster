#!/usr/bin/env python3
"""Probe: let Codex play A/B/C on the minority-GT clips, reading frames instead of video.

Standalone on purpose. It imports pilot.py for the prompts, schemas and rules but never
modifies it, so a pilot run can keep going while this is used.

Codex reads still images, not video, so each clip is sampled into timestamped JPEGs
(`t03.500s.jpg`) and the prompt tells the agent that the filename is the timestamp. Agent
times are then normalized to the 0.5s grid the rules expect.

    python3 multi-agent/codex_probe.py --clip SNGS-137 --clip SNGS-019
    python3 multi-agent/codex_probe.py --minority --fps 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs/codex_probe"
MAJORITY = {"counter-attack", "long-ball"}
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "xhigh"
FRAME_NOTE = (
    "\n\n本次输入不是视频，而是从同一段30秒视频里等间隔抽出的 {n} 张静止图片。\n"
    "图片文件名就是该帧的精确时刻：t03.125s.jpg 表示第 3.125 秒。\n"
    "相邻两帧间隔 {step:.3f} 秒，两帧之间发生的事情你看不到，不要凭想象补全。\n"
    "输出中的所有时间仍须落在0.5秒网格。请取最近的0.5秒；正中边界时取较早值。\n"
    "看不清的动作宁可不写，也不要猜。"
)
WINDOW_PROBE_NOTE = (
    "\n\n这是事后诊断，不是正式分类。只记录 {lo:.3f}–{hi:.3f} 秒内能确认的事件。"
    "GT区间只用于判断全片事件预算是否漏掉了局部动作。"
)


def quantize_half_second(value):
    """Round to the nearest 0.5 seconds, resolving exact ties toward the earlier time."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    lower = math.floor(float(value) * 2) / 2
    upper = lower + 0.5
    return round(lower if float(value) - lower <= upper - float(value) else upper, 1)


def frame_name(index: int, fps: float) -> str:
    return f"t{index / fps:06.3f}s.jpg"


def extract_frames(video: Path, out_dir: Path, fps: float, height: int) -> list[Path]:
    """Sample `video` into timestamped JPEGs; the filename carries the only time signal Codex gets."""
    if fps <= 0 or height <= 0:
        raise ValueError("fps and height must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps},scale=-2:{height}", "-q:v", "6", str(out_dir / "raw_%04d.jpg")],
        check=True, capture_output=True, text=True,
    )
    frames = []
    for index, raw in enumerate(sorted(out_dir.glob("raw_*.jpg"))):
        target = out_dir / frame_name(index, fps)
        raw.rename(target)
        frames.append(target)
    return frames


def normalize_agent_times(role: str, payload: dict) -> dict:
    if role == "A":
        return {**payload, "possession_change": quantize_half_second(payload.get("possession_change"))}
    if role == "B":
        return {
            **payload,
            "events": [
                {**event, "time": quantize_half_second(event.get("time"))}
                for event in payload.get("events", [])
            ],
        }
    return payload


def call_codex(prompt: str, schema: dict, frames: list[Path], workdir: Path, role: str,
               model: str, reasoning_effort: str, timeout_s: float) -> tuple[dict, float]:
    """One `codex exec` call returning schema-conformant JSON.

    Notes on the flags, all of which were needed to make this work at all:
      * the prompt goes on stdin because `-i` is variadic and swallows a trailing positional;
      * `-s danger-full-access` because this kernel blocks unprivileged user namespaces, so
        Codex's bubblewrap sandbox cannot even open the image files (it silently returns an
        empty answer instead of erroring). Keep `workdir` disposable.
    """
    schema_path = workdir / f"{role}.schema.json"
    out_path = workdir / f"{role}.out.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
    out_path.unlink(missing_ok=True)
    command = [
        "codex", "--search", "exec", "--ephemeral", "--skip-git-repo-check",
        "-s", "danger-full-access",
        "--model", model,
        "-c", f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema", str(schema_path), "-o", str(out_path),
    ]
    for frame in frames:
        command += ["-i", str(frame)]
    started = time.monotonic()
    done = subprocess.run(command, input=prompt, cwd=workdir, capture_output=True,
                          text=True, timeout=timeout_s)
    elapsed = round(time.monotonic() - started, 1)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout)[-800:]
        raise RuntimeError(f"codex exited {done.returncode}: {detail}")
    if not out_path.is_file():
        raise RuntimeError("codex produced no output")
    text = out_path.read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("codex response must be a JSON object")
    return payload, elapsed


def base_result(clip: str, gt: dict, args) -> dict:
    return {
        "clip": clip,
        "gt": {"tactic": gt["tactic"], "interval": gt["interval"]},
        "mode": "window_probe" if args.window_probe else "full",
        "fps": args.fps,
        "height": args.height,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_cli_version": args.codex_cli_version,
        "agents": {},
    }


def resume_matches(result: dict, args) -> bool:
    return result.get("status") == "success" and all((
        result.get("mode", "full") == ("window_probe" if args.window_probe else "full"),
        result.get("fps") == args.fps,
        result.get("height") == args.height,
        result.get("model") == args.model,
        result.get("reasoning_effort") == args.reasoning_effort,
        result.get("codex_cli_version") == args.codex_cli_version,
    ))


def probe_clip(clip: str, gt: dict, criteria: dict, args) -> dict:
    video = pilot.VIDEO_ROOT / f"{clip}.mp4"
    result = base_result(clip, gt, args)
    with tempfile.TemporaryDirectory(prefix=f"codex-probe-{clip}-") as directory:
        workdir = Path(directory)
        frames = extract_frames(video, workdir / "frames", args.fps, args.height)
        note = FRAME_NOTE.format(n=len(frames), step=1 / args.fps)
        result["n_frames"] = len(frames)
        expected = round(30 * args.fps)
        if len(frames) != expected or len(set(frames)) != expected:
            raise RuntimeError(f"expected {expected} unique frames, got {len(frames)}")

        if args.window_probe:
            interval = gt["interval"]
            prompt = pilot.B_PROMPT + note + WINDOW_PROBE_NOTE.format(
                lo=interval["start_s"], hi=interval["end_s"],
            )
            payload, elapsed = call_codex(
                prompt, pilot.B_SCHEMA, frames, workdir, "B_window", args.model,
                args.reasoning_effort, args.timeout,
            )
            payload = normalize_agent_times("B", payload)
            pilot.validate_b(payload)
            result["agents"]["B"] = {"output": payload, "elapsed_s": elapsed}
            result["status"] = "success"
            return result

        for role, prompt, schema, validator in (
            ("A", pilot.A_PROMPT + note, pilot.A_SCHEMA, pilot.validate_a),
            ("B", pilot.B_PROMPT + note, pilot.B_SCHEMA, pilot.validate_b),
        ):
            try:
                payload, elapsed = call_codex(
                    prompt, schema, frames, workdir, role, args.model,
                    args.reasoning_effort, args.timeout,
                )
                payload = normalize_agent_times(role, payload)
                validator(payload)
                result["agents"][role] = {"output": payload, "elapsed_s": elapsed}
            except Exception as exc:
                result["agents"][role] = {"error_type": type(exc).__name__, "error": str(exc)[:800]}
                result["status"] = "failed"
                return result

        rule = pilot.classify(result["agents"]["A"]["output"], result["agents"]["B"]["output"])
        result["rule"] = rule
        candidates = pilot.c_candidates(rule)
        c_prompt = build_probe_c_prompt(result["agents"]["A"]["output"],
                                        result["agents"]["B"]["output"], rule, criteria) + note
        try:
            payload, elapsed = call_codex(
                c_prompt, pilot.C_SCHEMA, frames, workdir, "C", args.model,
                args.reasoning_effort, args.timeout,
            )
            pilot.validate_c(payload, candidates)
            result["agents"]["C"] = {"output": payload, "elapsed_s": elapsed}
        except Exception as exc:
            result["agents"]["C"] = {"error_type": type(exc).__name__, "error": str(exc)[:800]}
        verdict = result["agents"]["C"].get("output")
        result["final"] = {"label": verdict["label"] if verdict else rule.get("label"),
                           "source": "arbiter" if verdict else "rule", "candidates": candidates}
    result["status"] = "success" if verdict else "partial"
    return result


def build_probe_c_prompt(a: dict, b: dict, rule: dict, criteria: dict) -> str:
    """C prompt with the r3 over-rejection bias removed.

    On r3 the arbiter changed 7/19 clips: 5 correct->wrong, 0 wrong->correct. Its rationales
    show it treated the glossary's prose (numerical superiority, second-ball structure, live-ball
    trigger) as a conjunctive checklist, effectively reinstating the hard gates P0 removed. So
    here it sees only the operational video-side criteria, and rejection has to be argued for.
    """
    sections = []
    for candidate in rule["ranked"][:pilot.C_CANDIDATES]:
        row = criteria[candidate["tactic"]]
        lines = [f"候选：{candidate['tactic']}（rule 打分 {candidate['score']}）"]
        for field in ("视频侧-参与者", "视频侧-空间约束", "视频侧-时间窗", "视频侧-事件序列"):
            if row.get(field):
                lines.append(f"  {field}: {row[field]}")
        sections.append("\n".join(lines))
    names = "、".join(pilot.c_candidates(rule))
    return f"""你是最终裁决模块。A和B是只描述画面、不给战术名的模块，Rule是一套写死的打分规则。
从候选里选出最符合画面的那个战术。

重要的判断尺度：
- 下面每个候选给出的是「典型特征」，不是「必要条件清单」。不要因为某一条典型特征没出现就否决整个候选。
- 判断依据是哪个候选与画面的整体吻合度最高，而不是哪个候选挑不出毛病。
- Rule 的打分是有信息量的先验。只有当你有明确的画面证据表明它选错了，才改选别的。
- 只有当所有候选都与画面明显矛盾时才填 null。宁可选一个不完美但最接近的，也不要轻易全盘否定。

label只能是以下之一，或者null：{names}
rationale：一到两句话，引用画面里的具体依据。
objections：A或B中与画面明确不符的字段，最多两条，格式“字段路径: 具体错误”，路径以A.或B.开头。没有就留空。

A={json.dumps(a, ensure_ascii=False, sort_keys=True)}
B={json.dumps(b, ensure_ascii=False, sort_keys=True)}

候选及其典型特征：
{chr(10).join(sections)}

只输出严格JSON：label、rationale、objections。"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", dest="clips")
    parser.add_argument("--minority", action="store_true", help="all clips whose GT is not counter-attack/long-ball")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        choices=("low", "medium", "high", "xhigh", "max", "ultra"))
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true", help="skip matching successful outputs")
    parser.add_argument("--window-probe", action="store_true", help="diagnostic B-only run with the GT interval")
    args = parser.parse_args()

    if not shutil.which("codex"):
        raise SystemExit("codex CLI not found on PATH")
    if args.fps <= 0 or args.height <= 0 or args.timeout <= 0:
        raise SystemExit("fps, height and timeout must be positive")
    version = subprocess.run(["codex", "--version"], check=True, capture_output=True, text=True)
    args.codex_cli_version = version.stdout.strip()
    rows = {row["clip"]: row for row in pilot.load_gt()}
    criteria = pilot.load_criteria()
    if args.minority:
        selected = [c for c, r in rows.items() if r["tactic"] not in MAJORITY]
    else:
        selected = args.clips or []
    if not selected:
        raise SystemExit("nothing selected: pass --clip or --minority")

    args.out.mkdir(parents=True, exist_ok=True)
    for clip in selected:
        if clip not in rows:
            raise SystemExit(f"unknown clip {clip}")
        output_path = args.out / f"{clip}.json"
        if args.resume and output_path.is_file():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if resume_matches(existing, args):
                print(f"SKIP {clip}  matching success", flush=True)
                continue
        started = time.monotonic()
        try:
            result = probe_clip(clip, rows[clip], criteria, args)
        except Exception as exc:
            result = base_result(clip, rows[clip], args)
            result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:800])
        result["elapsed_s"] = round(time.monotonic() - started, 1)
        pilot.write_json(output_path, result)
        if args.window_probe:
            events = result.get("agents", {}).get("B", {}).get("output", {}).get("events", [])
            print(f"{clip}  window_probe status={result['status']} events={len(events)}  "
                  f"{result['elapsed_s']}s", flush=True)
            continue
        hit = "OK " if result.get("final", {}).get("label") == result["gt"]["tactic"] else "   "
        print(f"{hit}{clip}  GT={result['gt']['tactic']:<32} "
              f"rule={(result.get('rule') or {}).get('label')} "
              f"final={(result.get('final') or {}).get('label')}  "
              f"status={result['status']}  {result['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
