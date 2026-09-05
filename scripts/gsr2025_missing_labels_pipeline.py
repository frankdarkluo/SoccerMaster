"""GSR preprocess for GSR-2025 clips missing Labels-GameState.json.

Step1+2: fan out across clips (STEP12_WORKERS, default 3).
Step3: concurrency=2 (STEP3_WORKERS).
Within step2, SAM2 uses GPU_LIST / MAX_PROCESSES_PER_GPU (default 1).

# ponytail: single-GPU A6000; 4-way step2 OOMed — default 3 clips × 1 SAM2/GPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "GSR-2025" / "images"
OUT_ROOT = ROOT / "outputs" / "GSR-2025" / "preprocessing"
GSR_STEP = ROOT / "scripts" / "gsr_step.py"
PYTHON = os.environ.get(
    "GSR_PYTHON",
    str(Path.home() / "guoqing/miniconda3/envs/tracklab/bin/python"),
)


def missing_labels(images: Path) -> list[str]:
    names = []
    for d in sorted(images.glob("SNGS-*")):
        if not (d / "Labels-GameState.json").is_file():
            names.append(d.name)
    return names


def clip_dir(name: str) -> Path:
    return IMAGES / name


def predictions_done(name: str) -> bool:
    return (OUT_ROOT / name / "predictions.json").is_file()


def step_done(name: str, step: str) -> bool:
    out = OUT_ROOT / name
    if step == "1":
        return (out / "step1" / "states" / "sn-gamestate.pklz").is_file()
    if step == "2":
        return (out / "step2" / "refined_sn-gamestate.pklz").is_file()
    if step == "3":
        return predictions_done(name)
    raise ValueError(step)


def run_step(name: str, step: str, split: str, env: dict) -> None:
    cmd = [
        PYTHON,
        str(GSR_STEP),
        "--clip",
        name,
        "--step",
        step,
        "--split",
        split,
        "--output-root",
        str(OUT_ROOT),
        "--frames-root",
        str(IMAGES),
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def parallel(names: list[str], step: str, split: str, workers: int, env: dict) -> None:
    if not names:
        return
    workers = max(1, min(workers, len(names)))
    fails: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_step, n, step, split, env): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                fut.result()
                print(f"OK {n} step{step}", flush=True)
            except Exception as e:  # noqa: BLE001 — report & continue
                fails.append((n, str(e)))
                print(f"FAIL {n} step{step}: {e}", flush=True)
    if fails:
        raise SystemExit(f"step{step} failed for {len(fails)} clips: {[n for n, _ in fails]}")


def main() -> None:
    global IMAGES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=IMAGES)
    p.add_argument("--split", default="challenge")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if step artifacts exist")
    p.add_argument("--step12-workers", type=int, default=int(os.environ.get("STEP12_WORKERS", "3")))
    p.add_argument("--step3-workers", type=int, default=int(os.environ.get("STEP3_WORKERS", "2")))
    p.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=int(os.environ.get("MAX_PROCESSES_PER_GPU", "1")),
    )
    p.add_argument("--gpu-list", default=os.environ.get("GPU_LIST", "0"))
    p.add_argument(
        "--only-steps",
        default="1,2,3",
        help="comma steps to run, e.g. 2,3 (skips finished step artifacts unless --force)",
    )
    args = p.parse_args()
    only = {s.strip() for s in args.only_steps.split(",") if s.strip()}

    IMAGES = args.images.resolve()
    clips = missing_labels(IMAGES)
    print(f"missing Labels-GameState.json: {len(clips)}")
    for n in clips:
        print(
            f"  {n} s1={step_done(n,'1')} s2={step_done(n,'2')} pred={predictions_done(n)}"
        )
    if args.list_only:
        return

    todo = [n for n in clips if args.force or not predictions_done(n)]
    print(f"to_run={len(todo)} skip_pred_done={len(clips) - len(todo)}")
    if not todo:
        print("nothing to do")
        return

    for n in todo:
        cd = clip_dir(n)
        if not (cd / "img1").is_dir():
            raise SystemExit(f"clip dir missing img1: {cd}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["GPU_LIST"] = args.gpu_list
    env["MAX_PROCESSES_PER_GPU"] = str(args.max_processes_per_gpu)
    env["GSR_PYTHON"] = PYTHON
    env.setdefault("GSR_NUM_CORES", "8")
    env.setdefault("GSR_NUM_THREADS", "4")
    env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_list)

    print(
        f"plan GPU_LIST={env['GPU_LIST']} MAX_PROCESSES_PER_GPU={env['MAX_PROCESSES_PER_GPU']} "
        f"step12_workers={args.step12_workers} step3_workers={args.step3_workers} only={sorted(only)}",
        flush=True,
    )

    def need(step: str) -> list[str]:
        if step not in only:
            return []
        if args.force:
            return list(todo)
        return [n for n in todo if not step_done(n, step)]

    parallel(need("1"), "1", args.split, args.step12_workers, env)
    parallel(need("2"), "2", args.split, args.step12_workers, env)
    parallel(need("3"), "3", args.split, args.step3_workers, env)
    print("all done")


if __name__ == "__main__":
    main()
