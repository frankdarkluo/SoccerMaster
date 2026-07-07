"""Wrap SoccerMaster Steps 1-3 as subprocess calls."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from pipeline.config import GSR_ROOT, REPO_ROOT, PipelineConfig
from pipeline.stage1_inference.cpu_guard import (
    apply_cpu_limits,
    gsr_num_threads,
    hydra_cpu_overrides,
)

log = logging.getLogger(__name__)

_PYTHON = os.environ.get("GSR_PYTHON", sys.executable)


def _sequence_name(config: PipelineConfig) -> str:
    return Path(config.clip_dir).name


def _sequence_id(config: PipelineConfig) -> str:
    """SNGS-148 → 148."""
    match = re.match(r"SNGS-(\d+)", _sequence_name(config), re.IGNORECASE)
    return match.group(1) if match else "001"


def _split_name(config: PipelineConfig) -> str:
    """Infer split from clip_dir parent (test/sn500/train/…), else config.gsr_split."""
    parent = Path(config.clip_dir).parent.name
    if parent in {"train", "valid", "test", "challenge", "sn500"}:
        return parent
    return config.gsr_split


def _gsr_output_base(config: PipelineConfig) -> Path:
    """Pipeline output dir for GSR intermediates (e.g. outputs/SNGS-148)."""
    return Path(config.output_dir).resolve()


def _step1_dir(config: PipelineConfig) -> Path:
    return _gsr_output_base(config) / "step1"


def _step2_dir(config: PipelineConfig) -> Path:
    return _gsr_output_base(config) / "step2"


def _step3_dir(config: PipelineConfig) -> Path:
    return _gsr_output_base(config) / "step3"


def _rel_to_gsr(path: Path) -> str:
    """Path relative to sn-gamestate cwd for Hydra / CLI overrides."""
    return os.path.relpath(path.resolve(), GSR_ROOT.resolve())


def _artifact_ready(path: Path, min_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def _pklz_has_video(pklz: Path, video_id: str) -> bool:
    """True if pklz is a readable archive containing {video_id}.pkl."""
    if not _artifact_ready(pklz, min_bytes=100):
        return False
    try:
        with zipfile.ZipFile(pklz) as z:
            return f"{video_id}.pkl" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def _subprocess_env() -> dict[str, str]:
    """Repo PYTHONPATH + conservative CPU/thread limits for child processes."""
    env = apply_cpu_limits(os.environ.copy())
    repo = str(REPO_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + (os.pathsep + prev if prev else "")
    return env


def _torch_thread_bootstrap(env: dict[str, str]) -> str:
    threads = gsr_num_threads(env)
    interop = max(1, threads // 2)
    return (
        "import torch; "
        f"torch.set_num_threads({threads}); "
        f"torch.set_num_interop_threads({interop})"
    )


def _run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    log.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    if dry_run:
        return

    env = _subprocess_env()

    # PyTorch >= 2.6 defaults torch.load(weights_only=True), which breaks YOLO
    # and other full-pickle GSR checkpoints. Patch before launching tracklab.
    if len(cmd) >= 3 and os.path.basename(cmd[0]) in {"python", "python3"} and cmd[1] == "-m":
        module = cmd[2]
        argv = [module, *cmd[3:]]
        bootstrap = (
            f"{_torch_thread_bootstrap(env)}; "
            "from pipeline.stage1_inference.torch_compat import patch_torch_load; "
            "patch_torch_load(); "
            "import runpy, sys; "
            f"sys.argv = {argv!r}; "
            f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
        )
        subprocess.run([_PYTHON, "-c", bootstrap], check=True, cwd=str(cwd), env=env)
        return

    if cmd and os.path.basename(cmd[0]) in {"python", "python3"}:
        cmd = [_PYTHON, *cmd[1:]]

    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def run_step1(config: PipelineConfig, dry_run: bool = False) -> Path:
    """Run YOLO detection + tracking for one sequence. Returns states directory."""
    split = _split_name(config)
    seq_name = _sequence_name(config)
    seq_id = _sequence_id(config)
    out_dir = _step1_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    states_dir = out_dir / "states"
    pklz = states_dir / "sn-gamestate.pklz"
    if not config.force and _pklz_has_video(pklz, seq_id):
        log.info("Step 1 already done, skipping: %s", pklz)
        return states_dir

    log.info("Running GSR Step 1 on %s/%s → %s", split, seq_name, out_dir)
    cmd = [
        "python", "-m", "tracklab.main",
        "-cn", "gsr_step_1_example",
        f"experiment_subname=step_1_{split}_{seq_name}",
        f"dataset.eval_set={split}",
        f"dataset.start_vid={seq_id}",
        f"dataset.end_vid={seq_id}",
        f"hydra.run.dir={_rel_to_gsr(out_dir)}",
        f"use_rich={os.environ.get('GSR_USE_RICH', 'false')}",
        *hydra_cpu_overrides(),
    ]
    _run(cmd, GSR_ROOT, dry_run=dry_run)
    return states_dir


def run_step2(config: PipelineConfig, dry_run: bool = False) -> Path:
    """Run SAM2 segmentation refinement for one sequence."""
    split = _split_name(config)
    seq_name = _sequence_name(config)
    seq_id = _sequence_id(config)

    sam2_dir = GSR_ROOT.parent / "sam2" / "step_2"
    input_pklz = _step1_dir(config) / "states" / "sn-gamestate.pklz"
    output_dir = _step2_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pklz = output_dir / "refined_sn-gamestate.pklz"
    if not config.force and _pklz_has_video(save_pklz, seq_id):
        log.info("Step 2 already done, skipping: %s", save_pklz)
        return output_dir

    log.info("Running GSR Step 2 (SAM2) on %s/%s → %s", split, seq_name, output_dir)

    infer_cmd = [
        "python", "inference.py",
        "--sam_checkpoint", "../checkpoints/sam2.1_hiera_large.pt",
        "--sam_config", "configs/sam2.1/sam2.1_hiera_l.yaml",
        "--input_pklz", str(input_pklz),
        "--dataset_root", str(GSR_ROOT / "datasets" / "SoccerNetGS"),
        "--output_dir", str(output_dir),
        "--split", split,
        "--fps", str(config.fps),
        "--best_iou_threshold", "0.5",
        "--best_seg_bbox_be_overlapped_ratio_threshold", "0.7",
        "--mask_iou_threshold", "0.6",
        "--seg_mask_be_overlapped_ratio_threshold", "0.6",
        "--max_expansion_ratio", "1.0",
        "--max_width_offset", "30",
        "--max_height_offset", "60",
        "--kernel_size", "10",
        "--fix_duplicate_track_ids",
        "--gpu_list", os.environ.get("GPU_LIST", "0"),
        "--max_processes_per_gpu", os.environ.get("MAX_PROCESSES_PER_GPU", "1"),
        "--video_id_list", seq_id,
    ]
    _run(infer_cmd, sam2_dir, dry_run=dry_run)

    merge_cmd = [
        "python", "merge_pkl.py",
        "--input_pklz", str(input_pklz),
        "--dataset_root", str(GSR_ROOT / "datasets" / "SoccerNetGS"),
        "--output_dir", str(output_dir),
        "--split", split,
        "--fix_duplicate_track_ids",
        "--save_refined_pklz",
        "--save_pklz_path", str(save_pklz),
        "--output_pkl", str(output_dir / "results.pkl"),
        "--include_unmatched_segments",
        "--video_id_list", seq_id,
    ]
    _run(merge_cmd, sam2_dir, dry_run=dry_run)
    return output_dir


def run_step3(config: PipelineConfig, dry_run: bool = False) -> Path:
    """Run calibration + jersey + team for one sequence. Returns final pklz path."""
    split = _split_name(config)
    seq_name = _sequence_name(config)
    seq_id = _sequence_id(config)
    out_dir = _step3_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_pklz = out_dir / "states" / "sn-gamestate.pklz"
    if not config.force and _pklz_has_video(final_pklz, seq_id):
        log.info("Step 3 already done, skipping: %s", final_pklz)
        return final_pklz
    load_file = str((_step2_dir(config) / "refined_sn-gamestate.pklz").resolve())

    log.info("Running GSR Step 3 on %s/%s → %s", split, seq_name, out_dir)
    cmd = [
        "python", "-m", "tracklab.main",
        "-cn", config.step3_config,
        f"experiment_subname=step_3_{split}_{seq_name}",
        f"dataset.eval_set={split}",
        f"dataset.start_vid={seq_id}",
        f"dataset.end_vid={seq_id}",
        f"state.load_file={load_file}",
        f"hydra.run.dir={_rel_to_gsr(out_dir)}",
        f"use_rich={os.environ.get('GSR_USE_RICH', 'false')}",
        *hydra_cpu_overrides(),
    ]
    _run(cmd, GSR_ROOT, dry_run=dry_run)

    if not dry_run:
        from pipeline.stage1_inference.calibration_guard import check_pklz
        report_path = out_dir / "calibration_report.json"
        check_pklz(final_pklz, seq_id, report_path=report_path)

    return final_pklz


def run_full_gsr(config: PipelineConfig, dry_run: bool = False) -> Path:
    """Run all 3 GSR steps for the sequence in config.clip_dir. Returns final pklz."""
    run_step1(config, dry_run=dry_run)
    if not config.skip_sam2:
        run_step2(config, dry_run=dry_run)
    return run_step3(config, dry_run=dry_run)
