"""Conservative CPU/thread limits for Stage 1 GSR subprocesses.

Shared servers often SIGKILL Stage 1 when tracklab defaults (num_cores=64) spawn
too many workers or BLAS threads compete for RAM.
"""
from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parse_positive_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    raw = (env or os.environ).get(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _cpu_count() -> int:
    return os.cpu_count() or 4


def gsr_num_cores(env: dict[str, str] | None = None) -> int:
    """Hydra num_cores (dataloaders, viz pool, eval parallelism)."""
    default = max(2, min(8, _cpu_count() // 4))
    return _parse_positive_int("GSR_NUM_CORES", default, env=env)


def gsr_num_threads(env: dict[str, str] | None = None) -> int:
    """BLAS / OpenMP threads per process."""
    default = max(1, min(4, _cpu_count() // 4))
    return _parse_positive_int("GSR_NUM_THREADS", default, env=env)


def apply_cpu_limits(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return env with thread caps; does not lower values already set by the user."""
    out = (env or os.environ).copy()
    threads = str(gsr_num_threads(out))
    for var in _THREAD_VARS:
        out.setdefault(var, threads)
    out.setdefault("TOKENIZERS_PARALLELISM", "false")
    return out


def hydra_cpu_overrides(env: dict[str, str] | None = None) -> list[str]:
    """CLI overrides to tame tracklab worker pools (yaml defaults use 32–64 cores)."""
    src = env or os.environ
    cores = gsr_num_cores(src)
    viz_workers = max(1, min(cores, 2))
    return [
        f"num_cores={cores}",
        f"visualization.cfg.num_workers={viz_workers}",
        "modules.reid.cfg.data.workers=0",
    ]
