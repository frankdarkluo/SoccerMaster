"""Atomic replacement helpers for pipeline artifacts."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def _sibling_temp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, value) -> Path:
    path = Path(path)
    temporary = _sibling_temp(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_copy(source: Path, target: Path) -> Path:
    target = Path(target)
    temporary = _sibling_temp(target)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
